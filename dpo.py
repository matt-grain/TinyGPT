"""
DPO (Direct Preference Optimization) on top of SFT-tuned TinyGPT.

THE FULL PIPELINE:
1. Pre-training (gpt2_v3.py): Learn French from raw Hugo text → "autocomplete"
2. SFT (sft_v3.py): Learn Q&A format from structured pairs → "follows instructions"
3. DPO (this file): Learn to prefer good answers over bad ones → "gives quality answers"

WHAT IS DPO:
- Simpler alternative to RLHF (no reward model, no PPO)
- Takes preference pairs: (question, good_answer, bad_answer)
- Trains the model to increase P(good) and decrease P(bad)
- Uses a FROZEN reference model to prevent drifting too far

THE DPO LOSS (the math that makes it work):
    loss = -log(sigmoid(β × (log_ratio_good - log_ratio_bad)))

    where log_ratio = log(P_model(answer)) - log(P_reference(answer))

    Intuition: "prefer good over bad, but don't drift from what you already know"
    β controls how aggressively to optimize preferences (higher = more aggressive)

WHY A REFERENCE MODEL:
Without it, the model could "cheat" — crank good answer probability to 99.99%
and lose all nuance. The reference is the anchor: "stay close to your SFT self."
DPO is ADJUSTMENT, not relearning.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from tinygpt.checkpoint import auto_detect_latest, load_checkpoint_with_resize
from tinygpt.device import get_device
from tinygpt.generate import generate_answer

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def load_config() -> dict:
    """Load the 'dpo' section from config.json at the project root."""
    config_path = Path(__file__).parent / "config.json"
    return json.loads(config_path.read_text(encoding="utf-8"))["dpo"]


# ---------------------------------------------------------------------------
# Preference data
# ---------------------------------------------------------------------------
# Each pair has: question, chosen (good answer), rejected (bad answer).
# "chosen" = what we want the model to prefer.
# "rejected" = what we want the model to avoid.
#
# In production: generate multiple answers, have humans rank them.
# Here: chosen = expert-written, rejected = typical model failures.
PREFERENCE_PAIRS: list[dict[str, str]] = json.loads(
    Path("datasets/training/dpo_preference_pairs.json").read_text(encoding="utf-8")
)

# ---------------------------------------------------------------------------
# Log-probability scoring (core DPO building block)
# ---------------------------------------------------------------------------


def get_sequence_log_prob(
    model: torch.nn.Module,
    prompt_ids: list[int],
    answer_ids: list[int],
    context_length: int,
) -> torch.Tensor:
    """Compute log P(answer | prompt) for a given model.

    This is the core building block of DPO. We need to compare:
    - How likely the ACTIVE model thinks the answer is
    - How likely the REFERENCE model thinks the answer is

    The difference tells us: "has the model drifted from its SFT baseline?"

    WHY LOG PROBABILITIES instead of raw probabilities?
    Raw P(sentence) = P(word1) × P(word2) × ... × P(word50)
    Example: 0.15 × 0.08 × 0.25 × 0.30 × 0.02 = 0.0000018
    With 50 tokens this becomes ~0.00000000000000... → UNDERFLOW (computer sees 0.0)

    Fix: log turns multiplication into addition:
        log(A × B × C) = log(A) + log(B) + log(C)

    Example:  log(0.15) + log(0.08) + log(0.25) + log(0.30) + log(0.02)
            = -1.90    + -2.53    + -1.39     + -1.20     + -3.91
            = -10.93   ← perfectly normal number, more negative = less likely
    """
    full_ids = prompt_ids + answer_ids
    if len(full_ids) > context_length:
        full_ids = full_ids[:context_length]

    x = torch.tensor(full_ids[:-1]).unsqueeze(0)  # input (all but last)
    y = torch.tensor(full_ids[1:]).unsqueeze(0)  # target (all but first)

    # TEACHER FORCING: we don't ask the model to generate — we force-feed the
    # answer and ask "what probability did you assign to each word?"
    # The model outputs probabilities for ALL vocab words at each position.
    # Same forward pass as generation, but instead of sampling one token,
    # we READ the probability of the specific token that was in the answer.
    #
    # Model's job is always the same: "here are my opinions about all 10K words."
    # What we DO with those opinions changes:
    #   - Generation: sample one token (temperature, top-k, etc.)
    #   - Training: cross-entropy loss against target
    #   - DPO: read the probability of a specific token (scoring)
    with torch.set_grad_enabled(model.training):
        logits = model(x)  # [1, seq_len-1, vocab_size] → 10K probs per position

    # log_softmax = log(softmax(x)) but computed in a numerically stable way.
    # Each value is negative: more negative = less likely.
    log_probs = F.log_softmax(logits, dim=-1)  # [1, seq_len-1, vocab_size]

    # Gather the log prob of each actual target token.
    # From the full vocab distribution, pick only the log prob of the token
    # that was actually in the answer. Like looking up each word's score.
    token_log_probs = log_probs[0].gather(1, y[0].unsqueeze(1)).squeeze(1)

    # Only sum over the ANSWER portion (skip prompt tokens).
    # Why skip the prompt? Both "chosen" and "rejected" share the SAME prompt
    # (<|user|> question <|assistant|>). Including it would add identical noise
    # to both scores, drowning out the actual difference between good/bad answers.
    # Like grading two essays but including the exam question in the grade.
    answer_start = len(prompt_ids) - 1  # -1 because of next-token shift
    return token_log_probs[answer_start:].sum()


# ---------------------------------------------------------------------------
# DPO loss
# ---------------------------------------------------------------------------


def dpo_loss(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    prompt_ids: list[int],
    chosen_ids: list[int],
    rejected_ids: list[int],
    context_length: int,
    beta: float = 0.1,
) -> tuple[torch.Tensor, float]:
    """DPO loss for one preference pair.

    The math:
        log_ratio_chosen  = log P_model(chosen)  - log P_ref(chosen)
        log_ratio_rejected = log P_model(rejected) - log P_ref(rejected)
        loss = -log(sigmoid(β × (log_ratio_chosen - log_ratio_rejected)))

    Concrete example for "Qui est Valjean?":
        Reference (frozen SFT):
            log P_ref(chosen="ancien forçat condamné...")  = -9.0
            log P_ref(rejected="il fait des choses...")    = -6.5  ← ref prefers bad!

        Active model (during training):
            log P_model(chosen)  = -8.2    log_ratio_chosen  = -8.2 - (-9.0) = +0.8
            log P_model(rejected) = -6.1   log_ratio_rejected = -6.1 - (-6.5) = +0.4

        Loss pushes for: log_ratio_chosen > log_ratio_rejected
        → "increase chosen MORE than rejected, relative to where you started"

        After training:
            log P_model(chosen)  = -7.0    (went up a lot)
            log P_model(rejected) = -6.8   (barely moved)
            → model now PREFERS the good answer!

    β controls strength: high β = aggressive optimization, risk of overfitting.
    sigmoid squashes to [0,1], -log makes it a proper loss (0 = perfect).
    """
    model.train()
    log_prob_chosen = get_sequence_log_prob(
        model, prompt_ids, chosen_ids, context_length
    )
    log_prob_rejected = get_sequence_log_prob(
        model, prompt_ids, rejected_ids, context_length
    )

    with torch.no_grad():
        ref_log_prob_chosen = get_sequence_log_prob(
            ref_model, prompt_ids, chosen_ids, context_length
        )
        ref_log_prob_rejected = get_sequence_log_prob(
            ref_model, prompt_ids, rejected_ids, context_length
        )

    log_ratio_chosen = log_prob_chosen - ref_log_prob_chosen
    log_ratio_rejected = log_prob_rejected - ref_log_prob_rejected

    loss = -F.logsigmoid(beta * (log_ratio_chosen - log_ratio_rejected))

    # Reward margin > 0 means model prefers chosen over rejected.
    # Reward margin growing over epochs means preferences are being learned.
    chosen_reward = beta * log_ratio_chosen.item()
    rejected_reward = beta * log_ratio_rejected.item()
    reward_margin = chosen_reward - rejected_reward

    return loss, reward_margin


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    device = get_device()
    config = load_config()

    # -----------------------------------------------------------------------
    # 1. Load SFT checkpoint and add special tokens
    # -----------------------------------------------------------------------
    # We start from the SFT checkpoint, not the pre-trained one.
    # DPO refines what SFT learned — it's the final polish.
    snapshots_dir = Path(__file__).parent / "snapshots"
    sft_path = snapshots_dir / "tinygpt_sft.pt"

    if not sft_path.exists():
        detected = auto_detect_latest(snapshots_dir, prefix="tinygpt")
        if detected is None:
            raise FileNotFoundError(
                f"No SFT checkpoint found at {sft_path} and no epoch checkpoints in {snapshots_dir}"
            )
        sft_path = detected
        print(f"tinygpt_sft.pt not found — using latest checkpoint: {sft_path.name}")

    special_tokens: list[str] = config["special_tokens"]
    new_vocab_size = None  # determined after loading to know base vocab size

    # Peek at the base vocab size to compute new_vocab_size before loading.
    raw_ckpt: dict = torch.load(sft_path, weights_only=False)
    base_vocab_size: int = raw_ckpt["vocab_size"]
    # Count only tokens not already present in the checkpoint vocab.
    base_word_to_id: dict[str, int] = raw_ckpt["word_to_id"]
    tokens_to_add = [t for t in special_tokens if t not in base_word_to_id]
    new_vocab_size = base_vocab_size + len(tokens_to_add)
    del raw_ckpt  # free memory before the proper load

    model, tokenizer, hparams = load_checkpoint_with_resize(
        sft_path, new_vocab_size, device
    )
    context_length: int = hparams["context_length"]

    tokenizer.add_special_tokens(special_tokens)

    user_id: int = tokenizer._word_to_id["<|user|>"]
    assistant_id: int = tokenizer._word_to_id["<|assistant|>"]
    end_id: int = tokenizer._word_to_id["<|end|>"]

    # -----------------------------------------------------------------------
    # 2. Create frozen reference model
    # -----------------------------------------------------------------------
    # Why deepcopy? `ref_model = model` would create a second NAME pointing to the
    # SAME object — modifying one would modify both. deepcopy creates a truly
    # independent clone with its own weights.
    # eval() = inference mode (no dropout). requires_grad=False = PyTorch won't
    # compute gradients → saves memory and guarantees the reference never moves.
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    print(
        f"Model loaded. Vocab: {tokenizer.vocab_size}, "
        f"Params: {sum(p.numel() for p in model.parameters()):,}"
    )
    print("Reference model frozen (identical copy, will NOT be updated)")

    # -----------------------------------------------------------------------
    # 3. DPO training loop
    # -----------------------------------------------------------------------
    # Very low learning rate — DPO is fine adjustment, not relearning.
    # β = 0.1 is standard — controls preference strength.
    learning_rate: float = config["learning_rate"]
    beta: float = config["beta"]
    nb_epochs: int = config["epochs"]

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"\nDPO Training: {len(PREFERENCE_PAIRS)} preference pairs")
    print(f"β={beta}, lr={learning_rate}")
    print("=" * 60)

    for epoch in range(nb_epochs):
        total_loss = 0.0
        total_margin = 0.0

        for pair in PREFERENCE_PAIRS:
            prompt_ids = [user_id] + tokenizer.encode(pair["question"]) + [assistant_id]
            chosen_ids = tokenizer.encode(pair["chosen"]) + [end_id]
            rejected_ids = tokenizer.encode(pair["rejected"]) + [end_id]

            loss, margin = dpo_loss(
                model,
                ref_model,
                prompt_ids,
                chosen_ids,
                rejected_ids,
                context_length,
                beta,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_margin += margin

        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(PREFERENCE_PAIRS)
            avg_margin = total_margin / len(PREFERENCE_PAIRS)
            # Reward margin > 0 means model prefers chosen over rejected. Good!
            # Reward margin growing means preferences are being learned.
            print(
                f"Epoch {epoch + 1:3d} | Loss: {avg_loss:.4f} | Reward margin: {avg_margin:+.4f}"
            )

    # -----------------------------------------------------------------------
    # 4. Compare SFT (reference) vs DPO (tuned) answers
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("COMPARISON: Reference (SFT) vs DPO-tuned")
    print("=" * 60)

    test_questions = [
        "Qui est Jean Valjean ?",
        "Que symbolise la cathédrale Notre - Dame ?",
        "Que représente Javert dans Les Misérables ?",
        # New question — generalization test
        "Qui est Cosette ?",
    ]

    for question in test_questions:
        print(f"\nQ: {question}")
        ref_answer = generate_answer(
            ref_model,
            tokenizer,
            question,
            context_length,
            user_id,
            assistant_id,
            end_id,
            device=device,
        )
        dpo_answer = generate_answer(
            model,
            tokenizer,
            question,
            context_length,
            user_id,
            assistant_id,
            end_id,
            device=device,
        )
        print(f"  SFT (reference): {ref_answer}")
        print(f"  DPO (tuned):     {dpo_answer}")


if __name__ == "__main__":
    main()
