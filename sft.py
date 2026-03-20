"""
SFT (Supervised Fine-Tuning) on top of pre-trained TinyGPT.

WHAT THIS DOES:
Pre-training taught the model to predict next words in raw Victor Hugo text.
SFT teaches it to follow a Q&A format: answer questions about Hugo's novels.

WHAT CHANGES vs PRE-TRAINING:
- Same model architecture (TinyGPT unchanged)
- Same objective (next-token prediction, cross-entropy loss)
- Different DATA: structured <|user|> question <|assistant|> answer pairs
- Loss is MASKED on the prompt portion — we only train on the answer tokens.
  Why? We don't want the model to learn to generate questions, only answers.

WHAT THE MODEL LEARNS:
- Turn-taking format: "after <|assistant|>, produce an answer"
- Domain vocabulary associations (métaphore, Valjean, justice → response patterns)
- The shape of an analytical response
- NOT actual literary understanding (3M params is too small for that)

THIS IS THE SAME PROCESS used to turn GPT-4 base into ChatGPT.
The only differences are scale (billions vs millions of params) and
data quality (thousands of expert-written pairs vs our handful).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data

from tinygpt.checkpoint import (
    auto_detect_latest,
    load_checkpoint,
    load_checkpoint_with_resize,
    save_checkpoint,
)
from tinygpt.data import SFTDataset
from tinygpt.device import get_device
from tinygpt.generate import generate_answer

# ---------------------------------------------------------------------------
# Configuration — config.json centralizes all hyperparameters so that
# pretrain.py, sft.py, and dpo.py share a single source of truth.
# ---------------------------------------------------------------------------


def load_config() -> dict:  # type: ignore[type-arg]
    return json.loads(Path("config.json").read_text(encoding="utf-8"))["sft"]


SNAPSHOT_DIR = Path("snapshots")
SFT_OUTPUT_PATH = SNAPSHOT_DIR / "tinygpt_sft.pt"


def main() -> None:
    device = get_device()

    config = load_config()
    NB_EPOCHS: int = config["epochs"]
    LEARNING_RATE: float = config["learning_rate"]
    BATCH_SIZE: int = config["batch_size"]
    SPECIAL_TOKENS: list[str] = config["special_tokens"]

    QA_PAIRS: list[dict[str, str]] = json.loads(
        Path("datasets/training/sft_qa_pairs.json").read_text(encoding="utf-8")
    )

    # -----------------------------------------------------------------------
    # 1. Locate pre-trained checkpoint
    # -----------------------------------------------------------------------
    checkpoint_path = auto_detect_latest(SNAPSHOT_DIR)
    if checkpoint_path is None:
        raise FileNotFoundError(f"No pre-trained checkpoint found in {SNAPSHOT_DIR}. Run pretrain.py first.")
    print(f"Using pre-trained checkpoint: {checkpoint_path}")

    # -----------------------------------------------------------------------
    # 2. Determine new vocab size, then load + resize
    # -----------------------------------------------------------------------
    # Strategy: load the checkpoint at its original vocab size just to read the
    # tokenizer, add special tokens to that tokenizer to count the new size,
    # then reload with load_checkpoint_with_resize so the model is built at the
    # right size from the start. The first load is lightweight (no training).
    _, probe_tokenizer, _ = load_checkpoint(checkpoint_path, device)
    probe_tokenizer.add_special_tokens(SPECIAL_TOKENS)
    new_vocab_size = probe_tokenizer.vocab_size
    original_vocab_size = new_vocab_size - len(SPECIAL_TOKENS)
    del probe_tokenizer

    # load_checkpoint_with_resize creates a TinyGPT at new_vocab_size and copies
    # all pre-trained weights. New special-token rows stay randomly initialised.
    model, tokenizer, hparams = load_checkpoint_with_resize(checkpoint_path, new_vocab_size, device)

    # Add the special tokens to the tokenizer vocabulary so encode() and
    # decode() know about <|user|>, <|assistant|>, <|end|>.
    tokenizer.add_special_tokens(SPECIAL_TOKENS)

    user_id = tokenizer._word_to_id["<|user|>"]
    assistant_id = tokenizer._word_to_id["<|assistant|>"]
    end_id = tokenizer._word_to_id["<|end|>"]

    context_length: int = hparams["context_length"]

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    print(f"Vocabulary: {original_vocab_size} → {new_vocab_size} (+{len(SPECIAL_TOKENS)} special tokens)")

    # -----------------------------------------------------------------------
    # 3. Build SFT dataset
    # -----------------------------------------------------------------------
    dataset = SFTDataset(
        qa_pairs=QA_PAIRS,
        max_len=context_length,
        encode_fn=tokenizer.encode,
        user_id=user_id,
        assistant_id=assistant_id,
        end_id=end_id,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # -----------------------------------------------------------------------
    # 4. Training loop
    # -----------------------------------------------------------------------
    # Lower learning rate than pre-training! Pre-trained weights are already good.
    # Too high LR would destroy what the model learned about French.
    # This is called "catastrophic forgetting" — a key SFT risk.
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\nSFT Training: {len(dataset)} samples, {len(dataloader)} batches/epoch")
    print("=" * 60)

    for epoch in range(NB_EPOCHS):
        model.train()
        total_loss = 0.0
        total_tokens = 0.0

        for batch in dataloader:
            input_ids: torch.Tensor = batch["input_ids"].to(device)  # [B, seq_len]
            loss_mask: torch.Tensor = batch["loss_mask"].to(device)  # [B, seq_len]

            # Targets = input shifted by 1 (next token prediction, same as pre-training)
            x = input_ids[:, :-1]  # [B, seq_len-1]
            y = input_ids[:, 1:]  # [B, seq_len-1]
            mask = loss_mask[:, 1:]  # [B, seq_len-1] — aligned with targets

            logits = model(x)  # [B, seq_len-1, vocab_size]

            # Compute loss ONLY on answer tokens (where mask=1).
            # Without masking: model learns to generate questions too (bad!).
            # With masking: model only learns "given this question, produce this answer".
            # <|end|> is included in the loss mask so the model learns to stop.
            loss_per_token = F.cross_entropy(
                logits.reshape(-1, new_vocab_size),
                y.reshape(-1),
                reduction="none",  # don't average yet — we have a custom mask
            )
            loss_per_token = loss_per_token.reshape(y.shape)  # [B, seq_len-1]
            masked_loss = (loss_per_token * mask).sum()
            num_answer_tokens = mask.sum()

            if num_answer_tokens == 0:
                continue

            loss = masked_loss / num_answer_tokens

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += masked_loss.item()
            total_tokens += num_answer_tokens.item()

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:3d} | Loss: {avg_loss:.4f} | Perplexity: {np.exp(avg_loss):.1f}")

    # -----------------------------------------------------------------------
    # 5. Save SFT checkpoint
    # -----------------------------------------------------------------------
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        model,
        tokenizer,
        SFT_OUTPUT_PATH,
        embed_dim=hparams["embed_dim"],
        num_heads=hparams["num_heads"],
        ff_dim=hparams["ff_dim"],
        num_blocks=hparams["num_blocks"],
        context_length=context_length,
    )

    # -----------------------------------------------------------------------
    # 6. Evaluation
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TESTING FINE-TUNED MODEL")
    print("=" * 60)

    # Test on training questions — should work well (memorisation check)
    print("\n--- Questions from training data ---")
    for qa in QA_PAIRS[:3]:
        print(f"\nQ: {qa['question']}")
        print(f"Expected: {qa['answer']}")
        answer = generate_answer(
            model=model,
            tokenizer=tokenizer,
            question=qa["question"],
            context_length=context_length,
            user_id=user_id,
            assistant_id=assistant_id,
            end_id=end_id,
            device=device,
        )
        print(f"Model:    {answer}")

    # Test on new questions — the real test: can the model generalise?
    print("\n--- New questions (generalisation test) ---")
    new_questions = [
        "Qui est Cosette ?",
        "Que symbolise la barricade ?",
        "Pourquoi Valjean change de nom ?",
    ]
    for question in new_questions:
        print(f"\nQ: {question}")
        answer = generate_answer(
            model=model,
            tokenizer=tokenizer,
            question=question,
            context_length=context_length,
            user_id=user_id,
            assistant_id=assistant_id,
            end_id=end_id,
            device=device,
        )
        print(f"Model: {answer}")


if __name__ == "__main__":
    main()
