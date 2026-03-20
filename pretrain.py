"""Pre-training script for TinyGPT on a French text corpus (Victor Hugo).

Pipeline:
  1. Load raw text from datasets/hugo/*.txt
  2. Build a word-level tokenizer from the full corpus
  3. Encode the corpus and build a random-sampling dataset + dataloader
  4. Create the model — or resume from the latest checkpoint
  5. Training loop with periodic checkpoint saves and sample generation
  6. Generate a sample from the final model

Run:  python pretrain.py
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tinygpt import TinyGPT, Tokenizer, generate, get_device
from tinygpt.checkpoint import auto_detect_latest, load_checkpoint, save_checkpoint
from tinygpt.data import TextDatasetSmall

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_DIR = Path("datasets/hugo/")
SNAPSHOT_DIR = Path("snapshots/")
CHECKPOINT_PREFIX = "tinygpt_pretrain"

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
# All hyperparameters are centralised in config.json so that experiments can
# be re-run or compared by editing a single file instead of hunting through
# source code.


def load_config() -> dict:
    """Read config.json from the project root and return the 'pretrain' section."""
    config_path = Path(__file__).parent / "config.json"
    with config_path.open() as f:
        return json.load(f)["pretrain"]


cfg = load_config()

# Context length in WORDS — 64 words ≈ 2-3 sentences.
# Using word tokens (not chars) means this 64-token window represents much
# more semantic content than 64 characters would.
WORD_CONTEXT_LENGTH: int = cfg["context_length"]

EMBED_DIM: int = cfg["embed_dim"]  # bigger embeddings to handle large vocab
NUM_HEADS: int = cfg["num_heads"]
FF_DIM: int = cfg["ff_dim"]  # wider FFN for more capacity
NUM_BLOCKS: int = cfg["num_blocks"]
DROPOUT: float = cfg["dropout"]
MAX_VOCAB: int = cfg["max_vocab"]

BATCH_SIZE: int = cfg["batch_size"]
NUM_SAMPLES: int = cfg["num_samples"]  # random windows drawn per epoch (see TextDatasetSmall)
NB_EPOCHS: int = cfg["epochs"]
LR: float = cfg["learning_rate"]
SAVE_EVERY: int = 3  # save checkpoint + generate sample every N epochs
SEED_TEXT: str = "Il était une fois"


# ---------------------------------------------------------------------------
# Inference-only helper
# ---------------------------------------------------------------------------


def run_inference(snapshot_dir: Path, device: torch.device) -> None:
    """Load the latest checkpoint and generate a sample passage."""
    latest = auto_detect_latest(snapshot_dir, prefix=CHECKPOINT_PREFIX)
    if latest is None:
        print("No checkpoint found in", snapshot_dir)
        return

    model, tokenizer, hparams = load_checkpoint(latest, device)
    context_length: int = hparams["context_length"]

    sample = generate(
        model=model,
        tokenizer=tokenizer,
        seed_text=SEED_TEXT,
        context_length=context_length,
        num_words=200,
        temperature=0.8,
        device=device,
    )
    print("\n--- Generated sample ---")
    print(sample)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# __name__ guard: without this, `from pretrain import TinyGPT` would execute
# the entire training block below.  The guard ensures this code only runs when
# pretrain.py is called directly, not when imported as a module.
if __name__ == "__main__":
    device = get_device()
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Inference-only path: if any checkpoint exists, generate and exit.
    # auto_detect_latest() finds the highest-epoch file automatically —
    # no hardcoded filename needed.
    # ------------------------------------------------------------------
    # If a checkpoint exists, we'll resume training from it (not inference-only).
    # The resume logic is in section 5 below, after model construction.

    # ------------------------------------------------------------------
    # 1. Load raw text
    # ------------------------------------------------------------------
    txt_files = sorted(DATASET_DIR.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {DATASET_DIR}")

    full_text = ""
    for txt_path in txt_files:
        full_text += txt_path.read_text(encoding="utf-8")
        print(f"Loaded {txt_path.name}  ({txt_path.stat().st_size // 1024} KB)")

    print(f"\nTotal characters: {len(full_text):,}")
    print(f"Sample:\n{full_text[:200]}\n")

    # ------------------------------------------------------------------
    # 2. Word-level tokenizer
    #
    # WORD-LEVEL TOKENIZER: moves complexity from sequence length to the
    # vocabulary dimension.
    # Char-level (v2): vocab=126, model learns spelling + grammar + meaning.
    #   Slow to converge.
    # Word-level (v3): vocab=~10K, spelling is "free" (baked into embeddings),
    #   model focuses on word RELATIONSHIPS (grammar, meaning). Faster
    #   convergence, but the embedding table explodes:
    #   126*64=8K params → 10003*128=1.28M params (160x bigger).
    # Real GPT-2 uses BPE (byte-pair encoding) — a middle ground between
    # char and word tokenization.
    # ------------------------------------------------------------------
    tokenizer = Tokenizer.from_corpus(full_text, max_vocab=MAX_VOCAB)
    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")

    # ------------------------------------------------------------------
    # 3. Encode corpus and build dataset
    # ------------------------------------------------------------------
    tokens = re.findall(Tokenizer.PATTERN, full_text, re.UNICODE)
    # Map each token to its ID, falling back to <UNK> for out-of-vocabulary words.
    # We access _word_to_id directly (same as tokenizer.encode but avoids re-running
    # the regex on each already-split token).
    encoded: np.ndarray = np.array(
        [tokenizer._word_to_id.get(t, tokenizer.unk_id) for t in tokens],
        dtype=np.int32,
    )

    print(f"Encoded length: {len(encoded):,} tokens")
    print(f"First 20 tokens: {tokens[:20]}")
    print(f"First 20 IDs:    {encoded[:20].tolist()}\n")

    # TextDatasetSmall draws NUM_SAMPLES random windows per epoch instead of
    # iterating every possible window.  For a 1 M-token corpus with L=64, the
    # exhaustive sliding window gives ~1 M near-duplicate samples per epoch —
    # wasteful and slow.  Random sampling gives a manageable epoch size while
    # still covering the whole corpus over many epochs.
    dataset = TextDatasetSmall(encoded, WORD_CONTEXT_LENGTH, num_samples=NUM_SAMPLES)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # ------------------------------------------------------------------
    # 4. Peek at one batch (educational sanity check)
    # ------------------------------------------------------------------
    x_batch, y_batch = next(iter(dataloader))
    print(f"Input batch shape:  {x_batch.shape}")  # (64, 64)
    print(f"Target batch shape: {y_batch.shape}")  # (64, 64) — shifted by 1
    print(f"First sequence input:  {' '.join(tokenizer._id_to_word[i.item()] for i in x_batch[0][:20])}")
    print(f"First sequence target: {' '.join(tokenizer._id_to_word[i.item()] for i in y_batch[0][:20])}")
    print(f"\nDataset size:      {len(dataset):,}")
    print(f"Batches per epoch: {len(dataloader)}")  # ≈ 312 with NUM_SAMPLES=20000, BATCH_SIZE=64

    # ------------------------------------------------------------------
    # 5. Create model
    #
    # Architecture note — parameter budget at these settings:
    #   Embedding:   vocab_size * embed_dim = 10003 * 128 = 1.28M params
    #   Output head: embed_dim * vocab_size = 128 * 10003 = 1.28M params
    #   These two layers dominate the total count (was 8K params for char-level).
    #   3 TransformerBlocks add relatively few parameters on top.
    # ------------------------------------------------------------------
    model = TinyGPT(
        vocab_size=vocab_size,
        context_length=WORD_CONTEXT_LENGTH,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        num_blocks=NUM_BLOCKS,
        dropout=DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    # LOSS BASELINE: initial random loss ≈ log(vocab_size).
    # Char-level: log(126) ≈ 4.84.  Word-level: log(10003) ≈ 9.21.
    # Don't panic at loss=9.2 on epoch 1 — it's random guessing over more choices.
    # Track PERPLEXITY (e^loss) to compare across vocabularies:
    #   random perplexity ≈ vocab_size.  Perplexity = 100 means the model is
    #   as uncertain as a uniform distribution over 100 words at each step.
    print(f"Loss baseline (random):    {math.log(vocab_size):.2f}  (= log({vocab_size}))")
    print(f"Perplexity baseline:       {vocab_size}  (random guessing)\n")

    # ------------------------------------------------------------------
    # 5b. Resume from checkpoint if available
    #
    # RESUME TRAINING: load last saved weights to continue where we left off.
    # The optimizer is re-created fresh (so Adam momentum/variance state is
    # lost), but for continued training at the same lr this is acceptable.
    # For a precise resume, also save/load optimizer.state_dict().
    # auto_detect_latest() scans the snapshot dir and picks the highest
    # epoch number — no hardcoded filename.
    # ------------------------------------------------------------------
    start_epoch = 0
    resume_path = auto_detect_latest(SNAPSHOT_DIR, prefix=CHECKPOINT_PREFIX)

    if resume_path is not None:
        _, _, hparams = load_checkpoint(resume_path, device)
        # Reload weights into the already-constructed model so we keep our
        # freshly built tokenizer (same vocab, just re-using weights).
        ckpt_raw = torch.load(resume_path, weights_only=False)
        filtered_state = {k: v for k, v in ckpt_raw["model_state_dict"].items() if "causal_mask" not in k}
        model.load_state_dict(filtered_state, strict=False)
        # Infer start epoch from the filename (e.g. tinygpt_pretrain_epoch18.pt → 18)
        match = re.search(r"_epoch(\d+)\.pt$", resume_path.name)
        start_epoch = int(match.group(1)) if match else 0
        print(f"Resumed from {resume_path}  (epoch {start_epoch})\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ------------------------------------------------------------------
    # 6. Training loop
    #
    # The loop is intentionally inline (not a function) — educational scripts
    # benefit from being readable top-to-bottom without jumping between
    # definitions.
    # ------------------------------------------------------------------
    print("Training started...\n")
    for epoch in range(start_epoch, NB_EPOCHS):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for i, (x_b, y_b) in enumerate(dataloader):
            x_b = x_b.to(device)
            y_b = y_b.to(device)

            logits = model(x_b)
            # logits: (B, T, vocab_size) → flatten to (B*T, vocab_size) so
            # cross_entropy can compare every position against its target word.
            loss = F.cross_entropy(logits.view(-1, vocab_size), y_b.view(-1))

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping prevents exploding gradients — especially
            # important in early training when the model is far from convergence.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Progress print every 50 batches so you know it's alive
            if i % 50 == 0:
                print(
                    f"  Epoch {epoch + 1}/{NB_EPOCHS} | Batch {i}/{len(dataloader)} | Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / num_batches
        perplexity = math.exp(avg_loss)
        print(f"\nEpoch {epoch + 1} complete | Avg loss: {avg_loss:.4f} | Perplexity: {perplexity:.1f}")

        # Save checkpoint + generate sample every SAVE_EVERY epochs
        if (epoch + 1) % SAVE_EVERY == 0:
            ckpt_path = SNAPSHOT_DIR / f"{CHECKPOINT_PREFIX}_epoch{epoch + 1}.pt"
            save_checkpoint(
                model,
                tokenizer,
                ckpt_path,
                embed_dim=EMBED_DIM,
                num_heads=NUM_HEADS,
                ff_dim=FF_DIM,
                num_blocks=NUM_BLOCKS,
                context_length=WORD_CONTEXT_LENGTH,
            )

            sample = generate(
                model=model,
                tokenizer=tokenizer,
                seed_text=SEED_TEXT,
                context_length=WORD_CONTEXT_LENGTH,
                num_words=100,
                temperature=0.8,
                device=device,
            )
            print(f"Sample: {sample}\n")

    # ------------------------------------------------------------------
    # 7. Save final model and generate a long sample
    # ------------------------------------------------------------------
    final_path = SNAPSHOT_DIR / f"{CHECKPOINT_PREFIX}_final.pt"
    save_checkpoint(
        model,
        tokenizer,
        final_path,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        num_blocks=NUM_BLOCKS,
        context_length=WORD_CONTEXT_LENGTH,
    )

    sample = generate(
        model=model,
        tokenizer=tokenizer,
        seed_text=SEED_TEXT,
        context_length=WORD_CONTEXT_LENGTH,
        num_words=200,
        temperature=0.8,
        device=device,
    )
    print("\n--- Final generated sample ---")
    print(sample)
