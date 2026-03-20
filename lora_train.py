"""LoRA (Low-Rank Adaptation) — Fine-tune TinyGPT for Balzac using adapters.

THE IDEA:
We have a model trained on Hugo (pre-training + SFT). We want it to also
work for Balzac. Instead of retraining ALL 3M parameters, we:
1. FREEZE all existing weights
2. Inject tiny trainable matrices (LoRA adapters) alongside frozen layers
3. Train only the adapters (~50K params instead of 3M)

WHY IT WORKS (the math):
A full weight update ΔW has shape [128, 128] = 16,384 parameters.
But most of the information can be captured by a LOW-RANK decomposition:
    ΔW ≈ A × B    where A=[128,4], B=[4,128] → only 1,024 params

The rank (4) controls expressiveness vs efficiency:
- A (128→4): "compress — what are the important directions for this task?"
- B (4→128): "expand — how to apply those directions back"
The model LEARNS which directions matter during training.
Same math as PCA/SVD — most info in a big matrix lives in a few components.

KILLER FEATURE — SWAPPABLE ADAPTERS:
Same frozen model + Hugo adapter → explains Hugo
Same frozen model + Balzac adapter → explains Balzac
Same frozen model + Zola adapter → explains Zola
Swap in milliseconds. Each adapter is ~50KB.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data

from tinygpt.checkpoint import auto_detect_latest, load_checkpoint
from tinygpt.data import TextDatasetSmall
from tinygpt.device import get_device
from tinygpt.generate import generate
from tinygpt.lora import apply_lora, save_lora_adapter

SNAPSHOTS_DIR = Path("snapshots")
BALZAC_DIR = Path("datasets/balzac")
ADAPTER_PATH = SNAPSHOTS_DIR / "lora_balzac.pt"

# All hyperparameters are centralized in config.json ("lora" section) so that
# every training script reads from a single source of truth.


def load_config() -> dict:
    """Read config.json and return the 'lora' hyperparameter section."""
    config_path = Path(__file__).parent / "config.json"
    with config_path.open(encoding="utf-8") as f:
        return json.load(f)["lora"]


def load_balzac_corpus(balzac_dir: Path) -> str:
    """Read and concatenate all .txt files found under balzac_dir."""
    text_files = list(balzac_dir.glob("*.txt"))
    if not text_files:
        print(f"WARNING: no .txt files found in {balzac_dir}", file=sys.stderr)
    corpus = ""
    for txt_path in sorted(text_files):
        corpus += txt_path.read_text(encoding="utf-8")
    print(f"Balzac corpus: {len(corpus):,} characters from {len(text_files)} file(s)")
    return corpus


def build_balzac_dataset(
    corpus: str,
    tokenizer_word_to_id: dict[str, int],
    unk_id: int,
    context_length: int,
    num_samples: int,
) -> TextDatasetSmall:
    """Tokenize the Balzac corpus with Hugo's vocabulary and build a dataset.

    Words not in Hugo's vocab become <UNK> — this is a limitation of word-level
    tokenization. BPE would handle Balzac's distinct vocabulary much better
    (Balzac uses legal/financial terms absent from Hugo's romantic corpus).
    """
    import re

    from tinygpt.tokenizer import Tokenizer

    tokens = re.findall(Tokenizer.PATTERN, corpus, re.UNICODE)
    encoded = np.array([tokenizer_word_to_id.get(t, unk_id) for t in tokens], dtype=np.int32)

    unk_count = sum(1 for t in tokens if t not in tokenizer_word_to_id)
    print(f"Balzac tokens:    {len(tokens):,}")
    print(
        f"Tokens as <UNK>:  {unk_count:,} ({100 * unk_count / len(tokens):.1f}%)"
        " — word-level tokenization limitation, BPE would reduce this"
    )

    return TextDatasetSmall(encoded, context_length, num_samples=num_samples)


def train_lora(
    model,
    dataloader,
    vocab_size: int,
    learning_rate: float,
    num_epochs: int,
    device: torch.device | None = None,
) -> None:
    """Run the LoRA training loop.

    Only parameters with requires_grad=True go to the optimizer.
    That's only the LoRA A and B matrices — everything else is frozen.
    Higher LR (1e-3) is safe here because the frozen weights cannot drift;
    only the tiny adapters update, so there is far less risk of catastrophic
    forgetting compared to full fine-tuning.
    """
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
    )

    print(f"\nLoRA Training: {len(dataloader.dataset)} samples, {num_epochs} epochs")
    print("=" * 60)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y_batch.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch + 1:3d} | Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    device = get_device()
    cfg = load_config()

    # ----------------------------------------
    # 1. Load pre-trained Hugo checkpoint
    # ----------------------------------------
    checkpoint_path = auto_detect_latest(SNAPSHOTS_DIR)
    if checkpoint_path is None:
        print(f"ERROR: no checkpoint found in {SNAPSHOTS_DIR}", file=sys.stderr)
        sys.exit(1)

    model, tokenizer, hparams = load_checkpoint(checkpoint_path, device)
    context_length: int = hparams["context_length"]
    vocab_size: int = hparams["vocab_size"]
    print(f"Base model loaded (Hugo pre-trained) from {checkpoint_path}\n")

    # ----------------------------------------
    # 2. Generate BEFORE LoRA (Hugo baseline)
    # ----------------------------------------
    print("=== BEFORE LoRA (Hugo style) ===")
    before_text = generate(model, tokenizer, "La maison était", context_length, device=device)
    print(before_text)
    print()

    # ----------------------------------------
    # 3. Apply LoRA adapters
    # ----------------------------------------
    print("Applying LoRA adapters...")
    model = apply_lora(model, rank=cfg["rank"], alpha=cfg["alpha"])

    # ----------------------------------------
    # 4. Load Balzac texts and build dataset
    # ----------------------------------------
    balzac_corpus = load_balzac_corpus(BALZAC_DIR)
    dataset = build_balzac_dataset(
        balzac_corpus,
        tokenizer._word_to_id,
        tokenizer.unk_id,
        context_length,
        num_samples=cfg["num_samples"],
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)

    # ----------------------------------------
    # 5. Train LoRA (only adapters update!)
    # ----------------------------------------
    train_lora(
        model,
        dataloader,
        vocab_size,
        learning_rate=cfg["learning_rate"],
        num_epochs=cfg["epochs"],
        device=device,
    )

    # ----------------------------------------
    # 6. Save the LoRA adapter
    # ----------------------------------------
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    save_lora_adapter(model, ADAPTER_PATH)

    # ----------------------------------------
    # 7. Generate AFTER LoRA (Balzac-adapted)
    # ----------------------------------------
    print("\n=== AFTER LoRA (Balzac-adapted) ===")
    after_text = generate(model, tokenizer, "La maison était", context_length, device=device)
    print(after_text)
    print()

    # ----------------------------------------
    # 8. Comparison across seeds
    # ----------------------------------------
    print("=== COMPARISON ===")
    seeds = ["Le vieillard", "Paris", "L' argent"]
    for seed in seeds:
        print(f"\nSeed: '{seed}'")
        adapted = generate(model, tokenizer, seed, context_length, num_words=30, device=device)
        print(f"  Balzac LoRA: {adapted}")

    # ----------------------------------------
    # 9. Adapter swapping demo
    # ----------------------------------------
    print("\n" + "=" * 60)
    print("ADAPTER SWAPPING DEMO")
    print("=" * 60)
    print("""
In production, you would:

1. Load base model (3M params, ~12MB)
2. Apply LoRA structure (adds empty A,B matrices)
3. Load adapter file:
   - load_lora_adapter(model, Path("snapshots/lora_balzac.pt"))   → Balzac expert
   - load_lora_adapter(model, Path("snapshots/lora_zola.pt"))      → Zola expert
   - load_lora_adapter(model, Path("snapshots/lora_hugo_sft.pt"))  → Hugo expert

Same model, different personality. Swap in <1ms.
Each adapter: ~50KB. Base model: ~12MB.
Serve 100 specialized models from 12MB + 100×50KB = 17MB total.
Without LoRA: 100 × 12MB = 1.2GB.
""")
