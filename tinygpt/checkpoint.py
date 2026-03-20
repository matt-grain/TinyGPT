"""Checkpoint save/load logic for TinyGPT.

All three training scripts (pretrain, SFT, DPO) use the same checkpoint format.
Centralising here prevents the three copies from drifting out of sync.

Checkpoint format (torch.save dict):
    model_state_dict  – model weights
    word_to_id        – str → int vocab mapping
    id_to_word        – int → str vocab mapping (keys stored as int in .pt files,
                        torch.load with weights_only=False preserves that)
    embed_dim, num_heads, ff_dim, num_blocks, context_length – constructor args
    vocab_size        – derived from word_to_id but stored explicitly for clarity
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch

from tinygpt.model import TinyGPT
from tinygpt.tokenizer import Tokenizer

# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def auto_detect_latest(snapshot_dir: Path, prefix: str = "tinygpt") -> Path | None:
    """Return the checkpoint with the highest epoch number in *snapshot_dir*.

    Scans for files matching ``<prefix>*_epoch<N>.pt`` and returns the one with
    the largest N.  Falls back to any ``<prefix>*.pt`` file (e.g. *_final.pt)
    if no epoch-numbered files are found.
    Returns None when the directory is empty or no file matches.

    Example filenames: tinygpt_pretrain_epoch18.pt, tinygpt_pretrain_final.pt
    """
    pattern = re.compile(rf"^{re.escape(prefix)}.*_epoch(\d+)\.pt$")
    best_path: Path | None = None
    best_epoch = -1

    for candidate in snapshot_dir.glob(f"{prefix}*.pt"):
        match = pattern.match(candidate.name)
        if match:
            epoch = int(match.group(1))
            if epoch > best_epoch:
                best_epoch = epoch
                best_path = candidate

    # Fallback: if no epoch-numbered file, pick any matching .pt file
    # (e.g. tinygpt_pretrain_final.pt)
    if best_path is None:
        fallbacks = sorted(snapshot_dir.glob(f"{prefix}*.pt"))
        if fallbacks:
            best_path = fallbacks[-1]  # most recent by name

    return best_path


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_checkpoint(
    model: TinyGPT,
    tokenizer: Tokenizer,
    path: Path,
    **hparams: Any,
) -> None:
    """Save model weights, vocabulary, and hyperparameters to *path*.

    The caller supplies hyperparameters as keyword arguments; the required set is:
        embed_dim, num_heads, ff_dim, num_blocks, context_length

    vocab_size is derived automatically from the tokenizer.

    Usage::

        save_checkpoint(
            model, tokenizer,
            Path("snapshots/tinygpt_epoch30.pt"),
            embed_dim=128, num_heads=4, ff_dim=512,
            num_blocks=3, context_length=64,
        )
    """
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "word_to_id": tokenizer._word_to_id,
            "id_to_word": tokenizer._id_to_word,
            "vocab_size": tokenizer.vocab_size,
            **hparams,
        },
        path,
    )
    print(f"Checkpoint saved to {path}")


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------


def _filter_causal_mask(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove causal_mask buffers from a raw state dict.

    causal_mask is a registered buffer (not a learned parameter) that is
    recomputed during TransformerBlock.__init__ via
    nn.Transformer.generate_square_subsequent_mask(context_length).

    We skip it during loading because:
    1. It is always correct after __init__ — copying from the file adds nothing.
    2. If the context_length changed between checkpoints (e.g. v3 used 128,
       SFT uses 64) the shapes would mismatch and strict=True would raise.
    We load with strict=False so the missing keys (causal_mask) are silently
    accepted — the freshly-initialised buffers stay in place.
    """
    return {k: v for k, v in state_dict.items() if "causal_mask" not in k}


def _build_tokenizer(checkpoint: dict[str, Any]) -> Tokenizer:
    """Reconstruct a Tokenizer from checkpoint vocabulary mappings.

    id_to_word keys may be stored as integers by torch.save; convert them back
    to int explicitly to guarantee the dict[int, str] contract.
    """
    word_to_id: dict[str, int] = checkpoint["word_to_id"]
    id_to_word: dict[int, str] = {int(k): v for k, v in checkpoint["id_to_word"].items()}
    return Tokenizer(word_to_id, id_to_word)


def _hparams_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Extract the canonical hyperparameter dict from a raw checkpoint."""
    return {
        "embed_dim": checkpoint["embed_dim"],
        "num_heads": checkpoint["num_heads"],
        "ff_dim": checkpoint["ff_dim"],
        "num_blocks": checkpoint["num_blocks"],
        "context_length": checkpoint["context_length"],
        "vocab_size": checkpoint["vocab_size"],
    }


# ---------------------------------------------------------------------------
# Load — exact vocab
# ---------------------------------------------------------------------------


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[TinyGPT, Tokenizer, dict[str, Any]]:
    """Load a TinyGPT checkpoint and return (model, tokenizer, hparams).

    The model is constructed from the hyperparameters stored in the file, so
    the caller does not need to know the architecture in advance.  This is the
    standard load path for generation and continued pre-training.

    Args:
        path:   Path to the .pt checkpoint file.
        device: Target device (e.g. torch.device("cpu") or "cuda").

    Returns:
        model     – TinyGPT instance with weights loaded, moved to *device*.
        tokenizer – Tokenizer reconstructed from the saved vocabulary.
        hparams   – Dict of embed_dim, num_heads, ff_dim, num_blocks,
                    context_length, vocab_size — useful for logging or resuming.
    """
    checkpoint: dict[str, Any] = torch.load(path, weights_only=False, map_location=device)
    hparams = _hparams_from_checkpoint(checkpoint)

    model = TinyGPT(
        vocab_size=hparams["vocab_size"],
        context_length=hparams["context_length"],
        embed_dim=hparams["embed_dim"],
        num_heads=hparams["num_heads"],
        ff_dim=hparams["ff_dim"],
        num_blocks=hparams["num_blocks"],
    )

    # strict=False: causal_mask keys are absent from filtered_state (intentionally).
    # All learned parameters ARE present, so strict=False only forgives the buffers.
    filtered_state = _filter_causal_mask(checkpoint["model_state_dict"])
    model.load_state_dict(filtered_state, strict=False)

    model.to(device)
    print(f"Checkpoint loaded from {path}")
    return model, _build_tokenizer(checkpoint), hparams


# ---------------------------------------------------------------------------
# Load — with embedding resize (SFT / DPO)
# ---------------------------------------------------------------------------


def load_checkpoint_with_resize(
    path: Path,
    new_vocab_size: int,
    device: torch.device,
) -> tuple[TinyGPT, Tokenizer, dict[str, Any]]:
    """Load a checkpoint and resize the embedding/head layers to *new_vocab_size*.

    Used by SFT and DPO after calling tokenizer.add_special_tokens(), which
    grows the vocabulary by a few tokens (e.g. <|user|>, <|assistant|>, <|end|>).

    Weight copying strategy:
    - Layers whose shape is unchanged: copy directly.
    - Embedding (token_emb.weight) and output head (head.weight / head.bias):
      copy the rows/columns that fit; the new rows for the added special tokens
      keep their random initialisation from TinyGPT.__init__.

    Why start new token embeddings random?
      The model has never seen these tokens, so there is no meaningful pre-trained
      representation to copy.  Random init is the correct starting point; the
      fine-tuning training loop will push them to useful positions in embedding space.

    Args:
        path:          Path to the pre-trained .pt checkpoint.
        new_vocab_size: Target vocab size (original + number of special tokens added).
        device:        Target device.

    Returns:
        model     – TinyGPT resized to new_vocab_size, pre-trained weights copied in.
        tokenizer – Tokenizer from the checkpoint (caller should call
                    add_special_tokens() on it afterwards).
        hparams   – Hyperparameter dict reflecting the ORIGINAL checkpoint architecture
                    (vocab_size is the original value, not new_vocab_size).
    """
    checkpoint: dict[str, Any] = torch.load(path, weights_only=False, map_location=device)
    hparams = _hparams_from_checkpoint(checkpoint)

    model = TinyGPT(
        vocab_size=new_vocab_size,
        context_length=hparams["context_length"],
        embed_dim=hparams["embed_dim"],
        num_heads=hparams["num_heads"],
        ff_dim=hparams["ff_dim"],
        num_blocks=hparams["num_blocks"],
    )

    pretrained_state = checkpoint["model_state_dict"]
    model_state = model.state_dict()

    for key, old_tensor in pretrained_state.items():
        if "causal_mask" in key:
            # Recomputed at init — never copy from file (see _filter_causal_mask).
            continue

        new_tensor = model_state[key]

        if old_tensor.shape == new_tensor.shape:
            # Identical shape: attention weights, LayerNorm, FFN — copy directly.
            model_state[key] = old_tensor
        elif old_tensor.dim() == 2:
            # 2-D mismatch: token_emb.weight [vocab, embed] or head.weight [vocab, embed].
            # Copy the rows/cols that exist in both; new rows stay random.
            min_rows = min(old_tensor.shape[0], new_tensor.shape[0])
            min_cols = min(old_tensor.shape[1], new_tensor.shape[1])
            model_state[key][:min_rows, :min_cols] = old_tensor[:min_rows, :min_cols]
        elif old_tensor.dim() == 1:
            # 1-D mismatch: head.bias [vocab].
            min_len = min(old_tensor.shape[0], new_tensor.shape[0])
            model_state[key][:min_len] = old_tensor[:min_len]

    model.load_state_dict(model_state)
    model.to(device)
    print(
        f"Checkpoint loaded from {path} with vocab resized "
        f"{hparams['vocab_size']} → {new_vocab_size}. "
        "New token embeddings initialised randomly."
    )
    return model, _build_tokenizer(checkpoint), hparams
