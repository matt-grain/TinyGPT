"""
Interactive CLI chatbot for TinyGPT.

Two modes:
  - "complete" (default): autocomplete mode — type a seed, get Hugo-style continuation
  - "chat": Q&A mode — uses the SFT model with <|user|>/<|assistant|> tokens

Usage:
  uv run python chat.py                          # autocomplete with latest pretrain checkpoint
  uv run python chat.py --mode chat              # Q&A with SFT checkpoint
  uv run python chat.py --checkpoint path.pt     # use a specific checkpoint
  uv run python chat.py --temperature 0.5        # lower = more focused, higher = more creative
  uv run python chat.py --words 200              # max words to generate
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tinygpt.checkpoint import auto_detect_latest, load_checkpoint
from tinygpt.device import get_device
from tinygpt.generate import generate, generate_answer

SNAPSHOTS_DIR = Path("snapshots")
SPECIAL_TOKENS = ["<|user|>", "<|assistant|>", "<|end|>"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TinyGPT interactive CLI")
    parser.add_argument(
        "--mode",
        choices=["complete", "chat"],
        default="complete",
        help="complete = autocomplete, chat = Q&A with SFT model",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="path to a specific .pt checkpoint")
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="sampling temperature (0.1=focused, 1.5=creative)"
    )
    parser.add_argument("--words", type=int, default=100, help="max words to generate")
    return parser.parse_args()


def load_model_for_mode(
    mode: str,
    checkpoint_path: Path | None,
    device: torch.device,
) -> tuple:
    """Load the appropriate model for the selected mode."""
    if checkpoint_path is not None:
        ckpt_path = checkpoint_path
    elif mode == "chat":
        sft_path = SNAPSHOTS_DIR / "tinygpt_sft.pt"
        if sft_path.exists():
            ckpt_path = sft_path
        else:
            print(f"SFT checkpoint not found at {sft_path}. Run sft.py first.")
            raise SystemExit(1)
    else:
        ckpt_path = auto_detect_latest(SNAPSHOTS_DIR)
        if ckpt_path is None:
            print(f"No checkpoint found in {SNAPSHOTS_DIR}. Run pretrain.py first.")
            raise SystemExit(1)

    if mode == "chat":
        # Load with resize to accommodate special tokens
        model, tokenizer, hparams = load_checkpoint(ckpt_path, device)
        # Add special tokens if not already present
        tokenizer.add_special_tokens(SPECIAL_TOKENS)
        return model, tokenizer, hparams
    else:
        model, tokenizer, hparams = load_checkpoint(ckpt_path, device)
        return model, tokenizer, hparams


def run_complete_mode(
    model, tokenizer, hparams: dict, device: torch.device, args: argparse.Namespace
) -> None:
    """Autocomplete mode — type a seed, get Hugo-style continuation."""
    print("\n" + "=" * 60)
    print("TinyGPT — Autocomplete Mode")
    print("Type a French text seed and press Enter.")
    print("Type 'quit' to exit.")
    print("=" * 60)

    while True:
        try:
            seed = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAu revoir!")
            break

        if not seed or seed.lower() == "quit":
            print("Au revoir!")
            break

        text = generate(
            model,
            tokenizer,
            seed,
            context_length=hparams["context_length"],
            num_words=args.words,
            temperature=args.temperature,
            device=device,
        )
        print(f"\n{text}")


def run_chat_mode(model, tokenizer, hparams: dict, device: torch.device, args: argparse.Namespace) -> None:
    """Q&A mode — ask questions about Hugo's works."""
    user_id = tokenizer._word_to_id.get("<|user|>")
    assistant_id = tokenizer._word_to_id.get("<|assistant|>")
    end_id = tokenizer._word_to_id.get("<|end|>")

    if any(tid is None for tid in [user_id, assistant_id, end_id]):
        print("Special tokens not found in vocabulary. Make sure to use an SFT checkpoint.")
        raise SystemExit(1)

    print("\n" + "=" * 60)
    print("TinyGPT — Chat Mode (Hugo Literary Analysis)")
    print("Ask questions about Victor Hugo's works.")
    print("Type 'quit' to exit.")
    print("=" * 60)

    while True:
        try:
            question = input("\nYou > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAu revoir!")
            break

        if not question or question.lower() == "quit":
            print("Au revoir!")
            break

        answer = generate_answer(
            model,
            tokenizer,
            question,
            context_length=hparams["context_length"],
            user_id=user_id,
            assistant_id=assistant_id,
            end_id=end_id,
            max_new_tokens=args.words,
            temperature=args.temperature,
            device=device,
        )
        print(f"\nHugo > {answer}")


def main() -> None:
    args = parse_args()
    device = get_device()

    model, tokenizer, hparams = load_model_for_mode(args.mode, args.checkpoint, device)
    print(f"Model loaded. Vocab: {tokenizer.vocab_size:,}, Context: {hparams['context_length']}")
    print(f"Mode: {args.mode} | Temperature: {args.temperature} | Max words: {args.words}")

    if args.mode == "chat":
        run_chat_mode(model, tokenizer, hparams, device, args)
    else:
        run_complete_mode(model, tokenizer, hparams, device, args)


if __name__ == "__main__":
    main()
