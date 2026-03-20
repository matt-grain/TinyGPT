"""
TinyGPT — A minimal GPT-2 implementation for learning the full LLM pipeline.

Built as a teaching project: pre-training → SFT → DPO → LoRA.
Every module contains detailed comments explaining WHY things work.
"""

from __future__ import annotations

from tinygpt.model import TinyGPT, TransformerBlock
from tinygpt.tokenizer import Tokenizer
from tinygpt.generate import generate, generate_answer
from tinygpt.device import get_device

__all__ = [
    "TinyGPT",
    "TransformerBlock",
    "Tokenizer",
    "generate",
    "generate_answer",
    "get_device",
]
