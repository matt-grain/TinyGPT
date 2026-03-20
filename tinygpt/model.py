"""Model architecture classes for TinyGPT.

TransformerBlock implements a single decoder block with causal self-attention,
feed-forward network, and post-norm residual connections. TinyGPT stacks N such
blocks between token/positional embeddings and a linear projection head.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        context_length: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # SELF-ATTENTION: Q, K, V all come from the same x (same sentence).
        # Inside this layer, 3 learned matrices (W_q, W_k, W_v) project x into:
        #   Q = x @ W_q  → "what am I looking for?"
        #   K = x @ W_k  → "what do I advertise as matchable?"
        #   V = x @ W_v  → "what information do I carry?"
        # Attention: softmax(Q·K^T / sqrt(d)) × V → context-enriched output.
        # If Q came from a DIFFERENT source than K,V → cross-attention (translation).
        # See attention_qkv_explained.md for full worked example.
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, embed_dim)
        )
        # LayerNorm (not BatchNorm!) — normalizes across the FEATURE dimension per token.
        # Why not BatchNorm? (1) Variable sequence lengths: position 50 in different sentences
        # are different words, unlike pixel (14,14) which is always "center of image".
        # (2) Causal mask: BatchNorm would leak statistical info across sequences.
        # LayerNorm asks "are YOUR features balanced?" per token, independently.
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        # CAUSAL MASK — what makes this model AUTOREGRESSIVE (GPT-style).
        # Creates a triangular matrix: token 3 can attend to [1,2,3] but not [4,5,...].
        # Without it → BERT (bidirectional, sees everything, great for understanding
        # but can't generate: it was trained to fill GAPS using both sides, so it has
        # no concept of "what comes next" — only "what fits HERE given left AND right").
        # With it → GPT (left-to-right, each position predicts next token from past only).
        # Bonus: during training, ALL 64 positions predict simultaneously in one forward pass
        # — that's why loss uses logits.view(-1, vocab_size) over all positions at once.
        self.register_buffer(
            "causal_mask",
            nn.Transformer.generate_square_subsequent_mask(context_length),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attention(
            x,
            x,
            x,
            attn_mask=self.causal_mask[: x.size(1), : x.size(1)],  # type: ignore[index]  # registered buffer is a Tensor at runtime
            is_causal=True,
        )
        # POST-NORM: residual add THEN normalize. Works fine for shallow models (3 blocks).
        # GPT-2 uses PRE-NORM (normalize BEFORE attention/FFN) — the residual stream stays
        # clean as a highway, each block adds small well-behaved corrections.
        # Pre-Norm trains more stably at depth (48+ blocks), often skips LR warmup.
        x = self.norm1(x + self.dropout(attn_out))  # residual connection: x + attn
        x = self.norm2(x + self.dropout(self.ffn(x)))  # residual connection: x + ffn
        return x


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        num_blocks: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(context_length, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, ff_dim, context_length, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        positions = torch.arange(T, device=x.device)
        x = self.dropout(self.token_emb(x) + self.pos_emb(positions))
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))
