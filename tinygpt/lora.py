"""LoRA (Low-Rank Adaptation) — Fine-tune a frozen TinyGPT with lightweight adapters.

THE IDEA:
We have a model trained on one corpus. We want it to also work for a different style.
Instead of retraining ALL parameters, we:
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
Same frozen model + adapter_A → style A expert
Same frozen model + adapter_B → style B expert
Same frozen model + adapter_C → style C expert
Swap in milliseconds. Each adapter is ~50KB.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from tinygpt.model import TinyGPT


# ----------------------------------------
# 1. LoRA Layer
# ----------------------------------------
class LoRALinear(nn.Module):
    """
    Wraps an existing frozen Linear layer with a LoRA adapter.

    Forward pass:
        output = frozen_linear(x) + (x @ A) @ B * scaling

    The frozen part gives us everything the base model learned.
    The A×B part adds task-specific adjustments.

    Parameters:
        original: the frozen nn.Linear layer
        rank: LoRA rank (4-8 typical). Lower = fewer params, less expressive.
        alpha: scaling factor. Controls how much the adapter influences output.
               scaling = alpha / rank. Higher alpha = stronger adapter effect.
    """

    def __init__(self, original: nn.Linear, rank: int = 4, alpha: int = 8) -> None:
        super().__init__()
        self.original = original
        in_features: int = original.in_features
        out_features: int = original.out_features

        # Freeze the original weights — the base model's knowledge stays intact
        for param in self.original.parameters():
            param.requires_grad = False

        # LoRA matrices — these are the ONLY trainable parameters
        # A: compresses input to rank dimensions ("what matters for this task?")
        # B: expands back to output dimensions ("how to apply it")
        # A is initialized with small random values (Kaiming), B starts at zero.
        # Starting B at zero means the adapter has NO effect at the start —
        # the model begins identical to the frozen one and gradually learns.
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * math.sqrt(2.0 / in_features))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scaling: float = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Frozen path: original computation (base model's knowledge)
        frozen_out = self.original(x)
        # LoRA path: small trainable adjustment (task-specific adaptation)
        lora_out = (x @ self.lora_A) @ self.lora_B * self.scaling
        return frozen_out + lora_out


# ----------------------------------------
# 2. Apply LoRA to a TinyGPT model
# ----------------------------------------
def apply_lora(model: TinyGPT, rank: int = 4, alpha: int = 8) -> TinyGPT:
    """
    Inject LoRA adapters into the FFN layers and output head of the model.

    First freezes ALL parameters, then replaces target Linear layers with
    LoRALinear wrappers. Only the LoRA A and B matrices will have
    requires_grad=True after this call.

    NOTE: We intentionally skip MultiheadAttention's internal projections
    (in_proj_weight, out_proj). PyTorch's MultiheadAttention accesses these
    as raw .weight attributes in its C++ kernel during forward(). Wrapping
    them in a LoRALinear module would break that internal attribute lookup.
    In production LoRA libraries (e.g. HuggingFace PEFT), this is handled
    by monkey-patching the MultiheadAttention.forward() method directly.
    For learning purposes, adapting FFN layers + the output head is sufficient
    and covers the majority of the model's representational capacity.
    """
    # Step 1: Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Step 2: Replace Linear layers in FFN blocks with LoRA-wrapped versions
    lora_params: int = 0
    total_params: int = sum(p.numel() for p in model.parameters())

    for block in model.blocks:
        ffn = block.ffn  # type: ignore[union-attr]  # PyTorch Sequential is dynamically typed
        for i, layer in enumerate(ffn):  # type: ignore[arg-type]  # Sequential is iterable at runtime
            if isinstance(layer, nn.Linear):
                ffn[i] = LoRALinear(layer, rank, alpha)  # type: ignore[index]
                lora_params += rank * layer.in_features
                lora_params += rank * layer.out_features

    # Also wrap the output head (largest Linear layer in the model)
    original_head = model.head
    model.head = LoRALinear(original_head, rank, alpha)  # type: ignore[assignment]  # LoRALinear wraps Linear
    lora_params += rank * original_head.in_features
    lora_params += rank * original_head.out_features

    trainable: int = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total_params:,}")
    print(f"LoRA params:      {trainable:,}")
    print(f"Trainable ratio:  {100 * trainable / total_params:.1f}%")

    return model


# ----------------------------------------
# 3. Save and load LoRA adapters
# ----------------------------------------
def save_lora_adapter(model: TinyGPT, path: Path) -> None:
    """Save ONLY the LoRA weights — tiny file, ~50KB.

    This enables the swappable-adapter pattern: ship one frozen base model
    and many small adapter files, loading whichever personality you need.
    """
    lora_state = {k: v for k, v in model.state_dict().items() if "lora_A" in k or "lora_B" in k}
    torch.save(lora_state, path)
    size_kb = path.stat().st_size / 1024
    print(f"LoRA adapter saved: {path} ({size_kb:.0f} KB)")


def load_lora_adapter(model: TinyGPT, path: Path) -> None:
    """Load LoRA weights into a model that already has LoRA layers injected.

    Call apply_lora() on the model before calling this function.
    Uses strict=False so that frozen base weights are left untouched.
    """
    lora_state: dict[str, torch.Tensor] = torch.load(path, weights_only=True)
    model.load_state_dict(lora_state, strict=False)
    print(f"LoRA adapter loaded: {path}")
