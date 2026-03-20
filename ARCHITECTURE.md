# Architecture

## Overview

TinyGPT is a minimal GPT-2 implementation built from scratch for learning the complete modern LLM training pipeline: pre-training, supervised fine-tuning (SFT), direct preference optimization (DPO), and LoRA adaptation. Every module contains detailed comments explaining **why** things work, not just what they do.

## Tech Stack

- **Language:** Python 3.12+
- **Framework:** PyTorch (no HuggingFace, no abstractions — raw tensors)
- **Package manager:** uv
- **Linting:** ruff, pyright (strict)
- **Training data:** Victor Hugo (pre-training/SFT/DPO), Balzac (LoRA)
- **Target:** CPU-friendly (3M parameters), Colab-ready with GPU auto-detection

## Project Structure

```
tinyGPT/
├── tinygpt/                 # Python package — reusable components
│   ├── __init__.py          # Re-exports: TinyGPT, Tokenizer, generate, get_device
│   ├── model.py             # TransformerBlock, TinyGPT (architecture only)
│   ├── tokenizer.py         # Tokenizer class (word-level, replaces globals)
│   ├── data.py              # TextDataset, TextDatasetSmall, SFTDataset
│   ├── checkpoint.py        # save/load/auto-detect checkpoints
│   ├── generate.py          # generate(), generate_answer()
│   ├── lora.py              # LoRALinear, apply_lora, save/load adapters
│   └── device.py            # GPU/CPU/MPS auto-detection
├── pretrain.py              # Step 1: Pre-training on raw Hugo text
├── sft.py                   # Step 2: Supervised fine-tuning with Q&A pairs
├── dpo.py                   # Step 3: Direct preference optimization
├── lora_train.py            # Step 4: LoRA adaptation for Balzac
├── datasets/
│   ├── hugo/                # Victor Hugo novels (93, Les Misérables, Notre-Dame)
│   └── balzac/              # Balzac novels (Comédie Humaine, Eugénie Grandet, etc.)
├── snapshots/               # Saved checkpoints and training logs
└── docs/
    ├── attention_qkv_explained.md   # Worked example: Q, K, V with numbers
    └── llm_pipeline_summary.md      # Full pipeline reference
```

## Layer Responsibilities

### `tinygpt/` package — Reusable components, no training logic

- **model.py:** Architecture only. TransformerBlock (self-attention + FFN + residual + LayerNorm) and TinyGPT (embeddings + blocks + output head). No global state.
- **tokenizer.py:** Tokenizer class wrapping word_to_id/id_to_word dicts. Provides encode/decode, special token management, and corpus-based construction.
- **data.py:** PyTorch Dataset classes for pre-training (sliding window, random sampling) and SFT (structured Q&A with loss masking).
- **checkpoint.py:** All save/load complexity centralized. Handles causal_mask filtering, embedding resize for vocab changes, auto-detection of latest checkpoint by epoch number.
- **generate.py:** Autoregressive text generation with temperature sampling and optional stop tokens.
- **lora.py:** Low-rank adaptation layer (LoRALinear) and utilities to inject adapters into a frozen model.
- **device.py:** CUDA > MPS > CPU detection for Colab compatibility.

### Training scripts — Orchestration, no reusable logic

Each script is a self-contained pipeline step:
- **pretrain.py:** Loads Hugo text → builds tokenizer → trains on next-token prediction
- **sft.py:** Loads pre-trained model → adds special tokens → trains on Q&A pairs with masked loss
- **dpo.py:** Loads SFT model → creates frozen reference → trains on preference pairs
- **lora_train.py:** Loads pre-trained model → injects LoRA adapters → trains on Balzac text

## Data Flow

**Pre-training:**
```
Hugo .txt files → Tokenizer.from_corpus() → word IDs → TextDatasetSmall → DataLoader
→ TinyGPT forward pass → cross_entropy(logits, next_token) → optimizer.step()
```

**SFT:**
```
QA_PAIRS → SFTDataset (with loss mask) → DataLoader
→ TinyGPT forward pass → cross_entropy × mask (answer tokens only) → optimizer.step()
```

**DPO:**
```
PREFERENCE_PAIRS → get_sequence_log_prob(chosen) vs get_sequence_log_prob(rejected)
→ DPO loss: -logsigmoid(β × (ratio_chosen - ratio_rejected)) → optimizer.step()
```

**LoRA:**
```
Balzac .txt files → same tokenizer → TextDatasetSmall → DataLoader
→ TinyGPT with LoRA adapters → cross_entropy → only adapter params update
```

## Key Domain Concepts

| Concept | Where | What it means |
|---------|-------|---------------|
| Causal mask | model.py | Triangular matrix preventing tokens from seeing the future (GPT vs BERT) |
| LayerNorm | model.py | Per-token normalization (not per-batch like BatchNorm for images) |
| Residual connections | model.py | Skip connections enabling gradient flow in deep networks |
| Loss masking | data.py, sft.py | Training only on answer tokens, not the question |
| Teacher forcing | dpo.py | Scoring a given answer's probability without generating |
| Reference model | dpo.py | Frozen copy preventing DPO from drifting too far |
| LoRA rank | lora.py | Dimensionality of the low-rank update (PCA analogy) |
