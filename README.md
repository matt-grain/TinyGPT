# TinyGPT

A minimal GPT-2 implementation built from scratch in PyTorch to learn the complete modern LLM training pipeline.

**No HuggingFace. No abstractions. Raw tensors, raw learning.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matt-grain/TinyGPT/blob/main/notebooks/train_full_pipeline.ipynb)

## What This Is

A 3M-parameter transformer trained on Victor Hugo novels, implementing the same 4-step pipeline that turns raw text into ChatGPT:

| Step | Script | What it does | What changes |
|------|--------|-------------|--------------|
| **Pre-training** | `pretrain.py` | Learn French from raw Hugo text | Next-token prediction on 2.4M characters |
| **SFT** | `sft.py` | Learn to answer questions | Same loss, but on structured Q&A pairs with masked prompts |
| **DPO** | `dpo.py` | Prefer good answers over bad | Preference pairs, no reward model needed |
| **LoRA** | `lora_train.py` | Adapt to Balzac with 1.8% of params | Frozen base + tiny trainable adapters |

Every module contains detailed comments explaining **why** things work, not just what they do.

## The Key Insight

The model architecture never changes. Only the **loss function** and **data format** change:

```
Same transformer + pre-training loss    → learns language
Same transformer + SFT loss (masked)    → learns to follow instructions
Same transformer + DPO loss             → learns human preferences
Same transformer + LoRA + same loss     → adapts to new domain (1.8% params)
```

## Quick Start

### On Google Colab (recommended, free GPU)

Click the badge above, select **Runtime > Change runtime type > T4 GPU**, and run all cells.

### Locally

```bash
git clone https://github.com/matt-grain/TinyGPT.git
cd TinyGPT
uv sync
uv run python pretrain.py   # Step 1: Pre-training (~30 min on CPU, ~2 min on GPU)
uv run python sft.py         # Step 2: Supervised fine-tuning
uv run python dpo.py         # Step 3: Direct preference optimization
uv run python lora_train.py  # Step 4: LoRA adaptation for Balzac
```

All hyperparameters are in `config.json` — edit epochs, learning rate, batch size, etc.

## Project Structure

```
TinyGPT/
├── tinygpt/                 # Reusable package
│   ├── model.py             # TransformerBlock + TinyGPT architecture
│   ├── tokenizer.py         # Word-level tokenizer (with BPE discussion)
│   ├── data.py              # Dataset classes (sliding window, SFT with loss mask)
│   ├── checkpoint.py        # Save/load/auto-detect checkpoints
│   ├── generate.py          # Autoregressive text generation
│   ├── lora.py              # LoRA adapter layer + injection
│   └── device.py            # GPU/CPU/MPS auto-detection
├── pretrain.py              # Step 1: Pre-training
├── sft.py                   # Step 2: Supervised fine-tuning
├── dpo.py                   # Step 3: DPO
├── lora_train.py            # Step 4: LoRA adaptation
├── config.json              # All hyperparameters
├── datasets/
│   ├── hugo/                # Les Miserables, Notre-Dame, Quatrevingt-treize
│   ├── balzac/              # Comedie Humaine, Eugenie Grandet, La Maison du Chat
│   └── training/            # SFT Q&A pairs + DPO preference pairs (JSON)
└── docs/
    ├── attention_qkv_explained.md   # Worked Q/K/V example with numbers
    └── llm_pipeline_summary.md      # Full pipeline reference
```

## Concepts Covered (with code comments)

### Architecture (`tinygpt/model.py`)
- **Self-attention Q/K/V** — the fuzzy hashmap analogy, with a worked numerical example
- **Causal mask** — why GPT is autoregressive and BERT is bidirectional
- **LayerNorm vs BatchNorm** — why images use batch norm but text uses layer norm
- **Residual connections** — gradient highways through deep networks
- **Post-norm vs Pre-norm** — why GPT-2 normalizes before attention, not after

### Training Pipeline
- **Loss baseline** — why initial loss = log(vocab_size), and what perplexity means
- **Loss masking** (SFT) — training only on answer tokens, not the question
- **Teacher forcing** (DPO) — scoring answers without generating them
- **Log probabilities** — why log(P) + sum instead of P * product (underflow)
- **Reference model** (DPO) — frozen copy preventing catastrophic drift
- **Catastrophic forgetting** — why SFT uses 20x lower learning rate

### Adaptation (`tinygpt/lora.py`)
- **LoRA rank decomposition** — the PCA/SVD analogy
- **B starts at zero** — adapter has no effect initially, learns gradually
- **Swappable adapters** — same base model, different personalities in <1ms

## Training Results (30 epochs, CPU)

```
Epoch  1 | Loss: 9.37 → random (baseline = log(10003) = 9.21)
Epoch 10 | Loss: 3.14
Epoch 20 | Loss: 2.87
Epoch 30 | Loss: 2.72

Sample at epoch 30:
"Il était une fois . Il répondit : « Ho ! » Il remua au bout de quelques
minutes , et un tremblement parcourut la terre . Une voix brève , et qui
jurait mieux garder le regard et se remit à marcher ..."
```

## Built With

- **PyTorch** — raw tensors, no HuggingFace
- **Python 3.12+** — type annotations throughout
- **uv** — package management
- **ruff + pyright** — linting and type checking (0 errors)

## Acknowledgments

Built as a hands-on learning project, pair-programming with [Claude](https://claude.ai). Every concept was learned through building, breaking, and understanding — not just reading about it.

The training data is Victor Hugo and Honore de Balzac, sourced from [Project Gutenberg](https://www.gutenberg.org/).
