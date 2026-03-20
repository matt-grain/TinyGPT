# TinyGPT Refactoring Plan

## Target Structure

```
tinyGPT/
├── tinygpt/                    # Python package
│   ├── __init__.py             # Re-exports: TinyGPT, Tokenizer, generate, get_device
│   ├── model.py                # TinyGPT, TransformerBlock (all learning comments preserved)
│   ├── tokenizer.py            # Tokenizer class (replaces word_to_id/id_to_word globals)
│   ├── data.py                 # TextDataset, TextDatasetSmall, SFTDataset
│   ├── checkpoint.py           # save/load/auto-detect latest checkpoint
│   ├── generate.py             # Single generate() function with stop_token support
│   ├── lora.py                 # LoRALinear, apply_lora, save/load adapter
│   └── device.py               # GPU/CPU/MPS auto-detection for Colab
├── pretrain.py                 # Pre-training script (was gpt2_v3.py)
├── sft.py                      # SFT script (was sft_v3.py)
├── dpo.py                      # DPO script (was dpo_v3.py)
├── lora_train.py               # LoRA training script (was lora_v3.py)
├── datasets/
│   ├── hugo/
│   └── balzac/
├── docs/
├── snapshots/
└── ARCHITECTURE.md
```

## Implementation Waves

### Wave 1 — Independent modules (parallel)
- [ ] `tinygpt/model.py` — extract TransformerBlock + TinyGPT from gpt2_v3.py, add types
- [ ] `tinygpt/tokenizer.py` — wrap build_word_tokenizer into Tokenizer class, add encode/decode
- [ ] `tinygpt/device.py` — new module, GPU/CPU/MPS detection
- [ ] `tinygpt/data.py` — extract TextDataset, TextDatasetSmall, SFTDataset

### Wave 2 — Modules with dependencies (parallel)
- [ ] `tinygpt/checkpoint.py` — consolidate save/load from all 4 files, add auto_detect_latest()
- [ ] `tinygpt/generate.py` — single generate() accepting Tokenizer explicitly
- [ ] `tinygpt/lora.py` — extract LoRALinear, apply_lora, save/load adapter
- [ ] `tinygpt/__init__.py` — re-exports

### Wave 3 — Training scripts (parallel)
- [ ] `pretrain.py` — rewrite gpt2_v3.py using tinygpt/ package
- [ ] `sft.py` — rewrite sft_v3.py
- [ ] `dpo.py` — rewrite dpo_v3.py
- [ ] `lora_train.py` — rewrite lora_v3.py

### Wave 4 — Cleanup
- [ ] Run ruff check --fix, ruff format, pyright
- [ ] Write ARCHITECTURE.md
- [ ] Delete old *_v3.py files

## Key Rules
- ALL learning comments MUST be preserved in their modules
- Use pathlib.Path for all paths
- Use get_device() for all tensor placement
- Tokenizer class replaces globals (word_to_id, id_to_word)
- auto_detect_latest() replaces hardcoded resume paths
- Disable ruff T20 (print statements are intentional teaching output)
- Type annotations on all functions
- Checkpoint compatibility: load old .pt files with word_to_id/id_to_word dicts
