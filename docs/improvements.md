# Possible Improvements — T4 GPU (16GB VRAM)

All improvements below are achievable on a free Google Colab T4 GPU.
Ordered by expected impact.

---

## High Impact

### 1. BPE Tokenization (eliminate `<UNK>`)
**Current problem:** Word-level tokenizer maps 5-10% of tokens to `<UNK>`. Rare but important words (proper nouns, conjugations, literary terms) are lost.
**Solution:** Byte-Pair Encoding (BPE) — the tokenizer GPT-2 actually uses. Splits rare words into subwords: `"révolutionnaire"` → `["ré", "volution", "naire"]`. No word is ever unknown.
**Impact:** Eliminates `<UNK>` entirely. Generated text becomes much more natural. Balzac LoRA would work dramatically better (his legal/financial vocabulary wouldn't be lost).
**Effort:** Replace `tinygpt/tokenizer.py` with a BPE implementation. HuggingFace `tokenizers` library can train a BPE tokenizer in seconds. Or implement from scratch for learning (merge most frequent byte pairs iteratively).

### 2. More SFT Training Data (Distillation)
**Current problem:** Only 10 hand-written Q&A pairs. The model memorizes templates instead of learning to reason.
**Solution:** Use Claude or GPT-4 to generate 500+ Q&A pairs from the Hugo corpus. Feed it passages and ask it to generate literary analysis questions and answers.
**Impact:** The model would generalize to new questions instead of keyword-matching to the closest memorized answer. This is the single biggest quality improvement for the chatbot.
**Effort:** Write a distillation script that sends Hugo excerpts to an API and collects Q&A pairs into `sft_qa_pairs.json`.

### 3. KV Cache for Inference
**Current problem:** Each generation step recomputes attention for ALL previous tokens. Token 100 recomputes tokens 1-99 even though they haven't changed.
**Solution:** Cache K and V matrices from previous positions. Each step only computes Q, K, V for the new token, then matches Q against cached K's.
**Impact:** ~50x faster generation at sequence length 50. Essential for interactive chatbot experience.
**Effort:** Modify `TransformerBlock.forward()` to accept and return cached K,V tensors. See `docs/kv_cache_explained.md` for the full explanation.

---

## Medium Impact

### 4. Pre-Norm Architecture (GPT-2 Style)
**Current:** Post-Norm — normalize after the residual add.
**GPT-2:** Pre-Norm — normalize before attention/FFN, keep the residual stream clean.
**Impact:** More stable training at depth. Would allow increasing from 3 to 6-8 blocks without instability. Minimal code change (swap two lines in `TransformerBlock.forward()`).

### 5. GELU Instead of ReLU
**Current:** `nn.ReLU()` in the FFN.
**GPT-2:** Uses GELU (Gaussian Error Linear Unit) — smoother activation, doesn't kill gradients at negative values like ReLU does.
**Impact:** Slightly better training dynamics. One-line change: `nn.GELU()` instead of `nn.ReLU()`.

### 6. Learning Rate Scheduling
**Current:** Fixed learning rate throughout training.
**Better:** Warmup (linearly increase LR for first 5-10% of steps) + cosine decay (gradually reduce LR to near-zero).
**Impact:** Faster convergence and better final loss. The model learns aggressively early, then fine-tunes gently.
**Effort:** Add `torch.optim.lr_scheduler.CosineAnnealingLR` or a custom warmup scheduler.

### 7. Weight Tying (Embedding ↔ Output Head)
**Current:** `token_emb` (10K × 128) and `head` (128 × 10K) are separate matrices — 2.56M parameters.
**GPT-2:** Shares the same matrix for both — `head.weight = token_emb.weight`. Same 10K × 128 matrix used for both input embedding and output prediction.
**Impact:** Cuts ~1.28M parameters (40% of the model!). Often improves quality because the output learns to predict in the same space as the input.
**Effort:** One line: `self.head.weight = self.token_emb.weight` in `TinyGPT.__init__()`.

### 8. Mixed Precision Training (fp16)
**Current:** All computations in float32 (32-bit).
**Better:** Use `torch.autocast` for automatic mixed precision — most operations in float16, critical ones in float32.
**Impact:** ~2x faster training on T4 (which has dedicated fp16 tensor cores), halves memory usage. Free speedup.
**Effort:** Wrap the forward pass in `with torch.autocast(device_type='cuda'):` and use `GradScaler`.

---

## Lower Impact (but educational)

### 9. Bigger Model
**T4 budget:** With 16GB VRAM, you could scale to ~50M parameters comfortably:
- `embed_dim=256` (from 128)
- `num_blocks=6` (from 3)
- `ff_dim=1024` (from 512)
- `num_heads=8` (from 4)
**Impact:** Dramatically better text quality and reasoning capability. The jump from 3M to 50M params is where models start producing genuinely interesting text.

### 10. Gradient Accumulation
**Current:** Effective batch size = 64 (physical batch size).
**Better:** Accumulate gradients over N mini-batches before stepping. `effective_batch_size = batch_size × accumulation_steps`.
**Impact:** Larger effective batch size without more memory. Smoother training, especially useful if scaling up model size.

### 11. Validation Split
**Current:** No validation — we only see training loss, can't detect overfitting.
**Better:** Hold out 10% of text for validation. Track val_loss alongside train_loss. If val_loss starts rising while train_loss drops → overfitting.
**Impact:** Know when to stop training. Currently we guess based on epoch count.

### 12. Dropout Tuning
**Current:** Fixed dropout=0.1 everywhere.
**Better:** Different dropout rates for attention (lower, ~0.05) vs FFN (higher, ~0.15). Or experiment with no dropout + more data.
**Impact:** Minor quality improvement. More relevant when scaling up.

---

## Implementation Priority (recommended order)

1. **BPE tokenization** — eliminates the biggest quality bottleneck
2. **Weight tying** — free parameter reduction, one line
3. **GELU activation** — one-line swap
4. **Pre-Norm** — two-line swap, enables deeper models
5. **Mixed precision** — free 2x speedup
6. **KV cache** — essential for interactive chatbot
7. **More SFT data** — biggest impact on chat quality
8. **LR scheduling** — better convergence
9. **Bigger model** — once other improvements are in place
10. **Validation split** — know when to stop
