# LLM Training Pipeline — From Raw Text to ChatGPT

## The Key Insight

A transformer is ONE architecture reused for everything. Only the **loss function** and **data format** change:

```
Same transformer → different loss → different capability

Pre-training loss    → learns language
SFT loss             → learns to follow instructions
DPO loss             → learns human preferences
Translation loss     → learns to translate (cross-attention)
Classification loss  → learns sentiment/categories (BERT-style)
Image+text loss      → learns to describe images (multimodal)
```

The model is a feature extractor. The loss function is the steering wheel.

---

## Step 1: Pre-Training (gpt2_v3.py)

**Goal:** Learn language from raw text.
**Data:** Unstructured text (Victor Hugo novels).
**Loss:** Cross-entropy — predict the next token.
**Result:** "Autocomplete" — speaks French but can't follow instructions.

### Key Concepts Learned

**Tokenization tradeoff:**
- Char-level (v2): vocab=126, model must learn spelling + grammar + meaning. Slow.
- Word-level (v3): vocab=10K, spelling is "free" (baked into embeddings). Faster convergence but embedding table explodes (160x bigger).
- BPE (real GPT-2): middle ground. Splits rare words into subwords, no `<UNK>`.
- Principle: moves complexity from sequence length (temporal) to vocabulary size (spatial).

**Self-Attention (Q, K, V) — the fuzzy hashmap:**
- Q = "what am I looking for?" (query)
- K = "what do I advertise as matchable?" (key)
- V = "what information do I carry?" (value)
- All three come from the same x (self-attention) via learned projection matrices W_q, W_k, W_v.
- Cross-attention: Q from one source, K/V from another (translation, image captioning).
- See `attention_qkv_explained.md` for full worked example with numbers.

**x is the ENTIRE sequence, not one word:**
- `self.attention(x, x, x)` — x has shape [batch, 64, 128] = all 64 words at once.
- Each word's Q matches against every other word's K → 64×64 attention matrix.
- ONE forward pass trains ALL positions in parallel.

**Causal mask — GPT vs BERT:**
- With mask → GPT: autoregressive, left-to-right, can generate text.
- Without mask → BERT: bidirectional, sees everything, great for understanding but can't generate (trained to fill gaps using both sides, no concept of "what comes next").
- The mask is a triangular matrix: token i can attend to tokens 1..i, not i+1, i+2...

**LayerNorm (not BatchNorm):**
- BatchNorm: normalizes across batch dimension. Works for images (pixel 14,14 is always "center") but not text (position 50 in different sentences = different words).
- LayerNorm: normalizes across feature dimension per token. "Are YOUR features balanced?"
- Also: BatchNorm would leak info across sequences, breaking the causal contract.

**Residual connections:**
- `x = norm(x + attn_out)` — the `+` shortcut lets gradient flow even if layers saturate.
- Without them, deep transformers don't train.

**Post-Norm vs Pre-Norm:**
- Post-Norm (our code): residual add THEN normalize. Fine for shallow models (3 blocks).
- Pre-Norm (GPT-2): normalize BEFORE attention/FFN. Residual stream stays clean. Trains more stably at depth (48+ blocks). Often skips LR warmup.

**Loss baseline:**
- Initial loss ≈ log(vocab_size). Char: log(126)≈4.84. Word: log(10003)≈9.21.
- Don't panic at high initial loss — it's random guessing over more options.
- Perplexity (e^loss) normalizes across vocabularies.

---

## Step 2: SFT — Supervised Fine-Tuning (sft_v3.py)

**Goal:** Learn to follow instructions.
**Data:** Structured Q&A pairs: `<|user|> question <|assistant|> answer <|end|>`
**Loss:** Same cross-entropy, but MASKED on the prompt portion.
**Result:** "Follows instructions" — answers questions about Hugo.

### Key Concepts Learned

**What changes vs pre-training:**
- NOT the architecture (same model, same layers)
- NOT the objective (still next-token prediction)
- ONLY the data format (structured pairs instead of raw text)

**Special tokens:** `<|user|>`, `<|assistant|>`, `<|end|>` — added to vocabulary. The model learns: "after `<|assistant|>`, produce an answer."

**Loss masking:** Only compute loss on answer tokens (mask=0 on prompt, mask=1 on answer). Otherwise the model would learn to generate questions too.

**`<|end|>` in the loss:** Must be included (mask=1) so the model learns WHEN TO STOP. Without it → rambling, hallucination by exhaustion.

**Lower learning rate (5e-5 vs 1e-3):** Prevents catastrophic forgetting — destroying the French language knowledge from pre-training.

**Distillation:** Using a strong model (Claude/GPT-4) to generate training data for a smaller model. Common industry practice.

**`if __name__ == "__main__"` guard:** Without it, `from gpt2_v3 import TinyGPT` executes the entire training block. Python executes the full file on import.

---

## Step 3: DPO — Direct Preference Optimization (dpo_v3.py)

**Goal:** Learn to prefer good answers over bad ones.
**Data:** Preference pairs: (question, chosen_answer, rejected_answer)
**Loss:** DPO loss — increases P(chosen) relative to P(rejected), anchored to a reference model.
**Result:** "Gives quality answers" — steers toward human-preferred responses.

### Key Concepts Learned

**DPO vs RLHF:**
- RLHF: train a reward model on human rankings, then use PPO (reinforcement learning) to optimize. Complex, unstable.
- DPO: skip the reward model, train directly on preference pairs. Same result, fewer moving parts. This is what Claude uses.

**The reference model:**
- Frozen `deepcopy` of the SFT model. Never updates.
- Prevents drifting: without it, model could crank good answer probability to 99.99% and lose all nuance.
- DPO is ADJUSTMENT, not relearning. It steers within existing capabilities.
- `deepcopy` (not `ref = model`) creates truly independent clone. `requires_grad=False` saves memory.

**Log probabilities (why not raw probabilities):**
- P(sentence) = P(w1) × P(w2) × ... × P(w50) → underflow (too many small numbers multiplied)
- log P = log(w1) + log(w2) + ... + log(w50) → normal numbers, addition instead of multiplication
- Always negative. More negative = less likely.

**Teacher forcing (scoring, not generating):**
- The model outputs probabilities for ALL 10K words at every position.
- During generation: sample one token (temperature, top-k, etc.)
- During DPO: READ the probability of a specific token (scoring)
- Same forward pass, different post-processing.

**The model always does the same thing:**
- Forward pass → logits [10,003 probabilities per position]
- Post-processing decides what to do: generate, compute loss, or score.
- Temperature doesn't change the model — it changes sampling. Same logits, different strategy.

**The DPO loss formula:**
```
log_ratio_chosen   = log P_model(chosen)  - log P_ref(chosen)
log_ratio_rejected = log P_model(rejected) - log P_ref(rejected)
loss = -logsigmoid(β × (log_ratio_chosen - log_ratio_rejected))
```
"Increase chosen MORE than rejected, relative to where you started."

---

## The Full Pipeline Cost

| Step | Data | Compute (GPT-4 scale) |
|------|------|-----------------------|
| Pre-training | Terabytes of raw text | ~$100M |
| SFT | ~100K expert Q&A pairs | ~$100K |
| DPO/RLHF | ~50K preference rankings | ~$50K |

Same architecture throughout. 1000x cheaper to steer than to build.

---

## The Universal Training Loop

All three steps use the same mechanism:
```python
logits = model(x)                    # forward pass (always the same)
loss = some_loss_function(logits, y)  # ← THIS is what changes
loss.backward()                       # compute gradients
optimizer.step()                      # update weights
```

Pre-training, SFT, DPO — three loss functions, one learning loop.
The model architecture never changes. Only what you optimize for changes.

---

## Step 4: LoRA — Low-Rank Adaptation (lora_v3.py)

**Goal:** Adapt the model to a new domain (Balzac) without retraining everything.
**Data:** Raw Balzac text (same format as pre-training).
**Loss:** Same cross-entropy as pre-training, but only LoRA params update.
**Result:** Balzac-adapted model. Only 1.8% of parameters trained.

### Key Concepts Learned

**The problem LoRA solves:**
Full fine-tuning updates ALL parameters. For a 70B model, that means storing 70B gradients + optimizer states = hundreds of GB of VRAM. LoRA freezes everything and injects tiny trainable matrices, making fine-tuning possible on consumer hardware.

**The math (PCA/SVD intuition):**
A full weight update ΔW has shape [128, 128] = 16,384 params.
Most of the information lives in a few principal components:
```
ΔW ≈ A × B    where A=[128,4], B=[4,128] → only 1,024 params
```
- A (128→4): "compress — what are the important directions for this task?"
- B (4→128): "expand — apply those directions back to full space"
- Rank (4): controls expressiveness vs efficiency. Like choosing how many PCA components to keep.
- The model LEARNS which directions matter — they're not pre-selected.

**B starts at zero:**
At initialization, `B = zeros` → `A × B = 0` → adapter has NO effect.
The model starts identical to the frozen one and gradually learns adjustments.
This is important: the starting point is the pre-trained model, not random noise.

**Swappable adapters — the killer feature:**
```
Same frozen model ──┬── Hugo adapter (50KB)      → explains Hugo
                    ├── Balzac adapter (50KB)     → explains Balzac
                    ├── Zola adapter (50KB)       → explains Zola
                    └── Baudelaire adapter (50KB) → explains Baudelaire
```
Swap in <1ms. Each adapter is tiny. This is how companies serve dozens of specialized models from one base model in production.

**Cost comparison:**
```
100 fully fine-tuned models: 100 × 12MB = 1.2GB
1 base model + 100 adapters: 12MB + 100 × 50KB = 17MB
```

**Higher learning rate is OK:**
LoRA can use higher LR (1e-3) than full SFT (5e-5) because only tiny adapters update.
The frozen base can't be damaged — it's not receiving gradients.

**Limitation with word-level tokenization:**
Balzac's vocabulary differs from Hugo's. 10% of Balzac tokens became `<UNK>` (vs 5% for Hugo).
LoRA can't fix this — the vocabulary is frozen. BPE tokenization would solve it.

### Our Results

| Metric | Value |
|--------|-------|
| Total params | 3,174,035 |
| LoRA params | 55,884 (1.8%) |
| Adapter file size | 223 KB |
| Training epochs | 10 |
| Balzac UNK rate | 10% |

---

## The Complete Pipeline

| Step | File | What | Params trained | Data |
|------|------|------|----------------|------|
| Pre-training | `gpt2_v3.py` | Learn French | 3,174,035 (100%) | Raw Hugo text |
| SFT | `sft_v3.py` | Learn Q&A format | 3,174,035 (100%) | Structured Q&A pairs |
| DPO | `dpo_v3.py` | Prefer good answers | 3,174,035 (100%) | Preference pairs |
| LoRA | `lora_v3.py` | Adapt to Balzac | 55,884 (1.8%) | Raw Balzac text |

From raw text to human-aligned, domain-adaptable LLM.
Same architecture throughout. Different data + loss + which params update.

---

## The Universal Training Loop

All four steps use the same mechanism:
```python
logits = model(x)                    # forward pass (always the same)
loss = some_loss_function(logits, y)  # ← THIS is what changes
loss.backward()                       # compute gradients (LoRA: only flows to A,B)
optimizer.step()                      # update weights (LoRA: only updates A,B)
```
