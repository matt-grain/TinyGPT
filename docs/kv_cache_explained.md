# KV Cache — Why Generation Is Slow Without It

## The Problem

During generation, we produce tokens one at a time. For each new token, the model runs a **full forward pass** on the entire sequence so far:

```
Step 1:  input = [Il]                    → compute Q,K,V for 1 token   → predict "était"
Step 2:  input = [Il, était]             → compute Q,K,V for 2 tokens  → predict "une"
Step 3:  input = [Il, était, une]        → compute Q,K,V for 3 tokens  → predict "fois"
...
Step 50: input = [Il, était, une, ..., ] → compute Q,K,V for 50 tokens → predict next
```

At step 50, we recompute K and V for tokens 1-49 even though **they haven't changed**. The K and V for "Il" are identical at step 2, step 3, step 50...

Total computation: 1 + 2 + 3 + ... + 50 = **1,275 forward passes** worth of K,V computation.

## The Fix: Cache K and V

Instead of recomputing everything, **store the K and V vectors** from previous positions:

```
Step 1:  compute Q₁,K₁,V₁  → cache [K₁], [V₁]           → Q₁ attends to [K₁]
Step 2:  compute Q₂,K₂,V₂  → cache [K₁,K₂], [V₁,V₂]    → Q₂ attends to [K₁,K₂]
Step 3:  compute Q₃,K₃,V₃  → cache [K₁,K₂,K₃], [V₁,V₂,V₃] → Q₃ attends to [K₁,K₂,K₃]
...
Step 50: compute Q₅₀,K₅₀,V₅₀ → append to cache           → Q₅₀ attends to all 50 K's
```

At each step, we only compute Q, K, V for **1 token** (the new one). Then we match Q against the full cached K to get attention weights, and multiply by cached V.

Total computation: 1 + 1 + 1 + ... + 1 = **50 single-token passes**. That's 25x faster than without cache at step 50.

## Why Not Cache Q?

Q is the **query** — "what am I looking for?" It changes for every new token. Token 50's question is different from token 49's question. There's nothing to cache.

K and V from previous tokens **never change** because of the causal mask: token 30's K and V are computed without seeing tokens 31+, so adding token 31 can't affect them.

## The Tradeoff: Memory

The cache grows with sequence length:
```
Cache size = num_layers × 2 (K+V) × seq_length × embed_dim × batch_size
```

For our TinyGPT (3 blocks, embed_dim=128, seq_len=64):
  → 3 × 2 × 64 × 128 = 49,152 floats = ~192 KB (trivial)

For GPT-4 (estimated 120 layers, embed_dim=12288, seq_len=128K):
  → 120 × 2 × 128,000 × 12,288 = ~377 billion floats = ~1.4 TB per batch!

This is why you hear about:
- **GQA** (Grouped Query Attention): share K,V across multiple Q heads → smaller cache
- **KV cache quantization**: store cached K,V in 4-bit instead of 16-bit → 4x smaller
- **Sliding window attention**: only cache the last N tokens → fixed memory
- **PagedAttention** (vLLM): manage cache like OS virtual memory pages

## Visual Summary

```
WITHOUT KV CACHE (current TinyGPT):
┌──────────────────────────────────────┐
│ Step 50: feed ALL 50 tokens          │
│ [Il, était, une, fois, ..., token49] │
│  ↓   ↓    ↓    ↓          ↓         │
│  Q₁   Q₂   Q₃   Q₄  ...  Q₅₀       │  ← compute ALL Q's (only use Q₅₀!)
│  K₁   K₂   K₃   K₄  ...  K₅₀       │  ← compute ALL K's (1-49 same as before!)
│  V₁   V₂   V₃   V₄  ...  V₅₀       │  ← compute ALL V's (1-49 same as before!)
└──────────────────────────────────────┘

WITH KV CACHE:
┌──────────────────────────────────────┐
│ Step 50: feed ONLY token 50          │
│ [token49]                            │
│    ↓                                 │
│   Q₅₀  (new)                        │  ← compute 1 Q
│   K₅₀  (new, append to cache)       │  ← compute 1 K, cache has [K₁...K₅₀]
│   V₅₀  (new, append to cache)       │  ← compute 1 V, cache has [V₁...V₅₀]
│                                      │
│   Q₅₀ × [K₁,K₂,...,K₅₀]ᵀ → weights │  ← match against ALL cached K's
│   weights × [V₁,V₂,...,V₅₀] → out   │  ← weighted sum of ALL cached V's
└──────────────────────────────────────┘
```
