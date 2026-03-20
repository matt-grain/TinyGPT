# Attention Q, K, V — Concrete Example

## The Hashmap Analogy
Attention works like a **fuzzy hashmap lookup**:
- **Q (Query):** "What am I looking for?"
- **K (Key):** "What do I advertise as matchable?"
- **V (Value):** "What information do I carry when matched?"

## Setup

Say `embed_dim=4` (instead of 128, for readability). After the embedding layer, each word is a raw vector:

```
roi      → [0.8, 0.1, 0.6, 0.3]
couronne → [0.7, 0.2, 0.9, 0.1]
portait  → [0.1, 0.8, 0.2, 0.5]
une      → [0.3, 0.4, 0.1, 0.2]
```

## The Three Learned Projections

`nn.MultiheadAttention` contains **three learned weight matrices** (W_q, W_k, W_v).
Each one transforms the **same input x** into a different role:

```
Q = x @ W_q    →  "what am I looking for?"
K = x @ W_k    →  "what do I advertise as matchable?"
V = x @ W_v    →  "what information do I carry?"
```

### For "roi" (the word doing the looking):
```
Q_roi = [0.8,0.1,0.6,0.3] @ W_q = [0.9, 0.7, 0.1, 0.8]
        → "I'm looking for crown-related, power-related context"
```

### For "couronne" (a highly relevant word):
```
K_couronne = [0.7,0.2,0.9,0.1] @ W_k = [0.8, 0.6, 0.2, 0.7]
             → "I match well on royalty/power queries"

V_couronne = [0.7,0.2,0.9,0.1] @ W_v = [0.3, 0.9, 0.5, 0.1]
             → "My actual content: golden, head-worn, ceremonial"
```

### For "une" (a low-relevance word):
```
K_une = [0.3,0.4,0.1,0.2] @ W_k = [0.1, 0.2, 0.3, 0.1]
        → "I match on... not much, I'm just an article"

V_une = [0.3,0.4,0.1,0.2] @ W_v = [0.2, 0.1, 0.1, 0.4]
        → "My content: generic, determiner-ish"
```

## Step-by-Step Attention Computation

### Step 1 — Match Q against all K's (dot product)
```
Q_roi · K_couronne = 0.9×0.8 + 0.7×0.6 + 0.1×0.2 + 0.8×0.7 = 1.72  ← HIGH MATCH
Q_roi · K_une      = 0.9×0.1 + 0.7×0.2 + 0.1×0.3 + 0.8×0.1 = 0.34  ← LOW MATCH
```

### Step 2 — Softmax → attention weights (sum to 1.0)
```
raw scores: [1.72, 0.34, ...]
softmax:    [0.45, 0.08, ...]
```

### Step 3 — Weighted sum of V's
```
output_roi = 0.45 × V_couronne + 0.08 × V_une + ...
           = 0.45 × [0.3,0.9,0.5,0.1] + 0.08 × [0.2,0.1,0.1,0.4] + ...
           = new 4-dim vector
```

## The Result

- **Before attention:** `roi` = just "king" in isolation
- **After attention:** `roi` = "king in context of wearing a golden crown"

Every token's representation gets **rewritten** based on relevant context, at every layer.
By block 3, `roi` has been contextualized three times.

## Why K and V Are Different

K and V start from the **same x** but W_k and W_v are different learned matrices:
- **W_k** is optimized for **matching** (like the hashmap key)
- **W_v** is optimized for **carrying useful content** (like the hashmap value)

The network learns these matrices during training to separate the two concerns.

## Critical: x Is the ENTIRE Sequence, Not One Word

A common confusion: `self.attention(x, x, x)` — the three `x`'s are NOT three different words.
`x` is the **entire sequence** of all 64 words at once:

```
x.shape = [batch_size, 64, 128]
              │         │    │
              │         │    └─ embed_dim (each word's vector)
              │         └─ all 64 words in the sequence
              └─ batch of sentences
```

So ALL 64 words produce Q, K, V simultaneously:
```
Q = ALL 64 words @ W_q  →  every word asks "what am I looking for?"
K = ALL 64 words @ W_k  →  every word advertises "here's what I match on"
V = ALL 64 words @ W_v  →  every word offers "here's my content"
```

Each word's Q gets matched against every other word's K → a 64×64 matrix of attention scores.
The **causal mask** zeros out the upper triangle so word _i_ can only attend to words 1.._i_:

```
         word1  word2  word3  word4
word1  [  0.9    -∞     -∞     -∞  ]   ← can only see itself
word2  [  0.3    0.7    -∞     -∞  ]   ← can see word1, word2
word3  [  0.1    0.2    0.8    -∞  ]   ← can see word1, word2, word3
word4  [  0.05   0.4    0.1    0.6 ]   ← can see all four
```

After softmax, -∞ becomes 0 → future tokens contribute nothing.

This is why ONE forward pass trains ALL 64 positions in parallel.
Without the mask → BERT (bidirectional). With it → GPT (autoregressive).

## Self-Attention vs Cross-Attention

- **Self-attention:** Q, K, V all come from the same sequence x.
  → Each word attends to other words in the SAME sentence.
  → Used in GPT, BERT, and encoder/decoder stacks.

- **Cross-attention:** Q comes from one source, K and V from another.
  → Example: Q = French sentence, K/V = English sentence.
  → Used in translation models and encoder-decoder architectures.
  → Also how image captioning works: Q = text so far, K/V = image features.
