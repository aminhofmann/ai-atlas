---
id: positional-encoding
title: Positional Encoding
category: foundations
level: 1
year: 2017
summary: >-
  Injects information about each token's position into the model, because
  attention by itself treats its input as an unordered set.
requires: [embeddings]
links:
  - { to: transformer, type: part-of }
  - { to: self-attention, type: related }
  - { to: rotary-position-embedding, type: related }
  - { to: bert, type: applied-in }
sources:
  - { title: "Attention Is All You Need (Vaswani et al., 2017)", url: "https://arxiv.org/abs/1706.03762" }
  - { title: "Self-Attention with Relative Position Representations (Shaw et al., 2018)", url: "https://arxiv.org/abs/1803.02155" }
  - { title: "Train Short, Test Long: Attention with Linear Biases (ALiBi) (Press et al., 2021)", url: "https://arxiv.org/abs/2108.12409" }
  - { title: "Transformer Language Models without Positional Encodings Still Learn Positional Information (Haviv et al., 2022)", url: "https://arxiv.org/abs/2203.16634" }
---

## Intuition
"Dog bites man" and "man bites dog" contain the same tokens. An RNN reads them in order, so order is built in. Attention, however, compares every token with every other token at once — it sees a bag of words. Shuffle the input and each token's output is simply shuffled along with it.

Positional encoding is like stamping a seat number onto each token before it enters the room: the content stays the same, but now the model can tell who sat where, and how far apart two tokens are.

## Mechanics
**Why it is needed:** self-attention is *permutation-equivariant*: for a permutation matrix $P$, $\text{Attn}(PX) = P\,\text{Attn}(X)$. Without position signals, order is invisible.

**1. Sinusoidal (absolute, fixed)** — original Transformer. For position $pos$ and dimension pair $i$:
$$PE_{(pos,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right),\quad PE_{(pos,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$
Added to the embeddings: $X' = X + PE$. Wavelengths form a geometric series from $2\pi$ to $10000\cdot 2\pi$. Because each dimension pair is a sine/cosine at one frequency, $PE_{pos+k}$ is a fixed linear (rotation) function of $PE_{pos}$, which makes relative offsets easy to read. No parameters; in principle defined for any length, though extrapolation is poor in practice.

**2. Learned absolute** — a trainable table $P \in \mathbb{R}^{L_{\max}\times d}$ added to embeddings (BERT: 512 positions, GPT-2: 1024). Vaswani et al. found it performs about the same as sinusoidal, but it cannot represent positions beyond $L_{\max}$.

**3. Relative** — encode the offset $j-i$ inside attention instead of the input:
- Shaw et al. (2018): learned vectors for clipped relative distances added to keys (and values).
- T5: a learned scalar bias per head and distance *bucket*, added to attention logits.
- **ALiBi:** add $-m_h\,|i-j|$ to the logits, with a fixed head-specific slope $m_h$; no embeddings at all, and good length extrapolation.

**4. Rotary (RoPE)** — rotate queries and keys by a position-dependent angle so their dot product depends only on the offset; the default in most current LLMs (see its own note).

**5. NoPE** — causal decoder-only models trained without any positional encoding still learn positional information, because the causal mask itself breaks permutation symmetry.

Key distinction: *absolute* schemes modify the input once (bottom of the stack); *relative* schemes act inside every attention layer's score computation.

## Code
```python
import torch

def sinusoidal_pe(n_pos: int, d: int) -> torch.Tensor:
    pos = torch.arange(n_pos).unsqueeze(1)                 # (n, 1)
    i = torch.arange(0, d, 2)                              # even dims
    freq = 1.0 / (10000 ** (i / d))                        # (d/2,)
    pe = torch.zeros(n_pos, d)
    pe[:, 0::2] = torch.sin(pos * freq)
    pe[:, 1::2] = torch.cos(pos * freq)
    return pe                                              # (n, d)

X = torch.randn(10, 64)            # token embeddings
X = X + sinusoidal_pe(10, 64)      # absolute PE is simply added

# The PE of nearby positions is more similar than that of distant ones
pe = sinusoidal_pe(100, 64)
print(pe[10] @ pe[11], pe[10] @ pe[60])
```

## Why it matters
Without positional information a transformer would be a powerful set function but useless for language, where order carries meaning. The design choice also governs *context-length generalization*: learned absolute tables hard-cap length, while relative schemes (ALiBi, RoPE plus scaling tricks) made long-context models practical. The field has largely moved from additive absolute encodings to RoPE.

## Self-test
Q: Why does a transformer need positional encoding but an RNN does not?
A: Self-attention is permutation-equivariant and ignores order; an RNN processes tokens sequentially, so order is built into the computation.

Q: Write the sinusoidal encoding for even dimension $2i$.
A: $PE_{(pos,2i)} = \sin(pos / 10000^{2i/d})$ (cosine for $2i+1$).

Q: What is the main limitation of learned absolute position embeddings?
A: They only exist for positions up to the trained maximum length $L_{\max}$.

Q: How does ALiBi encode position?
A: By adding a head-specific linear penalty $-m_h|i-j|$ to the attention logits, with no position embeddings at all.

Q: Where do absolute vs relative schemes act?
A: Absolute encodings are added to the input embeddings once; relative schemes modify the attention scores in every layer.
