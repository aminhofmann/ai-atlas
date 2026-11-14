---
id: multi-head-attention
title: Multi-Head Attention
category: foundations
level: 2
year: 2017
summary: >-
  Runs several attention operations in parallel, each in its own learned
  subspace, so the model can attend to different kinds of relationships at once.
requires: [self-attention]
links:
  - { to: transformer-block, type: part-of }
  - { to: grouped-query-attention, type: related }
  - { to: multi-head-latent-attention, type: related }
  - { to: kv-cache, type: related }
  - { to: mechanistic-interpretability, type: related }
sources:
  - { title: "Attention Is All You Need (Vaswani et al., 2017)", url: "https://arxiv.org/abs/1706.03762" }
  - { title: "Are Sixteen Heads Really Better than One? (Michel et al., 2019)", url: "https://arxiv.org/abs/1905.10650" }
  - { title: "Fast Transformer Decoding: One Write-Head is All You Need (Shazeer, 2019) — multi-query attention", url: "https://arxiv.org/abs/1911.02150" }
---

## Intuition
A single attention head produces one weighted average per token — it has to commit to one "way of looking". But a word often needs several things at once: its grammatical subject, the previous word, a matching bracket far back.

Multi-head attention is like a panel of specialists reading the same sentence. Each specialist (head) has its own notion of relevance and reports what it found; the reports are then stitched together and summarized. Splitting the width across heads keeps the total cost about the same as one big head.

## Mechanics
With $h$ heads and $d_{\text{head}} = d_{\text{model}}/h$:
$$\text{head}_i = \text{Attention}(XW_Q^{(i)},\, XW_K^{(i)},\, XW_V^{(i)}),\qquad W^{(i)}_{Q,K,V} \in \mathbb{R}^{d_{\text{model}}\times d_{\text{head}}}$$
$$\text{MHA}(X) = \text{Concat}(\text{head}_1,\dots,\text{head}_h)\,W_O,\qquad W_O \in \mathbb{R}^{h\,d_{\text{head}}\times d_{\text{model}}}.$$
Original Transformer base: $d_{\text{model}}=512$, $h=8$, $d_{\text{head}}=64$. BERT-base: 768 wide, 12 heads.

**Implementation:** one fused projection $X \to [Q;K;V] \in \mathbb{R}^{n\times 3d}$, reshape $(B, n, d) \to (B, h, n, d_{\text{head}})$, run batched attention over the head axis, transpose back to $(B, n, h\,d_{\text{head}})$, apply $W_O$.

**Cost:** parameters $4d_{\text{model}}^2$ (Q, K, V, O, ignoring biases) — the same as a single full-width head. FLOPs for projections are also unchanged; only the $h$ separate $n\times n$ score matrices are new.

**Why multiple heads help:**
- Softmax is a single distribution per query; one head cannot sharply attend to two unrelated positions for two different reasons. $h$ heads give $h$ independent distributions.
- Each head compares tokens through its own rank-$d_{\text{head}}$ QK matrix — a different "similarity notion".
- Concatenation then $W_O$ is equivalent to summing per-head contributions $\sum_i \text{head}_i W_O^{(i)}$, each head writing its own update into the residual stream.

**Observed behaviour:** heads specialize — previous-token heads, positional heads, syntactic heads, and *induction heads* that copy patterns, studied in mechanistic interpretability. Many heads are redundant: Michel et al. (2019) pruned a large fraction at test time with little loss.

**Variants that share K/V to shrink the KV cache:**
- **Multi-query attention (MQA):** all query heads share one K/V head.
- **Grouped-query attention (GQA):** groups of query heads share K/V heads — a middle ground used by many current LLMs.
- **Multi-head latent attention (MLA):** compresses K/V into a low-rank latent.

## Code
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.d_head = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)   # W_O

    def forward(self, x, causal=False):                      # x: (B, n, d)
        B, n, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=-1)
        # (B, n, d) -> (B, h, n, d_head): each head works in its own subspace
        q, k, v = (t.view(B, n, self.h, self.d_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        y = y.transpose(1, 2).reshape(B, n, d)               # concat heads
        return self.out(y)

mha = MultiHeadAttention()
print(mha(torch.randn(2, 10, 512), causal=True).shape)      # (2, 10, 512)
print(sum(p.numel() for p in mha.parameters()))             # 4 * 512**2 = 1,048,576
```

## Why it matters
Multi-head attention is the form in which attention is actually used in every transformer: it adds representational diversity at essentially no extra parameter cost. It is also the unit that interpretability research dissects (individual heads as circuits) and the unit that inference optimizations target — GQA, MQA and MLA all modify how heads share keys and values to cut memory bandwidth during generation.

## Self-test
Q: How is $d_{\text{head}}$ chosen in standard multi-head attention?
A: $d_{\text{head}} = d_{\text{model}}/h$, so all heads together have the model width.

Q: How many parameters does an MHA layer have, ignoring biases?
A: $4d_{\text{model}}^2$ — for the Q, K, V and output projections, independent of the number of heads.

Q: What does $W_O$ do?
A: It projects the concatenated head outputs back to $d_{\text{model}}$, mixing information across heads.

Q: Why can't a single head replace several heads?
A: One softmax gives one weighted average per query, so it can't sharply attend to several positions for different reasons at once.

Q: What do MQA and GQA change relative to MHA?
A: They let several query heads share the same key/value heads, shrinking the KV cache.
