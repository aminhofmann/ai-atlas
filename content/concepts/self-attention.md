---
id: self-attention
title: Self-Attention
category: foundations
level: 1
year: 2017
summary: >-
  Attention in which a sequence attends to itself, so every token's new
  representation is a learned mixture of all tokens in the same sequence.
requires: [query-key-value]
links:
  - { to: transformer-block, type: part-of }
  - { to: positional-encoding, type: related }
  - { to: flash-attention, type: related }
  - { to: bert, type: applied-in }
  - { to: vision-transformer, type: applied-in }
sources:
  - { title: "Attention Is All You Need (Vaswani et al., 2017)", url: "https://arxiv.org/abs/1706.03762" }
---

## Intuition
Read "The animal didn't cross the street because **it** was too tired." To understand "it", you glance back at "animal". Self-attention lets every word do this glancing simultaneously: each token asks the rest of its own sentence "who is relevant to me?" and rewrites itself as a blend of the answers.

After one round, "it" carries information about "animal"; after many rounds (layers), each token's vector reflects its meaning *in this specific context*.

## Mechanics
All three projections come from the same input $X \in \mathbb{R}^{n\times d}$:
$$Y = \text{softmax}\!\left(\frac{(XW_Q)(XW_K)^\top}{\sqrt{d_k}} + M\right)XW_V .$$
- The attention matrix $A = \text{softmax}(\cdot) \in \mathbb{R}^{n\times n}$; row $i$ is a distribution over the positions token $i$ reads from. $M$ is an optional mask (0 or $-\infty$).
- $y_i = \sum_j A_{ij}\, v_j$: every output is a convex combination of all value vectors — *contextualization*.
- **Bidirectional** (encoders, BERT): $M = 0$ except padding; every token sees all others.
- **Causal** (decoders, GPT): $M$ hides future positions.

Properties versus recurrence and convolution (as analysed by Vaswani et al.):

| | per-layer cost | sequential ops | max path length |
|---|---|---|---|
| self-attention | $O(n^2 d)$ | $O(1)$ | $O(1)$ |
| RNN | $O(n d^2)$ | $O(n)$ | $O(n)$ |
| convolution (kernel $k$) | $O(k n d^2)$ | $O(1)$ | $O(\log_k n)$ |

- **Constant path length:** any two positions interact directly in one layer, which helps learn long-range dependencies.
- **Fully parallel:** all positions are computed at once in training — the property that made large-scale pretraining practical on GPUs.
- **Quadratic cost:** the $n\times n$ score matrix grows quadratically with sequence length in time and (naively) memory.
- **Permutation-equivariant:** without positional information, shuffling the input just shuffles the output.
- **Input-dependent weights:** unlike a convolution kernel, the mixing weights $A$ are computed from the data itself on every forward pass.

In practice self-attention is always used in its multi-head form, with an output projection, wrapped in a residual connection and layer normalization inside a transformer block.

## Code
```python
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k):
        super().__init__()
        self.qkv = nn.Linear(d_model, 3 * d_k, bias=False)  # fused projections
        self.d_k = d_k

    def forward(self, x, mask=None):            # x: (batch, n, d_model)
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # each (batch, n, d_k)
        scores = q @ k.transpose(-2, -1) / self.d_k**0.5   # (batch, n, n)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        A = scores.softmax(-1)                  # attention matrix
        return A @ v, A

x = torch.randn(2, 6, 128)
y, A = SelfAttention(128, 64)(x)
print(y.shape, A.shape)                         # (2, 6, 64), (2, 6, 6)

# Permutation-equivariance: shuffle the input, the output shuffles identically
perm = torch.randperm(6)
attn = SelfAttention(128, 64)
print(torch.allclose(attn(x[:, perm])[0], attn(x)[0][:, perm], atol=1e-5))
```

## Why it matters
Self-attention replaced recurrence as the way sequence models share information between positions, removing the sequential bottleneck of RNNs and enabling training on vastly larger datasets. It is the mixing operation inside BERT, GPT, ViT and nearly every foundation model. Its $O(n^2)$ cost is the main limit on context length, which drives work on FlashAttention (exact but IO-efficient), sparse and sliding-window attention, and sub-quadratic alternatives such as state-space models.

## Self-test
Q: What distinguishes self-attention from cross-attention?
A: In self-attention queries, keys and values all come from the same sequence; in cross-attention keys and values come from a different one.

Q: What is the time complexity of one self-attention layer?
A: $O(n^2 d)$ for sequence length $n$ and width $d$.

Q: What is the maximum path length between two tokens in self-attention versus an RNN?
A: $O(1)$ for self-attention, $O(n)$ for an RNN.

Q: Why does self-attention need positional information?
A: It is permutation-equivariant: without position signals, reordering the input only reorders the output.

Q: What does row $i$ of the attention matrix represent?
A: A probability distribution over the positions that token $i$ reads from.
