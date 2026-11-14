---
id: attention
title: Attention
category: foundations
level: 1
year: 2014
summary: >-
  A differentiable lookup: each query computes similarity scores against a set
  of keys and returns the score-weighted average of the matching values.
requires: [embeddings]
links:
  - { to: transformer, type: part-of }
  - { to: encoder-decoder, type: related }
  - { to: flash-attention, type: related }
  - { to: attention-visualization, type: related }
sources:
  - { title: "Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau et al., 2014)", url: "https://arxiv.org/abs/1409.0473" }
  - { title: "Effective Approaches to Attention-based Neural Machine Translation (Luong et al., 2015)", url: "https://arxiv.org/abs/1508.04025" }
  - { title: "Attention Is All You Need (Vaswani et al., 2017)", url: "https://arxiv.org/abs/1706.03762" }
---

## Intuition
Think of a library. You walk in with a question (the **query**). Every book has a label on its spine (the **key**) and contents inside (the **value**). Instead of picking exactly one book, you skim how well each label matches your question, then read a blend of all books — mostly the well-matching ones, a little of the rest.

That "soft" choice is the key trick: because nothing is picked hard, the whole operation is differentiable, so the network can *learn* what to look for and what to advertise.

## Mechanics
General form: given one query $q$ and pairs $(k_j, v_j)_{j=1}^m$,
$$\alpha_j = \frac{\exp(s(q,k_j))}{\sum_{l}\exp(s(q,k_l))},\qquad \text{out} = \sum_j \alpha_j v_j .$$
The weights $\alpha$ are non-negative and sum to 1 — a probability distribution over positions.

Score functions $s$:
- **Additive** (Bahdanau, 2014): $s(q,k) = w^\top \tanh(W_q q + W_k k)$ — a small MLP.
- **Dot-product** (Luong, 2015): $s(q,k) = q^\top k$.
- **Scaled dot-product** (Transformer, 2017): $s(q,k) = q^\top k/\sqrt{d_k}$.

In matrix form with $Q \in \mathbb{R}^{n\times d_k}$, $K \in \mathbb{R}^{m\times d_k}$, $V \in \mathbb{R}^{m\times d_v}$:
$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V \;\in \mathbb{R}^{n\times d_v}.$$
Softmax runs over each row (over the $m$ keys). The score matrix $QK^\top$ is $n\times m$.

**Why divide by $\sqrt{d_k}$?** If the components of $q$ and $k$ are independent with mean 0 and variance 1, then $q^\top k = \sum_{i=1}^{d_k} q_i k_i$ has variance $d_k$. For large $d_k$ the logits get large, softmax saturates towards one-hot, and its gradients become vanishingly small. Dividing by $\sqrt{d_k}$ restores unit variance, keeping training stable. Dot-product attention is preferred over additive because it is a pair of matrix multiplications — fast on GPUs.

Further details:
- **Masks** are added to the logits before softmax ($-\infty$ to forbid a position): padding masks, causal masks.
- **Cost:** $O(n\,m\,d_k)$ time and $O(n\,m)$ memory for the score matrix.
- **Order-blindness:** the output is unchanged if key–value pairs are permuted together — positions must be supplied separately.
- Where $Q$, $K$, $V$ come from defines the flavour: same sequence → self-attention; different sequences → cross-attention.

## Code
```python
import math
import torch
import torch.nn.functional as F

def attention(Q, K, V, mask=None):
    """Q: (..., n, d_k), K: (..., m, d_k), V: (..., m, d_v)."""
    scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))  # (..., n, m)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))      # mask: True = keep
    weights = scores.softmax(dim=-1)                          # rows sum to 1
    return weights @ V, weights                               # (..., n, d_v)

Q, K, V = torch.randn(2, 5, 64), torch.randn(2, 7, 64), torch.randn(2, 7, 32)
out, w = attention(Q, K, V)
print(out.shape, w.sum(-1))   # (2, 5, 32), all ones

# PyTorch's fused kernel computes the same thing
print(torch.allclose(out, F.scaled_dot_product_attention(Q, K, V), atol=1e-5))
```

## Why it matters
Attention was introduced to fix the bottleneck of RNN encoder–decoders, which squeezed a whole source sentence into one fixed vector; it let the decoder look back at every source word. The Transformer then showed that attention alone — without recurrence — is enough, making it the core operation of modern language, vision and biology models. Its quadratic cost in sequence length is the main motivation for FlashAttention, sparse attention and attention alternatives such as state-space models.

## Self-test
Q: Write the scaled dot-product attention formula.
A: $\text{softmax}(QK^\top/\sqrt{d_k})\,V$, with softmax over the key dimension.

Q: Why scale the logits by $1/\sqrt{d_k}$?
A: The dot product of unit-variance vectors has variance $d_k$; without scaling, large logits saturate softmax and shrink its gradients.

Q: What shape does $\text{Attention}(Q,K,V)$ have for $n$ queries, $m$ keys and value width $d_v$?
A: $n \times d_v$ — one output per query.

Q: What problem did Bahdanau attention originally solve?
A: The fixed-length vector bottleneck of RNN encoder–decoders in machine translation.

Q: How is a position excluded from attention?
A: Its logit is set to $-\infty$ (via a mask) before the softmax, giving it weight 0.
