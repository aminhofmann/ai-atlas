---
id: query-key-value
title: Queries, Keys and Values
category: foundations
level: 1
year: 2017
summary: >-
  Three learned linear projections of each token: the query says what it looks
  for, the key what it can be matched on, and the value what it passes along.
requires: [attention]
links:
  - { to: multi-head-attention, type: part-of }
  - { to: kv-cache, type: related }
  - { to: mechanistic-interpretability, type: related }
sources:
  - { title: "Attention Is All You Need (Vaswani et al., 2017)", url: "https://arxiv.org/abs/1706.03762" }
  - { title: "A Mathematical Framework for Transformer Circuits (Elhage et al., 2021)", url: "https://transformer-circuits.pub/2021/framework/index.html" }
---

## Intuition
At a networking event, every person carries three things: a question they want answered ("who knows about protein folding?"), a name tag describing what they can offer ("structural biologist"), and the actual knowledge they would share in a conversation.

Queries, keys and values are those three things for each token. Matching happens between questions and name tags; what gets passed along is the knowledge. Keeping the name tag separate from the knowledge lets a token be *found* for one reason while *contributing* something else.

## Mechanics
Given hidden states $X \in \mathbb{R}^{n \times d_{\text{model}}}$:
$$Q = XW_Q,\quad K = XW_K,\quad V = XW_V,$$
with $W_Q, W_K \in \mathbb{R}^{d_{\text{model}}\times d_k}$ and $W_V \in \mathbb{R}^{d_{\text{model}}\times d_v}$. In multi-head attention $d_k = d_v = d_{\text{model}}/h$ per head (original base model: $512/8 = 64$). Many recent models drop the bias terms.

Roles:
- **Query** $q_i$: what position $i$ is searching for.
- **Key** $k_j$: what position $j$ advertises for matching. Score: $q_i^\top k_j / \sqrt{d_k}$.
- **Value** $v_j$: the content position $j$ delivers if attended to. Output $i$: $\sum_j \alpha_{ij} v_j$.

Why three separate projections rather than comparing raw $x_i^\top x_j$:
- **Asymmetry:** $q_i^\top k_j \ne q_j^\top k_i$ in general, so "i looks at j" need not imply "j looks at i" (e.g. a pronoun attending to its noun).
- **Low-rank bilinear form:** the score equals $x_i^\top W_Q W_K^\top x_j$; the combined $W_{QK} = W_Q W_K^\top \in \mathbb{R}^{d\times d}$ has rank $\le d_k$, so each head compares tokens in a small learned subspace. Elhage et al. call this the **QK circuit** (where to attend).
- **Decoupled content:** what is moved is $W_V$ followed by the output projection $W_O$; their product $W_V W_O$ is the **OV circuit** (what is copied). Matching and content are learned independently.

Where they come from defines the attention type:
- **Self-attention:** $Q$, $K$, $V$ all from the same sequence $X$.
- **Cross-attention:** $Q$ from one sequence (e.g. decoder), $K$ and $V$ from another (e.g. encoder output).

During autoregressive generation, $K$ and $V$ of past tokens never change, which is why they — not $Q$ — are stored in the KV cache.

## Code
```python
import torch
import torch.nn as nn

d_model, d_k, d_v, n = 512, 64, 64, 10
W_q = nn.Linear(d_model, d_k, bias=False)
W_k = nn.Linear(d_model, d_k, bias=False)
W_v = nn.Linear(d_model, d_v, bias=False)

X = torch.randn(n, d_model)            # one sequence of hidden states
Q, K, V = W_q(X), W_k(X), W_v(X)       # (n, 64) each

scores = Q @ K.T / d_k**0.5            # (n, n): row i = how much i wants each j
print(torch.allclose(scores, scores.T))  # False — attention is asymmetric

# The same scores via the combined low-rank QK matrix (rank <= d_k)
W_qk = W_q.weight.T @ W_k.weight       # (d_model, d_model)
print(torch.allclose(scores, X @ W_qk @ X.T / d_k**0.5, atol=1e-4))
print(torch.linalg.matrix_rank(W_qk))  # 64
```

## Why it matters
The Q/K/V decomposition is what turns attention from a fixed similarity measure into a learned, content-dependent routing mechanism, and it is shared by every transformer variant. Modern efficiency work operates directly on it: multi-query and grouped-query attention share K/V across heads, latent attention compresses them, and interpretability research analyses heads through their QK and OV circuits.

## Self-test
Q: What are the shapes of $W_Q$ and $W_V$ in a single head?
A: $W_Q \in \mathbb{R}^{d_{\text{model}}\times d_k}$ and $W_V \in \mathbb{R}^{d_{\text{model}}\times d_v}$.

Q: Why is attention between two tokens generally asymmetric?
A: Because queries and keys use different projections, $q_i^\top k_j \ne q_j^\top k_i$.

Q: What is the maximum rank of $W_Q W_K^\top$ in one head?
A: $d_k$, the head dimension.

Q: In cross-attention, which sequence provides the queries?
A: The sequence being updated (e.g. the decoder); keys and values come from the other sequence (e.g. the encoder output).

Q: Why are keys and values cached during generation but queries are not?
A: Past keys and values never change and are reused at every step; a query is only needed once, for the current token.
