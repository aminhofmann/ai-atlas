---
id: rotary-position-embedding
title: Rotary Position Embedding (RoPE)
category: foundations
level: 2
year: 2021
summary: >-
  Encodes position by rotating query and key vectors by an angle proportional to
  their position, so attention scores depend only on the relative distance.
requires: [positional-encoding, query-key-value]
links:
  - { to: llama, type: applied-in }
  - { to: esm2, type: applied-in }
  - { to: multi-head-attention, type: part-of }
  - { to: kv-cache, type: related }
  - { to: multi-head-latent-attention, type: related }
sources:
  - { title: "RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021)", url: "https://arxiv.org/abs/2104.09864" }
  - { title: "Extending Context Window of Large Language Models via Positional Interpolation (Chen et al., 2023)", url: "https://arxiv.org/abs/2306.15595" }
  - { title: "YaRN: Efficient Context Window Extension of Large Language Models (Peng et al., 2023)", url: "https://arxiv.org/abs/2309.00071" }
---

## Intuition
Imagine each query and key as a set of clock hands. Before comparing them, you turn every hand of a token at position $m$ forward by an amount proportional to $m$ — some hands turn fast, some slowly. When you then compare a query with a key, what matters is only the *difference* in how far they were turned, i.e. how far apart the two tokens are.

So RoPE gives the model relative position "for free", without adding anything to the token vectors themselves.

## Mechanics
Take a query or key vector $q \in \mathbb{R}^{d_{\text{head}}}$ ($d_{\text{head}}$ even) and split it into $d/2$ two-dimensional pairs. Pair $i$ at position $m$ is rotated by angle $m\theta_i$:
$$\begin{pmatrix} q'_{2i} \\ q'_{2i+1}\end{pmatrix} = \begin{pmatrix}\cos m\theta_i & -\sin m\theta_i\\ \sin m\theta_i & \cos m\theta_i\end{pmatrix}\begin{pmatrix} q_{2i} \\ q_{2i+1}\end{pmatrix},\qquad \theta_i = b^{-2i/d},\ b = 10000.$$
Write the full block-diagonal rotation as $R_m$. Since rotations satisfy $R_m^\top R_n = R_{n-m}$:
$$\langle R_m q,\; R_n k\rangle = q^\top R_{n-m}\, k,$$
so the attention logit depends on content and on the **offset** $n-m$ only. In complex notation: treat each pair as $z \in \mathbb{C}$ and multiply by $e^{i m \theta_i}$.

Properties and practice:
- Applied to $Q$ and $K$ **after** projection, in **every layer and head**; not to $V$, and nothing is added to the residual stream.
- Fast pairs (small $i$) resolve local order; slow pairs carry long-range position. The RoFormer paper argues the scheme yields a decay of interaction with distance.
- Two layouts exist: interleaved pairs $(0,1),(2,3),\dots$ (paper) and split halves $(i, i+d/2)$ via `rotate_half` (GPT-NeoX / HuggingFace Llama). They are equivalent up to a fixed permutation of weights, but checkpoints are not interchangeable without it.
- Some models rotate only a fraction of each head's dimensions ("partial rotary").
- **KV cache:** keys are cached already rotated at their absolute position; because scores only depend on offsets, nothing must be recomputed later.
- **Context extension:** beyond the training length, angles land outside the trained range. Fixes: *Position Interpolation* (scale positions $m \to m\,L/L'$), NTK-aware scaling (raise the base $b$), and YaRN (frequency-dependent interpolation plus attention temperature). Llama 3 raised the base to 500,000.

## Code
```python
import torch

def rope(x, base=10000.0):
    """x: (batch, heads, seq, d_head). 'rotate_half' layout (GPT-NeoX/Llama)."""
    *_, n, d = x.shape
    inv_freq = base ** (-torch.arange(0, d, 2).float() / d)   # theta_i, (d/2,)
    angles = torch.arange(n).float()[:, None] * inv_freq      # m*theta_i, (n, d/2)
    cos = torch.cat([angles.cos()] * 2, -1)                   # (n, d)
    sin = torch.cat([angles.sin()] * 2, -1)
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    rotated = torch.cat([-x2, x1], -1)                        # rotate_half
    return x * cos + rotated * sin

q = torch.randn(1, 1, 8, 64); k = torch.randn(1, 1, 8, 64)
# Shift both sequences by the same amount: relative scores are unchanged
q_shift = rope(torch.cat([torch.zeros(1, 1, 3, 64), q], 2))[..., 3:, :]
k_shift = rope(torch.cat([torch.zeros(1, 1, 3, 64), k], 2))[..., 3:, :]
s1 = rope(q) @ rope(k).transpose(-1, -2)
s2 = q_shift @ k_shift.transpose(-1, -2)
print(torch.allclose(s1, s2, atol=1e-4))   # True
```

## Why it matters
RoPE combines the cheapness of absolute encodings (one elementwise operation per layer) with the relative-position behaviour of methods like T5's bias, adds no parameters and works naturally with a KV cache. It became the default positional scheme in open LLMs (GPT-NeoX, Llama and most of their descendants) and appears in biological models such as ESM-2. Its frequency structure is also what context-extension methods manipulate.

## Self-test
Q: To which tensors is RoPE applied?
A: To the queries and keys of every attention head in every layer — not to the values or the residual stream.

Q: Why does the RoPE attention score depend only on relative position?
A: Because $\langle R_m q, R_n k\rangle = q^\top R_m^\top R_n k = q^\top R_{n-m}k$ — rotations compose by subtracting angles.

Q: What is the rotation frequency of pair $i$?
A: $\theta_i = b^{-2i/d}$ with base $b$ (classically 10000); position $m$ rotates by $m\theta_i$.

Q: Name two ways to extend a RoPE model's context window.
A: Position Interpolation (rescale positions into the trained range) and increasing the base (NTK-aware scaling / YaRN).

Q: Why doesn't the KV cache need recomputation under RoPE?
A: Cached keys are rotated once at their absolute position, and scores only depend on offsets, which stay correct as new tokens arrive.
