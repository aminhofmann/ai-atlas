---
id: flash-attention
title: FlashAttention
category: innovations
level: 3
year: 2022
summary: >-
  Computes exact attention without ever writing the full n×n score matrix to
  GPU memory, by processing it in tiles kept in fast on-chip memory — much
  faster and with memory linear in sequence length.
requires: [attention, self-attention]
links:
  - { to: multi-head-attention, type: applied-in }
  - { to: causal-masking, type: uses }
  - { to: grouped-query-attention, type: related }
  - { to: state-space-models, type: related }
  - { to: mamba, type: related }
sources:
  - { title: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Dao et al., 2022)", url: "https://arxiv.org/abs/2205.14135" }
  - { title: "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning (Dao, 2023)", url: "https://arxiv.org/abs/2307.08691" }
  - { title: "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision (Shah et al., 2024)", url: "https://arxiv.org/abs/2407.08608" }
  - { title: "Online normalizer calculation for softmax (Milakov & Gimelshein, 2018)", url: "https://arxiv.org/abs/1805.02867" }
---

## Intuition
A GPU has a small, very fast scratchpad (on-chip SRAM) and a big, much slower
main memory (HBM). Naive attention computes the full score matrix, writes it
to main memory, reads it back for the softmax, writes again, reads again for
multiplying by $V$. For long sequences, almost all the time goes to this
shuttling, not to arithmetic.

FlashAttention reorganises the work so a block of queries and a block of keys
are brought into the scratchpad, fully processed there, and only the running
result is kept. The big matrix never exists in main memory. The answer is
*exactly* the same — this is not an approximation, it's better bookkeeping.

## Mechanics
Standard attention: $S = QK^\top/\sqrt{d}$, $P = \mathrm{softmax}(S)$, $O = PV$,
with $Q,K,V \in \mathbb{R}^{N \times d}$. Materialising $S$ and $P$ costs
$O(N^2)$ HBM reads/writes and memory.

**Tiling + online softmax.** Split $Q$ into row blocks and $K,V$ into column
blocks sized to fit SRAM. For a query block, iterate over key blocks $j$ and
keep per-row running statistics: max $m$, normaliser $\ell$, and unnormalised
output $O$:

$$m^{\text{new}} = \max(m, \operatorname{rowmax}(S_j)), \quad
\ell^{\text{new}} = e^{m - m^{\text{new}}}\ell + \operatorname{rowsum}\big(e^{S_j - m^{\text{new}}}\big)$$
$$O^{\text{new}} = e^{m - m^{\text{new}}} O + e^{S_j - m^{\text{new}}} V_j$$

At the end, $O \leftarrow O / \ell$. Rescaling by $e^{m - m^{\text{new}}}$ fixes
earlier partial sums when a larger max appears, so the result equals the exact
softmax.

**Kernel fusion.** Matmul, masking, softmax, dropout and the $PV$ matmul run in
one CUDA kernel.

**Recomputation in backward.** Instead of storing $P$ ($N^2$), store only $O$
and the softmax statistics ($m,\ell$; $O(N)$), and recompute score blocks in
the backward pass. More FLOPs, fewer memory accesses — faster overall.

**IO complexity.** With SRAM size $M$, FlashAttention needs
$\Theta(N^2 d^2 / M)$ HBM accesses versus $\Theta(Nd + N^2)$ for standard
attention; extra memory is $O(N)$ instead of $O(N^2)$.

**Versions:** FlashAttention-2 (2023) reduces non-matmul FLOPs and parallelises
over the sequence dimension too, roughly 2× faster than v1. FlashAttention-3
(2024) targets Hopper GPUs with asynchronous warp-specialised pipelines and
FP8 support.

Compute is still $O(N^2 d)$: FlashAttention removes the memory bottleneck, not
the quadratic arithmetic.

## Code
```python
import torch

def flash_attention_ref(q, k, v, block=64):
    """Pure-PyTorch reference of the tiled online-softmax algorithm (one head)."""
    N, d = q.shape
    out = torch.zeros_like(q)
    m = torch.full((N, 1), float("-inf"))       # running row max
    l = torch.zeros(N, 1)                       # running normaliser
    for j in range(0, N, block):                # stream key/value blocks
        kj, vj = k[j:j+block], v[j:j+block]
        s = q @ kj.T / d**0.5                   # (N, block) – only one tile at a time
        m_new = torch.maximum(m, s.max(-1, keepdim=True).values)
        p = torch.exp(s - m_new)
        scale = torch.exp(m - m_new)
        l = scale * l + p.sum(-1, keepdim=True)
        out = scale * out + p @ vj
        m = m_new
    return out / l

q, k, v = (torch.randn(256, 64) for _ in range(3))
ref = torch.softmax(q @ k.T / 8, -1) @ v
print(torch.allclose(flash_attention_ref(q, k, v), ref, atol=1e-5))
# In practice: torch.nn.functional.scaled_dot_product_attention dispatches to a flash kernel.
```

## Why it matters
FlashAttention made long-context training practical and sped up training of
virtually every transformer; it is now the default attention kernel in PyTorch,
HuggingFace and inference engines. It also shifted how the field thinks about
efficiency: count memory traffic, not just FLOPs ("IO-awareness"). That same
hardware-aware style shaped Mamba's selective-scan kernel.

## Self-test
Q: Is FlashAttention an approximation of attention?
A: No; it computes exact attention, only reorganising the computation to reduce memory traffic.

Q: What large intermediate does FlashAttention avoid writing to HBM?
A: The N×N attention score/probability matrix.

Q: How does the online softmax let attention be computed block by block?
A: It keeps a running row max and normaliser and rescales previous partial outputs whenever a new block raises the max.

Q: Why does FlashAttention recompute attention scores in the backward pass?
A: Storing the N×N matrix would cost quadratic memory and traffic; recomputing tiles from Q, K and the saved statistics is faster on memory-bound hardware.

Q: Does FlashAttention change the O(N²) compute of attention?
A: No; FLOPs stay quadratic, but extra memory becomes linear and HBM accesses drop sharply.
