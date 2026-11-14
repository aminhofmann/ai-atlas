---
id: mamba
title: Mamba (Selective SSM) and Mamba-2
category: innovations
level: 3
year: 2023
summary: >-
  A state space model whose parameters depend on the current input, letting it
  choose what to remember or forget, with a hardware-aware scan that gives
  linear-time training and constant-memory generation.
requires: [state-space-models]
links:
  - { to: attention, type: alternative-to }
  - { to: hybrid-architectures, type: part-of }
  - { to: flash-attention, type: related }
  - { to: caduceus, type: related }
  - { to: kv-cache, type: related }
sources:
  - { title: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao, 2023)", url: "https://arxiv.org/abs/2312.00752" }
  - { title: "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality — Mamba-2 (Dao & Gu, 2024)", url: "https://arxiv.org/abs/2405.21060" }
---

## Intuition
Earlier SSMs treat every token the same way: the rules for updating memory
never change. That is fine for audio but bad for language, where you must
decide on the fly "this name matters, keep it" or "this filler word, ignore
it". Mamba lets each token set the knobs: how much to write into memory, how
fast to forget, and what to read out.

The cost is that the clever convolution trick no longer works, because the
rules change at every step. Mamba compensates with a carefully engineered GPU
scan, so it still trains fast and generates with a fixed-size memory instead
of a growing KV cache.

## Mechanics
**Selective SSM (S6).** For input $x \in \mathbb{R}^{B \times L \times D}$ and
state size $N$, compute per token:
$$B_t = \mathrm{Linear}_N(x_t), \quad C_t = \mathrm{Linear}_N(x_t), \quad \Delta_t = \mathrm{softplus}\big(\mathrm{Linear}_D(x_t) + b\big)$$
$A$ stays an input-independent diagonal (negative real) parameter of shape
$D \times N$. Discretise per token: $\bar A_t = \exp(\Delta_t A)$, $\bar B_t \approx \Delta_t B_t$, then
$$h_t = \bar A_t \odot h_{t-1} + \bar B_t\, x_t, \qquad y_t = C_t h_t \;(+ D x_t)$$

**Why $\Delta$ is the key knob:** large $\Delta_t$ → $\bar A_t \to 0$, the state
resets and focuses on the current token; small $\Delta_t$ → $\bar A_t \to 1$,
the token is mostly ignored and memory persists. This is a learned,
content-dependent gate, closely related to RNN gating.

**Hardware-aware selective scan.** Time-varying parameters rule out the
convolution, so Mamba uses a **parallel associative scan** ($O(L)$ work) in a
fused kernel: parameters are loaded into SRAM, the expanded state
$(B, L, D, N)$ is materialised only in SRAM, never in HBM, and intermediate
states are **recomputed** in the backward pass — the FlashAttention playbook.

**Mamba block.** Replaces both attention and MLP: input projection that
expands width (factor 2) into two branches; one branch → short causal conv1d →
SiLU → selective SSM; the other → SiLU gate; multiply, project back. Blocks
are stacked homogeneously with normalisation and residuals.

**Results reported:** about **5× higher generation throughput** than
transformers of similar size, linear scaling in sequence length, and Mamba-3B
matching transformers about twice its size on language modeling.

**Mamba-2 (SSD).** Restricts $A$ to a **scalar times identity** per head. The
SSM then equals a form of masked (causally structured, "semiseparable")
linear attention, which can be computed mostly with **matrix multiplications**
on tensor cores. Benefits: 2–8× faster core layer than Mamba-1's scan, much
larger state sizes (e.g. $N = 64$–$256$ vs $16$), multi-head structure, and
easier tensor parallelism.

**Weaknesses:** a fixed-size state cannot store everything; pure SSMs lag
attention on exact recall/copying tasks — the main motivation for hybrids.

## Code
```python
import torch, torch.nn as nn, torch.nn.functional as F

class SelectiveSSM(nn.Module):
    """Sequential reference of Mamba's S6 (real kernels use a fused parallel scan)."""
    def __init__(self, d, n=16):
        super().__init__()
        self.A_log = nn.Parameter(torch.log(torch.arange(1, n + 1).float()).repeat(d, 1))
        self.to_B, self.to_C = nn.Linear(d, n), nn.Linear(d, n)
        self.to_dt = nn.Linear(d, d)
        self.D = nn.Parameter(torch.ones(d))

    def forward(self, x):                                   # x: (batch, L, d)
        A = -torch.exp(self.A_log)                          # (d, n), negative real
        B, C = self.to_B(x), self.to_C(x)                   # (batch, L, n) input-dependent
        dt = F.softplus(self.to_dt(x))                      # (batch, L, d) input-dependent Δ
        h = x.new_zeros(x.size(0), x.size(2), A.size(1))    # (batch, d, n) fixed-size state
        ys = []
        for t in range(x.size(1)):
            Abar = torch.exp(dt[:, t, :, None] * A)         # per-token decay
            Bbar = dt[:, t, :, None] * B[:, t, None, :]     # ≈ Δ·B
            h = Abar * h + Bbar * x[:, t, :, None]
            ys.append((h * C[:, t, None, :]).sum(-1) + self.D * x[:, t])
        return torch.stack(ys, dim=1)

print(SelectiveSSM(8)(torch.randn(2, 10, 8)).shape)         # (2, 10, 8)
```

## Why it matters
Mamba was the first attention-free architecture widely seen as competitive
with transformers on language modeling at small-to-medium scale, while
offering linear-time training and constant-memory inference (no KV cache).
Mamba layers are now a standard ingredient in hybrid LLMs (Jamba and others)
and in long-sequence biology models such as Caduceus. Mamba-2's duality result
also tied SSMs and linear attention into one framework.

## Self-test
Q: Which SSM parameters are input-dependent in Mamba?
A: Δ (the step size), B and C; A remains an input-independent learned parameter.

Q: What does a large versus small Δ_t do?
A: Large Δ_t resets the state and focuses on the current token; small Δ_t keeps the previous state and largely ignores the current input.

Q: Why can't Mamba use the convolutional mode of S4?
A: Its parameters vary per token, so the system is no longer time-invariant and has no single convolution kernel.

Q: How does Mamba's kernel avoid HBM bottlenecks?
A: It fuses the scan, keeps the expanded state only in SRAM, and recomputes intermediate states in the backward pass.

Q: What structural restriction defines Mamba-2's SSD layer?
A: The state matrix A is a scalar times the identity, which makes the layer equivalent to a structured masked attention computable with matrix multiplications.
