---
id: state-space-models
title: State Space Models (S4 and lineage)
category: innovations
level: 3
year: 2021
summary: >-
  Sequence layers built on a linear dynamical system with a hidden state, which
  can run as a fast recurrence or as a long convolution and scale linearly
  with sequence length.
requires: [embeddings]
links:
  - { to: attention, type: alternative-to }
  - { to: mamba, type: related }
  - { to: hybrid-architectures, type: part-of }
  - { to: hyenadna, type: related }
  - { to: caduceus, type: related }
  - { to: evo, type: related }
  - { to: flash-attention, type: related }
sources:
  - { title: "HiPPO: Recurrent Memory with Optimal Polynomial Projections (Gu et al., 2020)", url: "https://arxiv.org/abs/2008.07669" }
  - { title: "Efficiently Modeling Long Sequences with Structured State Spaces — S4 (Gu, Goel & Ré, 2021)", url: "https://arxiv.org/abs/2111.00396" }
  - { title: "On the Parameterization and Initialization of Diagonal State Space Models — S4D (Gu et al., 2022)", url: "https://arxiv.org/abs/2206.11893" }
  - { title: "Hungry Hungry Hippos: Towards Language Modeling with State Space Models — H3 (Fu et al., 2022)", url: "https://arxiv.org/abs/2212.14052" }
  - { title: "Hyena Hierarchy: Towards Larger Convolutional Language Models (Poli et al., 2023)", url: "https://arxiv.org/abs/2302.10866" }
---

## Intuition
Attention compares every token with every other token, so cost grows with the
square of the length. An RNN instead carries a running memory forward, cheap
per step but hard to train and forgetful. State space models are a principled
middle ground: a hidden state evolves according to fixed linear rules borrowed
from control theory, carefully designed to remember the history well.

The neat part: because the rules are linear and the same at every step, the
whole layer is *also* one long convolution. So you train it in parallel like a
CNN and run it step by step like an RNN — whichever is cheaper at the moment.

## Mechanics
**Continuous system.** For input signal $u(t)$, state $x(t) \in \mathbb{R}^N$,
output $y(t)$:
$$x'(t) = A x(t) + B u(t), \qquad y(t) = C x(t) + D u(t)$$
with $A \in \mathbb{R}^{N\times N}$, $B \in \mathbb{R}^{N\times 1}$, $C \in \mathbb{R}^{1\times N}$.
One SSM maps one channel; a layer runs $d$ independent SSMs, one per channel,
followed by a mixing projection.

**Discretisation** with step size $\Delta$ (e.g. zero-order hold):
$\bar A = \exp(\Delta A)$, $\bar B = (\Delta A)^{-1}(\exp(\Delta A) - I)\,\Delta B$.

**Two equivalent computations:**
- **Recurrent:** $x_k = \bar A x_{k-1} + \bar B u_k$, $y_k = C x_k$ — $O(1)$ per
  token at inference, constant memory.
- **Convolutional:** $y = \bar K * u$ with kernel
  $\bar K = (C\bar B,\, C\bar A\bar B,\, C\bar A^2\bar B,\, \dots)$, computed with
  FFTs in $O(L \log L)$ for training.

**What made S4 work:**
- **HiPPO initialisation** of $A$: a matrix derived so the state optimally
  compresses the input history onto a polynomial basis — the source of
  long-range memory. Random $A$ performs badly.
- **Structured parameterisation:** HiPPO-type $A$ written as normal plus
  low-rank (NPLR / "diagonal plus low-rank"), allowing the kernel to be
  computed stably and efficiently.
- S4 was the first model to solve **Path-X** (length 16,384) in the Long Range
  Arena benchmark, where transformers did no better than chance.

**Lineage:**
- **DSS / S4D:** purely diagonal $A$ is almost as good and far simpler.
- **S5:** a single multi-input multi-output SSM computed with a parallel scan.
- **H3:** SSMs plus multiplicative gating to handle associative recall, closing
  much of the language-modeling gap.
- **Hyena:** replaces the SSM with implicitly parameterised long convolutions
  plus gating (not strictly an SSM, same family of ideas).
- **Mamba:** makes $\Delta, B, C$ input-dependent ("selective"), abandoning the
  convolution view for a hardware-aware scan.

**Key limitation of LTI SSMs:** $A, B, C, \Delta$ are the same for every input
(linear time-invariant), so the model cannot choose what to remember based on
content — the gap Mamba targets.

## Code
```python
import torch

def discretize(A, B, dt):                    # zero-order hold, diagonal A (S4D-style)
    Ab = torch.exp(dt * A)                   # (N,)
    Bb = (Ab - 1) / A * B                    # (N,)
    return Ab, Bb

def ssm_recurrent(u, A, B, C, dt):           # u: (L,) one channel
    Ab, Bb = discretize(A, B, dt)
    x, ys = torch.zeros_like(A), []
    for uk in u:
        x = Ab * x + Bb * uk                 # state update
        ys.append((C * x).sum().real)
    return torch.stack(ys)

def ssm_convolutional(u, A, B, C, dt):       # same output via a kernel
    Ab, Bb = discretize(A, B, dt)
    L = u.shape[0]
    K = torch.stack([(C * Ab**k * Bb).sum().real for k in range(L)])
    return torch.stack([(K[:k+1].flip(0) * u[:k+1]).sum() for k in range(L)])

N = 16
A = -0.5 + 1j * torch.pi * torch.arange(N)   # S4D-Lin-style init
B, C = torch.ones(N, dtype=torch.cfloat), torch.randn(N, dtype=torch.cfloat)
u = torch.randn(32)
print(torch.allclose(ssm_recurrent(u, A, B, C, 0.1), ssm_convolutional(u, A, B, C, 0.1), atol=1e-4))
```

## Why it matters
S4 showed that a recurrence with the right structure can model very long
dependencies and be trained in parallel, reviving recurrent ideas after
transformers. Its lineage (H3, Hyena, Mamba) produced genuine linear-time
alternatives to attention, and SSM-family layers now power long-sequence
genomics models (HyenaDNA, Caduceus, Evo) and hybrid LLMs.

## Self-test
Q: What are the continuous-time equations of a state space model?
A: x'(t) = A x(t) + B u(t) and y(t) = C x(t) + D u(t).

Q: What two equivalent ways can a linear time-invariant SSM be computed?
A: As a recurrence (step by step, constant memory, good for inference) or as a long convolution with kernel (CB̄, CĀB̄, CĀ²B̄, …), good for parallel training.

Q: What role does HiPPO play in S4?
A: It provides the initialisation of A that lets the state optimally compress input history, giving long-range memory.

Q: Which Long Range Arena task did S4 famously solve?
A: Path-X, with sequences of length 16,384.

Q: What limitation of S4-style SSMs motivated Mamba?
A: They are time-invariant: their parameters don't depend on the input, so they can't selectively remember or ignore tokens based on content.
