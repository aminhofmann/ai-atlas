---
id: superposition
title: Superposition
category: interpretability
level: 3
year: 2022
summary: >-
  The hypothesis that networks represent more features than they have
  dimensions by storing them as nearly-orthogonal directions, which is why single
  neurons are often polysemantic.
requires: [embeddings, feed-forward-network, mechanistic-interpretability]
links:
  - { to: sparse-autoencoders, type: related }
  - { to: probing, type: related }
  - { to: residual-connections, type: related }
sources:
  - { title: "Toy Models of Superposition (Elhage et al., 2022)", url: "https://arxiv.org/abs/2209.10652" }
  - { title: "Zoom In: An Introduction to Circuits (Olah et al., Distill 2020) — polysemantic neurons", url: "https://distill.pub/2020/circuits/zoom-in/" }
  - { title: "Linear Algebraic Structure of Word Senses, with Applications to Polysemy (Arora et al., 2018)", url: "https://arxiv.org/abs/1601.03764" }
---

## Intuition
If a model had one neuron per concept, interpretability would be easy: find the
"cat" neuron and read it. In practice, many neurons fire for unrelated things —
cat faces, car fronts and a piece of text in some language. These are
**polysemantic** neurons.

Superposition explains why. The world contains far more useful features than a
layer has dimensions, but most features are **sparse** — rarely active at the
same time. A model can then cram many features into a smaller space by giving
each a direction that is *almost* orthogonal to the others. When only a few are
active, they barely interfere, like many radio stations sharing a band because
most are silent at any moment. The price is occasional crosstalk, and neurons
that no longer line up with single concepts.

## Mechanics
Elhage et al. (2022) study a toy model. Features $x \in \mathbb{R}^{n}$ (each
zero with probability $S$, otherwise uniform in $[0,1]$, with importance
weights $I_i$) are compressed into $m < n$ dimensions and reconstructed:

$$h = W x, \qquad \hat{x} = \mathrm{ReLU}(W^\top h + b), \qquad W \in \mathbb{R}^{m \times n},$$

trained on $\sum_i I_i (x_i - \hat{x}_i)^2$. Findings:

- **Dense features** ($S \approx 0$): the model behaves like PCA, keeping the
  $m$ most important features and dropping the rest.
- **Sparse features**: it stores *more* than $m$ features in superposition; the
  ReLU filters out small negative interference.
- **Phase changes**: whether a feature is ignored, gets its own dimension, or
  is superposed changes sharply with sparsity and importance.
- **Geometry**: superposed features arrange into regular structures (pairs
  antipodal, triangles, pentagons, tetrahedra), with fractional "dimensions per
  feature".
- Superposition can also happen in *computation*, not just storage.

Why high dimensions help: in $\mathbb{R}^{m}$ you can fit exponentially many
vectors with pairwise cosine below any fixed $\varepsilon$ (Johnson–Lindenstrauss
intuition), so the residual stream of a large model could hold vastly more
features than $d$.

Consequence: the natural unit of analysis is a **direction**, not a neuron, and
the directions form an overcomplete basis — exactly what
sparse autoencoders try to recover. A **privileged basis**
exists only where an elementwise nonlinearity (e.g. in the MLP) breaks rotational
symmetry; the residual stream has none.

## Code
```python
import torch

n, m, S = 20, 5, 0.95                       # 20 features, 5 dims, 95% sparse
W = torch.nn.Parameter(torch.randn(m, n) * 0.1)
b = torch.nn.Parameter(torch.zeros(n))
imp = 0.9 ** torch.arange(n)                # decreasing feature importance
opt = torch.optim.Adam([W, b], lr=1e-2)

for step in range(5000):
    x = torch.rand(1024, n) * (torch.rand(1024, n) > S)
    x_hat = torch.relu(x @ W.T @ W + b)
    loss = (imp * (x - x_hat) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()

norms = W.norm(dim=0)                       # feature i represented if ||W_i|| ~ 1
print("features represented:", (norms > 0.5).sum().item(), "in", m, "dims")
# Lower S (denser) -> about m features; higher S -> many more than m.
```

## Why it matters
Superposition is the leading explanation for why neuron-level interpretability
fails in transformers, and it reframed the goal of
mechanistic interpretability as finding an
overcomplete set of feature directions. It directly motivated dictionary
learning with sparse autoencoders. It also suggests why larger models
represent more concepts: more dimensions mean exponentially more room for
nearly-orthogonal features. Note it is a well-supported *hypothesis*, demonstrated
cleanly in toy models; how exactly it plays out in large models is still studied.

## Self-test
Q: What is a polysemantic neuron?
A: A neuron that responds to several unrelated features or concepts.

Q: What property of features makes superposition worthwhile?
A: Sparsity — if features are rarely active together, their nearly-orthogonal directions seldom interfere.

Q: In the toy model, what happens when features are dense?
A: The model keeps roughly the $m$ most important features in orthogonal directions (PCA-like) and drops the rest.

Q: Why does the residual stream lack a privileged basis?
A: It has no elementwise nonlinearity, so any rotation of it is equivalent; only nonlinearities like ReLU/GELU in MLPs make individual neuron axes special.
