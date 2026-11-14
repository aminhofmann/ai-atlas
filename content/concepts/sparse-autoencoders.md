---
id: sparse-autoencoders
title: Sparse Autoencoders (Dictionary Learning)
category: interpretability
level: 3
year: 2023
summary: >-
  Training a wide, sparsely-activating autoencoder on a model's activations to
  decompose them into many interpretable "features" — undoing superposition.
requires: [superposition, residual-connections]
links:
  - { to: mechanistic-interpretability, type: part-of }
  - { to: probing, type: alternative-to }
  - { to: feed-forward-network, type: related }
  - { to: gpt, type: applied-in }
  - { to: esm2, type: applied-in }
  - { to: biological-foundation-models, type: applied-in }
sources:
  - { title: "Sparse Autoencoders Find Highly Interpretable Features in Language Models (Cunningham et al., 2023)", url: "https://arxiv.org/abs/2309.08600" }
  - { title: "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning (Bricken et al., 2023)", url: "https://transformer-circuits.pub/2023/monosemantic-features" }
  - { title: "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet (Templeton et al., 2024)", url: "https://transformer-circuits.pub/2024/scaling-monosemanticity/" }
  - { title: "Scaling and evaluating sparse autoencoders (Gao et al., 2024)", url: "https://arxiv.org/abs/2406.04093" }
---

## Intuition
If superposition is right, a model's activation vector is a
sum of a few active features out of a huge vocabulary of possible ones, each a
direction in space. Sparse autoencoders (SAEs) try to recover that vocabulary.

Think of a chord played on a piano: you hear one sound, but it is made of a few
keys out of 88. An SAE learns the "keys" (a dictionary of directions) and, for
each activation, which few keys are pressed and how hard. Because each key ends
up meaning one thing — "Golden Gate Bridge", "Python code", "a DNA sequence" —
the features are far more interpretable than raw neurons.

## Mechanics
Collect activations $x \in \mathbb{R}^{d}$ (residual stream or MLP output) from a
frozen model. Train an encoder–decoder with a much wider hidden layer
($F \gg d$, e.g. 8× to millions of features):

$$f = \mathrm{ReLU}(W_{e}(x - b_d) + b_e) \in \mathbb{R}^{F}, \qquad \hat{x} = W_d f + b_d,$$

$$\mathcal{L} = \lVert x - \hat{x} \rVert_2^2 + \lambda \lVert f \rVert_1 .$$

- Columns of $W_d \in \mathbb{R}^{d \times F}$ are the **dictionary** (feature
  directions), usually constrained to unit norm.
- The $L_1$ penalty makes most $f_i = 0$ for any input (typical $L_0$: tens of
  active features).
- **Interpretation**: for each feature, look at the inputs that activate it most
  and the logits its decoder direction promotes (via $W_U$).
- **Validation**: replace $x$ with $\hat{x}$ inside the model and measure the
  increase in loss; steer by adding $\alpha W_d[:, i]$ and observe behaviour.

Variants: **TopK SAEs** (Gao et al., 2024) keep only the $k$ largest
pre-activations instead of an $L_1$ penalty, avoiding shrinkage; JumpReLU and
gated SAEs serve a similar purpose; **transcoders** map MLP input to MLP output.
Known issues: dead features, feature splitting (features refine as $F$ grows),
and "dark matter" — reconstruction error that no feature explains.

Milestones: Cunningham et al. (2023) and Bricken et al. (2023, one-layer
transformer) showed SAE features are more monosemantic than neurons; Templeton
et al. (2024) scaled to a production model (Claude 3 Sonnet) with up to ~34M
features, including multilingual, multimodal and safety-relevant ones.

## Code
```python
import torch, torch.nn as nn

class SAE(nn.Module):
    def __init__(self, d, F):
        super().__init__()
        self.enc = nn.Linear(d, F)
        self.dec = nn.Linear(F, d, bias=True)
    def forward(self, x):
        f = torch.relu(self.enc(x - self.dec.bias))     # sparse codes (b, F)
        return self.dec(f), f

d, F, lam = 768, 768 * 16, 5e-4
sae = SAE(d, F)
opt = torch.optim.Adam(sae.parameters(), lr=1e-4)
acts = torch.randn(100_000, d)          # stand-in for cached GPT-2 residual activations

for batch in acts.split(4096):
    x_hat, f = sae(batch)
    loss = (batch - x_hat).pow(2).sum(-1).mean() + lam * f.abs().sum(-1).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():               # keep dictionary atoms unit-norm
        sae.dec.weight /= sae.dec.weight.norm(dim=0, keepdim=True)
print("avg active features:", (f > 0).sum(-1).float().mean().item())
```

## Why it matters
SAEs turned interpretability from hand-crafted probes into unsupervised
discovery: you don't need to guess the concept in advance. They are now the main
tool for finding features in frontier LLMs, building feature-level circuits, and
steering models, and have been applied to protein language models such as
ESM-2 and to single-cell models. Caveat: features depend on SAE size and
training choices, and whether SAEs recover the model's "true" features is an
open question.

## Self-test
Q: Why is the SAE hidden layer wider than the input?
A: Superposition implies there are more features than dimensions, so an overcomplete dictionary is needed to give each feature its own direction.

Q: What two terms make up the classic SAE loss?
A: Reconstruction error $\lVert x - \hat{x} \rVert^2$ plus an $L_1$ sparsity penalty on the feature activations.

Q: What does a column of the decoder matrix represent?
A: One feature's direction in the model's activation space.

Q: What does a TopK SAE change?
A: It keeps only the k largest feature activations per input instead of using an L1 penalty, directly controlling sparsity.

Q: How do you check that an SAE preserves what the model needs?
A: Splice the reconstruction back into the model and measure the increase in the model's loss.
