---
id: feed-forward-network
title: Feed-Forward Network (MLP, SwiGLU)
category: foundations
level: 2
year: 2017
summary: >-
  A two-layer MLP applied to every token independently inside each transformer
  block; it holds most of the parameters and does the per-token "thinking".
requires: [embeddings]
links:
  - { to: transformer-block, type: part-of }
  - { to: mixture-of-experts, type: related }
  - { to: superposition, type: related }
  - { to: llama, type: applied-in }
sources:
  - { title: "Attention Is All You Need (Vaswani et al., 2017)", url: "https://arxiv.org/abs/1706.03762" }
  - { title: "GLU Variants Improve Transformer (Shazeer, 2020)", url: "https://arxiv.org/abs/2002.05202" }
  - { title: "Gaussian Error Linear Units (GELUs) (Hendrycks & Gimpel, 2016)", url: "https://arxiv.org/abs/1606.08415" }
  - { title: "Transformer Feed-Forward Layers Are Key-Value Memories (Geva et al., 2020)", url: "https://arxiv.org/abs/2012.14913" }
---

## Intuition
After attention has gathered information from other tokens, each token needs to *process* what it collected. The feed-forward network is that private processing step: the same small neural network is applied to every token separately, like every employee running the same checklist on their own file.

It first expands the vector into a much wider space (many detectors, each checking for a pattern), keeps the detectors that fire, and compresses the result back. A lot of the model's stored knowledge appears to live in these layers.

## Mechanics
**Classic FFN** for one token $x \in \mathbb{R}^{d_{\text{model}}}$:
$$\text{FFN}(x) = W_2\,\sigma(W_1 x + b_1) + b_2,\qquad W_1 \in \mathbb{R}^{d_{ff}\times d_{\text{model}}},\ W_2 \in \mathbb{R}^{d_{\text{model}}\times d_{ff}}.$$
- Typically $d_{ff} = 4\,d_{\text{model}}$ (original: 512 → 2048 → 512).
- **Position-wise:** the same weights for every position, no interaction between tokens — equivalent to two $1\times1$ convolutions over the sequence. On $X \in \mathbb{R}^{n\times d}$ it is just two matrix multiplications.
- **Activations:** ReLU (original), GELU $x\,\Phi(x)$ (BERT, GPT-2), SiLU/Swish $x\,\text{sigmoid}(x)$.

**Gated variants (GLU family)** — Shazeer (2020):
$$\text{FFN}_{\text{SwiGLU}}(x) = W_2\big(\text{SiLU}(W_1 x) \odot W_3 x\big).$$
One branch produces a gate, the other a value; their elementwise product passes through $W_2$. With three matrices instead of two, $d_{ff}$ is usually shrunk to about $\tfrac{2}{3}\cdot 4d = \tfrac{8}{3}d$ to keep the parameter count equal. GeGLU uses GELU instead of SiLU. SwiGLU is used in Llama, PaLM and many current LLMs; Shazeer reported better perplexity than ReLU/GELU FFNs at matched compute, without a theoretical explanation.

**Parameter budget:** with $d_{ff}=4d$ the FFN has $8d^2$ parameters versus $4d^2$ for attention — roughly two thirds of each block, and the majority of a dense transformer's weights. Its FLOPs per token also dominate for moderate sequence lengths.

**Interpretation:**
- **Key–value memory view** (Geva et al.): rows of $W_1$ act as keys that detect input patterns; the matching columns of $W_2$ are values written to the residual stream when the key fires.
- Individual hidden neurons are often **polysemantic** (respond to unrelated concepts), a symptom of superposition; sparse autoencoders try to recover cleaner features.

**Mixture-of-experts** replaces this single FFN with many expert FFNs and a router that sends each token to a few of them — more parameters at similar per-token compute.

## Code
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FFN(nn.Module):                      # classic: d -> 4d -> d
    def __init__(self, d, d_ff=None):
        super().__init__()
        d_ff = d_ff or 4 * d
        self.w1, self.w2 = nn.Linear(d, d_ff), nn.Linear(d_ff, d)
    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))

class SwiGLU(nn.Module):                   # Llama-style gated FFN, no biases
    def __init__(self, d, d_ff=None):
        super().__init__()
        d_ff = d_ff or int(8 * d / 3)       # keeps params ~ equal to the 4d FFN
        self.w1 = nn.Linear(d, d_ff, bias=False)   # gate branch
        self.w3 = nn.Linear(d, d_ff, bias=False)   # value branch
        self.w2 = nn.Linear(d_ff, d, bias=False)   # down projection
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

x = torch.randn(2, 10, 512)                # applied to every token independently
count = lambda m: sum(p.numel() for p in m.parameters())
print(FFN(512)(x).shape, count(FFN(512)), count(SwiGLU(512)))
# Position-wise: processing one token alone gives the same result
f = SwiGLU(512)
print(torch.allclose(f(x)[:, 3], f(x[:, 3]), atol=1e-6))
```

## Why it matters
The FFN is where most of a transformer's parameters — and much of its factual knowledge — sit, so it is the main target for scaling tricks: gated activations (SwiGLU) for quality, mixture-of-experts for capacity at constant compute, and pruning/quantization for efficiency. Attention routes information; the FFN transforms it. Neither alone gives a capable model.

## Self-test
Q: Write the classic transformer FFN.
A: $\text{FFN}(x) = W_2\,\sigma(W_1x + b_1) + b_2$ with hidden width $d_{ff}$, typically $4d_{\text{model}}$.

Q: What does "position-wise" mean for the FFN?
A: The same weights are applied to each token independently; tokens do not interact inside it.

Q: Write the SwiGLU FFN.
A: $W_2(\text{SiLU}(W_1x) \odot W_3x)$ — a gated product of two projections, then a down projection.

Q: Why is $d_{ff}$ reduced to about $\tfrac{8}{3}d$ in SwiGLU models?
A: SwiGLU has three weight matrices instead of two, so shrinking the hidden width keeps the parameter count equal to a $4d$ FFN.

Q: What share of a dense block's parameters does the FFN hold with $d_{ff}=4d$?
A: About two thirds ($8d^2$ of $\approx 12d^2$).
