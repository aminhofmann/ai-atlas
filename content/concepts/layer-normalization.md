---
id: layer-normalization
title: Layer Normalization (and RMSNorm)
category: foundations
level: 2
year: 2016
summary: >-
  Rescales each token's vector to zero mean and unit variance (or, in RMSNorm,
  unit root-mean-square) with learned gains, keeping activations stable through deep stacks.
requires: [embeddings]
links:
  - { to: transformer-block, type: part-of }
  - { to: residual-connections, type: related }
  - { to: llama, type: applied-in }
  - { to: vision-transformer, type: applied-in }
sources:
  - { title: "Layer Normalization (Ba et al., 2016)", url: "https://arxiv.org/abs/1607.06450" }
  - { title: "Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)", url: "https://arxiv.org/abs/1910.07467" }
  - { title: "On Layer Normalization in the Transformer Architecture (Xiong et al., 2020)", url: "https://arxiv.org/abs/2002.04745" }
  - { title: "Scaling Vision Transformers to 22 Billion Parameters (Dehghani et al., 2023) — QK-norm", url: "https://arxiv.org/abs/2302.05442" }
---

## Intuition
As a signal passes through dozens of layers, its numbers can drift — getting huge in one place, tiny in another — and training becomes unstable. Layer normalization is a volume knob applied to each token separately: before a layer reads the token's vector, it is re-centred and rescaled to a standard loudness. Learned gains then let the model turn individual channels up or down as it likes.

Because each token is normalized on its own, it doesn't matter how many other tokens or examples are in the batch.

## Mechanics
**LayerNorm** for one token's vector $x \in \mathbb{R}^d$:
$$\mu = \frac{1}{d}\sum_{i} x_i,\quad \sigma^2 = \frac{1}{d}\sum_i (x_i-\mu)^2,\quad \text{LN}(x) = \gamma \odot \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}} + \beta,$$
with learned $\gamma, \beta \in \mathbb{R}^d$ and small $\epsilon$ (e.g. $10^{-5}$). For $X \in \mathbb{R}^{B\times n\times d}$, statistics are computed over the **last axis only** — one $\mu,\sigma$ per token.

**Versus BatchNorm:** BatchNorm normalizes each feature over the batch (and needs running statistics at inference). That is awkward for variable-length sequences, small batches and autoregressive decoding; LayerNorm behaves identically in training and inference and for any batch size.

**RMSNorm** drops the mean-centring and the bias:
$$\text{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} .$$
It is cheaper (one reduction instead of two), and Zhang & Sennrich found comparable quality — re-scaling invariance seems to matter more than re-centring. T5 uses this simplified form, and Llama and most of its descendants use RMSNorm.

**Placement** in a transformer block:
- **Post-norm** (2017 original, BERT): $\text{LN}(x + F(x))$ — normalizes the residual stream itself; needs learning-rate warm-up.
- **Pre-norm** (GPT-2 onward): $x + F(\text{LN}(x))$ — normalizes only the sublayer's *input*; the residual stream stays un-normalized, so a final norm is applied before the unembedding. Easier to train deep.
- **QK-norm:** normalizing queries and keys before their dot product to prevent attention logits from growing without bound; used e.g. in ViT-22B.

Note that pre-norm makes each sublayer see a scale-free input while the residual stream's norm can grow with depth — one reason the logit lens applies the final norm before decoding intermediate states.

## Code
```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(d))  # gamma only

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * x / rms

def layer_norm(x, gamma, beta, eps=1e-5):
    mu = x.mean(-1, keepdim=True)                     # per token, over features
    var = x.var(-1, keepdim=True, unbiased=False)
    return gamma * (x - mu) / torch.sqrt(var + eps) + beta

x = torch.randn(2, 10, 64) * 5 + 3                    # (batch, tokens, d)
ln = nn.LayerNorm(64)
print(torch.allclose(layer_norm(x, ln.weight, ln.bias), ln(x), atol=1e-5))
y = RMSNorm(64)(x)
print(y.pow(2).mean(-1)[0, :3])                       # ~1 for every token
```

## Why it matters
Normalization is one of the ingredients — with residual connections and careful initialization — that makes deep transformers trainable at all. Its *placement* (pre vs post) was a major stability lever as models grew deeper, and RMSNorm has become the default in modern LLMs because it is simpler and faster with no measurable quality cost. LayerNorm's per-token design is also why transformers work the same at any batch size and during token-by-token generation.

## Self-test
Q: Over which axis does LayerNorm compute its statistics in a transformer?
A: Over the feature dimension $d_{\text{model}}$ of each token separately.

Q: What does RMSNorm remove compared with LayerNorm?
A: The mean subtraction and the bias $\beta$; it only divides by the root-mean-square and multiplies by $\gamma$.

Q: Why is LayerNorm preferred over BatchNorm in transformers?
A: It does not depend on batch statistics, so it works for variable-length sequences, small batches and step-by-step decoding identically in training and inference.

Q: Where is the norm applied in a pre-norm block?
A: On the input to each sublayer, $x + F(\text{LN}(x))$, plus one final norm before the output layer.

Q: What is QK-norm for?
A: Normalizing queries and keys before the dot product to keep attention logits from growing and destabilizing training.
