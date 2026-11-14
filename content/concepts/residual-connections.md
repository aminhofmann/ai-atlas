---
id: residual-connections
title: Residual Connections (the Residual Stream)
category: foundations
level: 1
year: 2015
summary: >-
  Adds each layer's output back onto its input (x + F(x)), giving gradients a
  direct path through the network and turning the hidden state into a shared
  "residual stream" that every layer reads from and writes to.
requires: []
links:
  - { to: transformer-block, type: part-of }
  - { to: layer-normalization, type: related }
  - { to: logit-lens, type: related }
  - { to: mechanistic-interpretability, type: related }
sources:
  - { title: "Deep Residual Learning for Image Recognition (He et al., 2015)", url: "https://arxiv.org/abs/1512.03385" }
  - { title: "Residual Networks Behave Like Ensembles of Relatively Shallow Networks (Veit et al., 2016)", url: "https://arxiv.org/abs/1605.06431" }
  - { title: "Attention is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth (Dong et al., 2021)", url: "https://arxiv.org/abs/2103.03404" }
  - { title: "A Mathematical Framework for Transformer Circuits (Elhage et al., 2021)", url: "https://transformer-circuits.pub/2021/framework/index.html" }
---

## Intuition
Instead of asking each layer to produce a brand-new representation, a residual connection asks it only for an *edit*: "here is what came in; what should be added?" If a layer has nothing useful to say, it can output roughly zero and the input passes through untouched.

In a transformer this turns the hidden state into a shared notebook — the **residual stream** — passed from layer to layer. Every attention head and FFN reads from it and writes a note into it, and nothing is ever overwritten, only added.

## Mechanics
A residual block computes
$$y = x + F(x),$$
so $F$ only has to learn the residual $y - x$. Requirement: $F(x)$ must have the same shape as $x$ — in transformers, every sublayer maps $\mathbb{R}^{d_{\text{model}}} \to \mathbb{R}^{d_{\text{model}}}$.

**Gradient flow:**
$$\frac{\partial y}{\partial x} = I + \frac{\partial F}{\partial x}.$$
The identity term carries gradient from the loss back to early layers without passing through every Jacobian in the chain, which counters vanishing gradients. He et al. (2015) used this to train CNNs with over 100 layers; deeper plain networks had *higher* training error, an optimization (not overfitting) problem.

**Ensemble view:** unrolling $L$ residual blocks gives a sum over $2^L$ paths of different lengths; Veit et al. (2016) found that ResNets behave partly like ensembles dominated by relatively short paths.

**In transformers** (pre-norm), after $L$ blocks:
$$x_L = x_0 + \sum_{\ell=1}^{L}\Big(\text{Attn}_\ell(\cdot) + \text{FFN}_\ell(\cdot)\Big),$$
where $x_0$ is the token embedding (plus position). The final state is literally the embedding plus the sum of all sublayer outputs.
- **Residual-stream view** (Elhage et al., 2021): the stream is a $d_{\text{model}}$-dimensional communication channel; components *read* via linear maps ($W_Q, W_K, W_V$, $W_1$) and *write* additively ($W_O$, $W_2$). Information written early can be read by any later layer, which is the basis of circuit analysis and the logit lens.
- **Necessity:** Dong et al. (2021) showed that pure self-attention stacks without skip connections (and MLPs) lose rank doubly exponentially with depth — token representations collapse towards each other; skip connections prevent this.
- **Initialization:** because outputs accumulate, GPT-2 scales the weights of residual-output layers at initialization by $1/\sqrt{N}$ ($N$ = number of residual layers) to keep the stream's variance controlled.
- **Normalization interplay:** pre-norm leaves the stream itself un-normalized (a clean identity path); post-norm renormalizes the stream after every addition.

## Code
```python
import torch
import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f
    def forward(self, x):
        return x + self.f(x)          # only an edit is learned

d, L = 64, 50
plain = nn.Sequential(*[nn.Sequential(nn.Linear(d, d), nn.Tanh()) for _ in range(L)])
resid = nn.Sequential(*[Residual(nn.Sequential(nn.Linear(d, d), nn.Tanh())) for _ in range(L)])

for name, net in [("plain", plain), ("residual", resid)]:
    x = torch.randn(8, d, requires_grad=True)
    net(x).sum().backward()
    print(name, "grad norm at input:", x.grad.norm().item())
# The plain 50-layer stack's input gradient is orders of magnitude smaller.

# Residual-stream bookkeeping: final state = input + sum of all writes
x0 = torch.randn(1, d); stream, writes = x0, []
for blk in resid:
    w = blk.f(stream); writes.append(w); stream = stream + w
print(torch.allclose(stream, x0 + sum(writes), atol=1e-5))
```

## Why it matters
Residual connections made depth trainable — first for CNNs (ResNet), then for transformers, which stack dozens to over a hundred sublayers. They are a precondition for the pre-norm design, for stable scaling, and for the modern mechanistic picture of a transformer as many components communicating through one shared, additive residual stream.

## Self-test
Q: What is the Jacobian of $y = x + F(x)$ with respect to $x$?
A: $I + \partial F/\partial x$ — the identity term gives gradients a direct path.

Q: Why must a sublayer's output have the same width as its input?
A: Because it is added elementwise to its input; in transformers every sublayer maps $d_{\text{model}} \to d_{\text{model}}$.

Q: In a pre-norm transformer, what is the final hidden state in terms of the sublayers?
A: The input embedding plus the sum of all attention and FFN sublayer outputs.

Q: What happens to deep pure-attention stacks without skip connections?
A: Their token representations lose rank rapidly (doubly exponentially with depth) and collapse towards each other.

Q: What problem did ResNet's residual connections originally address?
A: Very deep plain networks had higher training error than shallower ones — an optimization difficulty, not overfitting.
