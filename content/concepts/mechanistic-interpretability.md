---
id: mechanistic-interpretability
title: Mechanistic Interpretability
category: interpretability
level: 3
year: 2020
summary: >-
  Reverse-engineering a trained network into human-understandable algorithms —
  features, circuits and the attention heads that implement them — and testing
  those stories with causal interventions.
requires: [residual-connections, multi-head-attention, transformer-block]
links:
  - { to: attention-visualization, type: improves }
  - { to: probing, type: improves }
  - { to: superposition, type: related }
  - { to: sparse-autoencoders, type: uses }
  - { to: query-key-value, type: related }
  - { to: feed-forward-network, type: related }
  - { to: gpt, type: applied-in }
  - { to: in-silico-perturbation, type: related }
sources:
  - { title: "Zoom In: An Introduction to Circuits (Olah et al., Distill 2020)", url: "https://distill.pub/2020/circuits/zoom-in/" }
  - { title: "A Mathematical Framework for Transformer Circuits (Elhage et al., 2021)", url: "https://transformer-circuits.pub/2021/framework/index.html" }
  - { title: "In-context Learning and Induction Heads (Olsson et al., 2022)", url: "https://arxiv.org/abs/2209.11895" }
  - { title: "Locating and Editing Factual Associations in GPT (Meng et al., 2022)", url: "https://arxiv.org/abs/2202.05262" }
  - { title: "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small (Wang et al., 2022)", url: "https://arxiv.org/abs/2211.00593" }
---

## Intuition
Most interpretability asks *what* a model represents. Mechanistic
interpretability asks *how* it computes: which components read which
information, transform it, and write it back, so that you could in principle
rewrite the model as pseudo-code. The analogy is reverse-engineering a compiled
binary — you have all the weights, but not the source.

Its building blocks are **features** (directions in activation space that mean
something), **circuits** (sub-graphs of weights connecting features), and the
insistence that every story be **tested by intervening** on the model, not just
by looking at it.

## Mechanics
**Residual stream view** (Elhage et al., 2021). The residual stream
$x \in \mathbb{R}^{d}$ is a shared communication channel. Each attention head
and MLP *reads* from it via a linear map and *adds* its output back:
$x_{\ell+1} = x_\ell + \sum_h \mathrm{head}_h(x_\ell) + \mathrm{MLP}(x_\ell)$.
So the logits decompose into a sum of paths, and each head factors into
- a **QK circuit** $W_E^\top W_Q^\top W_K W_E$: *where* to attend, and
- an **OV circuit** $W_U W_O W_V W_E$: *what* to move once attending.

Attention-only toy models can be analysed exactly this way.

**Induction heads** (Olsson et al., 2022). A two-head circuit that completes
patterns `[A][B] … [A] → [B]`:
1. a *previous-token head* writes "the token before me was A" into position B;
2. the *induction head* at the second A queries for positions whose previous
   token was A (K-composition), attends to B, and copies it (OV).

They form abruptly during training, at the same time as a visible bump in the
loss curve and a jump in in-context learning; the paper argues they may
underlie much of in-context learning.

**Activation patching / causal tracing.** Run a *clean* prompt and a *corrupted*
one; copy one component's activation from the clean run into the corrupted run
and measure how much of the correct output is restored. Meng et al. (2022) used
this to localise factual recall to mid-layer MLPs in GPT; Wang et al. (2022)
mapped a circuit of attention heads, grouped into functional classes (e.g. "name movers"), for indirect-object identification in GPT-2 small.
Related: ablation (zero/mean), path patching, attribution patching (a
gradient approximation).

**Obstacle:** neurons are polysemantic because of
superposition, which is why
sparse autoencoders are now used to find features first.

## Code
```python
import torch
from transformer_lens import HookedTransformer   # pip install transformer_lens

model = HookedTransformer.from_pretrained("gpt2")
clean = model.to_tokens("When Mary and John went to the store, John gave a drink to")
corrupt = model.to_tokens("When Mary and John went to the store, Mary gave a drink to")
target = model.to_single_token(" Mary")
_, clean_cache = model.run_with_cache(clean)

def patch_head(layer, head):
    name = f"blocks.{layer}.attn.hook_z"           # per-head outputs (b, pos, H, d_head)
    def hook(z, hook):
        z[:, :, head] = clean_cache[name][:, :, head]
        return z
    logits = model.run_with_hooks(corrupt, fwd_hooks=[(name, hook)])
    return logits[0, -1, target].item()

base = model(corrupt)[0, -1, target].item()
for L, H in [(9, 6), (9, 9), (10, 0)]:              # candidate "name mover" heads
    print(L, H, patch_head(L, H) - base)            # >0: head carries the answer
```

## Why it matters
It is the strongest form of interpretability: claims are causal and specific
("head 9.9 copies the indirect object"). It produced reusable concepts —
the residual stream as a bus, QK/OV circuits, induction heads — and tools like
TransformerLens. The open problem is scale: hand-mapping circuits works for
small models and narrow tasks, which is why the field turned to automated
feature discovery with SAEs and circuit-tracing over those features. The same
"intervene and measure" logic underlies in-silico perturbation
in biology.

## Self-test
Q: In the residual stream view, how do attention heads and MLPs interact with each other?
A: Each reads from the shared residual stream through a linear map and adds its output back into it; they communicate only through that stream.

Q: What do the QK and OV circuits of a head determine?
A: QK determines which positions the head attends to; OV determines what information it moves from those positions into the output.

Q: Describe the induction-head pattern.
A: A previous-token head marks each position with its preceding token; the induction head then finds the earlier occurrence of the current token and copies the token that followed it ([A][B]…[A] → [B]).

Q: What is activation patching?
A: Replacing a component's activation in a corrupted run with its value from a clean run and measuring how much of the clean behaviour returns, to establish causal importance.
