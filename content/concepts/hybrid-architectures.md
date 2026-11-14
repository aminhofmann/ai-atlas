---
id: hybrid-architectures
title: Hybrid Architectures (Attention + SSM / Convolution)
category: innovations
level: 3
year: 2023
summary: >-
  Models that interleave a few attention layers with many linear-time layers
  such as Mamba or Hyena, aiming for attention's precise recall at close to
  recurrent-model cost.
requires: [attention, state-space-models, mamba]
links:
  - { to: transformer-block, type: uses }
  - { to: mixture-of-experts, type: uses }
  - { to: grouped-query-attention, type: uses }
  - { to: kv-cache, type: improves }
  - { to: evo, type: applied-in }
  - { to: hyenadna, type: related }
sources:
  - { title: "Jamba: A Hybrid Transformer-Mamba Language Model (Lieber et al., 2024)", url: "https://arxiv.org/abs/2403.19887" }
  - { title: "StripedHyena: Moving Beyond Transformers with Hybrid Signal Processing Models (Together AI, 2023)", url: "https://www.together.ai/blog/stripedhyena-7b" }
  - { title: "Griffin: Mixing Gated Linear Recurrences with Local Attention (De et al., 2024)", url: "https://arxiv.org/abs/2402.19427" }
  - { title: "Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling (Ren et al., 2024)", url: "https://arxiv.org/abs/2406.07522" }
  - { title: "Sequence modeling and design from molecular to genome scale with Evo (Nguyen et al., 2024)", url: "https://www.science.org/doi/10.1126/science.ado9336" }
---

## Intuition
Attention is like a perfect photographic memory: it can look up any earlier
token exactly, but it gets expensive as the text grows. SSMs and long
convolutions are like a good summary in your head: cheap and fixed-size, but
details fade. Hybrids give the model mostly "summary" layers plus a few
"photographic" ones, so it can still look things up precisely when needed.

Empirically, you don't need many attention layers: a small fraction is enough
to recover most of the recall ability, while the rest of the network runs in
linear time with a small memory footprint.

## Mechanics
**Design axes:**
- **Ratio and placement** of attention vs linear-time layers (e.g. one
  attention layer per $k$ blocks).
- **Global vs local attention:** full attention gives exact long-range recall;
  sliding-window attention keeps cost linear but only sees nearby tokens.
- **Which recurrent/convolutional mixer:** Mamba, Hyena-style gated long
  convolutions, gated linear recurrences (RG-LRU).
- **Positional information:** recurrent layers already encode order, so some
  hybrids use no explicit positional encoding in their attention layers.

**KV-cache arithmetic.** If only a fraction $\rho$ of layers are attention,
the KV cache is $\rho$ times that of an equal-depth transformer; SSM layers
add a fixed state per layer independent of context length. This is the main
practical win at long context.

**Examples:**
- **StripedHyena-7B** (Together AI, Dec 2023): Hyena gated-convolution blocks
  interleaved ("striped") with attention blocks using GQA. Its architecture is
  the basis of **Evo**, a DNA foundation model.
- **Jamba** (AI21, Mar 2024): Jamba blocks with an attention:Mamba ratio of
  **1:7**, and **MoE** in every other layer (16 experts, top-2); ≈52B total and
  ≈12B active parameters; 256K-token context. The paper reports no explicit
  positional encoding was needed.
- **Griffin / RecurrentGemma** (Google DeepMind, 2024): gated linear recurrence
  (RG-LRU) blocks mixed with *local* (sliding-window) attention.
- **Samba** (Microsoft, 2024): layer-wise mix of Mamba, sliding-window
  attention and MLPs.
- **Zamba** (Zyphra, 2024): Mamba backbone with a shared attention block
  applied periodically.

**Why mixing helps (the usual explanation):** fixed-state recurrences struggle
with associative recall and copying (retrieve exactly what appeared earlier),
which attention handles easily; recurrent layers in turn provide cheap,
smooth long-range context mixing.

**Costs:** more complex kernels and inference stacks (two kinds of state to
manage), and less mature tooling than pure transformers.

## Code
```python
import torch.nn as nn

def build_hybrid(n_layers, d, make_attn, make_mamba, make_mlp, make_moe,
                 attn_every=8, moe_every=2):
    """Jamba-style layout: 1 attention layer per `attn_every`, MoE every `moe_every`."""
    layers = []
    for i in range(n_layers):
        mixer = make_attn(d) if i % attn_every == attn_every // 2 else make_mamba(d)
        ffn = make_moe(d) if i % moe_every == 1 else make_mlp(d)
        layers.append(nn.ModuleDict({"norm1": nn.LayerNorm(d), "mixer": mixer,
                                     "norm2": nn.LayerNorm(d), "ffn": ffn}))
    return nn.ModuleList(layers)

def forward(layers, x):
    for blk in layers:                       # pre-norm residual, as in a transformer block
        x = x + blk["mixer"](blk["norm1"](x))
        x = x + blk["ffn"](blk["norm2"](x))
    return x
# Only the attention layers keep a KV cache; Mamba layers keep a fixed-size state.
```

## Why it matters
Hybrids are currently the most practical way to bring linear-time sequence
mixers into large language models: they retain most of a transformer's quality
while cutting KV-cache memory and long-context cost. They are also prominent in
genomics, where sequences are extremely long (Evo is a StripedHyena hybrid, not
a pure transformer). Whether hybrids or pure attention win at the very largest
scales is still open.

## Self-test
Q: What is the core idea of a hybrid architecture?
A: Interleave a small number of attention layers with many linear-time layers (SSM, long convolution or linear recurrence) in one stack.

Q: What attention-to-Mamba ratio does Jamba use, and what else does it add?
A: 1 attention layer per 7 Mamba layers, plus MoE in every other layer.

Q: Why does a hybrid have a smaller KV cache than an equal-depth transformer?
A: Only the attention layers keep a KV cache; the SSM layers store a fixed-size state independent of context length.

Q: What capability are attention layers usually credited with providing in hybrids?
A: Exact associative recall and copying of earlier tokens, which fixed-size recurrent states handle poorly.

Q: Which DNA model is built on the StripedHyena architecture?
A: Evo.
