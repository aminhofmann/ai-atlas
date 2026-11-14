---
id: time-series-transformers
title: Time-Series Transformers
category: domains
level: 2
year: 2022
summary: >-
  Transformers for forecasting numeric sequences — from efficient-attention
  forecasters (Informer, Autoformer) to patch-based models (PatchTST) and
  pretrained zero-shot foundation models (Chronos, TimesFM).
requires: [transformer-block, positional-encoding, tokenization]
links:
  - { to: vision-transformer, type: related }
  - { to: t5, type: related }
  - { to: gpt, type: related }
  - { to: masked-language-modeling, type: related }
  - { to: autoregressive-training, type: uses }
  - { to: pretraining-and-fine-tuning, type: uses }
  - { to: state-space-models, type: alternative-to }
  - { to: weather-transformers, type: related }
  - { to: transformers-in-other-domains, type: part-of }
sources:
  - { title: "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers (PatchTST, Nie et al., 2022)", url: "https://arxiv.org/abs/2211.14730" }
  - { title: "Chronos: Learning the Language of Time Series (Ansari et al., 2024)", url: "https://arxiv.org/abs/2403.07815" }
  - { title: "Are Transformers Effective for Time Series Forecasting? (Zeng et al., 2022)", url: "https://arxiv.org/abs/2205.13504" }
  - { title: "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting (Zhou et al., 2020)", url: "https://arxiv.org/abs/2012.07436" }
  - { title: "A decoder-only foundation model for time-series forecasting (TimesFM, Das et al., 2023)", url: "https://arxiv.org/abs/2310.10688" }
  - { title: "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting (Liu et al., 2023)", url: "https://arxiv.org/abs/2310.06625" }
---

## Intuition
A time series — electricity load, sales, sensor readings — looks like a
sentence of numbers, so the transformer seems a natural fit. The hard question
is what a *token* should be. A single timestep carries almost no meaning (one
number is like one letter), and long histories make attention expensive.

The field's answer evolved in three steps. First, make attention cheaper over
very long series (Informer, Autoformer). Then, after a simple linear model beat
those designs, rethink the token: group consecutive points into **patches**,
like ViT does with pixels (PatchTST). Finally, treat forecasting like language
modelling at scale: turn numbers into a vocabulary, pretrain on huge collections
of series, and forecast new series zero-shot (Chronos, TimesFM).

## Mechanics
**Efficient-attention era.** *Informer* (Zhou et al., AAAI 2021) used ProbSparse
attention and distillation for long inputs; *Autoformer* (Wu et al., 2021) built
in series decomposition and auto-correlation. **Reality check:** Zeng et al.
(2022) showed a one-layer linear model ("DLinear") often beat them, arguing that
permutation-invariant attention over single timesteps loses temporal order.

**PatchTST** (Nie et al., ICLR 2023). Two ideas:
- **Patching:** split a univariate series of length $L$ into patches of length
  $P$ with stride $S$, giving $N \approx \lfloor (L-P)/S \rfloor + 2$ tokens; each
  patch is linearly embedded, plus learned positions. Fewer, more meaningful tokens
  → longer look-back at lower cost.
- **Channel independence:** each variable of a multivariate series is processed
  separately by the same shared encoder.
- Instance normalisation (RevIN), a flatten + linear forecasting head, and an
  optional masked-patch self-supervised pretraining.

**iTransformer** (2023) inverts the view: each *variable's whole history* is one
token, so attention models cross-variable correlations.

**Foundation models.**
- **Chronos** (Ansari et al., 2024): scale each series by the mean absolute value
  of its context, clip and **quantise** into a fixed vocabulary of bins (4,094
  bins plus special tokens), then train an off-the-shelf T5 with plain
  cross-entropy. Forecasts are sampled autoregressively and de-quantised,
  giving probabilistic forecasts. Sizes 20M–710M; pretrained on public data plus
  synthetic Gaussian-process series; strong zero-shot results.
- **TimesFM** (Das et al., 2023): decoder-only, patch tokens in, longer patches
  out, pretrained on a large corpus of real and synthetic series.
- Others: Moirai, Lag-Llama, MOMENT.

## Code
```python
import torch

# --- PatchTST-style tokenisation ---
def patchify(x, P=16, S=8):                        # x: (B, C, L) multivariate
    B, C, L = x.shape
    x = x.reshape(B * C, L)                        # channel independence
    x = (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-5)  # instance norm
    return x.unfold(-1, P, S)                      # (B*C, N, P) patch tokens

tokens = patchify(torch.randn(8, 7, 512))
print(tokens.shape)                                # (56, 63, 16) -> Linear(P, d) -> encoder

# --- Chronos-style tokenisation ---
def chronos_tokens(ctx, n_bins=4094, lo=-15.0, hi=15.0):
    scale = ctx.abs().mean(-1, keepdim=True).clamp_min(1e-8)   # mean scaling
    edges = torch.linspace(lo, hi, n_bins - 1)
    return torch.bucketize((ctx / scale).clamp(lo, hi), edges), scale

ids, scale = chronos_tokens(torch.rand(2, 512) * 100)
# Real model: from chronos import ChronosPipeline; pipe.predict(context, prediction_length)
```

## Why it matters
Time series is where "just add a transformer" was most publicly questioned, and
the response — better tokens via patching, normalisation, channel independence —
is a template for any new numeric domain. Pretrained forecasters now give
usable zero-shot forecasts on series they never saw, mirroring the LLM shift from
task-specific models to foundation models. Simple linear and tree-based
baselines remain strong and should always be compared against.

## Self-test
Q: What did Zeng et al. (2022) show about early time-series transformers?
A: That a simple one-layer linear model (DLinear) often outperformed them on long-term forecasting benchmarks.

Q: What are PatchTST's two key ideas?
A: Grouping consecutive timesteps into patch tokens, and processing each channel independently with a shared model.

Q: How does Chronos turn real numbers into tokens?
A: It mean-scales the series, clips it, and quantises the values into a fixed set of bins that act as a vocabulary for a T5 model.

Q: What does iTransformer treat as a token?
A: The entire look-back window of one variable, so attention runs across variables.
