---
id: hyenadna
title: HyenaDNA
category: biology
level: 3
year: 2023
summary: >-
  An attention-free, decoder-style genomic model built on Hyena long
  convolutions that reads up to 1 million single nucleotides of context.
requires: [state-space-models, autoregressive-training, biological-foundation-models]
links:
  - { to: attention, type: alternative-to }
  - { to: gpt, type: related }
  - { to: causal-masking, type: related }
  - { to: tokenization, type: related }
  - { to: caduceus, type: related }
  - { to: evo, type: related }
  - { to: dnabert, type: related }
  - { to: nucleotide-transformer, type: alternative-to }
sources:
  - { title: "HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution (Nguyen et al., NeurIPS 2023)", url: "https://arxiv.org/abs/2306.15794" }
  - { title: "Hyena Hierarchy: Towards Larger Convolutional Language Models (Poli et al., ICML 2023)", url: "https://arxiv.org/abs/2302.10866" }
---

## Intuition

Genomes are enormously long, and single-letter changes (SNPs) matter.
Transformers force a bad trade-off: tokenize coarsely (k-mers, BPE) to fit
more sequence into a quadratic-cost attention window, and lose
single-nucleotide resolution.

HyenaDNA removes attention entirely. Its Hyena layers mix information with
**very long convolutions** (as long as the input) computed fast with the FFT,
plus element-wise gating. Cost grows roughly like $L \log L$ instead of
$L^2$, so it can read one letter per token over up to a million letters.

## Mechanics

- **Tokens**: single nucleotides (character-level: A, C, G, T, N + specials).
- **Hyena operator** (replaces attention): from input $u$, compute
  projections $v, x_1, \dots, x_N$; then alternate
  $z \leftarrow x_n \odot (h_n * z)$ starting from $z = v$, where $h_n$ is a
  **long, implicitly parameterised filter** (a small MLP of position,
  modulated by a decay window) and $*$ is causal convolution via FFT,
  $O(L \log L)$.
- **Architecture**: decoder-style stack of Hyena blocks with MLPs, residuals,
  layer norm; models are small (millions of parameters).
- **Pretraining**: next-nucleotide prediction on the **human reference
  genome**, contexts up to **1M tokens**, with a **sequence-length warm-up**
  schedule (start short, grow the context) for stability.
- **Downstream**: fine-tuning on Nucleotide Transformer and GenomicBenchmarks
  tasks, and **soft-prompt / in-context-style** tuning on long sequences;
  species classification from long reads.

What it is **not**: not a transformer and not attention-based; not
bidirectional (causal, left-to-right); and strictly speaking not a state-space
model either — Hyena is a related family of long-convolution operators.

## Code

```python
# Illustrative Hyena-style operator (order 2) with FFT long convolution.
import torch, torch.nn as nn

def fft_conv(u, h):                       # u: (B, D, L), h: (D, L), causal
    L = u.shape[-1]
    U = torch.fft.rfft(u, n=2 * L); H = torch.fft.rfft(h, n=2 * L)
    return torch.fft.irfft(U * H, n=2 * L)[..., :L]

class Hyena(nn.Module):
    def __init__(self, d, L):
        super().__init__()
        self.proj = nn.Linear(d, 3 * d)
        self.filters = nn.Parameter(torch.randn(2, d, L) * 0.01)  # implicit MLP in practice
        self.out = nn.Linear(d, d)
    def forward(self, u):                 # u: (B, L, d)
        v, x1, x2 = self.proj(u).transpose(1, 2).chunk(3, dim=1)
        z = x1 * fft_conv(v, self.filters[0])
        z = x2 * fft_conv(z, self.filters[1])
        return self.out(z.transpose(1, 2))

x = torch.randn(1, 4096, 128)             # 4,096 single-nucleotide tokens
print(Hyena(128, 4096)(x).shape)
```

## Why it matters

HyenaDNA (Stanford, Hazy Research) showed that sub-quadratic,
attention-free models can do genome-scale contexts at single-nucleotide
resolution, a big jump over transformer DNA models limited to a few
kilobases. It directly led to Evo's StripedHyena design and motivated
Mamba-based genomics models like Caduceus.

## Self-test

Q: What replaces attention in HyenaDNA?
A: Hyena operators — long, implicitly parameterised convolutions (computed via FFT) interleaved with element-wise gating.

Q: How does HyenaDNA tokenize DNA?
A: One token per nucleotide (character-level).

Q: What is HyenaDNA's maximum context length?
A: Up to 1 million nucleotides.

Q: What training trick helped HyenaDNA train at very long contexts?
A: A sequence-length warm-up schedule that gradually increases context length.

Q: Is HyenaDNA bidirectional?
A: No — it is causal (next-nucleotide prediction); Caduceus addresses bidirectionality.
