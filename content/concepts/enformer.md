---
id: enformer
title: Enformer (and Borzoi)
category: biology
level: 3
year: 2021
summary: >-
  A supervised convolution + transformer model that reads ~200 kb of DNA and
  predicts thousands of gene-regulation measurements along it; Borzoi extends
  the idea to RNA-seq coverage over ~524 kb.
requires: [transformer-block, self-attention, positional-encoding]
links:
  - { to: transformer, type: based-on }
  - { to: multi-head-attention, type: uses }
  - { to: in-silico-perturbation, type: applied-in }
  - { to: attention-visualization, type: applied-in }
  - { to: nucleotide-transformer, type: alternative-to }
  - { to: hyenadna, type: related }
  - { to: evo, type: related }
  - { to: biological-foundation-models, type: related }
sources:
  - { title: "Effective gene expression prediction from sequence by integrating long-range interactions (Avsec et al., Nature Methods 2021)", url: "https://www.nature.com/articles/s41592-021-01252-x" }
  - { title: "Predicting RNA-seq coverage from DNA sequence as a unifying model of gene regulation — Borzoi (Linder et al., Nature Genetics 2025)", url: "https://www.nature.com/articles/s41588-024-02053-6" }
---

## Intuition

Genes are switched on and off by enhancers that can sit tens of thousands of
bases away. Earlier convolutional models (Basenji) only "saw" a few tens of
kilobases, so they missed many of these distant regulators.

Enformer reads a ~200 kb window of DNA, uses convolutions to compress it into
128-bp chunks, and then lets transformer layers connect chunks across the
whole window. Its job is not to "understand DNA" in a language-model sense
but to directly predict measured experiments — how open the chromatin is,
which histone marks are present, how much transcription happens — at every
position. Because it maps sequence → function, you can mutate the input and
read off the predicted effect.

## Mechanics

- **Input**: 196,608 bp, one-hot encoded ($L \times 4$).
- **Conv tower**: convolution + attention pooling blocks downsample to
  **128-bp bins** (1,536 positions).
- **Transformer**: **11 transformer blocks** with multi-head attention and
  **relative positional encodings** (basis functions of distance), letting
  each bin attend across the whole window — effective receptive field
  ~100 kb versus ~20 kb for Basenji2.
- **Crop + heads**: centre **896 bins** (~114.7 kb) are kept; species-specific
  heads predict **5,313 human** and **1,643 mouse** genomic tracks (ChIP-seq,
  DNase/ATAC, CAGE…).
- **Loss**: Poisson negative log-likelihood on track coverage; trained
  supervised on ENCODE/FANTOM-style data from human and mouse.
- **Uses**: variant-effect prediction via
  $\Delta = f(\text{alt}) - f(\text{ref})$; contribution scores (gradients,
  attention) to find enhancers for a gene.

**Borzoi** (Calico, 2025): 524 kb input, conv + self-attention (at 128 bp)
plus a U-Net-style upsampling back to **32-bp** resolution, trained to
predict **RNA-seq coverage** along with other tracks — so it models
transcription, splicing and polyadenylation in one network.

What it is **not**: not self-supervised, not a language model, not a
pure transformer (convolutions do most of the local work), and it does not
generalise well across individuals for personal gene-expression prediction,
as later studies showed.

## Code

```python
# Illustrative Enformer-shaped forward pass (not the official model).
import torch, torch.nn as nn

L, d = 196_608, 768
x = torch.randn(1, 4, L)                                    # one-hot DNA
conv = nn.Sequential(nn.Conv1d(4, d, 15, padding=7),
                     *[nn.MaxPool1d(2) for _ in range(7)])  # 2^7 = 128 bp bins
h = conv(x).transpose(1, 2)                                 # (1, 1536, d)
layer = nn.TransformerEncoderLayer(d, nhead=8, batch_first=True)
h = nn.TransformerEncoder(layer, num_layers=11)(h)          # long-range mixing
h = h[:, 320:-320]                                          # crop to 896 bins
human_head = nn.Sequential(nn.Linear(d, 5313), nn.Softplus())
print(human_head(h).shape)                                  # (1, 896, 5313)
```

## Why it matters

Enformer (DeepMind with Calico) showed attention can capture distal enhancer
→ gene effects and became a standard tool for variant-effect prediction from
sequence. It defines the "supervised sequence-to-function" alternative to
self-supervised DNA language models, and Borzoi extended it to RNA-level
readouts.

## Self-test

Q: What does Enformer predict?
A: Thousands of experimental genomic tracks (e.g. 5,313 human tracks like CAGE, DNase, ChIP-seq) along a DNA window.

Q: What is Enformer's input length and bin size?
A: About 196 kb of one-hot DNA, compressed to 128-bp bins.

Q: Why is Enformer not a language model?
A: It is trained supervised to predict measured assay coverage, not by masked or next-token prediction.

Q: How does Borzoi differ from Enformer?
A: It reads ~524 kb, upsamples to 32-bp resolution, and predicts RNA-seq coverage.
