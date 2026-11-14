---
id: scfoundation
title: scFoundation
category: biology
level: 2
year: 2024
summary: >-
  A ~100M-parameter single-cell model (xTrimoGene backbone) trained on 50M+
  human cells with an asymmetric encoder–decoder and a read-depth-aware
  masked objective.
requires: [transformer, masked-language-modeling, biological-foundation-models]
links:
  - { to: bert, type: based-on }
  - { to: encoder-decoder, type: uses }
  - { to: pretraining-and-fine-tuning, type: uses }
  - { to: in-silico-perturbation, type: applied-in }
  - { to: scgpt, type: related }
  - { to: geneformer, type: related }
sources:
  - { title: "Large-scale foundation model on single-cell transcriptomics (Hao et al., Nature Methods 2024)", url: "https://www.nature.com/articles/s41592-024-02305-7" }
  - { title: "scFoundation code (biomap-research)", url: "https://github.com/biomap-research/scFoundation" }
---

## Intuition

Single-cell measurements are noisy in a very specific way: one lab sequences
each cell deeply, another shallowly, and many genes simply read as zero. A
model that learns "this cell has these counts" will partly learn the lab, not
the biology.

scFoundation's idea is to make **sequencing depth an explicit input**. During
training it is shown a down-sampled (shallower) version of a cell and asked to
reconstruct the deeper original, told both depths. Afterwards you can ask it to
"enhance" a shallow cell to a higher depth. It also saves compute by only
feeding the *expressed* genes into the big encoder.

## Mechanics

- **Data and scale**: >50 million human single-cell transcriptomes from public
  repositories; ~100M parameters; a fixed vocabulary of **19,264 genes**.
- **Value embedding**: expression values are continuous; instead of hard bins
  they are mapped to embeddings by a learned soft lookup, then added to the
  gene-identity embedding.
- **Asymmetric encoder–decoder (xTrimoGene)**:
  - the **encoder** (large transformer) sees only **non-zero, unmasked** genes
    — typically a small fraction of the 19k, so attention is cheap;
  - the **decoder** (lighter, efficient-attention) receives the encoder outputs
    plus zero and masked positions for all 19,264 genes and predicts values.
  This mirrors the masked-autoencoder idea of putting the heavy compute where
  information actually is.
- **Read-depth-aware (RDA) pretraining**: the input cell is a down-sampled
  version with total count $S$; two extra tokens give $S$ and the target
  total count $T$; the model regresses the original expression (masked
  regression loss). At inference, setting $T > S$ yields depth-enhanced
  profiles.
- **Downstream**: cell embeddings and gene embeddings fed to other models —
  e.g. drug-response prediction and as input features for perturbation
  predictors (the paper plugs them into GEARS).

What it is **not**: not a generative decoder-only model and not trained on
non-human cells.

## Code

```python
# Illustrative read-depth-aware input construction (not the official API).
import numpy as np

def make_rda_example(counts, rng, frac=0.3):
    T = counts.sum()                              # target (original) depth
    down = rng.binomial(counts.astype(int), frac) # shallower version
    S = down.sum()                                # input depth
    nonzero = np.nonzero(down)[0]                 # encoder sees only these genes
    return {"enc_genes": nonzero,
            "enc_values": np.log1p(down[nonzero]),
            "depth_tokens": (np.log10(S + 1), np.log10(T + 1)),
            "target": np.log1p(counts)}           # decoder predicts all 19,264

rng = np.random.default_rng(0)
ex = make_rda_example(rng.poisson(0.3, 19264), rng)
print(len(ex["enc_genes"]), ex["depth_tokens"])
```

## Why it matters

scFoundation showed that an asymmetric design lets a single-cell model cover
the full ~19k-gene vocabulary at 100M-parameter scale, and that conditioning
on read depth tackles one of the field's main sources of technical noise. It
sits alongside scGPT and Geneformer as a general-purpose embedding provider.

## Self-test

Q: Which genes does scFoundation's encoder process?
A: Only the non-zero (and unmasked) genes; the decoder handles all 19,264 genes.

Q: What two extra quantities are given to the model in read-depth-aware pretraining?
A: The total count of the (down-sampled) input, S, and the target total count, T.

Q: What does setting T greater than S at inference do?
A: It asks the model to produce a depth-enhanced expression profile.

Q: Roughly how large is scFoundation and how much data was it trained on?
A: About 100M parameters, trained on more than 50 million human cells.
