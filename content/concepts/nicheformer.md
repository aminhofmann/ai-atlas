---
id: nicheformer
title: Nicheformer
category: biology
level: 2
year: 2025
summary: >-
  A transformer foundation model pretrained jointly on dissociated and spatial
  single-cell transcriptomics so that cell embeddings carry information about
  a cell's tissue neighbourhood.
requires: [masked-language-modeling, biological-foundation-models]
links:
  - { to: bert, type: based-on }
  - { to: pretraining-and-fine-tuning, type: uses }
  - { to: geneformer, type: related }
  - { to: uce, type: related }
  - { to: scgpt, type: related }
sources:
  - { title: "Nicheformer: a foundation model for single-cell and spatial omics (Tejada-Lapuerta, Schaar et al., Nature Methods 2025)", url: "https://www.nature.com/articles/s41592-025-02814-z" }
  - { title: "Nicheformer preprint (bioRxiv 2024)", url: "https://www.biorxiv.org/content/10.1101/2024.04.15.589472" }
---

## Intuition

Standard single-cell RNA-seq grinds tissue up: you learn what each cell
expresses but lose *where* it was and who its neighbours were. Spatial
transcriptomics keeps location but usually measures only a few hundred genes.

Nicheformer trains one model on **both** kinds of data. Because the spatial
data comes with neighbourhood labels, the model learns that a cell's
expression carries faint fingerprints of its surroundings. It can then
"project" spatial context back onto ordinary dissociated cells — e.g. guess
which tissue niche a cell came from.

## Mechanics

- **SpatialCorpus-110M**: >110 million cells — over 57M dissociated and 53M
  spatially resolved — from human and mouse, across many organs.
- **Tokenization**: Geneformer-style **rank-value encoding** — each cell is
  scaled to 10,000 counts, divided by a **technology-specific non-zero mean**
  per gene (one vector for dissociated assays, separate ones for MERFISH,
  Xenium, CosMx, ISS), then expressed genes are ranked descending; sequences
  are truncated to 1,500 tokens.
- **Shared vocabulary across species**: human and mouse genes are matched by
  orthology, giving ~20k gene tokens; spatial panels measure only a subset,
  but use the same tokens.
- **Context tokens** `<ASSAY>`, `<MODALITY>`, `<ORGANISM>` are prepended,
  so the model can learn technology-specific biases explicitly.
- **Backbone**: a BERT-style transformer encoder (12 layers, 16 heads,
  512-dim embeddings in the preprint); pretraining by masked gene-token
  prediction.
- **Spatial downstream tasks** (introduced as a benchmark): predicting a cell's
  spatial label / niche, neighbourhood cell-type composition and local density
  from expression alone — evaluated with linear probes or fine-tuning on the
  cell embedding.

What it is **not**: it does not take coordinates or images as input tokens;
spatial information enters through the training data and labels, not through
a spatial positional encoding.

## Code

```python
# Illustrative Nicheformer-style input (not the official code).
import numpy as np

SPECIAL = {"<human>": 0, "<mouse>": 1, "<dissociated>": 2, "<spatial>": 3,
           "<merfish>": 4, "<10x>": 5}
def encode(counts, gene_ids, tech_mean, species, modality, assay, max_len=1500):
    scaled = counts / counts.sum() * 1e4 / tech_mean  # technology-specific normalisation
    nz = np.nonzero(scaled)[0]
    ranked = gene_ids[nz][np.argsort(-scaled[nz])]
    ctx = [SPECIAL[species], SPECIAL[modality], SPECIAL[assay]]
    return np.concatenate([ctx, ranked + len(SPECIAL)])[:max_len]

rng = np.random.default_rng(0)
toks = encode(rng.poisson(0.5, 300).astype(float), np.arange(300),
              np.full(300, 1 / 300), "<mouse>", "<spatial>", "<merfish>")
print(toks[:8])
```

## Why it matters

Nicheformer, from Helmholtz Munich and TUM (Theis lab), was among the first
foundation models to treat spatial context as a first-class pretraining
signal. It showed that models trained only on dissociated data miss spatial
structure, and released spatial benchmark tasks for future models.

## Self-test

Q: What two kinds of data make up SpatialCorpus-110M?
A: About 57M dissociated single-cell profiles and about 53M spatially resolved cells (human and mouse).

Q: Which tokenization does Nicheformer borrow, and from which model?
A: Rank-value encoding of genes, from Geneformer.

Q: How does Nicheformer tell the model about species and technology?
A: With special context tokens for species, modality and assay prepended to the gene sequence.

Q: Does Nicheformer take spatial coordinates as input tokens?
A: No — spatial context is learned from spatial training data and labels, then predicted from expression.
