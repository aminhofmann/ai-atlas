---
id: uce
title: Universal Cell Embeddings (UCE)
category: biology
level: 2
year: 2023
summary: >-
  A single-cell foundation model that represents genes by the ESM-2 embeddings
  of their proteins, so one model can embed cells from many species zero-shot.
requires: [transformer, embeddings, biological-foundation-models]
links:
  - { to: esm2, type: uses }
  - { to: bert, type: based-on }
  - { to: masked-language-modeling, type: uses }
  - { to: geneformer, type: related }
  - { to: scgpt, type: related }
  - { to: nicheformer, type: related }
sources:
  - { title: "Universal Cell Embeddings: A Foundation Model for Cell Biology (Rosen et al., bioRxiv 2023)", url: "https://www.biorxiv.org/content/10.1101/2023.11.28.568918" }
---

## Intuition

Most single-cell models have a fixed gene vocabulary — a list of human gene
names. A mouse or zebrafish cell uses different genes, so it doesn't fit.

UCE sidesteps vocabularies entirely: it describes each gene by **what protein
it makes**, using a protein language model (ESM-2) to turn the protein into a
vector. Similar proteins in different species get similar vectors
automatically. A cell then becomes "a bag of protein vectors, weighted by
expression", and the model learns to summarise that bag into one cell
embedding — usable even for species it never saw.

## Mechanics

- **Gene tokens**: each gene's protein product is embedded with ESM-2 (the
  large 15B model per the paper); these embeddings are frozen and projected.
- **Building a cell "sentence"**:
  - sample 1,024 gene slots from the cell's expressed genes **with
    replacement, weighted by (log-normalised) expression** — highly expressed
    genes appear more often;
  - group the sampled genes by **chromosome**, ordered by genomic position,
    with special chromosome-start/end tokens; chromosome groups are shuffled.
  - prepend a `<cls>` token whose final state is the cell embedding.
- **Backbone**: a 33-layer transformer encoder (~650M parameters).
- **Objective**: self-supervised — ~20% of expressed genes are masked out;
  for query genes (masked + unexpressed), an MLP on [cell embedding; gene
  protein embedding] predicts *expressed or not* (binary cross-entropy).
- **Training data**: the Integrated Mega-scale Atlas of ~36 million cells from
  8 species, mostly human and mouse.
- **Use**: zero-shot — the cell embedding is used as-is, without fine-tuning
  or dataset-specific integration; embeddings from new datasets and new
  species land in one shared space.

What it is **not**: not a generative model of expression values, and not
tied to a single species' gene list.

## Code

```python
# Illustrative UCE-style cell sentence (not the official code).
import numpy as np

rng = np.random.default_rng(0)
n_genes, d_esm = 500, 5120                   # ESM-2 15B embedding size
protein_emb = rng.normal(size=(n_genes, d_esm))  # stand-in for ESM-2 vectors
chrom = rng.integers(1, 23, n_genes)
pos = rng.integers(0, 10**8, n_genes)
expr = rng.poisson(1.0, n_genes).astype(float)

p = expr / expr.sum()
sampled = rng.choice(n_genes, size=1024, p=p)        # expression-weighted sampling
order = sorted(sampled, key=lambda g: (chrom[g], pos[g]))
tokens = protein_emb[order]                          # -> project -> 33-layer encoder
print(tokens.shape)                                  # CLS output = cell embedding
```

## Why it matters

UCE made cross-species, zero-shot cell embedding a concrete goal and showed
that protein language models can serve as a shared gene "dictionary" across
organisms. It is a clear example of models feeding models: ESM-2 provides the
inputs for a cell-level foundation model.

## Self-test

Q: How does UCE represent a gene?
A: By the ESM-2 protein language model embedding of the protein the gene encodes.

Q: Why does that choice enable cross-species use?
A: Homologous proteins in different species get similar embeddings, so no fixed species-specific gene vocabulary is needed.

Q: How are genes selected and ordered to build UCE's input?
A: Sampled with replacement weighted by expression, then grouped by chromosome and ordered by genomic position.

Q: About how many cells and species was UCE trained on?
A: About 36 million cells from 8 species.
