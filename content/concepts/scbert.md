---
id: scbert
title: scBERT
category: biology
level: 2
year: 2022
summary: >-
  An early BERT-style single-cell model that embeds every gene plus its binned
  expression level and is fine-tuned mainly for cell-type annotation.
requires: [masked-language-modeling, biological-foundation-models]
links:
  - { to: bert, type: based-on }
  - { to: pretraining-and-fine-tuning, type: uses }
  - { to: embeddings, type: uses }
  - { to: attention-visualization, type: applied-in }
  - { to: geneformer, type: related }
  - { to: scgpt, type: related }
sources:
  - { title: "scBERT as a large-scale pretrained deep language model for cell type annotation of single-cell RNA-seq data (Yang et al., Nature Machine Intelligence 2022)", url: "https://www.nature.com/articles/s42256-022-00534-z" }
---

## Intuition

scBERT asks: what if each cell were a "sentence" whose "words" are genes, and
each word carries a volume knob — how strongly that gene is expressed? By
hiding some volume knobs and asking the model to guess them from all the other
genes, the model learns which genes go up and down together. Afterwards, a
small classification head turns that knowledge into a cell-type label.

Its key engineering choice was to keep **all** genes (no selection of highly
variable genes), which makes the "sentence" over ten thousand tokens long.
Standard quadratic attention is too expensive there, so scBERT uses an
efficient-attention variant.

## Mechanics

- **Input**: one token per gene in a fixed gene vocabulary, so the sequence
  length is the number of genes.
- **Embedding** per position = gene embedding + expression embedding:
  - *gene embedding* initialised from **gene2vec** (co-expression-based gene
    vectors) — this plays the role of a positional/identity encoding, since
    genes have no order;
  - *expression embedding*: the log-normalised count is **binned** into
    discrete levels, each bin with a learned vector.
- **Backbone**: **Performer** — a transformer whose attention is approximated
  by random features (kernelised linear attention), giving roughly $O(n)$ cost
  instead of $O(n^2)$ for $n$ genes.
- **Pretraining**: masked-expression prediction on large unlabeled scRNA-seq
  data (PanglaoDB); masked bins are reconstructed from context.
- **Fine-tuning**: supervised cell-type annotation on labelled datasets,
  including detecting novel (unseen) cell types by low-confidence predictions.
- **Interpretability**: attention weights were used to highlight genes
  important for each cell type.

What it is **not**: not a generative model and not a decoder; and the Performer
attention is an approximation of softmax attention, not exact attention.

## Code

```python
# Illustrative scBERT-style input embedding (not the official code).
import torch, torch.nn as nn

n_genes, n_bins, d = 16000, 7, 200
gene_emb = nn.Embedding(n_genes, d)     # in scBERT: initialised from gene2vec
bin_emb = nn.Embedding(n_bins, d)

log_expr = torch.rand(1, n_genes) * 6   # one cell, all genes
bins = torch.clamp(log_expr.long(), max=n_bins - 1)
gene_ids = torch.arange(n_genes).unsqueeze(0)

x = gene_emb(gene_ids) + bin_emb(bins)  # (1, n_genes, d) -> Performer encoder
mask = torch.rand(1, n_genes) < 0.15    # pretraining: predict masked bins
print(x.shape, mask.sum().item())
```

## Why it matters

scBERT was one of the first demonstrations that BERT-style pretraining
transfers to single-cell data, framing cell-type annotation as fine-tuning a
pretrained language model. Later models (Geneformer, scGPT, scFoundation)
scaled up data and changed tokenization; independent benchmarks have since
shown that well-tuned simpler classifiers can match it on many annotation
tasks.

## Self-test

Q: What two embeddings are summed per gene token in scBERT?
A: A gene-identity embedding (initialised from gene2vec) and an embedding of the binned expression level.

Q: Why does scBERT use Performer instead of standard attention?
A: It keeps all genes as tokens (sequence length >10k), and Performer's linear-complexity attention approximation makes that affordable.

Q: What is scBERT's main downstream task?
A: Cell-type annotation of scRNA-seq data, including flagging novel cell types.

Q: What plays the role of positional encoding in scBERT?
A: The gene embedding — genes have no order, so identity replaces position.
