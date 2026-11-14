---
id: geneformer
title: Geneformer
category: biology
level: 2
year: 2023
summary: >-
  A BERT-style encoder that represents each cell as its genes sorted by
  normalised expression (rank-value encoding), pretrained on ~30 million
  cells and used for in-silico perturbation.
requires: [masked-language-modeling, biological-foundation-models]
links:
  - { to: bert, type: based-on }
  - { to: pretraining-and-fine-tuning, type: uses }
  - { to: in-silico-perturbation, type: applied-in }
  - { to: attention-visualization, type: applied-in }
  - { to: scbert, type: related }
  - { to: scgpt, type: related }
  - { to: nicheformer, type: related }
sources:
  - { title: "Transfer learning enables predictions in network biology (Theodoris et al., Nature 2023)", url: "https://www.nature.com/articles/s41586-023-06139-9" }
  - { title: "Geneformer model card (Hugging Face)", url: "https://huggingface.co/ctheodoris/Geneformer" }
  - { title: "Genecorpus-30M dataset card", url: "https://huggingface.co/datasets/ctheodoris/Genecorpus-30M" }
---

## Intuition

Instead of telling the model *how much* of each gene a cell expresses,
Geneformer just tells it the **order**: "this cell's most distinctive gene is
X, then Y, then Z…". The sentence for a cell is its gene list, sorted.

The trick is that genes are ranked *relative to how they usually behave*. A
housekeeping gene that is always high gets pushed down; a transcription factor
that is usually silent but is on in this cell jumps to the front. So the
first tokens of the sentence are the genes that make this cell *this* cell.

## Mechanics

- **Genecorpus-30M**: ~30 million human single-cell transcriptomes from public
  data, used for pretraining (assembled 2021).
- **Rank-value encoding**:
  1. For each gene $g$, compute the median of its **non-zero** expression
     across the whole corpus, $m_g$.
  2. In a cell, normalise counts by total counts, then divide by $m_g$.
  3. Sort genes by this value (descending) and emit their **gene-id tokens**
     in that order; expression values themselves are discarded.
  The rank is effectively the positional information.
- **Architecture (2023 paper)**: 6 transformer encoder layers, embedding
  dimension 256, input size 2,048 gene tokens; vocabulary ~25k genes.
- **Pretraining**: masked-gene prediction — mask ~15% of positions and predict
  *which gene* sits there from the rest of the ranking.
- **Usage**: cell embeddings (pooled token states) for classification, or
  fine-tuning for tasks like predicting dosage-sensitive genes or chromatin
  dynamics.
- **In-silico perturbation**: delete a gene's token (simulated knockout) or
  move it to the front (simulated overexpression), re-embed the cell, and
  measure the shift in embedding (cosine similarity) toward/away from a
  disease or healthy state. Used in the paper to nominate candidate
  therapeutic targets in cardiomyopathy.
- Later releases (V2) were trained on ~104M cells with input size 4,096.

What it is **not**: not generative, and the ranking discards absolute
expression magnitudes.

## Code

```python
# Illustrative rank-value encoding + Hugging Face loading.
import numpy as np
from transformers import BertForMaskedLM

def rank_value_encode(counts, gene_ids, medians, max_len=2048):
    norm = counts / counts.sum() * 1e4            # depth-normalise
    nz = norm > 0
    scaled = norm[nz] / medians[nz]               # divide by corpus nonzero median
    order = np.argsort(-scaled)
    return gene_ids[nz][order][:max_len]          # token ids, most distinctive first

model = BertForMaskedLM.from_pretrained("ctheodoris/Geneformer")  # repo id
# knockout of gene token t: tokens = tokens[tokens != t]; compare cell embeddings
```

## Why it matters

Geneformer popularised two ideas: an order-free cell turned into a sequence by
ranking, and **in-silico perturbation** as a cheap way to generate hypotheses
before wet-lab experiments. Its rank encoding was reused by later models such
as Nicheformer. Independent benchmarks have questioned how much its
zero-shot embeddings beat simple baselines, so its perturbation predictions
are best treated as hypotheses.

## Self-test

Q: What does rank-value encoding divide each gene's expression by before ranking?
A: That gene's median non-zero expression across the whole pretraining corpus.

Q: What does Geneformer predict during pretraining?
A: The identity of masked gene tokens in the ranked gene list.

Q: How is an in-silico knockout done in Geneformer?
A: Remove the gene's token from the ranked input and measure how the cell embedding shifts.

Q: What is Genecorpus-30M?
A: The ~30-million-cell human single-cell corpus used to pretrain the original Geneformer.
