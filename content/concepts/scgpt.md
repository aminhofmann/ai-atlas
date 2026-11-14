---
id: scgpt
title: scGPT
category: biology
level: 2
year: 2024
summary: >-
  A single-cell foundation model that pairs every gene token with a binned
  expression-value token and uses a special attention mask to do "generative"
  pretraining on unordered genes.
requires: [transformer, attention, biological-foundation-models]
links:
  - { to: gpt, type: based-on }
  - { to: causal-masking, type: related }
  - { to: self-attention, type: uses }
  - { to: flash-attention, type: uses }
  - { to: pretraining-and-fine-tuning, type: uses }
  - { to: in-silico-perturbation, type: applied-in }
  - { to: geneformer, type: related }
  - { to: scbert, type: related }
  - { to: scfoundation, type: related }
sources:
  - { title: "scGPT: toward building a foundation model for single-cell multi-omics using generative AI (Cui et al., Nature Methods 2024)", url: "https://www.nature.com/articles/s41592-024-02201-0" }
  - { title: "scGPT preprint (bioRxiv 2023)", url: "https://www.biorxiv.org/content/10.1101/2023.04.30.538439" }
---

## Intuition

GPT predicts the next word left to right. A cell has no "left to right" — its
genes are an unordered set. scGPT keeps the GPT spirit ("generate what you
haven't seen from what you have") but replaces "next" with "unknown": some
genes are revealed, the rest must be generated from them, and an attention
mask controls who may look at whom.

Each position carries two things: *which gene* and *how much* of it. That lets
the model both read and write expression values, which is what you need for
tasks like predicting the effect of a genetic perturbation.

## Mechanics

- **Pretraining data**: ~33 million human cells from CELLxGENE.
- **Tokens per position** (summed embeddings):
  - gene token $g_i$ (vocabulary of gene names, plus `<cls>` and `<pad>`);
  - **value token**: expression is binned *per cell* into a fixed number of
    relative bins, so values are comparable across sequencing depths;
  - optional **condition tokens** (e.g. batch, modality, perturbation flag).
  $h_i^{(0)} = E_g(g_i) + E_v(x_i) + E_c(c_i)$.
- **Backbone**: stacked transformer blocks (implemented with FlashAttention);
  no positional encoding, since genes are unordered.
- **Generative attention mask**: tokens are split into *known* (gene + value
  given) and *unknown* (gene given, value to predict).
  - known tokens attend to all known tokens;
  - unknown tokens attend to known tokens **and themselves**, not to other
    unknowns.
  Training proceeds over several iterations, moving the most confident
  predictions into the known set — an order-free analogue of autoregression.
  A cell-embedding-conditioned variant generates the whole profile from `<cls>`.
- **Loss**: MSE on predicted expression values of unknown genes.
- **Fine-tuning tasks**: cell-type annotation, batch integration, multi-omic
  integration, perturbation-response prediction, gene-network inference
  from attention.

What it is **not**: not a left-to-right causal decoder despite the name, and
its value tokens are coarse bins rather than exact counts.

## Code

```python
# Illustrative scGPT-style generative attention mask (not the official code).
import torch

n = 6
known = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.bool)  # 3 known, 3 unknown
allow = torch.zeros(n, n, dtype=torch.bool)
allow[:, known] = True                  # everyone may attend to known genes
allow |= torch.eye(n, dtype=torch.bool) # unknown genes also see themselves
# allow[i, j] == True  ->  query i may attend to key j
scores = torch.randn(n, n).masked_fill(~allow, float("-inf"))
attn = scores.softmax(-1)
print(allow.int())
```

## Why it matters

scGPT showed one pretrained model could be adapted to many single-cell tasks
and made perturbation prediction a headline use case. It also made the
"generative pretraining without an order" idea concrete. Its perturbation
claims have been contested: several benchmarks found simple baselines (e.g.
mean or linear models) competitive on perturbation-effect prediction.

## Self-test

Q: What two embeddings represent each gene position in scGPT?
A: A gene-identity token embedding and an embedding of the cell-relative binned expression value (plus optional condition tokens).

Q: In scGPT's generative mask, what can an "unknown" gene token attend to?
A: All known gene tokens and itself, but not other unknown tokens.

Q: Why can't scGPT simply use a standard causal mask?
A: Genes in a cell have no natural order, so "previous token" is meaningless.

Q: Roughly how many cells was scGPT pretrained on, and from where?
A: About 33 million human cells from CELLxGENE.
