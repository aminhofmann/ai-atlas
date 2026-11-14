---
id: in-silico-perturbation
title: In-Silico Perturbation
category: interpretability
level: 2
year: 2023
summary: >-
  Editing a single-cell foundation model's input — deleting or "overexpressing"
  a gene — and measuring how the cell's embedding moves, to predict which genes
  drive a cell state without running the wet-lab experiment.
requires: [geneformer, embeddings]
links:
  - { to: biological-foundation-models, type: applied-in }
  - { to: scgpt, type: applied-in }
  - { to: scfoundation, type: applied-in }
  - { to: mechanistic-interpretability, type: related }
  - { to: probing, type: related }
  - { to: attention-visualization, type: related }
  - { to: pretraining-and-fine-tuning, type: uses }
sources:
  - { title: "Transfer learning enables predictions in network biology (Theodoris et al., Nature 2023)", url: "https://www.nature.com/articles/s41586-023-06139-9" }
  - { title: "Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines (Ahlmann-Eltze, Huber & Anders, Nature Methods 2025)", url: "https://www.nature.com/articles/s41592-025-02772-6" }
---

## Intuition
A biologist asks "what happens to a heart cell if I knock out gene X?" and
answers it with CRISPR, weeks of work and one gene at a time. A single-cell
foundation model has seen tens of millions of cells and learned which gene
programs go together. In-silico perturbation asks the model instead: take a
real cell, remove gene X from its input, run it through the model again, and see
where its embedding lands. If the "diseased" cell's embedding moves toward the
"healthy" cluster, X becomes a candidate therapeutic target.

It is a counterfactual question posed to a learned simulator. Like any
simulator, it is only as good as what it learned — so it generates hypotheses
to be tested, not answers.

## Mechanics
In **Geneformer** (Theodoris et al., 2023) a cell is tokenised by *rank value
encoding*: each gene's expression is normalised by its median across the
pretraining corpus, and the genes are sorted from highest to lowest, so a cell
becomes an ordered list of gene tokens (up to 2,048). The model was pretrained
with masked gene prediction on ~30M cells (Genecorpus-30M).

Perturbations are edits to that list:
- **Deletion (knockout):** remove gene $g$'s token.
- **Overexpression (activation):** move $g$ to the front (highest rank).
- **Combinations:** edit several genes at once.

Then compare embeddings before and after. With $e(c)$ the cell embedding (mean
of final- or penultimate-layer gene embeddings):

$$\Delta_{\text{cell}} = \cos\big(e(c_{\text{perturbed}}),\, \mu_{\text{target}}\big) - \cos\big(e(c),\, \mu_{\text{target}}\big),$$

where $\mu_{\text{target}}$ is the mean embedding of the desired state (e.g.
non-failing cardiomyocytes). Repeating over many cells and genes yields a ranked
list of genes whose deletion shifts cells toward (or away from) a state;
statistical significance is assessed against random perturbations. The shift of
*other genes'* embeddings indicates which genes respond, giving a predicted
network.

Theodoris et al. applied this to cardiomyopathy, predicting genes whose
deletion moves diseased cardiomyocytes toward the healthy state, and tested
some candidates experimentally in an iPSC-derived disease model.

**Caveats.**
- The model never saw interventional data in pretraining; it extrapolates from
  co-expression patterns.
- Benchmarks are sobering: Ahlmann-Eltze, Huber & Anders (2025) found that
  foundation models (including Geneformer- and scGPT-based predictors) did not
  outperform deliberately simple baselines (additive model, mean prediction) at
  predicting transcriptome changes after perturbations.

## Code
```python
import torch
from transformers import BertForMaskedLM   # Geneformer is BERT-style

model = BertForMaskedLM.from_pretrained("ctheodoris/Geneformer",
                                        output_hidden_states=True).eval()

def embed(token_ids):                       # ids = genes sorted by normalised rank
    with torch.no_grad():
        hs = model(input_ids=torch.tensor([token_ids])).hidden_states[-2]
    return hs[0].mean(0)                    # cell embedding (d,)

def perturb(cell, gene, mode="delete"):
    rest = [t for t in cell if t != gene]
    return rest if mode == "delete" else [gene] + rest   # overexpress = move to front

cell = [...]                                # rank-encoded token ids from the Geneformer tokenizer
healthy_centroid = torch.randn(model.config.hidden_size)  # mean embedding of target state
cos = torch.nn.functional.cosine_similarity
shift = cos(embed(perturb(cell, gene=1234)), healthy_centroid, dim=0) \
      - cos(embed(cell), healthy_centroid, dim=0)
print("shift toward healthy:", shift.item())
```
(Illustrative; the Geneformer package ships an `InSilicoPerturber` class that
does this with proper tokenisation and statistics.)

## Why it matters
It turned a pretrained single-cell model from a classifier into a hypothesis
generator for network biology — cheap, genome-wide screens run on a GPU before
any wet-lab work. It is also the biological counterpart of interventional
interpretability (ablation, activation patching): change the input, measure the
internal shift. Its value is ranking candidates; benchmarks show predictions of
actual expression changes are not yet better than simple baselines.

## Self-test
Q: How is an in-silico deletion done in Geneformer?
A: The gene's token is removed from the cell's rank-ordered gene list before re-embedding the cell.

Q: How is overexpression simulated?
A: By moving the gene's token to the front of the rank list (highest expression).

Q: What quantity is usually measured after a perturbation?
A: The change in cosine similarity between the cell's embedding and the embedding of a target state (or the shifts of other genes' embeddings).

Q: Why are these predictions hypotheses rather than facts?
A: The model learned from observational co-expression, not interventions, and benchmarks found it does not beat simple linear/additive baselines at predicting perturbation effects.
