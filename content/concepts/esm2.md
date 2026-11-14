---
id: esm2
title: ESM-2
category: biology
level: 2
year: 2022
summary: >-
  Meta's family of BERT-style protein language models (8M to 15B parameters)
  trained by masked amino-acid prediction, whose embeddings encode protein
  structure well enough to drive ESMFold.
requires: [masked-language-modeling, biological-foundation-models]
links:
  - { to: bert, type: based-on }
  - { to: rotary-position-embedding, type: uses }
  - { to: scaling-laws, type: related }
  - { to: probing, type: applied-in }
  - { to: attention-visualization, type: applied-in }
  - { to: sparse-autoencoders, type: applied-in }
  - { to: esmfold, type: part-of }
  - { to: esm3, type: related }
  - { to: protgpt2, type: related }
sources:
  - { title: "Evolutionary-scale prediction of atomic-level protein structure with a language model (Lin et al., Science 2023)", url: "https://www.science.org/doi/10.1126/science.ade2574" }
  - { title: "Preprint (bioRxiv 2022)", url: "https://www.biorxiv.org/content/10.1101/2022.07.20.500902" }
---

## Intuition

Evolution has run a gigantic experiment: millions of protein sequences that
all still fold and function. If two positions in a protein touch in 3D, a
mutation at one tends to be compensated at the other — they co-vary. A model
that learns to fill in masked amino acids across all these sequences is
forced to learn those dependencies, and with them, implicitly, the shape of
the protein.

ESM-2 is exactly that: BERT for proteins, made large. One letter per amino
acid, hide 15%, predict them back. No structures, no labels.

## Mechanics

- **Tokens**: one per amino acid (20 standard + rare/ambiguous letters) plus
  special tokens (`<cls>`, `<eos>`, `<mask>`, `<pad>`). Max length ~1,024.
- **Architecture**: transformer encoder (pre-LayerNorm) with **rotary
  position embeddings**. Sizes: 8M, 35M, 150M, 650M, 3B and 15B parameters
  (6 to 48 layers).
- **Training data**: UniRef50 clusters, sampling member sequences from
  UniRef90 within each cluster — tens of millions of diverse sequences.
- **Objective**: masked language modeling,
  $\mathcal{L} = -\sum_{i \in M} \log p(x_i \mid x_{\setminus M})$.
- **Emergent structure**: attention maps (and linear probes on them) predict
  residue–residue contacts; contact precision and perplexity improve
  smoothly with scale.
- **Uses**:
  - per-residue and mean-pooled embeddings as features for function,
    stability, localisation predictors;
  - zero-shot **variant-effect scoring** via masked marginals,
    $\log p(x_i = \text{mut}) - \log p(x_i = \text{wt})$;
  - the trunk of **ESMFold**, which adds a folding head;
  - input embeddings for other models (e.g. UCE uses ESM-2 gene embeddings).

What it is **not**: not generative in the GPT sense (no left-to-right
sampling), does not use multiple sequence alignments, and on its own outputs
no 3D coordinates.

## Code

```python
# Uses the fair-esm package (pip install fair-esm).
import torch, esm

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()   # 33 layers, 650M
batch_converter = alphabet.get_batch_converter()
model.eval()

data = [("p1", "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ")]
_, _, tokens = batch_converter(data)
with torch.no_grad():
    out = model(tokens, repr_layers=[33], return_contacts=True)

per_residue = out["representations"][33]      # (1, L+2, 1280)
protein_emb = per_residue[0, 1:-1].mean(0)     # mean-pool, drop <cls>/<eos>
contacts = out["contacts"]                     # (1, L, L) predicted contacts
print(protein_emb.shape, contacts.shape)
```

## Why it matters

ESM-2 is the default protein embedding model in much of computational
biology. Its paper showed that structure emerges from sequence-only language
modeling at scale, enabling ESMFold and the ESM Metagenomic Atlas of
hundreds of millions of predicted structures. It is also a common target for
interpretability work (probing, sparse autoencoders on its activations).

## Self-test

Q: What is ESM-2's pretraining objective?
A: Masked language modeling — predict masked amino acids from the rest of the sequence.

Q: What positional encoding does ESM-2 use?
A: Rotary position embeddings (RoPE).

Q: What is the largest ESM-2 model?
A: 15 billion parameters (48 layers).

Q: How can ESM-2 score a mutation without fine-tuning?
A: Compare the model's log-probability of the mutant vs wild-type amino acid at that position (masked marginal).

Q: Does ESM-2 need a multiple sequence alignment?
A: No — it works from a single sequence.
