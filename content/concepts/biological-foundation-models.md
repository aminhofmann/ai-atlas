---
id: biological-foundation-models
title: Biological Foundation Models
category: biology
level: 1
year: 2019
summary: >-
  Large models pretrained on unlabeled biological sequences — cells, proteins
  or DNA treated as "sentences" of tokens — and then reused for many
  downstream tasks.
requires: [transformer, tokenization, pretraining-and-fine-tuning]
links:
  - { to: bert, type: based-on }
  - { to: gpt, type: based-on }
  - { to: masked-language-modeling, type: uses }
  - { to: autoregressive-training, type: uses }
  - { to: state-space-models, type: related }
  - { to: hybrid-architectures, type: related }
  - { to: in-silico-perturbation, type: related }
  - { to: probing, type: related }
  - { to: geneformer, type: related }
  - { to: scgpt, type: related }
  - { to: esm2, type: related }
  - { to: alphafold2, type: related }
  - { to: dnabert, type: related }
  - { to: evo, type: related }
sources:
  - { title: "Evolutionary-scale prediction of atomic-level protein structure with a language model (Lin et al., Science 2023)", url: "https://www.science.org/doi/10.1126/science.ade2574" }
  - { title: "Transfer learning enables predictions in network biology — Geneformer (Theodoris et al., Nature 2023)", url: "https://www.nature.com/articles/s41586-023-06139-9" }
  - { title: "Nucleotide Transformer (Dalla-Torre et al., Nature Methods 2024)", url: "https://www.nature.com/articles/s41592-024-02523-z" }
---

## Intuition

A language model learns grammar by reading huge amounts of text and guessing
hidden or next words. Biology has its own "texts": a protein is a string of
amino acids, a genome is a string of nucleotides, and a single cell can be
written down as a list of the genes it is currently expressing. If you treat
these as sentences, the same pretraining recipe — hide something, predict it —
forces a model to learn the "grammar" of life: which amino acids co-vary
because they touch in 3D, which DNA motifs regulate genes, which genes switch
on together in a heart cell.

After pretraining on millions of unlabeled examples, the model is reused —
via embeddings, fine-tuning, or zero-shot scoring — for tasks where labels are
scarce: cell-type annotation, variant-effect prediction, structure prediction,
protein design.

## Mechanics

**Three modalities, three tokenization problems.**

- **Single-cell transcriptomics** (scBERT, Geneformer, scGPT, scFoundation,
  UCE, Nicheformer). A cell is a vector of ~20k gene counts, not a sequence —
  there is no natural order. Models invent one:
  - *rank-value encoding*: sort genes by normalized expression, keep gene ids
    as tokens (Geneformer, Nicheformer);
  - *gene token + binned value token* summed per position (scBERT, scGPT);
  - *continuous value embeddings* over a fixed gene vocabulary (scFoundation);
  - *genes as protein embeddings* sampled by expression (UCE).
- **Proteins** (ESM-2, ESM3, ProtGPT2). One token per amino acid (~20 letters
  + specials) is standard; ProtGPT2 instead uses BPE sub-words. ESM3 adds
  discrete structure and function tokens.
- **DNA** (DNABERT, Nucleotide Transformer, Enformer, HyenaDNA, Caduceus, Evo).
  Options: overlapping $k$-mers (DNABERT), non-overlapping 6-mers (Nucleotide
  Transformer), BPE (DNABERT-2), or single nucleotides (HyenaDNA, Caduceus,
  Evo). Genomes demand very long contexts, which pushes DNA models toward
  sub-quadratic architectures.

**Encoder vs. decoder.**

| Style | Objective | Examples |
|---|---|---|
| Encoder (BERT-like) | masked prediction | scBERT, Geneformer, scFoundation, UCE, Nicheformer, ESM-2, DNABERT(-2), Nucleotide Transformer, Caduceus |
| Decoder (GPT-like) | next-token | ProtGPT2, HyenaDNA, Evo / Evo 2 |
| Custom masking | generative but order-free | scGPT, ESM3 |
| Supervised, not LM | predict measured tracks / structure | Enformer, Borzoi, AlphaFold 2/3 |

**Not all are transformers.** HyenaDNA (Hyena convolutions), Caduceus (Mamba)
and Evo (StripedHyena hybrid) replace most or all attention; AlphaFold's
Evoformer/Pairformer use attention inside a bespoke, geometry-aware design.

## Code

```python
# Illustrative: three ways to turn biology into token ids.
import numpy as np

protein = "MKTAYIAKQR"
aa_vocab = {a: i for i, a in enumerate("ACDEFGHIKLMNPQRSTVWY")}
protein_ids = [aa_vocab[a] for a in protein]          # 1 token per residue

dna = "ATGCGTACGTTAGC"
k = 6
kmers = [dna[i:i + k] for i in range(0, len(dna) - k + 1, k)]  # non-overlapping 6-mers

genes = np.array(["GAPDH", "NANOG", "ACTB", "SOX2"])
expr = np.array([900.0, 3.0, 800.0, 5.0])
median = np.array([1000.0, 1.0, 900.0, 2.0])          # corpus-wide gene medians
cell_tokens = genes[np.argsort(-(expr / median))]      # rank-value encoding
print(protein_ids, kmers, cell_tokens)                 # NANOG, SOX2 rank first
```

## Why it matters

Biology has far more unlabeled data (sequence databases, cell atlases) than
labeled experiments. Foundation models turn that into reusable
representations: ESM-2 embeddings power ESMFold structure prediction,
Geneformer enables in-silico perturbation screens, and Evo 2 scores human
variants zero-shot. The field is young and benchmarks are contested — several
studies report that simple baselines (e.g. PCA or linear models) remain
competitive on some single-cell tasks — so claims should be read critically.

## Self-test

Q: Why is tokenizing a single cell harder than tokenizing a protein?
A: A cell's gene-expression profile has no natural order, so models must invent one (e.g. ranking genes by expression) or use order-free embeddings.

Q: Name the three main modalities of biological foundation models.
A: Single-cell transcriptomics, proteins, and DNA/genomic sequence.

Q: Give two DNA models that are not attention-based.
A: HyenaDNA (Hyena long convolutions) and Caduceus (bidirectional Mamba); Evo is a hybrid that is mostly convolutional.

Q: Which objective do ESM-2 and Geneformer use, and which do ProtGPT2 and Evo use?
A: ESM-2 and Geneformer use masked prediction (encoder); ProtGPT2 and Evo use next-token prediction (decoder).
