---
id: nucleotide-transformer
title: Nucleotide Transformer
category: biology
level: 2
year: 2023
summary: >-
  InstaDeep's family of BERT-style DNA language models (50M to 2.5B
  parameters) trained on thousands of human genomes and hundreds of species
  using non-overlapping 6-mer tokens.
requires: [masked-language-modeling, tokenization]
links:
  - { to: bert, type: based-on }
  - { to: scaling-laws, type: related }
  - { to: lora, type: related }
  - { to: probing, type: applied-in }
  - { to: attention-visualization, type: applied-in }
  - { to: dnabert, type: related }
  - { to: enformer, type: related }
  - { to: hyenadna, type: related }
sources:
  - { title: "Nucleotide Transformer: building and evaluating robust foundation models for human genomics (Dalla-Torre et al., Nature Methods 2024)", url: "https://www.nature.com/articles/s41592-024-02523-z" }
  - { title: "Preprint (bioRxiv 2023)", url: "https://www.biorxiv.org/content/10.1101/2023.01.11.523679" }
---

## Intuition

If ESM showed that scaling protein language models pays off, does the same
work for DNA? The Nucleotide Transformer team (InstaDeep with NVIDIA and TUM)
trained a family of DNA encoders of increasing size on increasingly diverse
data — many human individuals, then hundreds of species — and systematically
measured how well their embeddings transfer to genomics tasks.

A key finding: **diversity beats sheer human data**. A model trained across
many species often transferred better to *human* tasks than one trained only
on human genomes, because evolution across species reveals which sequence
features are conserved and therefore functional.

## Mechanics

- **Tokenization**: **non-overlapping 6-mers** (4,096 6-mer tokens plus a few
  single-nucleotide tokens for leftovers and `N`); a sequence of ~1,000 tokens
  covers ~6 kb in the first-generation models (longer in v2).
- **Architecture**: BERT-style transformer encoder; sizes from **50M to
  2.5B** parameters.
- **Training sets** (separate models):
  - the human reference genome;
  - **3,202 human genomes** (1000 Genomes Project) — captures human variation;
  - a **multispecies** set of **850 genomes** across diverse species.
- **Objective**: masked language modeling (BERT-style 15% masking).
- **Evaluation**: 18 downstream genomics tasks (promoters, enhancers,
  splice sites, histone marks…), using
  - frozen embeddings + linear **probing**, and
  - parameter-efficient fine-tuning (IA³-style rescaling, a LoRA relative),
    which matched or beat probing at a small fraction of parameters.
- **Interpretability**: attention heads were found to specialise on genomic
  elements (e.g. introns, exons, TF binding sites) without supervision;
  zero-shot embedding distances helped prioritise variants.
- **v2 models** reworked the architecture (e.g. rotary embeddings, gated
  MLPs) at smaller sizes with longer contexts.

What it is **not**: not autoregressive/generative, not single-nucleotide
resolution at the token level, and not designed for 100 kb-scale contexts.

## Code

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

name = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForMaskedLM.from_pretrained(name)

seq = "ATGGCGTACGTTAGCATGCAAGTCCGATAA"
batch = tok([seq], return_tensors="pt")
print(tok.convert_ids_to_tokens(batch["input_ids"][0]))  # <cls> + 6-mers
with torch.no_grad():
    out = model(**batch, output_hidden_states=True)
emb = out.hidden_states[-1].mean(dim=1)     # sequence embedding for probing
print(emb.shape)
```

## Why it matters

Nucleotide Transformer was a systematic scaling study for DNA language models
and showed that multi-species pretraining and parameter-efficient fine-tuning
are strong recipes. Later benchmarks found that for many regulatory tasks,
supervised models like Enformer, or long-context architectures, still win —
DNA LMs remain an active, unsettled area.

## Self-test

Q: How does Nucleotide Transformer tokenize DNA?
A: Into non-overlapping 6-mers (with single-nucleotide fallback tokens).

Q: What were the three kinds of pretraining data?
A: The human reference genome, 3,202 diverse human genomes, and 850 genomes from many species.

Q: What surprising data finding did the paper report?
A: Multi-species training often transferred better to human tasks than training on human genomes alone.

Q: What parameter range does the model family span?
A: From 50 million to 2.5 billion parameters.
