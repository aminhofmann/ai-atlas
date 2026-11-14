---
id: evo
title: Evo and Evo 2
category: biology
level: 3
year: 2024
summary: >-
  Arc Institute's long-context genomic language models built on the
  StripedHyena hybrid (mostly convolutional, a few attention layers) that
  model and generate DNA at single-nucleotide resolution.
requires: [hybrid-architectures, autoregressive-training, hyenadna]
links:
  - { to: state-space-models, type: related }
  - { to: attention, type: uses }
  - { to: rotary-position-embedding, type: uses }
  - { to: gpt, type: related }
  - { to: scaling-laws, type: related }
  - { to: sparse-autoencoders, type: applied-in }
  - { to: mechanistic-interpretability, type: applied-in }
  - { to: caduceus, type: related }
  - { to: enformer, type: related }
  - { to: protgpt2, type: related }
  - { to: biological-foundation-models, type: part-of }
sources:
  - { title: "Sequence modeling and design from molecular to genome scale with Evo (Nguyen et al., Science 2024)", url: "https://www.science.org/doi/10.1126/science.ado9336" }
  - { title: "Genome modeling and design across all domains of life with Evo 2 (Brixi et al., Nature 2026; bioRxiv 2025)", url: "https://www.biorxiv.org/content/10.1101/2025.02.18.638918" }
  - { title: "Evo 2 code and model card", url: "https://github.com/ArcInstitute/evo2" }
---

## Intuition

DNA, RNA and proteins are all written in the genome. A model that reads raw
DNA at single-letter resolution over very long stretches could learn all three
"languages" at once — and generate them.

Evo does this with a GPT-style next-nucleotide objective, but on an
architecture that is mostly **not** a transformer. StripedHyena stacks many
Hyena long-convolution layers (cheap over long sequences) with only a few
attention layers interleaved for precise recall. That hybrid lets Evo handle
contexts of 131k nucleotides (Evo) and 1 million (Evo 2).

## Mechanics

**Evo (2024)**
- **7B parameters**, StripedHyena: Hyena (gated long-convolution) layers
  with a small number of rotary-attention layers "striped" between them.
- **Tokens**: single nucleotides (byte-level).
- **Data**: OpenGenome, ~300B nucleotides of **prokaryotic and phage**
  genomes (plus plasmids).
- **Training**: autoregressive, first at 8k context, then extended to
  **131k**.
- **Results**: zero-shot fitness prediction for proteins, ncRNAs and
  regulatory DNA from likelihoods; **generation** of functional CRISPR–Cas
  systems and transposable elements validated in the lab; genome-scale
  sequence sampling.

**Evo 2 (2025 preprint; Nature 2026)**
- **7B and 40B** parameters on **StripedHyena 2**, a "multi-hybrid" mixing
  convolution operators of different lengths (short, medium, long) with
  attention.
- **Data**: OpenGenome2, about 9.3 trillion base pairs (per the paper; the
  model card cites 8.8T tokens) spanning **all domains of life**, including
  eukaryotes.
- **Context**: up to **1 million** nucleotides (after midtraining extension).
- **Results**: zero-shot **variant-effect prediction**, e.g. classifying
  *BRCA1* variants; generation of genome-scale sequences; **sparse
  autoencoders** on its activations revealed features for exons, introns,
  TF motifs, protein structure and more.

What it is **not**: not a pure transformer (attention is a minority of
layers), not bidirectional, and Evo 1 was trained without eukaryotic
genomes, so it is weak on human sequences.

## Code

```python
# Evo 2 via the ArcInstitute package (illustrative; needs a large GPU).
import torch
from evo2 import Evo2

model = Evo2("evo2_7b")
ref = "ATGGCGTACGTTAGCATGCAAGTCCGATAAGGCTTACGATCGATCG"
alt = ref[:20] + "T" + ref[21:]                       # single-nucleotide variant

def loglik(seq):
    ids = torch.tensor(model.tokenizer.tokenize(seq)).unsqueeze(0).cuda()
    (logits, _), _ = model(ids)                      # README: outputs[0] = logits
    logp = torch.log_softmax(logits[0, :-1].float(), -1)
    return logp.gather(-1, ids[0, 1:, None]).sum().item()

print("variant score:", loglik(alt) - loglik(ref))    # < 0: likely deleterious
```

## Why it matters

Evo showed that a single DNA model can capture protein, RNA and regulatory
"grammar" and design multi-component biological systems; Evo 2 scaled this to
eukaryotes and million-base contexts, with open weights, code and data. Both
are flagship examples of hybrid (convolution + attention) architectures
beating pure transformers on very long sequences. Developed by the Arc
Institute with Stanford, TogetherAI (Evo) and NVIDIA (Evo 2).

## Self-test

Q: What architecture do Evo and Evo 2 use?
A: StripedHyena (Evo) and StripedHyena 2 (Evo 2) — hybrids of Hyena-style convolution layers with a few attention layers, not pure transformers.

Q: What was Evo 1 trained on and at what context?
A: About 300B nucleotides of prokaryotic and phage genomes (OpenGenome), up to 131k context.

Q: What sizes and context length does Evo 2 have?
A: 7B and 40B parameters, with up to 1 million nucleotides of context.

Q: How does Evo 2 score a variant zero-shot?
A: By comparing the model's log-likelihood of the variant sequence to the reference sequence.

Q: Which interpretability method was applied to Evo 2's activations?
A: Sparse autoencoders, revealing features such as exon/intron boundaries and TF motifs.
