---
id: esm3
title: ESM3
category: biology
level: 3
year: 2024
summary: >-
  EvolutionaryScale's multimodal generative protein model that reasons over
  sequence, structure and function as separate token tracks, used to design a
  new green fluorescent protein.
requires: [masked-language-modeling, transformer, biological-foundation-models]
links:
  - { to: esm2, type: improves }
  - { to: bert, type: based-on }
  - { to: tokenization, type: uses }
  - { to: scaling-laws, type: related }
  - { to: rlhf, type: related }
  - { to: protgpt2, type: related }
  - { to: alphafold3, type: related }
sources:
  - { title: "Simulating 500 million years of evolution with a language model (Hayes et al., Science 2025)", url: "https://www.science.org/doi/10.1126/science.ads0018" }
  - { title: "Preprint (bioRxiv 2024)", url: "https://www.biorxiv.org/content/10.1101/2024.07.01.600583" }
  - { title: "EvolutionaryScale blog: ESM3", url: "https://www.evolutionaryscale.ai/blog/esm3" }
---

## Intuition

ESM-2 reads only amino-acid letters. ESM3 reads and writes several
"languages" of a protein at once — its sequence, its 3D structure, and
descriptions of its function — each as its own track of tokens lined up per
residue. Because every track can be partially masked, you can prompt it
flexibly: "here is an active-site geometry and a function keyword, fill in a
sequence and the rest of the fold."

The headline demo: prompted with pieces of a fluorescent protein's active
site, ESM3 generated esmGFP, a bright protein with only 58% sequence identity
to the nearest known fluorescent protein — the authors estimate this distance
equals roughly 500 million years of natural evolution.

## Mechanics

- **Tracks** (aligned per residue, embeddings summed at input):
  - amino-acid **sequence** tokens;
  - **structure tokens**: local 3D neighbourhoods discretised by a learned
    VQ-VAE structure tokenizer (plus a decoder back to coordinates);
  - **secondary structure** (8 classes) and **solvent accessibility**;
  - **function** tokens (keywords from InterPro/GO annotations) and residue
    annotations.
- **Backbone**: a bidirectional transformer; the first block adds
  **geometric attention** that uses backbone frames when coordinates are given.
- **Objective**: generative **masked language modeling** with a variable
  masking rate per track (from little to everything masked), so the model
  learns to generate any track conditioned on any subset of the others.
- **Generation**: iterative decoding — unmask a few positions at a time,
  re-run, repeat — rather than left-to-right sampling.
- **Scale**: models up to 98B parameters; training on ~771B unique tokens
  drawn from ~3.15B sequences, ~236M structures (mostly predicted) and
  ~539M function-annotated proteins.
- **Alignment**: a preference-tuning step (pairs of good/bad generations,
  akin to DPO-style alignment) improved prompt-following on hard prompts.
- Smaller open-weight version: ESM3-open (1.4B).

What it is **not**: not an autoregressive GPT, not an extension of AlphaFold,
and the 98B model is not openly released.

## Code

```python
# pip install esm  (EvolutionaryScale SDK; API shown as illustrative)
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, GenerationConfig

model = ESM3.from_pretrained("esm3-open").to("cuda")   # 1.4B open model
prompt = ESMProtein(sequence="_" * 40 + "MKTAYIAK" + "_" * 40)  # "_" = masked
protein = model.generate(prompt, GenerationConfig(track="sequence", num_steps=8))
protein = model.generate(protein, GenerationConfig(track="structure", num_steps=8))
protein.to_pdb("design.pdb")                           # sequence + predicted fold
```

## Why it matters

ESM3 moved protein language models from *understanding* sequences to
*controllable multimodal design*, and gave a vivid experimental validation
(esmGFP). It shows the "tokenize everything, mask anything" recipe
generalising to non-text modalities like 3D structure.

## Self-test

Q: Which three main modalities does ESM3 represent as token tracks?
A: Sequence, structure (discrete VQ-VAE structure tokens) and function annotations.

Q: What is ESM3's training objective?
A: Generative masked language modeling with variable masking rates across tracks.

Q: What was esmGFP, and how far is it from known proteins?
A: A new fluorescent protein designed with ESM3, with 58% sequence identity to the closest known fluorescent protein.

Q: How large is the largest ESM3 model?
A: 98 billion parameters.
