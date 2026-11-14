---
id: esmfold
title: ESMFold
category: biology
level: 3
year: 2022
summary: >-
  A single-sequence structure predictor that replaces AlphaFold 2's MSA input
  with ESM-2 language-model embeddings, trading some accuracy for a large
  speed-up.
requires: [esm2, alphafold2]
links:
  - { to: biological-foundation-models, type: part-of }
  - { to: attention, type: uses }
  - { to: probing, type: related }
  - { to: esm3, type: related }
sources:
  - { title: "Evolutionary-scale prediction of atomic-level protein structure with a language model (Lin et al., Science 2023)", url: "https://www.science.org/doi/10.1126/science.ade2574" }
  - { title: "ESM GitHub repository (ESMFold)", url: "https://github.com/facebookresearch/esm" }
---

## Intuition

AlphaFold 2 gets its evolutionary information by searching huge databases for
related sequences and aligning them (an MSA) — slow, and impossible for
"orphan" proteins with few relatives. ESM-2 has already absorbed that
evolutionary information into its weights during pretraining.

ESMFold's idea: skip the database search. Feed the single sequence through
ESM-2, and hand its internal representations to a slimmed-down AlphaFold-style
folding head. The language model plays the role the MSA used to play.

## Mechanics

- **Language-model trunk**: a frozen-at-start ESM-2 (3B-parameter version)
  embeds the sequence; a learned weighted combination of its layer
  representations gives per-residue features, and attention-derived features
  can seed the pair representation.
- **Folding trunk**: a simplified Evoformer-style stack that keeps a
  **sequence** (single) representation and a **pair** representation and
  updates them with triangle-style pair updates — but there is **no MSA
  axis** (no row/column MSA attention), since there is only one sequence.
- **Structure module**: an AlphaFold 2-style structure module (invariant point
  attention over residue frames) outputs backbone frames and side-chain
  torsions → atomic coordinates; predictions are recycled.
- **Confidence**: per-residue pLDDT and predicted aligned error, as in AF2.
- **Training**: on experimental PDB structures plus distillation from
  predicted structures.
- **Scaling result**: structure accuracy tracks the language model's
  perplexity — sequences the LM "understands" better fold more accurately.
- **ESM Metagenomic Atlas**: ESMFold's speed allowed predicting structures for
  over 600 million metagenomic proteins.

What it is **not**: not more accurate than AlphaFold 2 on average (it is
generally somewhat lower, especially where good MSAs exist); its advantage is
speed and independence from MSA search. It predicts single chains primarily,
not general complexes with ligands like AlphaFold 3.

## Code

```python
# pip install "fair-esm[esmfold]"  (needs extra OpenFold deps)
import torch, esm

model = esm.pretrained.esmfold_v1().eval()    # ESM-2 3B trunk + folding head
seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQ"
with torch.no_grad():
    pdb_string = model.infer_pdb(seq)         # single sequence in, PDB out
open("pred.pdb", "w").write(pdb_string)       # B-factor column holds pLDDT
```

## Why it matters

ESMFold demonstrated that a protein language model can substitute for
explicit evolutionary search, making structure prediction fast enough for
whole metagenomes. It is the clearest evidence that pretrained protein LMs
encode 3D structure, and a template for "LM trunk + task head" designs.

## Self-test

Q: What does ESMFold use instead of an MSA?
A: Per-residue representations from the ESM-2 protein language model.

Q: Which part of ESMFold is borrowed from AlphaFold 2?
A: The folding design: a pair-representation trunk with triangle updates and an IPA-based structure module.

Q: Is ESMFold generally more accurate than AlphaFold 2?
A: No — it is usually somewhat less accurate but much faster and MSA-free.

Q: What large resource did ESMFold's speed make possible?
A: The ESM Metagenomic Atlas of over 600 million predicted protein structures.
