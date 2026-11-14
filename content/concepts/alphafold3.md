---
id: alphafold3
title: AlphaFold 3
category: biology
level: 3
year: 2024
summary: >-
  Predicts joint 3D structures of proteins, nucleic acids, ligands and ions by
  replacing AlphaFold 2's Evoformer with a Pairformer and its structure
  module with a diffusion model over raw atom coordinates.
requires: [attention, biological-foundation-models]
links:
  - { to: alphafold2, type: improves }
  - { to: self-attention, type: uses }
  - { to: cross-attention, type: uses }
  - { to: tokenization, type: uses }
  - { to: transformer, type: related }
  - { to: esm3, type: related }
  - { to: molecular-transformers, type: related }
sources:
  - { title: "Accurate structure prediction of biomolecular interactions with AlphaFold 3 (Abramson et al., Nature 2024)", url: "https://www.nature.com/articles/s41586-024-07487-w" }
  - { title: "AlphaFold 3 inference code", url: "https://github.com/google-deepmind/alphafold3" }
---

## Intuition

AlphaFold 2 handled protein chains, with residues as rigid frames and
hand-crafted side-chain geometry. Real biology is complexes: proteins bound to
DNA, RNA, drugs, metal ions, modified residues. Writing special geometry for
every kind of molecule doesn't scale.

AlphaFold 3 keeps the "pair representation" brain but simplifies the trunk,
and swaps the structure module for a **diffusion** model: start from noisy
atom positions and repeatedly denoise them, conditioned on the trunk's
understanding. Diffusion works directly on atoms, so it handles any chemistry
without special frames.

## Mechanics

- **Tokens**: one token per standard amino acid or nucleotide; **one token per
  atom** for ligands and non-standard residues. An input embedder also runs a
  small atom-level attention to build token features.
- **MSA module**: a lightweight block stack (4 blocks) that feeds MSA
  information into the pair representation — much smaller than the 48-block
  Evoformer.
- **Pairformer** (48 blocks): updates a **pair** representation
  $z \in \mathbb{R}^{N \times N \times c_z}$ and a **single** representation
  $s \in \mathbb{R}^{N \times c_s}$ with triangle multiplicative updates,
  triangle attention, and single-sequence attention with pair bias. Unlike the
  Evoformer, **no MSA representation** is carried through the trunk.
- **Diffusion module**: predicts denoised **raw atom coordinates** directly.
  It is transformer-based — atom-level local attention, token-level attention
  conditioned on $s$ and $z$, then back to atoms — and is trained at many
  noise levels. No residue frames, no torsion angles, no rotational
  equivariance built in (handled by data augmentation).
- **Hallucination control**: cross-distillation from AlphaFold-Multimer
  predictions teaches it to output extended, low-confidence regions for
  disordered parts rather than invented structure.
- **Confidence head**: pLDDT, PAE and PDE, trained by rolling out the
  diffusion sampler during training.
- Recycling of the trunk is kept.

What it is **not**: not a language model; not equivariant by construction;
and as a generative model it can hallucinate, especially for disordered
regions. Code and weights were released (Nov 2024) under restrictive,
non-commercial terms.

## Code

```python
# Illustrative diffusion sampling loop over atom coordinates (EDM-style).
import torch

def sample(denoiser, s, z, n_atoms, sigmas):
    x = sigmas[0] * torch.randn(n_atoms, 3)          # start from pure noise
    for s_hi, s_lo in zip(sigmas[:-1], sigmas[1:]):
        x0 = denoiser(x, s_hi, s, z)                  # predict clean coords
        d = (x - x0) / s_hi                           # score direction
        x = x + (s_lo - s_hi) * d                     # Euler step to lower noise
    return x

denoiser = lambda x, sig, s, z: x / (1 + sig**2) ** 0.5  # stand-in network
sigmas = torch.linspace(160.0, 0.0, 50)
coords = sample(denoiser, s=None, z=None, n_atoms=500, sigmas=sigmas)
print(coords.shape)                                  # (n_atoms, 3)
```

## Why it matters

AlphaFold 3 (Google DeepMind and Isomorphic Labs) extended accurate
structure prediction from single proteins to nearly all molecule types in the
PDB, with large reported gains on protein–ligand and protein–nucleic-acid
complexes over specialised tools. Its Pairformer + diffusion design has been
widely reimplemented in open models.

## Self-test

Q: What replaced AlphaFold 2's Evoformer in AlphaFold 3?
A: The Pairformer, which updates only pair and single representations (the MSA is processed by a small separate MSA module).

Q: What replaced the structure module?
A: A diffusion module that iteratively denoises raw atom coordinates.

Q: How are ligands tokenized in AlphaFold 3?
A: One token per atom (standard residues/nucleotides get one token each).

Q: Why was cross-distillation from AlphaFold-Multimer used?
A: To reduce hallucinated structure in disordered regions by teaching the model to output extended, low-confidence conformations.
