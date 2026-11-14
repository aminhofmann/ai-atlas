---
id: alphafold2
title: AlphaFold 2
category: biology
level: 3
year: 2021
summary: >-
  DeepMind's protein structure predictor: an attention-based Evoformer that
  jointly refines MSA and residue-pair representations, followed by a
  geometry-aware structure module with invariant point attention.
requires: [attention, multi-head-attention, biological-foundation-models]
links:
  - { to: self-attention, type: uses }
  - { to: layer-normalization, type: uses }
  - { to: residual-connections, type: uses }
  - { to: feed-forward-network, type: uses }
  - { to: transformer, type: related }
  - { to: alphafold3, type: related }
  - { to: esmfold, type: related }
  - { to: attention-visualization, type: related }
sources:
  - { title: "Highly accurate protein structure prediction with AlphaFold (Jumper et al., Nature 2021)", url: "https://www.nature.com/articles/s41586-021-03819-2" }
  - { title: "Highly accurate protein structure prediction for the human proteome (Tunyasuvunakool et al., Nature 2021)", url: "https://www.nature.com/articles/s41586-021-03828-1" }
---

## Intuition

To predict how a protein folds, AlphaFold 2 looks at two things at once: an
alignment of the protein's evolutionary relatives (which positions mutate
together?), and a table of every residue pair (how far apart are $i$ and $j$
likely to be?). These two views constantly update each other — co-evolution
hints at contacts, contacts constrain which co-evolution signals make sense.

The pair table also has to be geometrically consistent: if A is near B and B is
near C, then A cannot be arbitrarily far from C. AlphaFold builds that
"triangle" reasoning directly into its layers. Finally, a structure module
places each residue as a rigid body in 3D, using an attention that does not
care how the whole molecule is rotated.

## Mechanics

- **Inputs**: target sequence, MSA from database search, optional templates.
- **Representations**: MSA $m \in \mathbb{R}^{s \times r \times c_m}$
  ($s$ sequences, $r$ residues) and pair $z \in \mathbb{R}^{r \times r \times c_z}$.
- **Evoformer** (48 blocks), each with:
  - *row-wise gated self-attention* over residues in each MSA row, **biased by
    the pair representation**;
  - *column-wise gated self-attention* over sequences at each position;
  - MSA transition (MLP);
  - **outer product mean**: MSA → pair communication;
  - **triangle multiplicative updates** (outgoing and incoming edges): update
    $z_{ij}$ from $z_{ik}$ and $z_{jk}$ (or $z_{ki}, z_{kj}$) over all $k$;
  - **triangle self-attention** around the starting and ending node, with the
    third edge as bias;
  - pair transition.
- **Structure module** (8 layers, shared weights): each residue is a rigid
  frame (rotation + translation). **Invariant Point Attention (IPA)** combines
  ordinary attention, pair bias, and distances between 3D points expressed in
  each residue's frame — invariant to global rotations and translations.
  Outputs backbone frames and side-chain torsion angles → all-atom coordinates.
- **Loss**: frame-aligned point error (FAPE) plus auxiliary losses (distogram,
  masked MSA, confidence).
- **Recycling**: outputs are fed back as inputs several times (3 by default).
- **Confidence**: pLDDT per residue; predicted aligned error (PAE).

What it is **not**: not a standard transformer or language model — no token
sequence in/out, no next-token objective; attention operates over 2D grids
(MSA, pair) with geometric inductive biases. It needs an MSA (unlike ESMFold).

## Code

```python
# Illustrative triangle multiplicative update ("outgoing" edges).
import torch, torch.nn as nn

class TriangleMultOutgoing(nn.Module):
    def __init__(self, c_z=128, c=128):
        super().__init__()
        self.norm = nn.LayerNorm(c_z)
        self.a, self.b = nn.Linear(c_z, c), nn.Linear(c_z, c)
        self.ga, self.gb = nn.Linear(c_z, c), nn.Linear(c_z, c)
        self.g, self.out, self.norm_out = nn.Linear(c_z, c_z), nn.Linear(c, c_z), nn.LayerNorm(c)

    def forward(self, z):                         # z: (r, r, c_z)
        z = self.norm(z)
        a = torch.sigmoid(self.ga(z)) * self.a(z)
        b = torch.sigmoid(self.gb(z)) * self.b(z)
        x = torch.einsum("ikc,jkc->ijc", a, b)     # combine edges i-k and j-k
        return torch.sigmoid(self.g(z)) * self.out(self.norm_out(x))

z = torch.randn(64, 64, 128)
print(TriangleMultOutgoing()(z).shape)            # residual: z = z + update
```

## Why it matters

AlphaFold 2 achieved near-experimental accuracy for many proteins at CASP14
(2020), effectively solving single-chain structure prediction for a large
class of proteins. The AlphaFold Protein Structure Database (with EMBL-EBI)
made predictions for a vast number of proteins freely available. Hassabis and
Jumper shared the 2024 Nobel Prize in Chemistry for this work (with David
Baker, for protein design).

## Self-test

Q: What two representations does the Evoformer maintain?
A: An MSA representation (sequences × residues) and a pair representation (residues × residues).

Q: What is the purpose of the triangle updates?
A: To make pair features geometrically consistent by updating edge ij using edges ik and jk over all third residues k.

Q: What makes Invariant Point Attention invariant?
A: It computes attention from 3D points expressed in local residue frames, so global rotations and translations do not change it.

Q: What is "recycling" in AlphaFold 2?
A: Feeding the predicted outputs back in as inputs and re-running the network several times (3 by default).

Q: Why is AlphaFold 2 not a standard transformer?
A: Its attention runs over MSA rows/columns and pair grids with pair biases, triangle operations and geometric IPA, not over a 1D token sequence with an LM objective.
