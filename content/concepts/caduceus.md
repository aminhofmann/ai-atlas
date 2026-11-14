---
id: caduceus
title: Caduceus
category: biology
level: 3
year: 2024
summary: >-
  A bidirectional, reverse-complement-equivariant DNA language model built
  from Mamba blocks rather than attention.
requires: [masked-language-modeling, biological-foundation-models]
links:
  - { to: mamba, type: based-on }
  - { to: state-space-models, type: uses }
  - { to: attention, type: alternative-to }
  - { to: hyenadna, type: improves }
  - { to: nucleotide-transformer, type: alternative-to }
  - { to: evo, type: related }
  - { to: dnabert, type: related }
sources:
  - { title: "Caduceus: Bi-Directional Equivariant Long-Range DNA Sequence Modeling (Schiff et al., ICML 2024)", url: "https://arxiv.org/abs/2403.03234" }
  - { title: "Caduceus code (kuleshov-group)", url: "https://github.com/kuleshov-group/caduceus" }
---

## Intuition

DNA is double-stranded: the same stretch can be read on one strand or as its
**reverse complement** on the other (reverse the order, swap A↔T and C↔G).
Biologically these carry the same information, so a good model should give
consistent answers for both. Regulation also depends on what lies both
upstream and downstream of a position, so the model should read in both
directions.

HyenaDNA reads long DNA cheaply but only left-to-right, and has no built-in
strand symmetry. Caduceus keeps the cheap, long-range idea but uses Mamba
(a selective state-space model), runs it in both directions, and wires the
weights so that reverse-complement symmetry is guaranteed by construction —
not merely hoped for through data augmentation.

## Mechanics

- **Tokens**: single nucleotides.
- **BiMamba**: apply a Mamba block to the sequence and to its reversal,
  **sharing (tying) the projection weights** between the two directions, and
  combine outputs — bidirectional at little extra parameter cost.
- **MambaDNA block**: makes a block **RC-equivariant**. Split channels into
  two halves; run BiMamba on one half and on the *reverse complement* of the
  other half with shared parameters, then reverse-complement it back and
  concatenate. Formally, with $\mathrm{RC}$ the reverse-complement map,
  $f(\mathrm{RC}(x)) = \mathrm{RC}(f(x))$.
- **Caduceus-PS** (parameter sharing): RC-equivariant embeddings, MambaDNA
  blocks and an equivariant LM head → equivariance end to end.
- **Caduceus-Ph** (post-hoc): train with RC data augmentation, then average
  predictions over the sequence and its reverse complement at inference.
- **Pretraining**: masked language modeling on the human reference genome,
  with contexts up to ~131k bp.
- **Results**: strong on genomic benchmarks and, notably, long-range
  **variant effect prediction**, where it beat attention-based models roughly
  10× larger that lack bidirectionality or equivariance.

What it is **not**: not attention-based and not a transformer; not a
generative left-to-right model (it is a bidirectional masked LM).

## Code

```python
# Illustrative reverse-complement equivariance check.
import torch

COMP = torch.tensor([3, 2, 1, 0])          # A,C,G,T -> T,G,C,A (one-hot index map)
def rc(x):                                  # x: (L, 4) one-hot
    return x.flip(0)[:, COMP]

def rc_equivariant(f):
    # combine f on the strand and on its reverse complement (post-hoc style)
    return lambda x: 0.5 * (f(x) + rc(f(rc(x))))

f = torch.nn.Sequential(torch.nn.Linear(4, 4))      # stand-in per-position model
g = rc_equivariant(f)
x = torch.eye(4)[torch.randint(0, 4, (10,))]
print(torch.allclose(g(rc(x)), rc(g(x)), atol=1e-6)) # True: g(RC x) = RC g(x)
```

## Why it matters

Caduceus (Kuleshov group at Cornell, with Mamba co-authors Tri Dao and
Albert Gu) showed that building biological symmetries directly into a
sub-quadratic architecture pays off, especially for long-range variant
effects. It is a canonical example of Mamba used outside language.

## Self-test

Q: What two properties does Caduceus build in that HyenaDNA lacks?
A: Bidirectionality and reverse-complement equivariance.

Q: What is a reverse complement?
A: The DNA sequence read in reverse order with each base swapped for its pair (A↔T, C↔G).

Q: What is BiMamba?
A: A Mamba block applied in both directions with tied projection weights, combining the outputs.

Q: How does Caduceus-Ph differ from Caduceus-PS?
A: Ph uses RC augmentation in training plus post-hoc averaging over both strands; PS enforces equivariance through parameter sharing in the architecture.

Q: Is Caduceus a transformer?
A: No — it uses Mamba (selective state-space) blocks, not attention.
