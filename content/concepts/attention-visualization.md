---
id: attention-visualization
title: Attention Visualization
category: interpretability
level: 1
year: 2019
summary: >-
  Plotting a transformer's attention weights as heatmaps to see which tokens each
  position "looks at" — useful for building intuition, but not a faithful
  explanation of why the model made its prediction.
requires: [attention, multi-head-attention]
links:
  - { to: self-attention, type: uses }
  - { to: residual-connections, type: related }
  - { to: mechanistic-interpretability, type: related }
  - { to: probing, type: related }
  - { to: bert, type: applied-in }
  - { to: vision-transformer, type: applied-in }
  - { to: molecular-transformers, type: applied-in }
  - { to: geneformer, type: applied-in }
sources:
  - { title: "Attention is not Explanation (Jain & Wallace, 2019)", url: "https://arxiv.org/abs/1902.10186" }
  - { title: "Attention is not not Explanation (Wiegreffe & Pinter, 2019)", url: "https://arxiv.org/abs/1908.04626" }
  - { title: "Quantifying Attention Flow in Transformers (Abnar & Zuidema, 2020)", url: "https://arxiv.org/abs/2005.00928" }
  - { title: "What Does BERT Look At? An Analysis of BERT's Attention (Clark et al., 2019)", url: "https://arxiv.org/abs/1906.04341" }
---

## Intuition
Every attention head produces, for each token, a probability distribution over
the other tokens. Those numbers are begging to be drawn: put query tokens on one
axis, key tokens on the other, colour each cell by its weight, and you get a
heatmap that looks like "where the model is looking". Tools such as BertViz made
this the first thing many people did with a new transformer.

The catch is the difference between *where the eye points* and *what the brain
uses*. A head can attend strongly to a token whose value vector is nearly zero,
or whose contribution is later overwritten by other heads and the
feed-forward network. Attention weights describe one
mixing step inside one layer, not the causal story of the whole prediction —
which is what the "attention is not explanation" debate is about.

## Mechanics
For one head, the attention matrix is

$$A = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) \in \mathbb{R}^{n \times n},$$

each row summing to 1. A model with $L$ layers and $H$ heads yields $L \cdot H$
such matrices per input. Common views:

- **Per-head heatmaps** (row = query, column = key). Clark et al. (2019) found
  BERT heads that attend to the previous/next token, to `[SEP]`, or along
  syntactic relations (e.g. direct objects to their verbs).
- **Head averages** across heads or layers — easy, but they blur specialised heads.

**The debate.** Jain & Wallace (2019) showed, for RNN-attention text
classifiers, that attention weights correlate poorly with gradient- and
leave-one-out importance, and that very different ("adversarial") attention
distributions can give the same prediction. Wiegreffe & Pinter (2019) replied
that this does not prove attention is *useless*: an explanation need not be
unique, and they proposed stricter tests. The consensus today: attention is a
*plausible* view, not a guaranteed *faithful* one.

**Attention rollout** (Abnar & Zuidema, 2020) tries to account for depth. Because
of residual connections, each layer's effective mixing is
roughly $\tilde{A}_\ell = \tfrac{1}{2}(A_\ell + I)$ (heads averaged, rows
renormalised). Multiplying through the layers,

$$R = \tilde{A}_L \tilde{A}_{L-1} \cdots \tilde{A}_1,$$

gives an estimate of how much each input token flows into each output position.
The same paper's **attention flow** treats the layers as a graph and solves a
max-flow problem. Rollout is widely used for ViT saliency
maps; it still ignores the value vectors and MLPs.

Gradient-weighted variants (attention × gradient) and causal methods such as
activation patching (see mechanistic interpretability)
go further toward faithfulness.

## Code
```python
import torch
from transformers import AutoTokenizer, AutoModel

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased", output_attentions=True)
enc = tok("The cat sat on the mat because it was tired", return_tensors="pt")
with torch.no_grad():
    atts = model(**enc).attentions  # tuple of L tensors, each (1, H, n, n)

# raw head: layer 5, head 3
head = atts[5][0, 3]                                  # (n, n)

# attention rollout (Abnar & Zuidema 2020)
n = head.shape[-1]
R = torch.eye(n)
for A in atts:
    A = A[0].mean(0)                                  # average heads -> (n, n)
    A = 0.5 * A + 0.5 * torch.eye(n)                  # account for residual
    A = A / A.sum(-1, keepdim=True)
    R = A @ R
tokens = tok.convert_ids_to_tokens(enc["input_ids"][0])
print(list(zip(tokens, R[0].tolist())))               # [CLS] row: input relevance
```

## Why it matters
Attention maps are the cheapest window into a transformer and remain the default
first look in NLP, vision and biology (e.g. claims that a gene model's heads
"attend to" transcription factors). Knowing their limits matters just as much:
a heatmap is a hypothesis generator, not evidence. Faithful claims need
interventions — ablations, patching, or perturbations — which is the direction
the field moved in after 2019.

## Self-test
Q: What does one row of an attention matrix represent?
A: The softmax distribution over key tokens that one query token uses to mix value vectors in a single head of a single layer.

Q: What was the central claim of "Attention is not Explanation" (Jain & Wallace, 2019)?
A: Attention weights correlate weakly with other importance measures, and very different attention distributions can produce the same prediction, so they are not a faithful explanation.

Q: How does attention rollout account for residual connections?
A: It adds the identity to each layer's head-averaged attention (roughly $\tfrac12(A+I)$), renormalises, and multiplies the matrices across layers.

Q: Name one thing attention maps ignore that affects a token's actual contribution.
A: The value vectors' magnitudes (and the later MLP/output-projection processing), so a high weight on a small value can matter little.
