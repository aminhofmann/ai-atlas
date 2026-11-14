---
id: embeddings
title: Embeddings
category: foundations
level: 1
summary: >-
  A learned lookup table that turns each token id into a dense vector, so that
  tokens used in similar ways end up close together in vector space.
requires: [tokenization]
links:
  - { to: transformer, type: part-of }
  - { to: positional-encoding, type: related }
  - { to: logit-lens, type: related }
  - { to: geneformer, type: applied-in }
sources:
  - { title: "Attention Is All You Need (Vaswani et al., 2017)", url: "https://arxiv.org/abs/1706.03762" }
  - { title: "Efficient Estimation of Word Representations in Vector Space (Mikolov et al., 2013)", url: "https://arxiv.org/abs/1301.3781" }
  - { title: "Using the Output Embedding to Improve Language Models (Press & Wolf, 2017)", url: "https://arxiv.org/abs/1608.05859" }
---

## Intuition
A token id like `4821` is just a label — the number itself says nothing about meaning. An embedding gives every token a "location" in a high-dimensional space, like a coordinate on a map, and training moves these locations around until tokens that behave alike (e.g. "cat" and "dog") sit near each other.

The model never sees words, only these coordinates. Everything downstream — attention, feed-forward layers — works by pushing those coordinates around and combining them.

## Mechanics
- **Lookup:** an embedding matrix $E \in \mathbb{R}^{V \times d_{\text{model}}}$ (vocabulary size $V$, model width $d_{\text{model}}$). Token id $t$ selects row $E_t$. Mathematically this equals a one-hot vector $e_t \in \{0,1\}^V$ times $E$, but it is implemented as an index gather.
- **Sequence:** ids $x \in \{0..V-1\}^n$ become $X \in \mathbb{R}^{n \times d_{\text{model}}}$ (or $B \times n \times d$ with batch). Position information is added or injected separately, because the lookup itself is position-blind.
- **Learning:** $E$ is an ordinary parameter trained by backprop; on each step only rows of tokens present in the batch receive gradient.
- **Scale:** the original Transformer multiplies embeddings by $\sqrt{d_{\text{model}}}$ before adding positional encodings.
- **Size:** $V \cdot d$ parameters — e.g. GPT-2 small: $50{,}257 \times 768 \approx 38.6$M, a large share of a small model.
- **Unembedding:** at the output, a matrix $W_U \in \mathbb{R}^{d \times V}$ maps the final hidden state $h$ to logits $z = h W_U$, then softmax gives next-token probabilities. **Weight tying** sets $W_U = E^\top$ (Press & Wolf, 2017), saving parameters and often helping; the original Transformer and GPT-2 tie, while many larger recent models keep them separate.
- **Static vs contextual:** the input embedding of "bank" is the same in every sentence (like word2vec). Only after attention layers does the hidden state become *contextual* — that is the transformer's job.
- **Geometry:** similarity is usually measured by dot product or cosine similarity; directions can encode features (the famous $\text{king} - \text{man} + \text{woman} \approx \text{queen}$ analogy comes from word2vec-style embeddings).
- **Other modalities:** "embedding" generalizes to any learned map into $\mathbb{R}^d$ — a linear projection of image patches, of a gene's expression bin, of an amino acid.

## Code
```python
import torch
import torch.nn as nn

V, d = 50_257, 768
emb = nn.Embedding(V, d)                 # E: (V, d)
ids = torch.tensor([[464, 3797, 3332]])  # (batch=1, n=3) token ids
X = emb(ids)                             # (1, 3, 768) — one row per token
print(X.shape)

# Output side: logits over the vocabulary with tied weights (W_U = E^T)
lm_head = nn.Linear(d, V, bias=False)
lm_head.weight = emb.weight              # weight tying: same tensor
h = torch.randn(1, 3, d)                 # pretend final hidden states
logits = lm_head(h)                      # (1, 3, V)
probs = logits.softmax(-1)
print(probs.sum(-1))                     # each position sums to 1
```

## Why it matters
Dense embeddings replaced sparse one-hot or count-based word representations and made generalization across similar words possible. In transformers the embedding table is the entry point (and, via tying, often the exit point) of the model; the vectors it produces form the start of the residual stream that every layer reads from and writes to. Interpretability tools such as the logit lens rely on projecting hidden states back through the unembedding.

## Self-test
Q: What are the shapes of the embedding matrix and the embedded sequence?
A: $E \in \mathbb{R}^{V \times d_{\text{model}}}$; a sequence of $n$ ids becomes $X \in \mathbb{R}^{n \times d_{\text{model}}}$.

Q: What is weight tying?
A: Using the transpose of the input embedding matrix as the output projection to vocabulary logits ($W_U = E^\top$).

Q: Is the input embedding of a token contextual?
A: No — it is the same vector in every context; context is mixed in later by attention layers.

Q: Which rows of $E$ get gradient on a training step?
A: Only the rows of tokens that appear in the current batch.
