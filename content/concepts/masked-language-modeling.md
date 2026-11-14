---
id: masked-language-modeling
title: Masked Language Modeling
category: training
level: 1
year: 2018
summary: >-
  Hides a random subset of tokens and trains the model to recover them from
  context on both sides, producing strong bidirectional representations.
requires: [tokenization, self-attention]
links:
  - { to: bert, type: applied-in }
  - { to: autoregressive-training, type: alternative-to }
  - { to: pretraining-and-fine-tuning, type: part-of }
  - { to: t5, type: related }
  - { to: scbert, type: applied-in }
  - { to: geneformer, type: applied-in }
  - { to: esm2, type: applied-in }
  - { to: dnabert, type: applied-in }
sources:
  - { title: "BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)", url: "https://arxiv.org/abs/1810.04805" }
  - { title: "RoBERTa: A Robustly Optimized BERT Pretraining Approach (Liu et al., 2019)", url: "https://arxiv.org/abs/1907.11692" }
  - { title: "ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators (Clark et al., 2020)", url: "https://arxiv.org/abs/2003.10555" }
---

## Intuition
It is a fill-in-the-blanks exercise: "The cat sat on the ___ and purred." To
fill the gap you look both left AND right. A model trained on millions of such
gaps learns what each word means in its full surrounding context.

Unlike next-token prediction, the model never has to generate text left to
right, so it can use attention without a causal mask. The price: it is not a
natural text generator.

## Mechanics
Given a token sequence $x$, choose a set of positions $M$ and corrupt them to
get $\tilde{x}$. The loss is cross-entropy **only on the chosen positions**:

$$\mathcal{L} = -\sum_{i \in M} \log p_\theta(x_i \mid \tilde{x})$$

**BERT's recipe:**
- Select **15%** of tokens as prediction targets.
- Of those: **80%** replaced with `[MASK]`, **10%** replaced with a random
  token, **10%** left unchanged.
- Why the 80/10/10 split: `[MASK]` never appears at fine-tuning time, so
  always masking would create a train/test mismatch. Random and unchanged
  tokens force the model to keep a good representation of *every* input token,
  since it cannot tell which ones are being scored.
- The encoder is fully bidirectional: no causal mask. The final hidden states
  at masked positions go through an output layer tied to the input embeddings
  to predict over the vocabulary.
- BERT added a second objective, **Next Sentence Prediction (NSP)**, which
  later work showed was unnecessary.

**Refinements:**
- **RoBERTa**: dynamic masking (a new mask each epoch rather than one fixed at
  preprocessing), NSP removed, much more data and longer training.
- **Whole-word / span masking** (SpanBERT, T5's span corruption): mask
  contiguous spans so the model can't trivially reconstruct a word from its
  other sub-word pieces.
- **ELECTRA**: a small generator fills masked slots, and the main model
  classifies *every* token as original vs replaced, so all positions give a
  training signal instead of 15%.

The core inefficiency of MLM is exactly that: only the masked fraction of each
sequence contributes to the loss.

## Code
```python
import torch

def mask_tokens(input_ids, mask_id, vocab_size, special_mask, p=0.15):
    labels = input_ids.clone()
    probs = torch.full(labels.shape, p)
    probs[special_mask] = 0.0                       # never mask [CLS], [SEP], padding
    chosen = torch.bernoulli(probs).bool()
    labels[~chosen] = -100                          # ignored by cross_entropy

    ids = input_ids.clone()
    to_mask = torch.bernoulli(torch.full(ids.shape, 0.8)).bool() & chosen
    ids[to_mask] = mask_id                          # 80% -> [MASK]
    to_rand = torch.bernoulli(torch.full(ids.shape, 0.5)).bool() & chosen & ~to_mask
    ids[to_rand] = torch.randint(vocab_size, ids.shape)[to_rand]  # 10% -> random
    return ids, labels                              # remaining 10% unchanged
```

## Why it matters
MLM made bidirectional pretraining possible and, through BERT, set the state of
the art on classification, question answering and tagging benchmarks in 2018–
2019. It remains the default objective for encoder models wherever the goal is
a representation rather than generated text: search embeddings, protein
language models like ESM-2, DNA models and single-cell models such as
Geneformer and scBERT all use masked-token objectives.

## Self-test
Q: What fraction of tokens does BERT select for prediction, and how are they corrupted?
A: 15% are selected; of these 80% become [MASK], 10% a random token and 10% stay unchanged.

Q: Why not replace all selected tokens with [MASK]?
A: [MASK] never appears during fine-tuning, so always masking creates a pretrain/fine-tune mismatch; random and unchanged tokens make the model represent every token well.

Q: What did RoBERTa change about BERT's masking setup?
A: It used dynamic masking (a fresh mask each time a sequence is seen) and dropped Next Sentence Prediction.

Q: What efficiency problem does ELECTRA address?
A: In MLM only the ~15% masked positions give a loss; ELECTRA's replaced-token detection trains on every position.
