---
id: bert
title: BERT
category: architectures
level: 1
year: 2018
summary: >-
  An encoder-only transformer pretrained with masked language modeling, giving
  each token a context-aware representation that reads both left and right.
requires: [transformer, masked-language-modeling]
links:
  - { to: self-attention, type: uses }
  - { to: pretraining-and-fine-tuning, type: uses }
  - { to: tokenization, type: uses }
  - { to: gpt, type: alternative-to }
  - { to: probing, type: related }
  - { to: attention-visualization, type: related }
sources:
  - { title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (Devlin et al., 2018)", url: "https://arxiv.org/abs/1810.04805" }
  - { title: "RoBERTa: A Robustly Optimized BERT Pretraining Approach (Liu et al., 2019)", url: "https://arxiv.org/abs/1907.11692" }
---

## Intuition
BERT is a reader, not a writer. It takes a whole sentence at once and produces
a vector per token that reflects everything around it: "bank" in "river bank"
and "bank account" gets different vectors. It learned this by playing
fill-in-the-blanks on books and Wikipedia.

To use it, you put a small task-specific layer on top and fine-tune briefly.
That recipe beat specialised models on a wide range of benchmarks.

## Mechanics
**Architecture:** the Transformer's *encoder stack only* — no decoder, no
causal mask, so every token attends to every other token.
- **BERT-Base:** 12 layers, hidden 768, 12 heads, ≈110M parameters.
- **BERT-Large:** 24 layers, hidden 1024, 16 heads, ≈340M parameters.
- Learned absolute position embeddings, max length 512; GELU FFNs; post-norm.

**Input:** WordPiece tokens (~30k vocabulary). Every sequence starts with
`[CLS]`; sentence pairs are separated by `[SEP]`. The input vector is the sum of
three embeddings: **token + position + segment** (sentence A or B).

**Pretraining objectives** (on BooksCorpus + English Wikipedia):
1. **Masked LM:** select 15% of tokens; replace 80% of those with `[MASK]`,
   10% with a random token, keep 10% unchanged; predict the originals.
2. **Next Sentence Prediction (NSP):** given A and B, classify from `[CLS]`
   whether B actually followed A (50%) or is random (50%).

**Fine-tuning:**
- Sentence classification: linear layer on the final `[CLS]` vector.
- Token tagging (NER): linear layer on each token's vector.
- Extractive QA (SQuAD): predict start and end positions of the answer span.

**Descendants:**
- **RoBERTa:** same architecture; removed NSP, dynamic masking, bigger batches,
  far more data — showed original BERT was under-trained.
- **ALBERT:** factorised embeddings and cross-layer parameter sharing.
- **DistilBERT:** distilled smaller model.
- **ELECTRA:** replaced-token detection instead of MLM.

**What BERT is NOT:** a generator. Without a causal mask and next-token
objective it cannot naturally produce text left to right.

## Code
```python
# Illustrative HuggingFace usage: fill a mask and get contextual embeddings.
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
mlm = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

enc = tok("The river [MASK] was flooded.", return_tensors="pt")
# input_ids look like: [CLS] the river [MASK] was flooded . [SEP]
with torch.no_grad():
    out = mlm(**enc, output_hidden_states=True)

mask_pos = (enc.input_ids == tok.mask_token_id).nonzero()[0, 1]
print(tok.decode(out.logits[0, mask_pos].topk(5).indices))  # top guesses
cls_vec = out.hidden_states[-1][0, 0]   # (768,) final [CLS] representation
```

## Why it matters
BERT (2018) established "pretrain a big bidirectional encoder, then fine-tune"
as the default for NLP understanding tasks and topped GLUE, SQuAD and other
benchmarks at release. Decoder-only LLMs later took over most generative
work, but BERT-style encoders remain workhorses for classification, retrieval
embeddings and reranking, and the BERT recipe was copied directly into biology
(scBERT, Geneformer, DNABERT, ESM-style protein models).

## Self-test
Q: Which part of the original Transformer does BERT use?
A: Only the encoder stack, with bidirectional (unmasked) self-attention.

Q: What are BERT-Base's layers, hidden size, heads and parameter count?
A: 12 layers, hidden size 768, 12 heads, about 110M parameters.

Q: Which three embeddings are summed to form BERT's input?
A: Token, position and segment embeddings.

Q: What are BERT's two pretraining objectives?
A: Masked language modeling and Next Sentence Prediction.

Q: Which BERT objective did RoBERTa drop?
A: Next Sentence Prediction.
