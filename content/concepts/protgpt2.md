---
id: protgpt2
title: ProtGPT2
category: biology
level: 2
year: 2022
summary: >-
  A GPT-2-style autoregressive protein language model that generates novel
  protein sequences which look natural yet are distant from known ones.
requires: [autoregressive-training, biological-foundation-models]
links:
  - { to: gpt, type: based-on }
  - { to: tokenization, type: uses }
  - { to: causal-masking, type: uses }
  - { to: pretraining-and-fine-tuning, type: uses }
  - { to: esm2, type: alternative-to }
  - { to: esm3, type: related }
  - { to: evo, type: related }
sources:
  - { title: "ProtGPT2 is a deep unsupervised language model for protein design (Ferruz, Schmidt & Höcker, Nature Communications 2022)", url: "https://www.nature.com/articles/s41467-022-32007-7" }
  - { title: "ProtGPT2 model card (Hugging Face)", url: "https://huggingface.co/nferruz/ProtGPT2" }
---

## Intuition

ESM-style models are fill-in-the-blank readers. ProtGPT2 is a writer: like
GPT-2 producing text one word at a time, it produces a protein one chunk at a
time, each chunk conditioned on everything before it. Sample from it and you
get brand-new sequences that "sound" like proteins — similar amino-acid
statistics, predicted to fold, often with recognisable structural motifs —
but that don't exist in nature.

## Mechanics

- **Architecture**: GPT-2 large-style decoder-only transformer — 36 layers,
  ~738M parameters, causal self-attention.
- **Tokenizer**: **byte-pair encoding** trained on protein sequences
  (vocabulary ~50k), so a token covers several amino acids on average
  (around four), rather than one letter per token.
- **Data**: UniRef50 (~50M sequences); objective is next-token prediction,
  $\mathcal{L} = -\sum_t \log p(x_t \mid x_{<t})$.
- **Generation**: sampling with top-$k$ and a repetition penalty; generated
  sequences are often filtered by perplexity.
- **Findings (paper)**: generated sequences have natural-like amino-acid
  propensities and disorder content, are largely predicted to be globular by
  AlphaFold 2, and sit in under-explored regions of protein space while being
  only distantly related to training sequences.
- **Fine-tuning**: can be fine-tuned on a protein family to bias generation
  toward that family, the usual route to targeted design with it.
- **Context**: proteins are packed as FASTA-like text separated by the
  `<|endoftext|>` token, which also serves as the generation prompt.

What it is **not**: not structure-aware (sequence only) and not a
bidirectional encoder; its sequences are *predicted* to fold — the paper did
not experimentally validate function.

## Code

```python
from transformers import pipeline

gen = pipeline("text-generation", model="nferruz/ProtGPT2")
seqs = gen("<|endoftext|>",              # start-of-sequence prompt
           max_length=100, do_sample=True, top_k=950,
           repetition_penalty=1.2, num_return_sequences=3,
           eos_token_id=0)
for s in seqs:
    print(s["generated_text"].replace("\n", ""))  # model emits FASTA-style newlines
```

## Why it matters

ProtGPT2 was an early, openly released demonstration that decoder-only
language models can do *de novo* protein sequence generation. It sits on the
generative side of protein LMs, alongside later controllable models (ProGen,
ESM3).

## Self-test

Q: What objective does ProtGPT2 use?
A: Autoregressive next-token prediction (GPT-2 style).

Q: How does ProtGPT2 tokenize proteins?
A: With byte-pair encoding, so tokens span several amino acids rather than one.

Q: What data was ProtGPT2 trained on?
A: UniRef50 protein sequences.

Q: About how many parameters and layers does ProtGPT2 have?
A: About 738M parameters in 36 layers.
