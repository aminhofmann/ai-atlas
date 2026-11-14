---
id: gpt
title: GPT (GPT-1 → GPT-4)
category: architectures
level: 1
year: 2018
summary: >-
  OpenAI's family of decoder-only transformers trained by next-token
  prediction, whose scaling from GPT-1 to GPT-4 turned language models into
  general-purpose assistants.
requires: [transformer, autoregressive-training, causal-masking]
links:
  - { to: self-attention, type: uses }
  - { to: kv-cache, type: uses }
  - { to: tokenization, type: uses }
  - { to: scaling-laws, type: related }
  - { to: rlhf, type: uses }
  - { to: pretraining-and-fine-tuning, type: uses }
  - { to: bert, type: alternative-to }
  - { to: llama, type: related }
  - { to: logit-lens, type: related }
sources:
  - { title: "Improving Language Understanding by Generative Pre-Training — GPT-1 (Radford et al., 2018)", url: "https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf" }
  - { title: "Language Models are Unsupervised Multitask Learners — GPT-2 (Radford et al., 2019)", url: "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf" }
  - { title: "Language Models are Few-Shot Learners — GPT-3 (Brown et al., 2020)", url: "https://arxiv.org/abs/2005.14165" }
  - { title: "GPT-4 Technical Report (OpenAI, 2023)", url: "https://arxiv.org/abs/2303.08774" }
---

## Intuition
GPT is the writer counterpart to BERT's reader. It keeps only the decoder half
of the Transformer and learns one thing: given the text so far, predict the
next token. Repeat that, and it writes.

The surprise of the series was how much falls out of that single objective
once you scale it. GPT-1 needed fine-tuning for each task; GPT-2 could do some
tasks with no task training at all; GPT-3 could learn a new task from a few
examples in the prompt; GPT-4 combined scale with human-feedback training into
a broadly capable assistant.

## Mechanics
**Shared architecture:** a stack of decoder blocks *without* cross-attention:
causal multi-head self-attention + FFN, residuals, LayerNorm, learned absolute
position embeddings, output projection tied to the token embedding. Training
loss: next-token cross-entropy.

| | Year | Params | Context | Notable change |
|---|---|---|---|---|
| GPT-1 | 2018 | ≈117M (12 layers, $d=768$) | 512 | generative pretraining + discriminative fine-tuning |
| GPT-2 | 2019 | up to 1.5B (48 layers, $d=1600$) | 1024 | **pre-norm** (LN moved to sublayer inputs + final LN), byte-level BPE, WebText; zero-shot task transfer |
| GPT-3 | 2020 | 175B (96 layers, $d=12288$, 96 heads) | 2048 | alternating dense and locally banded sparse attention; **in-context / few-shot learning** |
| GPT-4 | 2023 | not disclosed | 8K / 32K at launch | accepts image inputs; post-trained with RLHF |

Key ideas, by step:
- **GPT-1:** pretrain on BooksCorpus, then fine-tune with task inputs serialised
  into one token sequence plus a linear head (auxiliary LM loss kept).
- **GPT-2:** "language models are unsupervised multitask learners" — tasks can
  be specified in natural language ("TL;DR:" for summarisation).
- **GPT-3:** no gradient updates at test time; $k$ demonstrations in the
  prompt. Few-shot ability improved sharply with model size.
- **GPT-4:** the report deliberately withholds architecture, size, data and
  compute. It reports predicting final loss from runs using ≤1/1000th of the
  compute (scaling laws in practice) and human-exam-level benchmark results.
  Any claims about GPT-4 being MoE or its parameter count are unconfirmed.

Between GPT-3 and GPT-4, **InstructGPT** (2022) added SFT + RLHF, the recipe
behind ChatGPT.

## Code
```python
# Illustrative HuggingFace usage with the open GPT-2 weights.
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")       # 124M-param release

prompt = "Translate English to French:\ncheese =>"         # zero/few-shot prompting
ids = tok(prompt, return_tensors="pt").input_ids
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=5, do_sample=False)  # uses a KV cache
print(tok.decode(out[0][ids.shape[1]:]))

# Training signal is just next-token loss:
loss = model(ids, labels=ids).loss   # HF shifts labels internally
```

## Why it matters
GPT made the decoder-only, next-token-prediction transformer the dominant
design for large language models; Llama, Mistral and most open models follow
the same template. GPT-3 demonstrated in-context learning and helped motivate
scaling laws as a planning tool, and ChatGPT (built on the GPT-3.5/4 line with
RLHF) brought LLMs to mainstream use.

## Self-test
Q: Which Transformer component does GPT drop compared with the original decoder?
A: Cross-attention (and the encoder entirely); it is a decoder-only stack of causal self-attention and FFN blocks.

Q: What normalisation change did GPT-2 introduce?
A: Pre-norm: LayerNorm moved to the input of each sublayer, plus a final LayerNorm.

Q: How many parameters does GPT-3 have, and what capability did its paper highlight?
A: 175 billion; few-shot in-context learning from examples in the prompt without gradient updates.

Q: What does the GPT-4 technical report disclose about its architecture and size?
A: Essentially nothing; architecture, parameter count, data and compute were deliberately withheld.
