---
id: pretraining-and-fine-tuning
title: Pretraining and Fine-Tuning
category: training
level: 1
year: 2018
summary: >-
  First train a large model on huge amounts of unlabelled data, then adapt the
  same weights to a specific task with a much smaller labelled dataset.
requires: [autoregressive-training, masked-language-modeling]
links:
  - { to: bert, type: applied-in }
  - { to: gpt, type: applied-in }
  - { to: t5, type: applied-in }
  - { to: lora, type: related }
  - { to: rlhf, type: related }
  - { to: biological-foundation-models, type: applied-in }
  - { to: scaling-laws, type: related }
sources:
  - { title: "Universal Language Model Fine-tuning for Text Classification (Howard & Ruder, 2018)", url: "https://arxiv.org/abs/1801.06146" }
  - { title: "Improving Language Understanding by Generative Pre-Training (Radford et al., 2018)", url: "https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf" }
  - { title: "BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)", url: "https://arxiv.org/abs/1810.04805" }
  - { title: "Finetuned Language Models Are Zero-Shot Learners (Wei et al., 2021)", url: "https://arxiv.org/abs/2109.01652" }
---

## Intuition
A medical student first spends years on general biology and chemistry, then
specialises in cardiology in a fraction of the time. Pretraining is the general
education: expensive, done once, learned from raw material nobody had to
label. Fine-tuning is the specialisation: cheap, repeated for each job.

The key bet is that the general knowledge transfers. A model that has learned
how language (or proteins, or cells) works needs only a few thousand labelled
examples to learn a particular task on top.

## Mechanics
**Stage 1 — pretraining.** Optimise a self-supervised loss on a large corpus
$\mathcal{D}_{\text{pre}}$:
- autoregressive next-token loss (GPT, Llama), or
- masked/denoising loss (BERT, T5).

This dominates the compute budget and produces weights $\theta_0$.

**Stage 2 — fine-tuning.** Initialise from $\theta_0$ and minimise a task loss
on a small labelled set $\mathcal{D}_{\text{task}}$, usually with a smaller
learning rate and few epochs:

$$\theta^{*} = \arg\min_\theta \; \mathbb{E}_{(x,y) \sim \mathcal{D}_{\text{task}}} \, \ell(f_\theta(x), y), \quad \theta \text{ initialised at } \theta_0$$

Common variants:
- **Task head.** BERT-style: add a small linear layer on the `[CLS]` vector
  (classification) or on each token (tagging), train everything end to end.
- **Text-to-text.** T5-style: express every task as input text → output text,
  so no new head is needed.
- **Instruction tuning / SFT.** Fine-tune a generative model on many tasks
  phrased as instructions with target responses (e.g. FLAN), improving
  zero-shot behaviour on unseen tasks. This is the first step of the RLHF
  pipeline.
- **Parameter-efficient fine-tuning (PEFT).** Freeze $\theta_0$ and train only a
  few added parameters (LoRA, adapters, prefix tuning).
- **Continued pretraining / domain adaptation.** Keep the self-supervised loss
  but switch to in-domain data before task fine-tuning.
- **No fine-tuning at all.** Large enough models can do tasks from a prompt
  with a few examples (in-context learning, shown at scale by GPT-3).

Risks: **catastrophic forgetting** of pretraining knowledge, and overfitting a
tiny dataset — both reasons for small learning rates or frozen backbones.

## Code
```python
# Illustrative HuggingFace usage: fine-tune a pretrained BERT for 2-class sentiment.
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2)          # pretrained body + new random head

opt = torch.optim.AdamW(model.parameters(), lr=2e-5)  # small LR: don't erase pretraining
batch = tok(["great film", "dull and slow"], return_tensors="pt", padding=True)
labels = torch.tensor([1, 0])

model.train()
out = model(**batch, labels=labels)             # cross-entropy on the head
out.loss.backward()
opt.step(); opt.zero_grad()
```

## Why it matters
Before 2018, NLP models were mostly trained from scratch per task, with only
the word embeddings pretrained. ULMFiT, GPT-1 and BERT showed that pretraining
the *whole network* and fine-tuning it beat task-specific architectures across
the board. The paradigm is now universal: "foundation models" in language,
vision, proteins and single-cell biology are all pretrained once and adapted
many times.

## Self-test
Q: Which stage of the paradigm consumes most of the compute, and what data does it use?
A: Pretraining; it uses large amounts of unlabelled data with a self-supervised objective.

Q: Why are fine-tuning learning rates typically small?
A: To adapt to the task without destroying the knowledge stored in the pretrained weights (catastrophic forgetting).

Q: What is instruction tuning?
A: Supervised fine-tuning of a generative model on many tasks phrased as natural-language instructions with target responses.

Q: How does parameter-efficient fine-tuning differ from full fine-tuning?
A: The pretrained weights are frozen and only a small number of added parameters (e.g. LoRA matrices) are trained.
