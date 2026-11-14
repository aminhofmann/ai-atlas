---
id: autoregressive-training
title: Autoregressive Training (Next-Token Prediction)
category: training
level: 1
year: 2018
summary: >-
  Trains a model to predict each token from the tokens before it, turning raw
  text into a nearly limitless supply of supervised examples.
requires: [tokenization, causal-masking]
links:
  - { to: gpt, type: applied-in }
  - { to: llama, type: applied-in }
  - { to: masked-language-modeling, type: alternative-to }
  - { to: pretraining-and-fine-tuning, type: part-of }
  - { to: kv-cache, type: related }
  - { to: scaling-laws, type: related }
sources:
  - { title: "Improving Language Understanding by Generative Pre-Training (Radford et al., 2018)", url: "https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf" }
  - { title: "Language Models are Unsupervised Multitask Learners (Radford et al., 2019)", url: "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf" }
---

## Intuition
Imagine reading a book with your hand covering everything after the current
word, and guessing the next word before sliding your hand along. Every word in
every book becomes a little quiz, and the answer key is the book itself. No
human labelling is needed.

A model trained this way must learn grammar, facts and reasoning patterns,
because all of them help it guess better. At generation time the same model
writes text by repeatedly guessing a token, appending it, and guessing again.

## Mechanics
A sequence $x_1, \dots, x_T$ is factorised by the chain rule of probability:

$$p_\theta(x_1,\dots,x_T) = \prod_{t=1}^{T} p_\theta(x_t \mid x_{<t})$$

Training minimises the negative log-likelihood (cross-entropy) averaged over
positions:

$$\mathcal{L}(\theta) = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})$$

- **Shifted targets.** The input is $x_{1:T-1}$, the target is $x_{2:T}$. The
  model outputs logits $\in \mathbb{R}^{T \times |V|}$; position $t$'s logits
  are scored against token $t+1$.
- **Causal mask.** A lower-triangular mask in self-attention stops position $t$
  from seeing $x_{>t}$. This is what lets all $T$ predictions be computed in
  ONE parallel forward pass instead of $T$ sequential ones.
- **Teacher forcing.** During training the model is always conditioned on the
  true previous tokens, never on its own guesses. At inference it consumes its
  own outputs, a mismatch sometimes called exposure bias.
- **Perplexity** $= \exp(\mathcal{L})$ is the standard metric: roughly how many
  tokens the model is "choosing between" on average.
- **Decoding** is separate from training: greedy, beam search, temperature,
  top-k or nucleus (top-p) sampling all read the same learned distribution.
  A KV cache avoids recomputing past keys and values at each generation step.

Every token in the corpus provides a training signal, which makes the objective
very data-efficient compared with objectives that score only a subset of
positions (BERT scores ~15%).

## Code
```python
import torch
import torch.nn.functional as F

def lm_loss(model, tokens):              # tokens: (B, T) int64
    inputs  = tokens[:, :-1]             # x_1 .. x_{T-1}
    targets = tokens[:, 1:]              # x_2 .. x_T
    logits = model(inputs)               # (B, T-1, V); model applies a causal mask
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    )

@torch.no_grad()
def generate(model, prompt, n_new, temperature=1.0):
    x = prompt
    for _ in range(n_new):
        logits = model(x)[:, -1, :] / temperature
        next_tok = torch.multinomial(F.softmax(logits, -1), 1)
        x = torch.cat([x, next_tok], dim=1)
    return x
```

## Why it matters
Next-token prediction is the pretraining objective behind the GPT family,
Llama and essentially every modern large language model. Its appeal is
simplicity and scale: any text is training data, the loss is smooth and
predictable (it is what scaling laws are measured on), and the same objective
gives you a generator for free. Its limits are also well known: it optimises
likelihood of internet text, not helpfulness or truthfulness, which is why
instruction tuning and RLHF follow it.

## Self-test
Q: How are inputs and targets related in autoregressive training?
A: The targets are the inputs shifted left by one token, so position t is trained to predict token t+1.

Q: Why can all T next-token predictions be computed in one forward pass?
A: The causal mask prevents each position from attending to future tokens, so the predictions are independent of later inputs and can be computed in parallel.

Q: What is teacher forcing?
A: Conditioning the model on the ground-truth previous tokens during training rather than on its own sampled outputs.

Q: How is perplexity defined?
A: Perplexity is the exponential of the average per-token cross-entropy loss.
