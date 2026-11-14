---
id: logit-lens
title: Logit Lens and Tuned Lens
category: interpretability
level: 2
year: 2020
summary: >-
  Decoding a language model's intermediate hidden states with its own output
  layer to watch the next-token prediction form, layer by layer; the tuned lens
  adds a small learned translator per layer to make this reliable.
requires: [residual-connections, gpt, layer-normalization]
links:
  - { to: autoregressive-training, type: related }
  - { to: transformer-block, type: uses }
  - { to: probing, type: related }
  - { to: mechanistic-interpretability, type: part-of }
  - { to: llama, type: applied-in }
sources:
  - { title: "interpreting GPT: the logit lens (nostalgebraist, LessWrong, 2020)", url: "https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens" }
  - { title: "Eliciting Latent Predictions from Transformers with the Tuned Lens (Belrose et al., 2023)", url: "https://arxiv.org/abs/2303.08112" }
---

## Intuition
In a decoder-only transformer, every layer adds its contribution to a shared
running vector — the residual stream — and only at the very end is that vector
turned into word probabilities. The logit lens asks: what if we did that final
conversion *early*? Apply the model's own unembedding to the stream after layer
5, layer 10, layer 20, and read off what the model "would say" if it stopped
there.

It is like peeking at a draft essay at each revision: early drafts are vague,
middle drafts already have the right topic, the final draft has the exact word.
In GPT-2 the right next token often appears well before the last layer.

## Mechanics
Write the residual stream at layer $\ell$ as $h_\ell \in \mathbb{R}^{d}$. The
model's real output is

$$\text{logits} = W_U \,\mathrm{LN}_f(h_L), \quad W_U \in \mathbb{R}^{|V| \times d}.$$

**Logit lens** (nostalgebraist, 2020) simply substitutes an earlier layer:

$$\text{logits}_\ell = W_U \,\mathrm{LN}_f(h_\ell).$$

This works because residual connections keep all layers in roughly one shared
basis. Observations on GPT-2: predictions converge gradually; the input token
dominates early layers; the final answer often becomes top-1 in the middle.

**Problems.** For other models (e.g. BLOOM, some GPT-Neo variants) the logit
lens is noisy or meaningless at early layers: intermediate representations are
not aligned with the output basis ("representational drift"), and some
dimensions have outsized norms.

**Tuned lens** (Belrose et al., 2023) trains, per layer, an affine "translator"
$A_\ell h + b_\ell$ (initialised at identity) with the model frozen:

$$\text{logits}_\ell = W_U \,\mathrm{LN}_f(A_\ell h_\ell + b_\ell),$$

minimising $\mathrm{KL}(p_{\text{final}} \,\Vert\, p_\ell)$. It is a
probe whose target is the model's own final distribution, so it needs
no labels. They report lower perplexity and less bias than the logit lens across
models up to ~20B parameters, and use the per-layer predictions to detect
prompt injection.

Related ideas: decoding attention-head or MLP outputs through $W_U$
("direct logit attribution"), a basic tool in
mechanistic interpretability.

## Code
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
enc = tok("The Eiffel Tower is located in the city of", return_tensors="pt")

with torch.no_grad():
    out = model(**enc, output_hidden_states=True)

ln_f, W_U = model.transformer.ln_f, model.lm_head
for layer, h in enumerate(out.hidden_states):     # embeddings + 12 layers
    logits = W_U(ln_f(h[0, -1]))                   # last position only
    top = tok.decode(logits.argmax())
    print(f"layer {layer:2d}: {top!r}")
# Tuned lens: replace h with (A_l @ h + b_l), trained to match final logits (KL).
```
(The last `hidden_states` entry in HF GPT-2 already has `ln_f` applied; the
others do not.)

## Why it matters
The logit lens turned "the model is a black box until the end" into a picture
of iterative refinement and supports the residual-stream view of
transformers. It is cheap, label-free and works on any decoder model, which
makes it a standard first diagnostic. The tuned lens fixes its main failure
mode and is used for analysing where a model "decides", early exiting, and
anomaly detection.

## Self-test
Q: What does the logit lens compute?
A: The output vocabulary distribution obtained by applying the model's final layer norm and unembedding to an intermediate residual-stream state.

Q: Why can the logit lens work at all?
A: Residual connections make every layer write into a shared stream in roughly the same basis that the unembedding reads from.

Q: What does the tuned lens add and how is it trained?
A: A learned affine translator per layer, trained with the model frozen to minimise KL divergence to the model's final output distribution.

Q: Name a failure mode of the plain logit lens.
A: In some models early-layer representations are not aligned with the output basis, so decoded predictions are noisy or biased.
