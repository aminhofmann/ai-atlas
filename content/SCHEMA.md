# Concept file format

One concept = one Markdown file at `content/concepts/<id>.md`. The file name IS
the id (kebab-case). The app reads every file in that folder at build/request
time; there is no other registry.

## Frontmatter (YAML)

```yaml
---
id: multi-head-attention          # must equal the file name
title: Multi-Head Attention
category: foundations             # foundations | training | architectures | innovations | biology | interpretability | domains
level: 2                          # 1 = start here, 2 = intermediate, 3 = advanced
year: 2017                        # optional: year introduced
summary: >-                       # ONE sentence, plain language
  Runs several attention operations in parallel, each in its own learned
  subspace, so the model can attend to different kinds of relationships at once.
requires: [attention, embeddings] # prerequisite ids — what you must understand first
links:                            # typed relationships to OTHER ids
  - { to: transformer-block, type: part-of }  # reads: multi-head-attention part-of transformer-block
sources:
  - { title: "Attention Is All You Need (Vaswani et al., 2017)", url: "https://arxiv.org/abs/1706.03762" }
---
```

**Direction rule for `links`:** every edge reads as a sentence with THIS concept
as the subject: `<this> <type> <to>`.
- `multi-head-attention part-of transformer-block`
- `grouped-query-attention variant-of multi-head-attention` (written in the GQA file)
- `mamba alternative-to attention`
- `geneformer is-a bert`-style → use `based-on` (`geneformer based-on bert`)

Allowed `type` values: `part-of`, `uses`, `based-on`, `variant-of`,
`alternative-to`, `improves`, `applied-in`, `related`.

Only link to ids in the list below. Don't duplicate `requires` in `links`.

## Body sections (exactly these H2 headings, in this order)

```markdown
## Intuition
Plain-language explanation with an analogy. 1–3 short paragraphs. No math.

## Mechanics
How it actually works. Use LaTeX: inline `$...$`, display `$$...$$`.
Name tensor shapes where it helps (e.g. $Q \in \mathbb{R}^{n \times d_k}$).
Bullet lists welcome. This is the "detailed explanation" layer — be thorough
and precise: 150–400 words.

## Code
One short, runnable-in-spirit PyTorch (or Python) snippet in a ```python block,
≤ 30 lines, that shows the core idea. For models (e.g. Geneformer), show how
the input is built or how the model is used (e.g. HuggingFace loading), and say
so if the exact API is illustrative.

## Why it matters
What problem it solved, what it replaced, where it sits in the field today.
Mention concrete numbers only if you are sure of them.

## Self-test
Q: question one?
A: answer one.

Q: question two?
A: answer two.
```

Self-test: 3–5 Q/A pairs, each `Q:` line followed by an `A:` line, blank line
between pairs. These become spaced-repetition flashcards, so answers must be
short (1–2 sentences) and checkable.

## Accuracy rules

- These are study notes. A wrong fact gets memorised. If you are not sure of a
  number, date, author or architecture detail, leave it out — or verify it
  with a web search.
- Every file needs at least one source: the original paper (arXiv / journal
  link) preferred.
- Be honest about what a model is NOT (e.g. Evo is a StripedHyena hybrid, not
  a pure transformer; AlphaFold 2's Evoformer is attention-based but not a
  standard transformer).

## The complete id list (link only to these)

foundations: tokenization, embeddings, positional-encoding, rotary-position-embedding, attention, query-key-value, self-attention, causal-masking, multi-head-attention, cross-attention, transformer-block, layer-normalization, feed-forward-network, residual-connections, encoder-decoder, kv-cache

training: autoregressive-training, masked-language-modeling, pretraining-and-fine-tuning, lora, scaling-laws, rlhf

architectures: transformer, bert, gpt, t5, llama

innovations: mixture-of-experts, grouped-query-attention, multi-head-latent-attention, flash-attention, state-space-models, mamba, hybrid-architectures

biology: biological-foundation-models, scbert, geneformer, scgpt, scfoundation, uce, nicheformer, esm2, esmfold, esm3, alphafold2, alphafold3, protgpt2, dnabert, nucleotide-transformer, enformer, hyenadna, caduceus, evo

interpretability: attention-visualization, probing, logit-lens, mechanistic-interpretability, superposition, sparse-autoencoders, in-silico-perturbation

domains: vision-transformer, clip, whisper, decision-transformer, time-series-transformers, weather-transformers, molecular-transformers, transformers-in-other-domains
