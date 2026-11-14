# Claims to check against the papers

The agents that drafted the seed pages left out anything they couldn't verify.
These are the claims they DID include but flagged as recalled rather than checked.
Tick them off as you confirm them in the source paper.

## Training / architectures / innovations
- [ ] `scaling-laws` — Chinchilla fitted constants in the code (E=1.69, A=406.4, B=410.7, α=0.34, β=0.28)
- [ ] `hybrid-architectures` / `mixture-of-experts` — Jamba: 16 experts, top-2, 52B total / 12B active
- [ ] `mixture-of-experts` — Mixtral 8x7B: 46.7B total / 12.9B active
- [ ] `t5` — C4 ≈ 750 GB
- [ ] `multi-head-latent-attention` — "equivalent to GQA with ~2.25 groups" wording (512 + 64 cache layout and 93.3 % KV reduction were verified)

## Foundations
- [ ] `rotary-position-embedding` — Llama 3 RoPE base θ = 500,000
- [ ] `rotary-position-embedding` — ESM-2 uses RoPE

## Biology
- [ ] `nucleotide-transformer` — ~6 kb context for v1, v2 architecture changes, 18-task benchmark
- [ ] `alphafold3` — 4-block MSA module
- [ ] `caduceus` — pretraining up to ~131 kbp
- [ ] `uce` — published version (cited as bioRxiv preprint)
- [ ] `evo` — Evo 2 training data: ~9.3T bp (paper) vs 8.8T tokens (model card) — both given, labelled

## Interpretability / domains
- [ ] `in-silico-perturbation` — that Geneformer- and scGPT-based predictors were among the models in Ahlmann-Eltze et al. 2025
- [ ] `weather-transformers` — Aurora published in Nature (2025)

## Code snippets
Only the tokenization (BPE) snippet was executed. Model-loading snippets for
Geneformer, ChemBERTa, Aurora, ESM3, Evo 2, HyenaDNA, Enformer and scGPT are
marked illustrative in their files — checkpoint names and APIs unverified.
