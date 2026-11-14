---
id: dnabert
title: DNABERT and DNABERT-2
category: biology
level: 2
year: 2021
summary: >-
  BERT applied to genomic DNA: DNABERT tokenizes sequence into overlapping
  k-mers, while DNABERT-2 switches to byte-pair encoding and multi-species
  pretraining.
requires: [masked-language-modeling, tokenization]
links:
  - { to: bert, type: based-on }
  - { to: pretraining-and-fine-tuning, type: uses }
  - { to: flash-attention, type: uses }
  - { to: attention-visualization, type: applied-in }
  - { to: nucleotide-transformer, type: related }
  - { to: hyenadna, type: related }
  - { to: biological-foundation-models, type: part-of }
sources:
  - { title: "DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome (Ji et al., Bioinformatics 2021)", url: "https://academic.oup.com/bioinformatics/article/37/15/2112/6128680" }
  - { title: "DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species Genome (Zhou et al., ICLR 2024)", url: "https://arxiv.org/abs/2306.15006" }
---

## Intuition

DNA has only four letters, so single letters carry little meaning on their
own. DNABERT's answer was to read DNA in overlapping "words" of $k$ letters
($k$-mers) — like reading `ATGCGT` as `ATGCGT`, `TGCGTA`, `GCGTAC`, … — and
then train a standard BERT on the human genome with masked-word prediction.
Fine-tuned, it predicts promoters, splice sites and transcription-factor
binding sites, and its attention highlights candidate motifs.

DNABERT-2 noticed that overlapping $k$-mers are wasteful and leaky: each
letter appears in $k$ tokens, so a masked token can be partly reconstructed
from its neighbours. It instead *learns* a vocabulary with byte-pair encoding,
the same way GPT tokenizers learn sub-words.

## Mechanics

**DNABERT (2021)**
- Tokens: overlapping $k$-mers, $k \in \{3,4,5,6\}$ (separate models), stride 1;
  vocabulary $4^k$ + specials.
- Backbone: BERT-base (12 layers); max 512 tokens (~0.5 kb of sequence).
- Pretraining data: the human reference genome.
- Masking: **contiguous spans of $k$ tokens** are masked together, since a
  single masked $k$-mer is trivially inferable from its overlapping neighbours.
- Fine-tuning: promoter/TSS prediction, splice sites, TF binding; attention
  maps visualised as motif evidence (DNABERT-viz).

**DNABERT-2 (2023, ICLR 2024)**
- Tokens: **BPE** learned on genomes — variable-length, non-overlapping tokens,
  shortening sequences several-fold.
- Positional information: **ALiBi** (attention biases) instead of learned
  absolute positions, so input length is not hard-capped.
- Efficiency: FlashAttention, low-precision training; ~117M parameters.
- Data: genomes from many species (multi-species corpus).
- Introduced **GUE** (Genome Understanding Evaluation), a multi-species
  benchmark of classification tasks.

What they are **not**: not long-context models (hundreds to a few thousand
bp), not generative, and DNABERT's overlapping $k$-mers are not
interchangeable with Nucleotide Transformer's non-overlapping 6-mers.

## Code

```python
# DNABERT-style overlapping k-mers vs DNABERT-2 BPE.
def kmers(seq, k=6):
    return [seq[i:i + k] for i in range(len(seq) - k + 1)]   # stride 1

print(kmers("ATGCGTACGT", k=6))   # 5 overlapping tokens for 10 bp

from transformers import AutoTokenizer, AutoModel
name = "zhihan1996/DNABERT-2-117M"
tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
model = AutoModel.from_pretrained(name, trust_remote_code=True)
ids = tok("ATGCGTACGTTAGCATGCAAGT", return_tensors="pt")["input_ids"]
print(tok.convert_ids_to_tokens(ids[0]))    # variable-length BPE tokens
hidden = model(ids)[0]                       # (1, n_tokens, 768)
emb = hidden.mean(dim=1)                     # sequence embedding
```

## Why it matters

DNABERT was one of the first transfer-learning "DNA language models" and
made tokenization of DNA an explicit design question. DNABERT-2's BPE result
— comparable accuracy to much larger $k$-mer models at a fraction of the
compute — fed into later work, while long-range genome modeling moved to
architectures like HyenaDNA, Caduceus and Evo.

## Self-test

Q: How does DNABERT tokenize DNA?
A: Into overlapping k-mers (k = 3–6) with stride 1.

Q: Why does DNABERT mask contiguous spans of k tokens?
A: Because overlapping k-mers share letters, a single masked token could be reconstructed from its neighbours.

Q: What tokenizer and positional scheme does DNABERT-2 use?
A: Byte-pair encoding and ALiBi attention biases.

Q: What benchmark did DNABERT-2 introduce?
A: GUE, the Genome Understanding Evaluation multi-species benchmark.
