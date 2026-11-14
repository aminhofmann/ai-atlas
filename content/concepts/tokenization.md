---
id: tokenization
title: Tokenization
category: foundations
level: 1
summary: >-
  Splits raw text into a sequence of discrete units (tokens) from a fixed
  vocabulary, each mapped to an integer id that the model can look up.
requires: []
links:
  - { to: transformer, type: part-of }
  - { to: embeddings, type: related }
  - { to: dnabert, type: applied-in }
  - { to: vision-transformer, type: related }
sources:
  - { title: "Neural Machine Translation of Rare Words with Subword Units (Sennrich et al., 2016) — BPE", url: "https://arxiv.org/abs/1508.07909" }
  - { title: "SentencePiece: A simple and language independent subword tokenizer (Kudo & Richardson, 2018)", url: "https://arxiv.org/abs/1808.06226" }
  - { title: "Subword Regularization (Kudo, 2018) — Unigram LM tokenizer", url: "https://arxiv.org/abs/1804.10959" }
---

## Intuition
A neural network cannot read letters; it needs numbers from a fixed menu. Tokenization is the step that chops text into pieces from that menu and replaces each piece with its menu number.

Think of Lego: very common words ("the", "and") are single pre-made bricks, while a rare word like "unbelievably" is assembled from smaller bricks ("un", "believ", "ably"). With a few tens of thousands of bricks you can build any text, without needing a brick for every word ever written.

The choice of bricks matters more than it seems: it decides how long a sequence gets (and thus how much compute is spent), and it explains odd model behaviour such as struggling to count the letters in a word it only ever saw as one brick.

## Mechanics
Three granularities trade off vocabulary size against sequence length:
- **Characters/bytes:** tiny vocabulary, very long sequences.
- **Words:** short sequences, but a huge vocabulary and unknown words (OOV).
- **Subwords:** the standard compromise — frequent strings get their own token, rare ones decompose.

Main subword algorithms (a vocabulary of size $V$ is *learned* from a corpus once, then frozen):
- **BPE (Byte-Pair Encoding):** start from characters; repeatedly count adjacent symbol pairs and merge the most frequent pair into a new symbol until $V$ is reached. Encoding replays the merges in learned order. **Byte-level BPE** (GPT-2) uses the 256 bytes as the base alphabet, so any string is encodable and there is no unknown token; GPT-2's vocabulary has 50,257 entries.
- **WordPiece** (BERT): also merges pairs, but picks the pair that most increases training-data likelihood, roughly $\frac{\text{freq}(ab)}{\text{freq}(a)\,\text{freq}(b)}$. Encoding is greedy longest-match; word-internal pieces carry a `##` prefix.
- **Unigram LM:** start from a large candidate vocabulary and prune the tokens whose removal hurts corpus likelihood least. Segmentation is probabilistic (Viterbi for the best split), which enables *subword regularization* — sampling different splits during training.
- **SentencePiece** is a library/framework, not a separate algorithm: it implements BPE and Unigram directly on raw text, treating whitespace as a symbol (`▁`), so no language-specific pre-tokenizer is needed. Used by T5 and the first Llama models.

The output is a vector of ids $x \in \{0,\dots,V-1\}^n$, usually wrapped with special tokens (`<bos>`, `<eos>`, `[CLS]`, `[SEP]`, `<pad>`). Consequences: compute scales with $n$ (quadratically in attention), languages under-represented in the tokenizer's corpus need more tokens per word, and digits/spelling are split inconsistently.

## Code
```python
from collections import Counter

def train_bpe(words, num_merges):
    # each word as a tuple of symbols, plus an end-of-word marker
    vocab = Counter(tuple(w) + ("</w>",) for w in words)
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for word, freq in vocab.items():
            for a, b in zip(word, word[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = max(pairs, key=pairs.get)          # most frequent adjacent pair
        merges.append(best)
        new_vocab = Counter()
        for word, freq in vocab.items():
            out, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best:
                    out.append(word[i] + word[i + 1]); i += 2
                else:
                    out.append(word[i]); i += 1
            new_vocab[tuple(out)] += freq
        vocab = new_vocab
    return merges

print(train_bpe(["low", "lower", "lowest", "newer", "wider"], 5))
```

## Why it matters
Subword tokenization solved the open-vocabulary problem of word-level models: any input can be represented, yet common words stay short. It is the first step of every language transformer and silently shapes cost (tokens = compute = price), multilingual fairness and failure modes. Outside text, "tokenization" generalizes to any discretization: image patches in ViT, k-mers or single nucleotides in DNA models, rank-ordered genes in single-cell models.

## Self-test
Q: What does one BPE training step do?
A: It counts all adjacent symbol pairs in the corpus and merges the most frequent pair into a new vocabulary symbol.

Q: Why does byte-level BPE never produce an unknown token?
A: Its base alphabet is all 256 byte values, so any string can be decomposed into known symbols.

Q: Is SentencePiece an algorithm like BPE?
A: No — it is a library that runs BPE or Unigram directly on raw text, treating whitespace as an ordinary symbol.

Q: How does WordPiece choose which pair to merge, compared with BPE?
A: BPE merges the most frequent pair; WordPiece merges the pair that most increases training-data likelihood (frequency normalized by the parts' frequencies).

Q: Why does a larger vocabulary make sequences cheaper to process?
A: More text fits into each token, so sequences get shorter, and attention cost grows with sequence length.
