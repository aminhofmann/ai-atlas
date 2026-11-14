---
id: causal-masking
title: Causal Masking
category: foundations
level: 2
year: 2017
summary: >-
  Blocks each position from attending to later positions, so a model can learn
  next-token prediction for a whole sequence in one parallel pass without cheating.
requires: [self-attention]
links:
  - { to: gpt, type: applied-in }
  - { to: autoregressive-training, type: related }
  - { to: masked-language-modeling, type: related }
  - { to: kv-cache, type: related }
  - { to: encoder-decoder, type: part-of }
sources:
  - { title: "Attention Is All You Need (Vaswani et al., 2017)", url: "https://arxiv.org/abs/1706.03762" }
  - { title: "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (Raffel et al., 2019) — causal vs prefix masks", url: "https://arxiv.org/abs/1910.10683" }
---

## Intuition
Imagine a fill-in-the-next-word exam where the whole text is printed on one sheet. To make it fair, each student gets a piece of cardboard that covers everything to the right of the word they are guessing. Now hundreds of students can each guess a different position at the same time, and nobody can peek at the answer.

Causal masking is that cardboard. It lets a decoder train on every position of a sequence in parallel while guaranteeing that each prediction only used the past — exactly the information it will have when generating text one token at a time.

## Mechanics
Add a mask $M \in \mathbb{R}^{n\times n}$ to the attention logits before softmax:
$$M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i\end{cases},\qquad Y = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V .$$
$M$ is strictly upper-triangular $-\infty$; $\exp(-\infty)=0$, so future weights are exactly zero and each row still sums to 1 over positions $\le i$. Because every layer is masked and the FFN is position-wise, the hidden state at position $i$ depends only on tokens $1..i$ in **every** layer.

Consequences:
- **Parallel training (teacher forcing):** one forward pass over $n$ tokens yields $n$ predictions $p(x_{t+1}\mid x_{\le t})$, and the loss is the sum of their cross-entropies. A sequence of length $n$ gives $n$ training signals at the cost of one pass.
- **Train/inference consistency:** computing position $t$ in a full pass gives the same result as generating step-by-step. That invariance is what makes the **KV cache** valid: past keys and values never change when a new token arrives.
- **Breaks permutation symmetry:** a causal model can infer position from how many tokens it can see, which is why decoder-only models can work even without explicit positional encodings.

Implementation notes:
- Built with `torch.triu(ones, diagonal=1)` (positions to hide) or a boolean lower-triangular "keep" mask; fused kernels take `is_causal=True`.
- Combined with padding masks by logical AND. A row that is fully masked yields NaN ($0/0$); implementations guard against it or use a large finite negative.
- Variants: **prefix-LM** masks (bidirectional over a prompt prefix, causal afterwards), **sliding-window** masks (attend only to the last $w$ tokens), block masks for packed sequences.

Do not confuse it with **masked language modeling** (BERT): there, *input tokens* are replaced by `[MASK]` and attention stays bidirectional. Causal masking hides *attention connections*, not tokens. In an encoder–decoder, only the decoder's self-attention is causal; encoder self-attention and cross-attention are not.

## Code
```python
import torch
import torch.nn.functional as F

n, d = 5, 16
q, k, v = torch.randn(3, 1, n, d).unbind(0)          # (1, n, d) each

future = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
scores = q @ k.transpose(-2, -1) / d**0.5
scores = scores.masked_fill(future, float("-inf"))   # hide j > i
A = scores.softmax(-1)
print(A[0].round(decimals=2))                        # lower-triangular weights

out = A @ v
ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
print(torch.allclose(out, ref, atol=1e-5))           # True

# Changing a future token does not change earlier outputs
v2 = v.clone(); v2[:, -1] += 100.0
print(torch.allclose((A @ v2)[:, :-1], out[:, :-1]))  # True
```

## Why it matters
Causal masking is what makes decoder-only transformers (GPT, Llama and most current LLMs) trainable efficiently: RNN language models also learned next-token prediction, but had to process positions sequentially. With the mask, the transformer gets full GPU parallelism during training and a consistent, cacheable computation during generation. It is the single architectural difference between a BERT-style encoder stack and a GPT-style decoder stack.

## Self-test
Q: What value does the causal mask put at position $(i, j)$ with $j > i$?
A: $-\infty$ (added to the logit), so the attention weight after softmax is exactly 0.

Q: How many next-token predictions does one masked forward pass over $n$ tokens give?
A: $n$ — one per position, all computed in parallel.

Q: How does causal masking differ from BERT's masked language modeling?
A: Causal masking removes attention to future positions; MLM replaces input tokens with `[MASK]` while attention stays bidirectional.

Q: Why does causal masking make a KV cache possible?
A: Earlier positions never depend on later tokens, so their keys and values stay fixed as new tokens are appended.

Q: In an encoder–decoder transformer, which attention is causally masked?
A: Only the decoder's self-attention.
