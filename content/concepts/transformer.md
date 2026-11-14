---
id: transformer
title: The Transformer (2017)
category: architectures
level: 1
year: 2017
summary: >-
  The original encoder-decoder architecture built entirely from attention and
  feed-forward layers, introduced for machine translation and the ancestor of
  nearly every modern language model.
requires: [encoder-decoder, multi-head-attention, positional-encoding]
links:
  - { to: transformer-block, type: uses }
  - { to: self-attention, type: uses }
  - { to: cross-attention, type: uses }
  - { to: causal-masking, type: uses }
  - { to: feed-forward-network, type: uses }
  - { to: residual-connections, type: uses }
  - { to: layer-normalization, type: uses }
  - { to: embeddings, type: uses }
  - { to: state-space-models, type: related }
sources:
  - { title: "Attention Is All You Need (Vaswani et al., 2017)", url: "https://arxiv.org/abs/1706.03762" }
---

## Intuition
Before 2017, translation models read a sentence one word at a time with a
recurrent network, passing a running summary along like a game of telephone.
Long sentences lost information, and training was slow because step 50 had to
wait for step 49.

The Transformer dropped recurrence. Every word looks directly at every other
word through attention, all at once. One stack (the encoder) reads the source
sentence; a second stack (the decoder) writes the translation one token at a
time while consulting the encoder's output. Because nothing is sequential
during training, it parallelises beautifully on GPUs.

## Mechanics
**Encoder** (N = 6 layers). Each layer: multi-head self-attention → add &
LayerNorm → position-wise FFN → add & LayerNorm. The original is **post-norm**:
$x \leftarrow \mathrm{LN}(x + \mathrm{Sublayer}(x))$.

**Decoder** (N = 6 layers). Each layer has three sublayers:
1. **Masked** (causal) self-attention over the target prefix,
2. **Cross-attention**: queries from the decoder, keys/values from the final
   encoder output,
3. FFN — each with residual + LayerNorm.

**Attention:**
$$\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$
with $h = 8$ heads, $d_k = d_v = d_{\text{model}}/h = 64$.

**Base model hyperparameters:** $d_{\text{model}} = 512$, FFN inner size
$d_{ff} = 2048$ with ReLU, dropout 0.1, ≈65M parameters. "Big": $d_{\text{model}}=1024$,
$d_{ff}=4096$, 16 heads.

**Position:** fixed sinusoidal encodings added to the token embeddings:
$PE_{(pos,2i)} = \sin(pos/10000^{2i/d})$, $PE_{(pos,2i+1)} = \cos(pos/10000^{2i/d})$.

**Embeddings:** shared byte-pair vocabulary; input embeddings, output embeddings
and the pre-softmax projection share weights (embeddings scaled by $\sqrt{d_{\text{model}}}$).

**Training:** teacher-forced cross-entropy on the target, Adam with a warm-up
then inverse-square-root learning-rate schedule (4000 warm-up steps), label
smoothing $\epsilon = 0.1$. Inference uses beam search.

**Cost:** self-attention is $O(n^2 d)$ in sequence length $n$, versus $O(n d^2)$
for a recurrent layer — but with $O(1)$ sequential steps instead of $O(n)$.

Later models kept the block but changed details: pre-norm instead of post-norm,
learned or rotary positions, GELU/SwiGLU FFNs, and usually only ONE of the two
stacks (encoder-only BERT, decoder-only GPT).

## Code
```python
import torch, torch.nn as nn

# PyTorch ships the original encoder-decoder layout.
model = nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6,
                       num_decoder_layers=6, dim_feedforward=2048,
                       dropout=0.1, batch_first=True)   # post-norm by default

src = torch.randn(2, 20, 512)   # embedded + position-encoded source (B, S, d)
tgt = torch.randn(2, 15, 512)   # embedded target prefix, shifted right (B, T, d)
causal = nn.Transformer.generate_square_subsequent_mask(15)

out = model(src, tgt, tgt_mask=causal)   # (2, 15, 512) -> project to vocab logits
print(out.shape)
```

## Why it matters
On WMT 2014 the Transformer set new translation records (28.4 BLEU English→German,
big model) at a fraction of previous training cost. More importantly, its
block became the universal building unit: BERT took the encoder, GPT took the
decoder, T5 kept both, and the same block now runs vision, audio, protein and
genomics models. Its quadratic attention cost is the main weakness later work
attacks (FlashAttention, state-space models, hybrids).

## Self-test
Q: What three sublayers does each decoder layer of the original Transformer contain?
A: Masked self-attention, cross-attention over the encoder output, and a position-wise feed-forward network.

Q: What are the base model's d_model, number of heads and FFN size?
A: d_model = 512, 8 heads (d_k = 64) and d_ff = 2048.

Q: Where is LayerNorm placed in the original Transformer?
A: After each residual addition (post-norm): LN(x + Sublayer(x)).

Q: How does the original Transformer encode position?
A: Fixed sinusoidal positional encodings added to the token embeddings.

Q: Why does attention scale scores by 1/√d_k?
A: Dot products grow with dimension; scaling keeps the softmax from saturating and its gradients from vanishing.
