---
id: cross-attention
title: Cross-Attention
category: foundations
level: 2
year: 2017
summary: >-
  Attention where queries come from one sequence and keys and values from
  another, letting one stream of tokens read information from a second source.
requires: [self-attention]
links:
  - { to: encoder-decoder, type: part-of }
  - { to: multi-head-attention, type: related }
  - { to: t5, type: applied-in }
  - { to: whisper, type: applied-in }
  - { to: kv-cache, type: related }
sources:
  - { title: "Attention Is All You Need (Vaswani et al., 2017)", url: "https://arxiv.org/abs/1706.03762" }
  - { title: "Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau et al., 2014)", url: "https://arxiv.org/abs/1409.0473" }
  - { title: "High-Resolution Image Synthesis with Latent Diffusion Models (Rombach et al., 2021)", url: "https://arxiv.org/abs/2112.10752" }
  - { title: "Perceiver: General Perception with Iterative Attention (Jaegle et al., 2021)", url: "https://arxiv.org/abs/2103.03206" }
---

## Intuition
A translator writing a German sentence keeps glancing at the English original. The sentence being written asks the questions ("what should come next?"), and the original supplies the answers.

Cross-attention is exactly that glance: one sequence (the one being built or updated) sends queries, and a second sequence (the source — a sentence, an audio clip, an image, a text prompt) provides the keys and values. Information flows in one direction, from source to target.

## Mechanics
Target states $Y \in \mathbb{R}^{n\times d}$ and source states $Z \in \mathbb{R}^{m\times d_z}$:
$$Q = YW_Q,\quad K = ZW_K,\quad V = ZW_V,\qquad \text{CrossAttn}(Y,Z) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V \in \mathbb{R}^{n\times d_v}.$$
- $W_Q \in \mathbb{R}^{d\times d_k}$, $W_K \in \mathbb{R}^{d_z\times d_k}$, $W_V \in \mathbb{R}^{d_z\times d_v}$: the two sequences may have different widths and lengths.
- The attention matrix is **rectangular**, $n\times m$; the output has the target's length $n$. Shape follows the query side.
- Cost $O(n\,m\,d)$ — linear in each length separately.
- **No causal mask** on the source (it is fully known); only a padding mask for the source. In practice cross-attention is multi-headed, just like self-attention.

**In the encoder–decoder Transformer** each decoder block has three sublayers: causal self-attention → cross-attention to the final encoder output → FFN (each with residual + norm). Every decoder layer attends to the *same* encoder output $Z$, through its own projections. Because $Z$ is fixed during decoding, each layer's $ZW_K$, $ZW_V$ are computed once and reused for every generated token.

**Other uses:**
- **Conditioning generative models:** latent diffusion (Stable Diffusion) injects the text prompt by letting image features cross-attend to text-encoder embeddings.
- **Speech:** Whisper's text decoder cross-attends to the audio encoder.
- **Latent bottlenecks:** Perceiver lets a small learned latent array cross-attend to a huge input (pixels, audio), making cost linear in input size.
- **Multimodal fusion:** vision–language models that interleave cross-attention layers into a language model.

Self-attention is the special case $Z = Y$.

## Code
```python
import torch
import torch.nn as nn

d_model, n_heads = 256, 4
cross = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

decoder_states = torch.randn(2, 7, d_model)    # Y: target, n = 7 tokens so far
encoder_out = torch.randn(2, 20, d_model)      # Z: source, m = 20 tokens
src_padding = torch.zeros(2, 20, dtype=torch.bool)
src_padding[1, 15:] = True                     # 2nd source sentence is shorter

out, weights = cross(query=decoder_states,     # Q from the decoder
                     key=encoder_out,          # K, V from the encoder
                     value=encoder_out,
                     key_padding_mask=src_padding)
print(out.shape)        # (2, 7, 256) — length follows the query side
print(weights.shape)    # (2, 7, 20)  — rectangular n x m attention map
print(weights[1, :, 15:].abs().max())          # 0: padded source ignored
```

## Why it matters
Cross-attention is the original form of attention in neural machine translation, where it replaced the single fixed context vector with a per-step alignment over the source. Today it is the standard way to condition one modality on another — text on audio (Whisper), images on text (diffusion), or a compact latent on a large input (Perceiver). Decoder-only LLMs do without it by putting everything into one sequence, which is one of the key design differences between the architecture families.

## Self-test
Q: In cross-attention, where do Q, K and V come from?
A: Queries from the target sequence being updated; keys and values from the source sequence.

Q: What is the shape of the cross-attention matrix for $n$ target and $m$ source tokens?
A: $n \times m$, and the output has length $n$.

Q: Does cross-attention in an encoder–decoder use a causal mask?
A: No — the whole source is visible; only padding in the source is masked.

Q: Why can the encoder's K and V be cached for the whole decoding run?
A: The encoder output does not change while the decoder generates, so its projections are computed once per layer.

Q: How does Stable Diffusion use cross-attention?
A: Image features in the denoising network cross-attend to text-encoder embeddings to condition generation on the prompt.
