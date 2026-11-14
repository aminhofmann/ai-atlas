---
id: encoder-decoder
title: Encoder–Decoder Architecture
category: foundations
level: 2
year: 2014
summary: >-
  Splits a sequence-to-sequence model into an encoder that reads the whole input
  bidirectionally and a decoder that generates the output token by token while
  cross-attending to the encoding.
requires: [cross-attention, causal-masking, transformer-block]
links:
  - { to: transformer, type: part-of }
  - { to: t5, type: applied-in }
  - { to: whisper, type: applied-in }
  - { to: bert, type: related }
  - { to: gpt, type: related }
sources:
  - { title: "Sequence to Sequence Learning with Neural Networks (Sutskever et al., 2014)", url: "https://arxiv.org/abs/1409.3215" }
  - { title: "Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau et al., 2014)", url: "https://arxiv.org/abs/1409.0473" }
  - { title: "Attention Is All You Need (Vaswani et al., 2017)", url: "https://arxiv.org/abs/1706.03762" }
  - { title: "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (Raffel et al., 2019)", url: "https://arxiv.org/abs/1910.10683" }
---

## Intuition
An interpreter at a conference first listens to a full sentence, forming an understanding of it (encoding), and then speaks the translation word by word, glancing back at what they understood whenever needed (decoding).

The encoder–decoder architecture builds exactly these two roles into the model: one part reads the entire input with no restrictions, the other writes the output one token at a time, consulting the encoder's notes through cross-attention.

## Mechanics
Task: model $p(y \mid x)$ for a source $x = (x_1..x_m)$ and target $y = (y_1..y_n)$:
$$p(y\mid x) = \prod_{t=1}^{n} p(y_t \mid y_{<t},\, x).$$

**Encoder** — $N$ blocks of *bidirectional* (unmasked, except padding) self-attention + FFN over the source, producing a memory $Z \in \mathbb{R}^{m\times d}$.

**Decoder** — $N$ blocks, each with three sublayers (residual + norm around each):
1. **causal** self-attention over the target prefix $y_{<t}$,
2. **cross-attention**: queries from the decoder, keys/values from $Z$,
3. FFN.
A final linear layer + softmax gives the next-token distribution. The original Transformer used $N = 6$ for both stacks.

**Training:** teacher forcing — feed the gold target shifted right (starting with a start token), predict every next token in parallel thanks to the causal mask, minimize summed cross-entropy.

**Inference:** run the encoder **once**; then decode autoregressively (greedy, beam search, sampling). Each decoder layer caches its cross-attention $K, V$ (computed once from $Z$) plus a growing self-attention KV cache.

**History:** RNN seq2seq (Sutskever et al., 2014; Cho et al., 2014) compressed the input into a single fixed vector — a bottleneck for long sentences. Bahdanau attention let the decoder attend over all encoder states; the Transformer kept the encoder–decoder shape but replaced the RNNs with attention.

**The three transformer families:**

| family | attention | typical use | examples |
|---|---|---|---|
| encoder-only | bidirectional | representations, classification, embedding | BERT, ESM-2 |
| decoder-only | causal | open-ended generation, LLMs | GPT, Llama |
| encoder–decoder | bidir. encoder + causal decoder + cross-attn | translation, summarization, speech-to-text | original Transformer, T5, BART, Whisper |

Trade-offs: the encoder–decoder gives the input full bidirectional context and separates "understanding" from "generating"; decoder-only models treat the input as a prefix of one causal sequence, which is simpler and scales uniformly. In T5's controlled comparison, the encoder–decoder with a denoising objective performed best; at LLM scale, decoder-only has nonetheless become dominant.

## Code
```python
import torch
import torch.nn as nn

d, V = 256, 1000
emb = nn.Embedding(V, d)
model = nn.Transformer(d_model=d, nhead=4, num_encoder_layers=3,
                       num_decoder_layers=3, dim_feedforward=1024, batch_first=True)
lm_head = nn.Linear(d, V)

src = torch.randint(0, V, (2, 12))           # source sentence ids
tgt = torch.randint(0, V, (2, 8))            # target ids (gold, for teacher forcing)
tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]    # shift right: predict the next token

causal = nn.Transformer.generate_square_subsequent_mask(tgt_in.size(1))
memory = model.encoder(emb(src))             # (2, 12, d) — encode once
h = model.decoder(emb(tgt_in), memory, tgt_mask=causal)   # self-attn + cross-attn
logits = lm_head(h)                          # (2, 7, V)

loss = nn.functional.cross_entropy(logits.reshape(-1, V), tgt_out.reshape(-1))
print(memory.shape, logits.shape, loss.item())
```
(Positional encodings are omitted for brevity.)

## Why it matters
The encoder–decoder was the original Transformer and the standard for machine translation, summarization and speech recognition. Understanding it explains why BERT (encoder half) and GPT (decoder half) look the way they do. It remains the architecture of choice where a fixed input is mapped to an output — T5-style models, Whisper, and many scientific models that translate between modalities.

## Self-test
Q: What three sublayers does each decoder block contain?
A: Causal self-attention, cross-attention to the encoder output, and a feed-forward network.

Q: Is the encoder's self-attention masked causally?
A: No — it is bidirectional; only padding is masked.

Q: How often is the encoder run when generating an output sequence?
A: Once; the decoder then reuses its output (and the cached cross-attention keys/values) at every step.

Q: What bottleneck of RNN seq2seq did attention remove?
A: Compressing the whole input into one fixed-length vector.

Q: Name one encoder-only, one decoder-only and one encoder–decoder model.
A: BERT (encoder-only), GPT (decoder-only), T5 or Whisper (encoder–decoder).
