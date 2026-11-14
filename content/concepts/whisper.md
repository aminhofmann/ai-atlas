---
id: whisper
title: Whisper (Speech Recognition)
category: domains
level: 2
year: 2022
summary: >-
  An encoder–decoder transformer trained on 680,000 hours of weakly-supervised
  web audio to transcribe and translate speech, steered by special task tokens.
requires: [encoder-decoder, cross-attention, autoregressive-training]
links:
  - { to: transformer, type: based-on }
  - { to: positional-encoding, type: uses }
  - { to: tokenization, type: uses }
  - { to: causal-masking, type: uses }
  - { to: kv-cache, type: uses }
  - { to: scaling-laws, type: related }
  - { to: transformers-in-other-domains, type: part-of }
sources:
  - { title: "Robust Speech Recognition via Large-Scale Weak Supervision (Radford et al., 2022)", url: "https://arxiv.org/abs/2212.04356" }
---

## Intuition
Before Whisper, the best speech recognisers were typically pretrained on
unlabelled audio and then fine-tuned on a carefully transcribed benchmark —
great on that benchmark, brittle elsewhere. Whisper skips the fine-tuning:
OpenAI gathered a huge, messy pile of audio with its existing internet
transcripts and trained a plain sequence-to-sequence transformer on it.

The model listens with its encoder (audio → acoustic features) and writes with
its decoder (text, one token at a time), like a translator who hears a sentence
and writes it down. Special tokens at the start of the decoder's output tell it
*which* job to do: transcribe in this language, translate to English, add
timestamps or not. One model handles many tasks and languages, and is robust in
noisy, real-world audio "zero-shot".

## Mechanics
**Input.** Audio resampled to 16 kHz, cut into 30-second chunks (padded),
converted to an 80-channel log-Mel spectrogram (25 ms windows, 10 ms stride) —
about 3,000 frames.

**Encoder.** Two 1D convolutions (filter width 3, GELU; the second with stride 2)
downsample to 1,500 positions; add sinusoidal positional encodings; then
pre-norm transformer blocks with bidirectional self-attention.

**Decoder.** Learned positional embeddings, causal self-attention, and
cross-attention into the encoder output; byte-level BPE vocabulary (GPT-2's,
extended for multilingual use). Input and output token embeddings are tied.

**Multitask format.** The decoder is prompted with a token sequence such as

`<|startoftranscript|> <|de|> <|transcribe|> <|notimestamps|>` → text … `<|endoftext|>`

- language token (also used for language ID),
- `<|transcribe|>` or `<|translate|>` (to English),
- timestamp tokens interleaved with text when timestamps are on,
- optional previous-segment text as context for long audio.

**Training.** 680,000 hours of audio–transcript pairs from the web,
multilingual, with heuristics to remove machine-generated transcripts. Plain
next-token cross-entropy. Model sizes range from Tiny (39M) to Large (1.55B
parameters).

**Inference.** Long audio is processed in sliding 30 s windows; beam search with
temperature fallback; a KV cache speeds decoding.

## Code
```python
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

proc = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").eval()

audio, sr = librosa.load("speech.wav", sr=16_000)           # 16 kHz mono
feats = proc(audio, sampling_rate=sr, return_tensors="pt").input_features
print(feats.shape)                                           # (1, 80, 3000) log-Mel

# task tokens: language + transcribe/translate are forced at the start of decoding
with torch.no_grad():
    ids = model.generate(feats, language="de", task="transcribe")
print(proc.batch_decode(ids, skip_special_tokens=True)[0])

with torch.no_grad():
    ids = model.generate(feats, language="de", task="translate")  # -> English
print(proc.batch_decode(ids, skip_special_tokens=True)[0])
```

## Why it matters
Whisper showed that for speech, scale and diversity of *weakly* labelled data
beat clever self-supervised pretraining plus narrow fine-tuning, giving a model
that works robustly out of the box across accents, noise and ~100 languages.
Its open weights made it the default ASR component in countless applications and
the audio front end for many speech-enabled LLM pipelines. Architecturally it is
deliberately boring — the original encoder-decoder
transformer — which is the point.

## Self-test
Q: What is the input representation of Whisper's encoder?
A: An 80-channel log-Mel spectrogram of a 30-second, 16 kHz audio chunk, downsampled by two convolution layers.

Q: How does Whisper know whether to transcribe or translate?
A: Special task tokens at the start of the decoder sequence specify the language and `<|transcribe|>` or `<|translate|>`.

Q: What kind of supervision was Whisper trained with?
A: Weak supervision: ~680,000 hours of audio paired with transcripts found on the internet, without benchmark-specific fine-tuning.

Q: Which positional schemes do Whisper's encoder and decoder use?
A: Sinusoidal positional encodings in the encoder and learned positional embeddings in the decoder.
