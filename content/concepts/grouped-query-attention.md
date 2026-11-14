---
id: grouped-query-attention
title: Grouped-Query Attention (GQA) and Multi-Query Attention (MQA)
category: innovations
level: 2
year: 2023
summary: >-
  Lets several query heads share one key/value head, shrinking the KV cache and
  speeding up decoding with little loss in quality.
requires: [kv-cache]
links:
  - { to: multi-head-attention, type: variant-of }
  - { to: llama, type: applied-in }
  - { to: multi-head-latent-attention, type: related }
  - { to: flash-attention, type: related }
sources:
  - { title: "Fast Transformer Decoding: One Write-Head is All You Need — MQA (Shazeer, 2019)", url: "https://arxiv.org/abs/1911.02150" }
  - { title: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints (Ainslie et al., 2023)", url: "https://arxiv.org/abs/2305.13245" }
---

## Intuition
When a model generates text, it keeps the keys and values of every past token
for every head in memory (the KV cache). For long contexts and big batches
that cache becomes enormous, and reading it from memory at every step — not the
arithmetic — is what makes generation slow.

MQA's idea: let all query heads look at the *same* keys and values, so you only
store one set. GQA is the compromise: query heads are split into a few groups,
and each group shares one key/value head. Each query head can still ask a
different question; they just consult a smaller shared library.

## Mechanics
With $H$ query heads of size $d_h$, choose $G$ KV heads ($1 \le G \le H$, $H$
divisible by $G$):
- $Q \in \mathbb{R}^{n \times H \times d_h}$, $K, V \in \mathbb{R}^{n \times G \times d_h}$.
- Query head $i$ uses KV head $\lfloor i / (H/G) \rfloor$.
- $G = H$ → standard MHA; $G = 1$ → **MQA**; in between → **GQA**.

**KV-cache memory** per token, per layer, is $2 \cdot G \cdot d_h$ values
(K and V). Across the model:

$$\text{KV bytes} = 2 \cdot L \cdot G \cdot d_h \cdot n_{\text{ctx}} \cdot B \cdot \text{bytes/elem}$$

So the cache shrinks by a factor $H/G$. Example: Llama 2 70B has $H = 64$ query
heads and $G = 8$ KV heads — an 8× smaller cache than MHA with identical query
heads.

**Why this speeds decoding:** each new token must read the entire cache from
GPU memory. Autoregressive decoding is memory-bandwidth-bound, so fewer bytes
read ≈ faster tokens, and the freed memory allows larger batches or longer
contexts.

**Quality trade-off:** MQA can lose quality and was reported to be less stable
in training; GQA with a moderate $G$ recovers quality close to MHA at speed
close to MQA (Ainslie et al.).

**Uptraining:** an existing MHA checkpoint can be converted by **mean-pooling**
the K and V projection heads within each group, then continuing pretraining
for a small fraction (≈5%) of the original compute.

Compute (FLOPs) for the attention scores is essentially unchanged; the savings
are in K/V projection parameters, cache memory and memory traffic.

## Code
```python
import torch, torch.nn.functional as F

def gqa(x, Wq, Wk, Wv, n_heads, n_kv_heads):
    B, T, D = x.shape
    hd = D // n_heads
    q = (x @ Wq).view(B, T, n_heads,    hd).transpose(1, 2)   # (B, H, T, hd)
    k = (x @ Wk).view(B, T, n_kv_heads, hd).transpose(1, 2)   # (B, G, T, hd)  <- cached
    v = (x @ Wv).view(B, T, n_kv_heads, hd).transpose(1, 2)   # (B, G, T, hd)  <- cached
    rep = n_heads // n_kv_heads
    k = k.repeat_interleave(rep, dim=1)                         # share each KV head
    v = v.repeat_interleave(rep, dim=1)                         # across its query group
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return out.transpose(1, 2).reshape(B, T, D)

D, H, G = 512, 8, 2
x = torch.randn(1, 16, D)
Wq = torch.randn(D, D); Wk = torch.randn(D, G * D // H); Wv = torch.randn(D, G * D // H)
print(gqa(x, Wq, Wk, Wv, H, G).shape)
```

## Why it matters
GQA is now standard in open LLMs (Llama 2 70B, all Llama 3 sizes, Mistral and
many others) because serving cost is dominated by the KV cache at long context.
It is one point on a spectrum of KV-compression ideas; DeepSeek's multi-head
latent attention goes further by caching a low-rank latent instead of
per-head keys and values.

## Self-test
Q: What are MHA, MQA and GQA in terms of the number of KV heads G versus query heads H?
A: MHA has G = H, MQA has G = 1, GQA uses 1 < G < H with each KV head shared by H/G query heads.

Q: By what factor does GQA shrink the KV cache relative to MHA?
A: By H/G, the number of query heads per KV head.

Q: Why does a smaller KV cache speed up generation?
A: Decoding is memory-bandwidth-bound; each step must read the whole cache, so fewer bytes means faster steps and room for bigger batches.

Q: How can an MHA checkpoint be converted to GQA?
A: Mean-pool the key and value heads within each group, then uptrain for a small fraction (about 5%) of the original pretraining compute.
