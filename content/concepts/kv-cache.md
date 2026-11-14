---
id: kv-cache
title: KV Cache
category: foundations
level: 2
summary: >-
  Stores the keys and values of already-processed tokens during generation, so
  each new token only computes its own and attends to the stored ones instead of
  recomputing the whole prefix.
requires: [causal-masking, multi-head-attention]
links:
  - { to: autoregressive-training, type: related }
  - { to: grouped-query-attention, type: related }
  - { to: multi-head-latent-attention, type: related }
  - { to: flash-attention, type: related }
  - { to: gpt, type: applied-in }
sources:
  - { title: "Efficiently Scaling Transformer Inference (Pope et al., 2022)", url: "https://arxiv.org/abs/2211.05102" }
  - { title: "Fast Transformer Decoding: One Write-Head is All You Need (Shazeer, 2019)", url: "https://arxiv.org/abs/1911.02150" }
  - { title: "Efficient Memory Management for Large Language Model Serving with PagedAttention (Kwon et al., 2023)", url: "https://arxiv.org/abs/2309.06180" }
---

## Intuition
When you write a story one word at a time, you don't re-read and re-analyse the whole story from scratch before every word — you remember your notes about what you've written. A naive transformer does re-analyse everything at every step.

The KV cache is the model's notepad: for every token already processed, each attention layer keeps that token's key ("how to find me") and value ("what I contribute"). A new token just adds its own entry and looks up the notepad. This makes generation dramatically faster — at the cost of memory that grows with every token.

## Mechanics
At decoding step $t$, layer $\ell$ and head $h$ need $q_t$ and all past $k_{1..t}$, $v_{1..t}$:
$$o_t = \text{softmax}\!\left(\frac{q_t\,[k_1;\dots;k_t]^\top}{\sqrt{d_k}}\right)[v_1;\dots;v_t].$$
**Why caching is valid:** with causal masking, the hidden state at position $j<t$ — and hence $k_j, v_j$ in every layer — never depends on later tokens. So they can be computed once and reused.

- **Without cache:** each step re-runs the full prefix through all layers, so generating $n$ tokens costs on the order of $n^2$ token-forward passes.
- **With cache:** each step processes one token: compute $q_t, k_t, v_t$, append $k_t, v_t$ to the cache, attend with a $1\times t$ score row. Projection/FFN work per step is constant; attention work grows linearly with $t$.

**Two phases of LLM inference:**
- **Prefill:** process the prompt in one parallel pass, filling the cache — compute-bound.
- **Decode:** one token per step — typically **memory-bandwidth-bound**, since every step must read all weights plus the entire cache for little arithmetic.

**Memory:** per sequence,
$$\text{bytes} = 2 \cdot L \cdot n_{kv} \cdot d_{\text{head}} \cdot T \cdot \text{bytes/elem}$$
(factor 2 for K and V; $L$ layers, $n_{kv}$ KV heads, $T$ tokens). Example: $L=32$, $n_{kv}=32$, $d_{\text{head}}=128$ in fp16 gives 512 KiB per token, i.e. 2 GiB for a 4,096-token sequence — multiplied by batch size. The cache, not the weights, often limits batch size and context length.

**Reducing it:**
- **MQA / GQA:** fewer KV heads ($n_{kv} < n_q$) shared by many query heads.
- **MLA:** cache a compressed low-rank latent instead of full K/V.
- **Quantized caches** (8-bit or lower), **sliding-window attention** (keep only the last $w$ tokens), token eviction.
- **PagedAttention** (vLLM): store the cache in fixed-size blocks like virtual-memory pages, removing fragmentation and allowing prefix sharing across requests.

Details: with RoPE, keys are cached already rotated. In encoder–decoder models, cross-attention K/V are computed once from the encoder output and cached separately. FlashAttention speeds up the attention computation but does not shrink the cache.

## Code
```python
import torch
import torch.nn.functional as F

d = 64
W_q, W_k, W_v = (torch.randn(d, d) / d**0.5 for _ in range(3))

def decode_step(x_t, cache):
    """x_t: (1, d) hidden state of the newest token; cache holds past K, V."""
    q, k, v = x_t @ W_q, x_t @ W_k, x_t @ W_v
    cache["k"] = k if cache["k"] is None else torch.cat([cache["k"], k])
    cache["v"] = v if cache["v"] is None else torch.cat([cache["v"], v])
    scores = q @ cache["k"].T / d**0.5          # (1, t) — no mask needed:
    return F.softmax(scores, -1) @ cache["v"]   # the cache only holds the past

xs = torch.randn(10, d)
cache = {"k": None, "v": None}
incremental = torch.cat([decode_step(xs[t:t + 1], cache) for t in range(10)])

# Identical to full causal attention over all 10 tokens at once
Q, K, V = xs @ W_q, xs @ W_k, xs @ W_v
future = torch.triu(torch.ones(10, 10, dtype=torch.bool), 1)
full = F.softmax((Q @ K.T / d**0.5).masked_fill(future, float("-inf")), -1) @ V
print(torch.allclose(incremental, full, atol=1e-5))   # True
print(cache["k"].shape)                               # (10, 64): grows per token
```
(Single layer, single head; a real model keeps one cache per layer and KV head.)

## Why it matters
Practically every autoregressive transformer deployment uses a KV cache; without it, generation cost would grow quadratically in output length. It also reshaped architecture design: GQA, MQA and MLA exist largely to shrink the cache, serving systems like vLLM are built around managing it, and long-context inference is often limited more by cache memory than by compute.

## Self-test
Q: Why is it correct to reuse cached keys and values of earlier tokens?
A: Because of causal masking, earlier positions' hidden states — and thus their K and V — never change when later tokens are added.

Q: Why are queries not cached?
A: Each query is used only once, at its own step; only past keys and values are needed again.

Q: Give the formula for KV-cache size per sequence.
A: $2 \cdot L \cdot n_{kv} \cdot d_{\text{head}} \cdot T \cdot$ bytes per element.

Q: Why is the decode phase usually memory-bandwidth-bound?
A: Each step does little arithmetic for one token but must read all model weights and the whole KV cache from memory.

Q: Name two architectural changes that shrink the KV cache.
A: Grouped-/multi-query attention (fewer KV heads) and multi-head latent attention (compressed latent KV).
