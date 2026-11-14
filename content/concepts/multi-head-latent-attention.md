---
id: multi-head-latent-attention
title: Multi-Head Latent Attention (MLA)
category: innovations
level: 3
year: 2024
summary: >-
  DeepSeek's attention variant that caches one small compressed latent vector
  per token instead of full per-head keys and values, cutting KV-cache memory
  far below GQA while keeping full multi-head expressiveness.
requires: [kv-cache, rotary-position-embedding]
links:
  - { to: multi-head-attention, type: variant-of }
  - { to: grouped-query-attention, type: alternative-to }
  - { to: mixture-of-experts, type: related }
  - { to: lora, type: related }
sources:
  - { title: "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model (DeepSeek-AI, 2024)", url: "https://arxiv.org/abs/2405.04434" }
  - { title: "DeepSeek-V3 Technical Report (DeepSeek-AI, 2024)", url: "https://arxiv.org/abs/2412.19437" }
---

## Intuition
GQA shrinks the KV cache by making heads share keys and values, which throws
away some per-head diversity. MLA instead notices that all the heads' keys and
values for a token are highly redundant, so it stores a short "summary" vector
per token and reconstructs every head's keys and values from it when needed.

It is like storing a compressed zip instead of many full copies of a file. The
trick that makes it cheap is algebraic: the "unzip" matrix can be folded into
the query and output projections, so at inference you never actually unzip.

## Mechanics
For hidden state $h_t \in \mathbb{R}^d$, with $n_h$ heads of size $d_h$:

**Low-rank joint KV compression:**
$$c_t^{KV} = W^{DKV} h_t \in \mathbb{R}^{d_c}, \quad k_t^{C} = W^{UK} c_t^{KV}, \quad v_t^{C} = W^{UV} c_t^{KV}$$
with $d_c \ll n_h d_h$. **Only $c_t^{KV}$ is cached.** (Queries are also
low-rank compressed during training, to save activation memory.)

**Weight absorption:** the score $q^\top k^C = q^\top W^{UK} c$ lets you
pre-multiply $W^{UK}$ into the query projection, and $W^{UV}$ can be merged
into the output projection. Attention then runs directly against the cached
latents.

**Decoupled RoPE:** RoPE applies a position-dependent rotation between $q$
and $k$, which would sit *between* $W^Q$ and $W^{UK}$ and block absorption. So
MLA splits each head's query/key into:
- a content part from the latent (no RoPE, absorbable), and
- a small RoPE part: per-head $q_t^R$ and ONE key $k_t^R = \mathrm{RoPE}(W^{KR} h_t)
  \in \mathbb{R}^{d_h^R}$ **shared by all heads** and cached alongside $c_t^{KV}$.

$q_{t,i} = [q_{t,i}^C; q_{t,i}^R]$, $k_{t,i} = [k_{t,i}^C; k_t^R]$.

**Cache size** per token per layer: $d_c + d_h^R$ elements, versus $2 n_h d_h$
for MHA and $2 G d_h$ for GQA. In DeepSeek-V2, $d_c = 4 d_h = 512$ and
$d_h^R = d_h/2 = 64$, i.e. **576** elements — the paper describes this as equal
to GQA with only 2.25 groups, while reporting performance stronger than MHA.
DeepSeek-V2 reports a **93.3%** smaller KV cache than DeepSeek 67B (a dense MHA
model).

Trade-off: more matrix multiplications per step (reconstruction or absorbed
projections), which is acceptable because decoding is memory-bound, not
compute-bound.

## Code
```python
import torch, torch.nn as nn

class MLAKV(nn.Module):
    """Illustrative: what MLA caches and how K/V are recovered (absorption omitted)."""
    def __init__(self, d=5120, n_h=128, d_h=128, d_c=512, d_r=64):
        super().__init__()
        self.n_h, self.d_h = n_h, d_h
        self.W_dkv = nn.Linear(d, d_c, bias=False)          # down-projection -> latent
        self.W_uk  = nn.Linear(d_c, n_h * d_h, bias=False)  # up-projection to keys
        self.W_uv  = nn.Linear(d_c, n_h * d_h, bias=False)  # up-projection to values
        self.W_kr  = nn.Linear(d, d_r, bias=False)          # shared RoPE key

    def cache_entry(self, h, rope):                         # h: (T, d)
        c_kv = self.W_dkv(h)                                # (T, 512)  cached
        k_r  = rope(self.W_kr(h))                           # (T, 64)   cached
        return c_kv, k_r                                    # 576 values/token/layer

    def expand(self, c_kv, k_r):
        T = c_kv.size(0)
        k_c = self.W_uk(c_kv).view(T, self.n_h, self.d_h)
        v   = self.W_uv(c_kv).view(T, self.n_h, self.d_h)
        k   = torch.cat([k_c, k_r[:, None].expand(T, self.n_h, -1)], -1)
        return k, v                                         # per-head keys/values
```

## Why it matters
MLA is one of the two pillars (with fine-grained MoE) behind DeepSeek-V2 and
V3's low serving cost; it makes long contexts and large batches affordable at a
cache size far below what GQA reaches at similar quality. It has since been
adopted or studied beyond DeepSeek, including methods to convert GQA models
into MLA form.

## Self-test
Q: What does MLA store in the KV cache instead of per-head keys and values?
A: A low-rank compressed latent vector c^KV per token, plus a small shared RoPE key.

Q: What is "weight absorption" in MLA?
A: Folding the key up-projection into the query projection (and the value up-projection into the output projection) so attention runs directly on cached latents.

Q: Why does MLA need decoupled RoPE?
A: RoPE's position-dependent rotation would sit between the query and key up-projection and prevent absorption, so position is carried by a separate small RoPE query/key part.

Q: How many cached elements per token per layer does DeepSeek-V2's MLA use?
A: 576: a 512-dimensional latent plus a 64-dimensional RoPE key.

Q: What KV-cache reduction does DeepSeek-V2 report versus DeepSeek 67B?
A: 93.3%.
