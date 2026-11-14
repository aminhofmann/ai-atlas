---
id: llama
title: Llama (1, 2, 3)
category: architectures
level: 2
year: 2023
summary: >-
  Meta's family of openly released decoder-only language models whose recipe —
  pre-RMSNorm, SwiGLU, rotary positions and grouped-query attention — became
  the default template for open LLMs.
requires: [gpt, rotary-position-embedding, layer-normalization, feed-forward-network]
links:
  - { to: grouped-query-attention, type: uses }
  - { to: kv-cache, type: uses }
  - { to: scaling-laws, type: related }
  - { to: rlhf, type: uses }
  - { to: lora, type: related }
  - { to: mixture-of-experts, type: related }
sources:
  - { title: "LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023)", url: "https://arxiv.org/abs/2302.13971" }
  - { title: "Llama 2: Open Foundation and Fine-Tuned Chat Models (Touvron et al., 2023)", url: "https://arxiv.org/abs/2307.09288" }
  - { title: "The Llama 3 Herd of Models (Grattafiori et al., 2024)", url: "https://arxiv.org/abs/2407.21783" }
  - { title: "GLU Variants Improve Transformer (Shazeer, 2020)", url: "https://arxiv.org/abs/2002.05202" }
---

## Intuition
Llama is a GPT-style model with a handful of careful upgrades, trained on far
more tokens than "compute-optimal" would suggest so that a *small* model ends
up very good. The design philosophy: pay more at training time once, so that
everyone running the model pays less at inference forever.

Because the weights were released, Llama became the base for thousands of
fine-tunes and the template other open models copied.

## Mechanics
**Block (all versions):** decoder-only, pre-norm.
- **RMSNorm** instead of LayerNorm, applied to each sublayer's input:
  $\mathrm{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \odot g$
  — no mean subtraction, no bias; cheaper and works as well.
- **SwiGLU FFN:** $\mathrm{FFN}(x) = W_2\big(\mathrm{SiLU}(W_1 x) \odot W_3 x\big)$.
  Three matrices instead of two, so the hidden size is reduced to about
  $\tfrac{2}{3}\cdot 4d$ to keep parameters comparable.
- **RoPE** (rotary embeddings) on queries and keys in every layer instead of
  absolute position embeddings; relative position enters the $q \cdot k$ dot
  product as a rotation.
- No bias terms; untied input and output embeddings in the main models.

**Llama 1 (Feb 2023):** 7B, 13B, 33B, 65B; 2048 context; SentencePiece BPE,
32k vocab; trained only on public data — 1.0T tokens (7B, 13B), 1.4T (33B,
65B). LLaMA-13B outperformed GPT-3 (175B) on most reported benchmarks.

**Llama 2 (Jul 2023):** 7B, 13B, 70B; context 4096; 2T tokens. **GQA** in the
70B model (fewer KV heads than query heads, shrinking the KV cache). Released
with **Llama 2-Chat**, aligned by SFT + RLHF (rejection sampling + PPO, separate
helpfulness and safety reward models). Commercial-use licence.

**Llama 3 (Apr 2024) / 3.1 (Jul 2024):** 8B and 70B, then 405B in 3.1. GQA in
all sizes (8 KV heads); a **128K-token vocabulary** (tiktoken-based) for better
compression; over 15T training tokens; context 8K, extended to 128K in 3.1 via
continued pretraining on long sequences. The 405B model is dense, not MoE.

What Llama is NOT: architecturally novel. Each component predates it; the
contribution is the combination, the data/training scale and the open weights.

## Code
```python
import torch, torch.nn as nn, torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__(); self.g = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.g

class SwiGLU(nn.Module):
    def __init__(self, d, hidden):                     # hidden ≈ 2/3 * 4d, rounded
        super().__init__()
        self.w1 = nn.Linear(d, hidden, bias=False)     # gate
        self.w3 = nn.Linear(d, hidden, bias=False)     # up
        self.w2 = nn.Linear(hidden, d, bias=False)     # down
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class LlamaBlock(nn.Module):
    def __init__(self, d, attn, hidden):               # attn: causal GQA with RoPE on q, k
        super().__init__()
        self.n1, self.n2, self.attn, self.ffn = RMSNorm(d), RMSNorm(d), attn, SwiGLU(d, hidden)
    def forward(self, x):
        x = x + self.attn(self.n1(x))                  # pre-norm residual
        return x + self.ffn(self.n2(x))
```

## Why it matters
Llama shifted the open-model landscape: strong models anyone could download,
study and fine-tune (often with LoRA). "Llama architecture" — RMSNorm, SwiGLU,
RoPE, GQA — is now the shared baseline that Mistral, Qwen and many others
start from, and the Llama papers are detailed references for data, scaling and
post-training choices. Its over-training strategy is the practical counterpoint
to Chinchilla-optimal sizing.

## Self-test
Q: What four architectural components define the Llama block relative to GPT-3?
A: Pre-RMSNorm, SwiGLU feed-forward layers, rotary position embeddings, and (from Llama 2 70B / Llama 3) grouped-query attention.

Q: How does RMSNorm differ from LayerNorm?
A: It rescales by the root-mean-square only, with no mean subtraction and no bias.

Q: Why is the SwiGLU hidden size about 2/3 of 4d?
A: SwiGLU uses three weight matrices instead of two, so shrinking the hidden size keeps the parameter count comparable.

Q: Which Llama 2 model used grouped-query attention?
A: The 70B model.

Q: Roughly how many tokens was Llama 3 trained on, and what vocabulary size does it use?
A: Over 15 trillion tokens, with a 128K-token vocabulary.
