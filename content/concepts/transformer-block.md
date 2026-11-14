---
id: transformer-block
title: Transformer Block
category: foundations
level: 2
year: 2017
summary: >-
  The repeated layer of a transformer: an attention sublayer that mixes
  information across tokens and a feed-forward sublayer that processes each
  token, each wrapped in a residual connection and normalization.
requires: [multi-head-attention, feed-forward-network, layer-normalization, residual-connections]
links:
  - { to: transformer, type: part-of }
  - { to: encoder-decoder, type: related }
  - { to: mixture-of-experts, type: related }
  - { to: llama, type: applied-in }
sources:
  - { title: "Attention Is All You Need (Vaswani et al., 2017)", url: "https://arxiv.org/abs/1706.03762" }
  - { title: "On Layer Normalization in the Transformer Architecture (Xiong et al., 2020)", url: "https://arxiv.org/abs/2002.04745" }
  - { title: "Language Models are Unsupervised Multitask Learners (Radford et al., 2019) — GPT-2, pre-norm", url: "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf" }
---

## Intuition
A transformer is a tall stack of identical floors. On each floor two things happen: first the tokens hold a meeting where everyone can listen to everyone (attention — *communication*), then each token goes back to its desk and thinks on its own (feed-forward — *computation*).

Each floor doesn't overwrite what came before; it adds notes to a shared running document (the residual stream). Normalization keeps the numbers in a sane range so dozens of floors can be stacked without the signal exploding or fading.

## Mechanics
Input and output have the same shape $X \in \mathbb{R}^{n\times d_{\text{model}}}$, which is what allows stacking $L$ blocks.

**Pre-norm block** (GPT-2 onward, Llama, most modern models):
$$H = X + \text{MHA}(\text{LN}(X)),\qquad Y = H + \text{FFN}(\text{LN}(H)).$$
**Post-norm block** (original 2017 Transformer, BERT):
$$H = \text{LN}(X + \text{MHA}(X)),\qquad Y = \text{LN}(H + \text{FFN}(H)).$$

Pre-norm vs post-norm:
- Pre-norm leaves an un-normalized identity path from input to output, so gradients flow well at initialization; Xiong et al. (2020) showed post-norm has large gradients near the output at init and needs learning-rate warm-up, while pre-norm can train without it.
- Pre-norm needs a **final LayerNorm** after the last block before the unembedding.
- Post-norm is sometimes reported to reach slightly better quality when it does train stably; pre-norm won mainly for robustness at depth.

**Decoder block in an encoder–decoder** adds a third sublayer: causal self-attention → cross-attention to the encoder → FFN, each residual + norm.

**Sizes:** original base: $L=6$ (per stack), $d_{\text{model}}=512$, $d_{ff}=2048$, $h=8$. Ignoring biases and norms, a block has about $4d^2$ attention parameters plus $8d^2$ FFN parameters (with $d_{ff}=4d$), i.e. $\approx 12d^2$ — two thirds in the FFN.

**Common modern recipe** ("Llama-style"): pre-norm with RMSNorm, RoPE inside attention, SwiGLU FFN, no biases, often grouped-query attention.

**Other variants:**
- **Parallel block** (GPT-J, PaLM): $Y = X + \text{MHA}(\text{LN}(X)) + \text{FFN}(\text{LN}(X))$ — both sublayers read the same input, which allows fusing their matrix multiplications.
- **MoE block:** the dense FFN is replaced by a routed mixture of expert FFNs.

Division of labour: attention is the only place information moves *between* positions; the FFN, LayerNorm and residual act on each position independently.

## Code
```python
import torch
import torch.nn as nn

class PreNormBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, causal_mask=None):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + self.drop(a)                          # communicate across tokens
        x = x + self.drop(self.ffn(self.ln2(x)))      # compute per token
        return x

n = 10
mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)  # True = blocked
blocks = nn.Sequential(*[PreNormBlock() for _ in range(6)])
x = torch.randn(2, n, 512)
for blk in blocks:
    x = blk(x, causal_mask=mask)
print(x.shape)   # (2, 10, 512) — shape preserved, so blocks stack
```

## Why it matters
The block is the unit of scale: model size is mostly chosen by picking $d_{\text{model}}$ and the number of blocks $L$, and nearly all architecture innovation (RMSNorm, SwiGLU, RoPE, GQA, MoE, parallel blocks) consists of swapping a component inside it. The move from post-norm to pre-norm was one of the key changes that made very deep transformers train reliably.

## Self-test
Q: Write the two equations of a pre-norm transformer block.
A: $H = X + \text{MHA}(\text{LN}(X))$ and $Y = H + \text{FFN}(\text{LN}(H))$.

Q: Why does pre-norm train more stably than post-norm?
A: It keeps a clean identity path through the residual stream, so gradients are well-behaved at initialization and warm-up is less critical.

Q: Which sublayer is the only one that mixes information between positions?
A: The attention sublayer; FFN, normalization and residuals act per position.

Q: Roughly how many parameters does a block have with $d_{ff}=4d$?
A: About $12d^2$: $4d^2$ in attention and $8d^2$ in the FFN.

Q: What extra sublayer does a decoder block in an encoder–decoder model have?
A: A cross-attention sublayer between the causal self-attention and the FFN.
