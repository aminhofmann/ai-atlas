---
id: mixture-of-experts
title: Mixture of Experts (MoE)
category: innovations
level: 2
year: 2017
summary: >-
  Replaces one big feed-forward layer with many "expert" FFNs and a router that
  sends each token to only a few of them, so total parameters grow while
  per-token compute stays small.
requires: [feed-forward-network, transformer-block]
links:
  - { to: scaling-laws, type: related }
  - { to: multi-head-latent-attention, type: related }
  - { to: hybrid-architectures, type: applied-in }
  - { to: llama, type: related }
sources:
  - { title: "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer (Shazeer et al., 2017)", url: "https://arxiv.org/abs/1701.06538" }
  - { title: "Switch Transformers (Fedus, Zoph & Shazeer, 2021)", url: "https://arxiv.org/abs/2101.03961" }
  - { title: "Mixtral of Experts (Jiang et al., 2024)", url: "https://arxiv.org/abs/2401.04088" }
  - { title: "DeepSeekMoE: Towards Ultimate Expert Specialization (Dai et al., 2024)", url: "https://arxiv.org/abs/2401.06066" }
  - { title: "DeepSeek-V3 Technical Report (DeepSeek-AI, 2024)", url: "https://arxiv.org/abs/2412.19437" }
---

## Intuition
A hospital doesn't make every patient see every doctor. A triage nurse looks at
each patient and sends them to one or two specialists. MoE does the same for
tokens: a small router looks at each token and sends it to the couple of
expert networks best suited to it.

The model can therefore "know" a lot (many experts, many parameters) while each
token only pays for the few experts it visits. The catch is logistics: if the
nurse sends everyone to the same doctor, that doctor is swamped and the others
sit idle.

## Mechanics
In a transformer block, the dense FFN is replaced by $E$ expert FFNs
$\{f_1,\dots,f_E\}$ and a router $W_r \in \mathbb{R}^{d \times E}$:

$$g(x) = \mathrm{softmax}(x W_r), \qquad y = \sum_{i \in \mathrm{TopK}(g(x),\,k)} \tilde g_i(x)\, f_i(x)$$

where $\tilde g$ are the gate values (renormalised over the selected $k$ in many
designs). Attention layers stay dense and shared.

- **Total vs active parameters.** Compute per token scales with $k$ experts, not
  $E$. **Mixtral 8×7B:** 8 experts per layer, top-2, ≈46.7B total, ≈12.9B active
  per token. **DeepSeek-V3:** ≈671B total, ≈37B active.
- **Top-k.** Shazeer 2017 used noisy top-k gating in LSTMs; GShard top-2;
  **Switch Transformer** showed top-1 works and simplifies everything.
- **Load balancing.** Without pressure, the router collapses onto a few experts.
  Switch adds an auxiliary loss $\mathcal{L}_{\text{aux}} = \alpha E \sum_{i=1}^{E} f_i P_i$,
  where $f_i$ is the fraction of tokens dispatched to expert $i$ and $P_i$ the
  mean router probability for $i$; it is minimised by a uniform distribution.
  **DeepSeek-V3** uses an *auxiliary-loss-free* scheme: a per-expert bias added
  to routing scores only for top-k selection, nudged up or down depending on
  whether the expert is under- or over-loaded.
- **Capacity.** Each expert processes at most
  $\text{capacity factor} \times \frac{\text{tokens}}{E}$ tokens per batch;
  overflow tokens are dropped (passed through the residual) in capacity-based
  implementations.
- **DeepSeekMoE.** (1) *Fine-grained segmentation*: split experts into many
  smaller ones and activate more of them, giving far more combinations;
  (2) *shared experts*: a few experts that every token always uses, to hold
  common knowledge so routed experts can specialise.
- **Systems.** Experts are spread across devices (expert parallelism), so
  routing requires all-to-all communication — often the real bottleneck. All
  experts must still sit in memory, so MoE saves FLOPs, not memory.

## Code
```python
import torch, torch.nn as nn, torch.nn.functional as F

class MoE(nn.Module):
    def __init__(self, d, hidden, n_experts=8, k=2):
        super().__init__()
        self.k, self.router = k, nn.Linear(d, n_experts, bias=False)
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(d, hidden), nn.SiLU(), nn.Linear(hidden, d))
            for _ in range(n_experts))

    def forward(self, x):                              # x: (tokens, d)
        probs = F.softmax(self.router(x), dim=-1)      # (tokens, E)
        topv, topi = probs.topk(self.k, dim=-1)
        topv = topv / topv.sum(-1, keepdim=True)       # renormalise over chosen k
        out = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            tok, slot = (topi == e).nonzero(as_tuple=True)
            if tok.numel():
                out[tok] += topv[tok, slot, None] * expert(x[tok])
        # Switch-style balance loss: E * sum_i f_i * P_i
        f = F.one_hot(topi[:, 0], probs.size(-1)).float().mean(0)
        aux = probs.size(-1) * (f * probs.mean(0)).sum()
        return out, aux
```

## Why it matters
MoE decouples model capacity from per-token compute, letting labs train much
larger models for the same FLOPs. Mixtral showed an open sparse model matching
or beating much larger dense ones, and DeepSeek-V2/V3 combined fine-grained
MoE with MLA for very cheap training and inference. Many frontier models are
believed to be MoE. Costs: memory for all experts, communication overhead,
and training instability around routing.

## Self-test
Q: In an MoE transformer, which sublayer is usually replaced by experts?
A: The feed-forward network; attention stays dense and shared.

Q: What are Mixtral 8×7B's expert count, top-k and approximate total and active parameters?
A: 8 experts, top-2 routing, about 46.7B total and 12.9B active parameters per token.

Q: Why is a load-balancing mechanism needed?
A: Without it the router tends to collapse onto a few experts, wasting capacity and overloading those devices.

Q: What two ideas define DeepSeekMoE?
A: Fine-grained expert segmentation (many smaller experts, more activated) and always-active shared experts.

Q: Does MoE reduce memory requirements compared with a dense model of the same total size?
A: No; all experts must be stored, MoE reduces compute per token, not parameter memory.
