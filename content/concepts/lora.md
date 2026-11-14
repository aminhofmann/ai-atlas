---
id: lora
title: LoRA (Low-Rank Adaptation)
category: training
level: 2
year: 2021
summary: >-
  Fine-tunes a frozen model by learning a small low-rank update to selected
  weight matrices, cutting trainable parameters by orders of magnitude.
requires: [pretraining-and-fine-tuning, multi-head-attention]
links:
  - { to: transformer-block, type: applied-in }
  - { to: llama, type: applied-in }
  - { to: gpt, type: applied-in }
  - { to: multi-head-latent-attention, type: related }
sources:
  - { title: "LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)", url: "https://arxiv.org/abs/2106.09685" }
  - { title: "QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023)", url: "https://arxiv.org/abs/2305.14314" }
---

## Intuition
Full fine-tuning rewrites every one of a model's billions of numbers, and you
need a full copy per task. LoRA bets that the *change* needed for a new task is
simple, so it can be described by a few directions rather than a whole new
matrix. It's like adjusting a photo with a couple of sliders instead of
repainting every pixel.

You keep the big model frozen and train a tiny "patch" next to it. Patches are
megabytes, swappable, and can be merged into the weights so inference costs
nothing extra.

## Mechanics
For a frozen weight $W_0 \in \mathbb{R}^{d \times k}$, LoRA parameterises the
update as a product of two thin matrices:

$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} B A, \qquad B \in \mathbb{R}^{d \times r},\; A \in \mathbb{R}^{r \times k},\; r \ll \min(d,k)$$

The forward pass becomes $h = W_0 x + \frac{\alpha}{r} B(Ax)$.

- **Parameter count.** $r(d+k)$ instead of $dk$. For $d=k=4096$, $r=8$:
  65,536 vs 16.8M parameters per matrix (~0.4%).
- **Initialisation.** $A$ is random Gaussian, $B$ is **zero**, so $\Delta W = 0$
  at step 0 and training starts exactly from the pretrained model.
- **Scaling.** $\alpha/r$ makes the effective step size less sensitive to the
  choice of $r$; $\alpha$ is a fixed hyperparameter.
- **Where.** The paper adapts attention projections (it found $W_q$ and $W_v$
  together worked well, and very small ranks — even $r=1$ or $2$ — were
  competitive). Common practice now adapts all attention and MLP projections.
- **No inference latency.** After training, compute $W = W_0 + \frac{\alpha}{r}BA$
  once and serve a normal dense model; or keep adapters separate to hot-swap
  tasks on one base model.
- **Why low rank works.** The paper's hypothesis, supported by its analysis, is
  that fine-tuning updates have low "intrinsic rank".
- **Memory.** Optimizer state (Adam's two moments) is needed only for $A, B$.
  On GPT-3 175B the paper reports ~10,000× fewer trainable parameters and ~3×
  less GPU memory than full fine-tuning.
- **QLoRA** stores the frozen base in 4-bit (NF4) and backpropagates through it
  into bf16 LoRA adapters, enabling fine-tuning of 65B models on a single
  48 GB GPU.

## Code
```python
import torch, torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)          # freeze W0
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))  # zero => ΔW = 0 at start
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T) @ self.B.T

    @torch.no_grad()
    def merge(self):                                     # fold into W0 for zero-overhead inference
        self.base.weight += self.scale * self.B @ self.A
```

## Why it matters
LoRA made fine-tuning large models practical on modest hardware and turned
"one model per task" into "one base model plus many small adapters". It is the
default PEFT method in open-model ecosystems (Llama fine-tunes, diffusion model
styles), and QLoRA pushed it onto single consumer and workstation GPUs. Its
limitation: a low-rank update can underfit tasks that need large changes, such
as teaching substantial new knowledge.

## Self-test
Q: Write the LoRA weight update.
A: W = W0 + (α/r)·B·A, with B of shape d×r, A of shape r×k and r much smaller than d and k.

Q: Why is B initialised to zero?
A: So the update BA is zero at the start and training begins exactly from the pretrained model's behaviour.

Q: Why does LoRA add no inference latency?
A: The product BA can be merged into W0 after training, leaving a normal dense matrix.

Q: How many trainable parameters does a rank-r LoRA add to a d×k matrix?
A: r(d + k).

Q: What does QLoRA add on top of LoRA?
A: It keeps the frozen base model quantised to 4 bits (NF4) while training LoRA adapters in higher precision.
