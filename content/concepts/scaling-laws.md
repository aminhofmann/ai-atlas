---
id: scaling-laws
title: Scaling Laws
category: training
level: 2
year: 2020
summary: >-
  Language-model loss falls as a smooth power law in parameters, data and
  compute, which lets you predict big-model performance and choose the best
  model size for a compute budget.
requires: [autoregressive-training]
links:
  - { to: gpt, type: applied-in }
  - { to: llama, type: applied-in }
  - { to: mixture-of-experts, type: related }
  - { to: pretraining-and-fine-tuning, type: related }
sources:
  - { title: "Scaling Laws for Neural Language Models (Kaplan et al., 2020)", url: "https://arxiv.org/abs/2001.08361" }
  - { title: "Training Compute-Optimal Large Language Models — Chinchilla (Hoffmann et al., 2022)", url: "https://arxiv.org/abs/2203.15556" }
---

## Intuition
If you plot a language model's test loss against its size on log-log axes, you
get an almost straight line across many orders of magnitude. The same holds for
dataset size and for training compute. That regularity means you can run cheap
small experiments and extrapolate to a model costing a hundred times more
before you spend the money.

It also answers a budgeting question: given a fixed amount of compute, should
you build a bigger model or train a smaller one on more data? The answer
changed in 2022, and it changed how open models are built.

## Mechanics
Let $N$ = non-embedding parameters, $D$ = training tokens, $C$ = compute. For a
dense transformer, training compute is approximately

$$C \approx 6ND \text{ FLOPs}$$

(≈2 FLOPs per parameter per token forward, ≈4 backward).

**Kaplan et al. (2020).** When the other factors are not bottlenecks, loss
follows power laws such as $L(N) = (N_c/N)^{\alpha_N}$ with $\alpha_N \approx 0.076$
and $L(D) = (D_c/D)^{\alpha_D}$ with $\alpha_D \approx 0.095$. Architecture details
(depth vs width) matter little at fixed $N$. Their compute-optimal fit
said to grow model size much faster than data ($N_{\text{opt}} \propto C^{0.73}$),
which encouraged very large, relatively under-trained models.

**Chinchilla (Hoffmann et al., 2022).** With learning-rate schedules matched to
each run's length, they fit

$$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}, \qquad \alpha \approx 0.34,\; \beta \approx 0.28$$

where $E$ is the irreducible loss of the data. Minimising under $C = 6ND$
gives $N_{\text{opt}} \propto C^{a}$, $D_{\text{opt}} \propto C^{b}$ with
$a \approx b \approx 0.5$: **scale parameters and tokens equally**, about
**20 tokens per parameter**. Chinchilla (70B params, 1.4T tokens) outperformed
Gopher (280B) at the same compute.

**Caveats worth knowing:**
- Compute-optimal ≠ deployment-optimal. Inference cost scales with $N$, so
  models like Llama are deliberately trained far past 20 tokens/parameter.
- Exponents depend on data, tokenizer and architecture; they are empirical fits,
  not laws of nature.
- Smooth loss does not imply smooth downstream metrics; some task scores jump
  at certain scales.
- MoE models break $C \approx 6ND$ in its simple form: compute tracks *active*
  parameters, not total.

## Code
```python
# Chinchilla-style compute-optimal split, using the rule of thumb D ≈ 20 N and C ≈ 6 N D.
def chinchilla_optimal(compute_flops, tokens_per_param=20):
    # C = 6 * N * (k * N)  =>  N = sqrt(C / (6k))
    n = (compute_flops / (6 * tokens_per_param)) ** 0.5
    return n, tokens_per_param * n

def chinchilla_loss(N, D, E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28):
    return E + A / N**alpha + B / D**beta     # parametric fit from Hoffmann et al.

C = 6 * 70e9 * 1.4e12                         # Chinchilla's own budget
N, D = chinchilla_optimal(C)
print(f"N ≈ {N/1e9:.0f}B params, D ≈ {D/1e12:.1f}T tokens, "
      f"predicted loss ≈ {chinchilla_loss(N, D):.2f}")
```

## Why it matters
Scaling laws turned model building into an engineering forecast: labs decide
model size, data needs and hardware from small-scale fits. Kaplan's results
helped justify GPT-3; Chinchilla's correction shifted the field toward more
tokens per parameter and is why recent open models are trained on many
trillions of tokens. Scaling-law fits are now also used for data mixtures,
MoE, and downstream capabilities.

## Self-test
Q: What is the approximate training compute of a dense transformer with N parameters on D tokens?
A: About 6·N·D FLOPs.

Q: What is Chinchilla's rule of thumb for compute-optimal training?
A: Scale parameters and tokens in equal proportion, roughly 20 training tokens per parameter.

Q: What did Chinchilla show against Gopher?
A: A 70B model trained on 1.4T tokens beat the 280B Gopher at the same compute, showing large models had been under-trained.

Q: Why are deployed models often trained well beyond the compute-optimal token count?
A: Inference cost grows with parameter count, so a smaller model trained on more tokens is cheaper to serve at the same quality.
