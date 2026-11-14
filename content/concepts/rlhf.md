---
id: rlhf
title: RLHF (Reinforcement Learning from Human Feedback)
category: training
level: 3
year: 2017
summary: >-
  Aligns a language model with human preferences by training a reward model on
  human comparisons and then optimising the model against that reward.
requires: [pretraining-and-fine-tuning, autoregressive-training]
links:
  - { to: gpt, type: applied-in }
  - { to: llama, type: applied-in }
sources:
  - { title: "Deep Reinforcement Learning from Human Preferences (Christiano et al., 2017)", url: "https://arxiv.org/abs/1706.03741" }
  - { title: "Learning to Summarize from Human Feedback (Stiennon et al., 2020)", url: "https://arxiv.org/abs/2009.01325" }
  - { title: "Training Language Models to Follow Instructions with Human Feedback — InstructGPT (Ouyang et al., 2022)", url: "https://arxiv.org/abs/2203.02155" }
  - { title: "Direct Preference Optimization (Rafailov et al., 2023)", url: "https://arxiv.org/abs/2305.18290" }
---

## Intuition
A pretrained model imitates internet text, which is not the same as being
helpful, honest or harmless. Writing down "good answer" as a formula is hard,
but people can easily say which of two answers is better. RLHF turns those
comparisons into a learned judge (the reward model), then lets the language
model practise against the judge.

A leash keeps the model close to where it started, so it doesn't find weird
outputs that fool the judge without being genuinely better.

## Mechanics
The InstructGPT pipeline has three stages:

1. **Supervised fine-tuning (SFT).** Fine-tune the pretrained model on
   human-written demonstrations, giving policy $\pi^{\text{SFT}}$.
2. **Reward model (RM).** Sample several responses per prompt, have labelers
   rank them, and train $r_\phi(x, y) \in \mathbb{R}$ with a Bradley–Terry
   pairwise loss on preferred $y_w$ vs dispreferred $y_l$:
   $$\mathcal{L}_{\text{RM}} = -\mathbb{E}\big[\log \sigma\big(r_\phi(x,y_w) - r_\phi(x,y_l)\big)\big]$$
3. **RL optimisation.** Optimise the policy with PPO to maximise reward minus a
   KL penalty against the reference (SFT) model:
   $$\max_{\pi_\theta}\; \mathbb{E}_{x,\, y\sim\pi_\theta}\big[r_\phi(x,y)\big] - \beta\, \mathrm{KL}\big(\pi_\theta(\cdot\mid x)\,\|\,\pi^{\text{SFT}}(\cdot\mid x)\big)$$
   The whole response is one "action sequence"; the reward arrives at the end,
   the KL term per token.

Why the KL term: the RM is only accurate near the data it was trained on.
Without it, the policy **reward-hacks** — exploits RM errors (e.g. verbosity,
flattery) — and also loses fluency.

**Variants and successors:**
- **DPO** shows the KL-regularised objective has a closed-form optimal policy,
  so you can skip the explicit RM and PPO and train directly on preference
  pairs with a classification-style loss involving log-probability ratios
  against a reference model.
- **RLAIF / Constitutional AI**: preferences come from an AI model guided by
  written principles instead of (only) humans.
- **RL with verifiable rewards**: for maths and code, the reward is a checker
  (tests pass, answer correct) rather than a learned preference model.

## Code
```python
import torch.nn.functional as F

def reward_model_loss(rm, prompt, chosen, rejected):
    # Bradley-Terry: preferred response should score higher
    return -F.logsigmoid(rm(prompt, chosen) - rm(prompt, rejected)).mean()

def dpo_loss(policy_logp_w, policy_logp_l, ref_logp_w, ref_logp_l, beta=0.1):
    # inputs: summed log-probs of chosen (w) / rejected (l) responses
    margin = (policy_logp_w - ref_logp_w) - (policy_logp_l - ref_logp_l)
    return -F.logsigmoid(beta * margin).mean()

def rlhf_reward(rm_score, logp_policy, logp_ref, beta=0.02):
    # per-sequence reward used by PPO: RM score minus KL penalty to the reference
    return rm_score - beta * (logp_policy - logp_ref)
```

## Why it matters
RLHF is what turned GPT-3-style text predictors into assistants. In InstructGPT,
labelers preferred outputs of the 1.3B RLHF model over the 175B GPT-3, a
striking demonstration that alignment fine-tuning can matter more than scale
for perceived quality. ChatGPT and most chat models since use some form of
preference optimisation. Known weaknesses: reward hacking, labeler
disagreement and bias, and "sycophancy" (telling users what they want to hear).

## Self-test
Q: What are the three stages of the InstructGPT RLHF pipeline?
A: Supervised fine-tuning on demonstrations, training a reward model on human rankings, and optimising the policy with PPO against the reward model.

Q: What loss trains the reward model?
A: A pairwise Bradley–Terry loss: −log σ(r(x, y_preferred) − r(x, y_rejected)).

Q: Why is there a KL penalty to the SFT model during RL?
A: To stop the policy drifting into regions where the reward model is unreliable (reward hacking) and to preserve fluency.

Q: What does DPO remove from the pipeline?
A: The explicit reward model and the RL loop; it trains directly on preference pairs using log-probability ratios against a reference model.
