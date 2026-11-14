---
id: decision-transformer
title: Decision Transformer
category: domains
level: 3
year: 2021
summary: >-
  Recasts offline reinforcement learning as sequence modelling: a GPT-style
  model reads (return-to-go, state, action) tokens and predicts the next action
  that achieves a requested return.
requires: [gpt, causal-masking, autoregressive-training]
links:
  - { to: embeddings, type: uses }
  - { to: positional-encoding, type: uses }
  - { to: transformers-in-other-domains, type: part-of }
sources:
  - { title: "Decision Transformer: Reinforcement Learning via Sequence Modeling (Chen et al., 2021)", url: "https://arxiv.org/abs/2106.01345" }
  - { title: "Offline Reinforcement Learning as One Big Sequence Modeling Problem (Trajectory Transformer, Janner et al., 2021)", url: "https://arxiv.org/abs/2106.02039" }
  - { title: "Online Decision Transformer (Zheng et al., 2022)", url: "https://arxiv.org/abs/2202.05607" }
---

## Intuition
Classic reinforcement learning learns *values* — how good is this state, how good
is this action — through Bellman updates, which are notoriously unstable when
learning only from a fixed log of past behaviour (offline RL). The Decision
Transformer sidesteps all that. It treats a recorded episode as a sentence:
"I wanted this much reward; I saw this; I did that; I now want this much
reward; I saw this; I did that…" and trains a GPT to predict the actions.

At test time you simply *ask* for a high return: put a large target reward at
the front of the prompt, and the model produces actions that, in the training
data, tended to go with achieving that much. It is behaviour cloning,
conditioned on the outcome you want.

## Mechanics
A trajectory of length $T$ becomes $3T$ tokens:

$$\tau = (\hat{R}_1, s_1, a_1,\; \hat{R}_2, s_2, a_2,\; \dots,\; \hat{R}_T, s_T, a_T),$$

where the **return-to-go** $\hat{R}_t = \sum_{t'=t}^{T} r_{t'}$ is the reward
still to be collected from step $t$.

- **Embedding.** Separate linear layers (or a CNN for Atari pixels) embed
  returns, states and actions to dimension $d$, plus a learned **timestep**
  embedding shared by the three tokens of a step (not a per-token position).
- **Backbone.** A causal GPT over the last $K$ timesteps ($3K$ tokens).
- **Output.** From the hidden state at each *state* token, a head predicts
  $a_t$; loss is MSE for continuous actions or cross-entropy for discrete
  ones. Returns and states are not predicted (optional in the paper).

**Inference.**
1. Set $\hat{R}_1$ to the desired return (e.g. the best seen in the dataset).
2. Predict $a_t$, execute it, observe $r_t, s_{t+1}$.
3. Update $\hat{R}_{t+1} = \hat{R}_t - r_t$, append tokens, repeat.

Results (Chen et al., 2021): competitive with or better than strong offline RL
baselines (e.g. CQL) on D4RL locomotion and Atari, and good at sparse /
delayed-reward credit assignment because attention links actions directly to
distant outcomes.

**Limits.** No dynamic programming means it cannot "stitch" together better
behaviour than any segment in the data as well as value-based methods; asking
for a return far above the data does not reliably produce it; in stochastic
environments, conditioning on returns can confuse luck with skill. The
**Trajectory Transformer** (Janner et al., 2021) is a sibling that models
discretised states, actions and rewards and plans with beam search; **Online
DT** adds online fine-tuning.

## Code
```python
import torch, torch.nn as nn

class DecisionTransformer(nn.Module):
    def __init__(self, s_dim, a_dim, d=128, K=20, max_t=1000, L=3, H=1):
        super().__init__()
        self.emb_R, self.emb_s, self.emb_a = nn.Linear(1, d), nn.Linear(s_dim, d), nn.Linear(a_dim, d)
        self.emb_t = nn.Embedding(max_t, d)                 # timestep, shared per step
        layer = nn.TransformerEncoderLayer(d, H, 4 * d, batch_first=True)
        self.gpt = nn.TransformerEncoder(layer, L)
        self.head = nn.Linear(d, a_dim)

    def forward(self, R, s, a, t):                          # (B,K,1) (B,K,s) (B,K,a) (B,K)
        te = self.emb_t(t)
        x = torch.stack([self.emb_R(R) + te, self.emb_s(s) + te, self.emb_a(a) + te], 2)
        x = x.flatten(1, 2)                                 # (B, 3K, d): R1 s1 a1 R2 ...
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1))
        h = self.gpt(x, mask=mask)
        return torch.tanh(self.head(h[:, 1::3]))            # predict a_t from s_t tokens

dt = DecisionTransformer(s_dim=17, a_dim=6)
B, K = 4, 20
pred = dt(torch.randn(B, K, 1), torch.randn(B, K, 17), torch.randn(B, K, 6),
          torch.arange(K).repeat(B, 1))
loss = nn.functional.mse_loss(pred, torch.randn(B, K, 6))   # vs. dataset actions
```

## Why it matters
It showed that the language-model toolkit — causal transformers, next-token
losses, prompting — transfers to control, and reframed offline RL as
conditional sequence modelling. That view fed later generalist agents trained on
mixed trajectories and the idea of "prompting" a policy with a goal. It is also
a good lesson in what a transformer does *not* give you for free: without value
learning it imitates the data rather than improving on it.

## Self-test
Q: What is the return-to-go at timestep t?
A: The sum of rewards from step t to the end of the episode.

Q: What tokens make up one timestep in a Decision Transformer?
A: Three: the return-to-go, the state, and the action.

Q: How do you make a trained Decision Transformer behave well?
A: Condition it on a high desired return-to-go at the start and decrease it by each received reward as the episode runs.

Q: What is a key weakness compared with value-based offline RL?
A: Without dynamic programming it struggles to stitch together sub-trajectories into behaviour better than what is present in the data.
