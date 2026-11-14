---
id: weather-transformers
title: Weather and Climate Transformers
category: domains
level: 3
year: 2023
summary: >-
  Transformers that forecast the global atmosphere by treating gridded
  weather fields as 3D patch tokens — e.g. Pangu-Weather and Microsoft Aurora —
  matching or beating physics-based forecasts at a fraction of the compute.
requires: [vision-transformer, transformer-block, positional-encoding]
links:
  - { to: pretraining-and-fine-tuning, type: uses }
  - { to: cross-attention, type: uses }
  - { to: time-series-transformers, type: related }
  - { to: scaling-laws, type: related }
  - { to: transformers-in-other-domains, type: part-of }
sources:
  - { title: "Accurate medium-range global weather forecasting with 3D neural networks (Pangu-Weather, Bi et al., Nature 2023)", url: "https://www.nature.com/articles/s41586-023-06185-3" }
  - { title: "Aurora: A Foundation Model of the Atmosphere / A Foundation Model for the Earth System (Bodnar et al., 2024)", url: "https://arxiv.org/abs/2405.13063" }
  - { title: "FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operators (Pathak et al., 2022)", url: "https://arxiv.org/abs/2202.11214" }
  - { title: "ClimaX: A foundation model for weather and climate (Nguyen et al., 2023)", url: "https://arxiv.org/abs/2301.10343" }
  - { title: "Learning skillful medium-range global weather forecasting (GraphCast, Lam et al., Science 2023) — a GNN, for contrast", url: "https://www.science.org/doi/10.1126/science.adi2336" }
---

## Intuition
A weather forecast is a giant image-to-image problem: the state of the
atmosphere now — temperature, wind, humidity, pressure on a global
latitude–longitude grid at many altitude levels — goes in, and the state six
hours later comes out. Repeat that step and you have a 10-day forecast.
Traditional numerical weather prediction solves physics equations on
supercomputers; learned models are trained on decades of reanalysis data
(ERA5) to make the same step directly.

Transformers fit because the grid can be cut into 3D patches (lat × lon ×
pressure level), exactly like ViT cuts images, and attention can relate distant
regions. Once trained, a forecast takes seconds on one GPU instead of hours on a
supercomputer.

Not every famous ML weather model is a transformer: **GraphCast** (DeepMind) is a
graph neural network on an icosahedral mesh, and **FourCastNet** mixes tokens
with Fourier neural operators (AFNO) rather than standard attention.

## Mechanics
**Pangu-Weather** (Bi et al., Huawei, Nature 2023).
- Input: upper-air variables on 13 pressure levels plus surface variables at
  0.25° resolution, embedded into 3D patches.
- Backbone: a **3D Earth-Specific Transformer (3DEST)** — a Swin-style
  encoder–decoder with windowed attention, where Swin's relative position bias is
  replaced by an **Earth-specific positional bias** (the atmosphere is not
  translation-invariant: latitude and altitude matter).
- **Hierarchical temporal aggregation:** separate models for 1 h, 3 h, 6 h and
  24 h lead times; a forecast greedily chains the largest steps possible (e.g.
  7 days = 7 × 24 h), reducing the number of autoregressive steps and thus error
  accumulation.
- Trained on 39 years of ERA5 (1979–2017); reported more accurate than ECMWF's
  operational IFS deterministic forecast on many variables, and more than
  10,000× faster.

**Aurora** (Bodnar et al., Microsoft, 2024; Nature 2025).
- A **1.3B-parameter** foundation model: a **3D Perceiver-based encoder** maps
  heterogeneous inputs (different variables, resolutions, pressure levels) into a
  fixed latent 3D representation using cross-attention; a
  **3D Swin Transformer U-Net** processor evolves it; a Perceiver decoder maps back
  to the requested variables.
- **Pretrain, then fine-tune:** pretrained on over a million hours of diverse
  Earth-system data (reanalyses, forecasts, climate simulations), then fine-tuned
  for high-resolution weather, atmospheric chemistry/air quality, ocean waves and
  tropical cyclone tracks.

**ClimaX** (Nguyen et al., 2023) is a simpler ViT with variable-wise tokenisation
and aggregation, pretrained on climate simulations.

Training objective (all of them): regression, typically latitude-weighted MSE
between predicted and true next state; some fine-tune on multi-step rollouts.

## Code
```python
import torch, torch.nn as nn

# Toy "Pangu-style" 3D patch embedding for one atmospheric state
V, Z, Hlat, Wlon = 5, 13, 721, 1440     # variables, pressure levels, 0.25° grid
x = torch.randn(1, V, Z, Hlat, Wlon)    # (B, vars, levels, lat, lon)

patch = nn.Conv3d(V, 192, kernel_size=(2, 4, 4), stride=(2, 4, 4))
x = nn.functional.pad(x, (0, 0, 0, 3, 0, 1))            # pad levels/lat to divide
tokens = patch(x).flatten(2).transpose(1, 2)            # (1, N_tokens, 192)
print(tokens.shape)                                      # -> Swin-style 3D blocks

def latitude_weighted_mse(pred, target, lat_deg):        # standard training loss
    w = torch.cos(torch.deg2rad(lat_deg))
    w = (w / w.mean()).view(1, 1, 1, -1, 1)
    return (w * (pred - target) ** 2).mean()

# Real Aurora (illustrative): from aurora import AuroraPretrained; model.load_checkpoint()
```

## Why it matters
Between 2022 and 2024, ML weather models went from curiosities to matching or
beating the world's best physics-based deterministic forecasts at a tiny
inference cost, and operational centres such as ECMWF now run their own
data-driven models. Transformers (Pangu, Aurora, FuXi, ClimaX) are one major
family alongside GNNs (GraphCast) and neural operators (FourCastNet). Aurora
pushes the LLM playbook — a large pretrained model fine-tuned for many
Earth-system tasks. All of these still depend on physics-based reanalysis and
analysis data for training and initial conditions.

## Self-test
Q: What is the token in Pangu-Weather and similar models?
A: A 3D patch of the gridded atmosphere spanning latitude, longitude and pressure levels, embedded like a ViT patch.

Q: Why does Pangu-Weather use separate 1 h, 3 h, 6 h and 24 h models?
A: Chaining the largest available lead times reduces the number of autoregressive steps and therefore accumulated error (hierarchical temporal aggregation).

Q: Is GraphCast a transformer?
A: No — it is a graph neural network operating on a multi-resolution icosahedral mesh.

Q: What role does the Perceiver encoder play in Aurora?
A: It uses cross-attention to map heterogeneous input variables and resolutions into a fixed latent 3D representation for the Swin backbone.

Q: Why is the loss latitude-weighted?
A: Grid cells near the poles cover less area on a lat–lon grid, so errors are weighted by cos(latitude).
