---
id: transformers-in-other-domains
title: Applying Transformers to a New Domain
category: domains
level: 2
summary: >-
  The general recipe for bringing transformers to any data type — decide what a
  token is, how to encode position and structure, which architecture and
  pretraining objective to use, how much data you need, and when not to use a
  transformer at all.
requires: [transformer, tokenization, positional-encoding, pretraining-and-fine-tuning]
links:
  - { to: vision-transformer, type: related }
  - { to: clip, type: related }
  - { to: whisper, type: related }
  - { to: decision-transformer, type: related }
  - { to: time-series-transformers, type: related }
  - { to: weather-transformers, type: related }
  - { to: molecular-transformers, type: related }
  - { to: biological-foundation-models, type: related }
  - { to: geneformer, type: related }
  - { to: esm2, type: related }
  - { to: alphafold2, type: related }
  - { to: encoder-decoder, type: uses }
  - { to: bert, type: uses }
  - { to: gpt, type: uses }
  - { to: masked-language-modeling, type: uses }
  - { to: autoregressive-training, type: uses }
  - { to: scaling-laws, type: related }
  - { to: state-space-models, type: related }
  - { to: mamba, type: related }
  - { to: hybrid-architectures, type: related }
  - { to: probing, type: related }
sources:
  - { title: "Attention Is All You Need (Vaswani et al., 2017)", url: "https://arxiv.org/abs/1706.03762" }
  - { title: "An Image is Worth 16x16 Words (Dosovitskiy et al., 2020)", url: "https://arxiv.org/abs/2010.11929" }
  - { title: "Perceiver: General Perception with Iterative Attention (Jaegle et al., 2021)", url: "https://arxiv.org/abs/2103.03206" }
  - { title: "On the Opportunities and Risks of Foundation Models (Bommasani et al., 2021)", url: "https://arxiv.org/abs/2108.07258" }
  - { title: "Are Transformers Effective for Time Series Forecasting? (Zeng et al., 2022)", url: "https://arxiv.org/abs/2205.13504" }
---

## Intuition
A transformer does not care what its tokens *are*. It takes a set of vectors,
lets every vector look at every other one, and repeats. Text was just the first
success. Everything else — images, audio, weather, molecules, cells, robot
trajectories — works the same way once you answer one question well: *how do I
turn my data into a sequence (or set) of vectors that each carry meaning?*

So bringing transformers to a new field is mostly a design exercise around the
model, not inside it: choose the token, tell the model about order or structure,
pick an architecture that matches the task, choose a self-supervised objective
your data supports, and check that you have enough data to pay for the lack of
built-in assumptions. And always compare against the simple baseline — sometimes
it wins.

## Mechanics
**1. What is a token?** It should be a meaningful unit at a manageable count
(attention cost is $O(n^2)$).
- Continuous grids → **patches** (ViT images, PatchTST series, Pangu 3D weather cubes).
- Signals → **spectrogram frames + conv downsampling** (Whisper).
- Discrete symbols → **characters/subwords** (SMILES, amino acids, k-mers in DNABERT).
- Unordered features → **ranked items** (Geneformer's genes by expression) or
  value **binning** (scBERT, Chronos).
- Heterogeneous parts → **typed tokens** (Decision Transformer's return/state/action).
- Very large or variable inputs → compress with **Perceiver-style cross-attention**
  into a fixed latent set (Aurora).

**2. Position and structure.** Attention is permutation-invariant; anything
not added is invisible. Options: learned/sinusoidal 1D, RoPE,
2D/3D or domain-specific biases (Pangu's Earth-specific bias), timestep embeddings,
or *no* positions when the input is genuinely a set (rank order or gene identity
carries the information). Graph structure can enter as attention biases
(Graphormer, AlphaFold 2's pair representation) or masks.

**3. Architecture.**
- **Encoder-only** (BERT-style) → representations, classification,
  regression (ChemBERTa, ESM-2, Geneformer, ViT).
- **Decoder-only** (GPT-style) → generation, forecasting, control
  (Decision Transformer, TimesFM, ProtGPT2).
- **Encoder–decoder** → map one modality/sequence to another (Whisper, Molecular
  Transformer, Chronos on T5).
- **Two towers + contrastive loss** → align modalities (CLIP).

**4. Pretraining objective.** Masked prediction (BERT, MAE), next-token
prediction, contrastive alignment, or direct regression of the next state
(weather). Discrete targets via binning often train more stably than raw MSE.

**5. Data scale.** Fewer inductive biases mean more data needed: ViT needed
JFT-300M-scale pretraining to beat CNNs. Rough checks: do you have millions of
unlabelled examples to pretrain on? Is there a scaling curve, not a single point?

**6. When NOT to use one.**
- Small, tabular datasets → gradient-boosted trees usually win.
- Strong known structure (symmetries, physics, graphs) with little data → CNNs,
  GNNs, equivariant nets.
- Very long sequences where $O(n^2)$ hurts → state-space models,
  Mamba, hybrids (e.g. HyenaDNA, Evo).
- A linear or additive baseline already matches it (DLinear for forecasting; linear
  baselines for gene perturbation prediction) — always run it.

## Code
```python
import torch, torch.nn as nn

class DomainTransformer(nn.Module):
    """Generic recipe: tokenizer -> embed + position/type -> encoder -> task head."""
    def __init__(self, tokenizer, in_dim, d=256, L=6, H=8, n_pos=1024, out_dim=1):
        super().__init__()
        self.tokenize = tokenizer                    # domain-specific: patches, bins, ranks...
        self.embed = nn.Linear(in_dim, d)            # or nn.Embedding for discrete tokens
        self.pos = nn.Embedding(n_pos, d)            # or RoPE / 2D / none for sets
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, H, 4 * d, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, L)   # swap for causal decoder / enc-dec
        self.head = nn.Linear(d, out_dim)

    def forward(self, raw):
        tok = self.tokenize(raw)                     # (B, n, in_dim)
        z = self.embed(tok) + self.pos(torch.arange(tok.size(1)))
        z = torch.cat([self.cls.expand(len(z), -1, -1), z], 1)
        return self.head(self.encoder(z)[:, 0])

# Example tokenizer: 1D signal -> patches of 16 samples
patchify = lambda x: x.unfold(-1, 16, 16)            # (B, L) -> (B, L//16, 16)
model = DomainTransformer(patchify, in_dim=16)
print(model(torch.randn(4, 512)).shape)              # (4, 1)
```

## Why it matters
The same architecture now underlies language, vision, speech, weather, chemistry,
biology and control — which is why a single set of tools (pretraining, fine-tuning,
scaling laws, interpretability methods like probing) transfers across fields.
Knowing the recipe lets you read any new "X-former" paper quickly: find the token,
the position scheme, the architecture type, the objective and the data size, and
then check whether it beat the simple baseline.

## Self-test
Q: Why must a new domain usually supply explicit position or structure information?
A: Self-attention is permutation-invariant, so order, spatial layout or graph structure is invisible unless encoded through positions, biases or masks.

Q: Which architecture type fits "map an audio clip to text"?
A: An encoder–decoder transformer, as in Whisper.

Q: Give two ways to tokenise continuous-valued data.
A: Group values into patches and linearly embed them, or quantise/bin values into a discrete vocabulary.

Q: Why do transformers need more data than CNNs or GNNs in many domains?
A: They have weaker built-in inductive biases (locality, equivariance, graph structure) and must learn them from data.

Q: Name a situation where a transformer is usually the wrong choice.
A: Small tabular datasets (where gradient-boosted trees usually win), or when a simple linear baseline already matches transformer performance.
