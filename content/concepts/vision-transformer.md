---
id: vision-transformer
title: Vision Transformer (ViT)
category: domains
level: 2
year: 2020
summary: >-
  Applies a standard transformer encoder to images by cutting them into
  fixed-size patches, treating each flattened patch as a token.
requires: [transformer-block, self-attention, positional-encoding]
links:
  - { to: bert, type: based-on }
  - { to: tokenization, type: uses }
  - { to: embeddings, type: uses }
  - { to: pretraining-and-fine-tuning, type: uses }
  - { to: masked-language-modeling, type: related }
  - { to: clip, type: part-of }
  - { to: scaling-laws, type: related }
  - { to: attention-visualization, type: related }
  - { to: transformers-in-other-domains, type: part-of }
sources:
  - { title: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (Dosovitskiy et al., 2020)", url: "https://arxiv.org/abs/2010.11929" }
  - { title: "Training data-efficient image transformers & distillation through attention (DeiT, Touvron et al., 2020)", url: "https://arxiv.org/abs/2012.12877" }
  - { title: "Masked Autoencoders Are Scalable Vision Learners (He et al., 2021)", url: "https://arxiv.org/abs/2111.06377" }
  - { title: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows (Liu et al., 2021)", url: "https://arxiv.org/abs/2103.14030" }
---

## Intuition
Attention over every pixel would be hopeless: a 224×224 image has ~50,000
pixels, and attention cost grows with the square of sequence length. ViT's
trick is to read the image like a page of words: chop it into a grid of 16×16
patches, flatten each patch into a vector, and feed the resulting sequence of
196 "visual words" to an ordinary BERT-style encoder.

Nothing in the architecture knows that images are 2D or that neighbouring
pixels are related — the inductive biases CNNs bake in (locality, translation
equivariance). ViT has to learn them from data, which is why it lagged CNNs on
small datasets but overtook them when pretrained on very large ones.

## Mechanics
1. **Patchify.** Image $x \in \mathbb{R}^{H \times W \times C}$ → $N = HW/P^2$
   patches $x_p \in \mathbb{R}^{N \times (P^2 C)}$. For $224^2$, $P=16$: $N=196$,
   each of length 768.
2. **Patch embedding.** Linear projection $E \in \mathbb{R}^{(P^2 C) \times D}$
   (equivalently a stride-$P$ convolution).
3. **[CLS] token and positions.** Prepend a learnable $x_{\text{class}}$, add
   learned 1D position embeddings $E_{pos} \in \mathbb{R}^{(N+1) \times D}$:
   $z_0 = [x_{\text{class}}; x_p^1 E; \dots; x_p^N E] + E_{pos}$. The paper found
   2D-aware position embeddings gave no clear gain.
4. **Encoder.** $L$ pre-norm blocks (multi-head self-attention + MLP with GELU).
   ViT-Base: $L=12$, $D=768$, 12 heads; Large: 24/1024/16; Huge: 32/1280/16.
5. **Head.** Classify from the final [CLS] state (or mean-pooled patches).

**Data.** Trained on ImageNet alone, ViT underperformed comparable ResNets;
pretrained on ImageNet-21k or the private JFT-300M and fine-tuned, it matched or
beat state-of-the-art CNNs with less pretraining compute. For fine-tuning at
higher resolution, the position embeddings are 2D-interpolated.

**Descendants.**
- **DeiT**: strong augmentation + distillation token → competitive with
  ImageNet-1k only.
- **Swin**: attention within shifted local windows + hierarchical patch merging,
  giving linear cost and multi-scale features for detection/segmentation.
- **MAE**: self-supervised pretraining by masking ~75% of patches and
  reconstructing pixels — the vision analogue of masked language modelling.
- ViTs are the image encoders in CLIP, most multimodal LLMs, and weather
  models such as ClimaX.

## Code
```python
import torch, torch.nn as nn

class TinyViT(nn.Module):
    def __init__(self, img=224, P=16, C=3, D=192, L=6, H=3, n_cls=1000):
        super().__init__()
        N = (img // P) ** 2
        self.patch = nn.Conv2d(C, D, kernel_size=P, stride=P)   # patchify + linear
        self.cls = nn.Parameter(torch.zeros(1, 1, D))
        self.pos = nn.Parameter(torch.randn(1, N + 1, D) * 0.02)
        layer = nn.TransformerEncoderLayer(D, H, 4 * D, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, L)
        self.head = nn.Linear(D, n_cls)

    def forward(self, x):                                       # (B, 3, 224, 224)
        z = self.patch(x).flatten(2).transpose(1, 2)           # (B, 196, D)
        z = torch.cat([self.cls.expand(len(z), -1, -1), z], 1) + self.pos
        return self.head(self.enc(z)[:, 0])                    # logits from [CLS]

print(TinyViT()(torch.randn(2, 3, 224, 224)).shape)            # (2, 1000)
```

## Why it matters
ViT showed that a nearly unmodified NLP transformer can be a top vision model if
given enough data, unifying architectures across modalities. That unification is
what made CLIP, multimodal LLMs, and "patches as tokens" designs in other
fields (time series, weather, microscopy) straightforward. It is the canonical
example of trading inductive bias for scale.

## Self-test
Q: How does ViT turn an image into tokens?
A: It splits the image into non-overlapping P×P patches, flattens each, and linearly projects it to the model dimension.

Q: How many tokens does a 224×224 image give with 16×16 patches (excluding [CLS])?
A: 196 (a 14×14 grid).

Q: Why did ViT underperform CNNs when trained on ImageNet-1k alone?
A: It lacks CNN inductive biases like locality and translation equivariance, so it needs much more data to learn them.

Q: What does MAE do?
A: Masks a large fraction (~75%) of image patches and trains a ViT to reconstruct the missing pixels, as self-supervised pretraining.
