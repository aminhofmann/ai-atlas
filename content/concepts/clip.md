---
id: clip
title: CLIP (Contrastive Language–Image Pretraining)
category: domains
level: 2
year: 2021
summary: >-
  Trains an image encoder and a text encoder together so that matching
  image–caption pairs land close in a shared embedding space, enabling
  zero-shot image classification from text prompts.
requires: [vision-transformer, embeddings, transformer]
links:
  - { to: gpt, type: related }
  - { to: pretraining-and-fine-tuning, type: uses }
  - { to: scaling-laws, type: related }
  - { to: cross-attention, type: alternative-to }
  - { to: transformers-in-other-domains, type: part-of }
sources:
  - { title: "Learning Transferable Visual Models From Natural Language Supervision (Radford et al., 2021)", url: "https://arxiv.org/abs/2103.00020" }
  - { title: "Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision (ALIGN, Jia et al., 2021)", url: "https://arxiv.org/abs/2102.05918" }
  - { title: "Sigmoid Loss for Language Image Pre-Training (SigLIP, Zhai et al., 2023)", url: "https://arxiv.org/abs/2303.15343" }
---

## Intuition
Classic image classifiers learn from a fixed list of labels ("1,000 ImageNet
classes"), so they cannot name anything outside it. CLIP learns from the
captions people already put next to images on the internet. It plays a matching
game: given a batch of images and a batch of captions, figure out which caption
belongs to which image.

To win, both encoders must map an image and its description to nearby points in
one shared space. Once trained, you classify an image zero-shot by writing the
candidate labels as sentences ("a photo of a dog", "a photo of a cat"),
embedding them, and picking the closest.

## Mechanics
Two encoders, no cross-attention between them:
- **Image encoder** $f$: a ResNet or a ViT (the best
  released model is ViT-L/14).
- **Text encoder** $g$: a GPT-style transformer with causal masking; the
  representation at the end-of-text token is used.

Each output is linearly projected and L2-normalised to $u_i, v_j \in \mathbb{R}^{D}$.
For a batch of $N$ pairs, the similarity matrix $S_{ij} = u_i^\top v_j / \tau$
($\tau$ a learned temperature) is trained with a **symmetric InfoNCE** loss —
cross-entropy over rows (image→text) and over columns (text→image), where the
diagonal is the correct pair:

$$\mathcal{L} = \tfrac{1}{2}\Big[\mathrm{CE}(S, \text{diag}) + \mathrm{CE}(S^\top, \text{diag})\Big].$$

Every off-diagonal pair is a negative, so large batches (32,768 in the paper)
matter. Training data: ~400M image–text pairs collected from the web (WIT).

**Zero-shot classification:** embed prompts $t_k$ = "a photo of a {class k}",
predict $\arg\max_k u^\top v_k$. Prompt ensembling over templates helps. The
paper reports zero-shot ImageNet accuracy matching the original supervised
ResNet-50, without using any ImageNet training labels, and better robustness
to distribution shift.

**Limitations:** weak on counting, fine-grained and abstract tasks; inherits
web biases; a bag-of-concepts tendency (poor at word order/composition).

**Successors:** ALIGN (noisier, larger data), OpenCLIP/LAION reproductions,
SigLIP (a pairwise sigmoid loss that removes the batch-wide softmax).

## Code
```python
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
labels = ["a photo of a dog", "a photo of a cat", "a photo of a car"]

inputs = proc(text=labels, images=Image.open("pet.jpg"), return_tensors="pt", padding=True)
with torch.no_grad():
    out = model(**inputs)
print(dict(zip(labels, out.logits_per_image.softmax(-1)[0].tolist())))

# The training loss, written out:
def clip_loss(img_emb, txt_emb, logit_scale):
    img = img_emb / img_emb.norm(dim=-1, keepdim=True)
    txt = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
    logits = logit_scale * img @ txt.T                     # (N, N)
    y = torch.arange(len(img))
    return (torch.nn.functional.cross_entropy(logits, y)
            + torch.nn.functional.cross_entropy(logits.T, y)) / 2
```

## Why it matters
CLIP made natural language the interface to vision: open-vocabulary
classification, image–text retrieval, and a text-aligned image space. Its
encoders became standard components — the text encoder conditions diffusion
models such as Stable Diffusion, and CLIP-style vision towers feed many
multimodal LLMs. The contrastive two-tower recipe has since been copied for
audio, molecules–text and cells–text.

## Self-test
Q: What does CLIP's training objective push together and apart?
A: It pulls matching image–caption embeddings together and pushes all other image–caption pairs in the batch apart (symmetric contrastive loss).

Q: How does CLIP classify an image zero-shot?
A: It embeds each class name as a text prompt and picks the class whose text embedding has the highest cosine similarity with the image embedding.

Q: Why does batch size matter for CLIP?
A: Every other pair in the batch acts as a negative, so larger batches give more and harder negatives.

Q: Do CLIP's image and text encoders attend to each other?
A: No — they are independent towers that only interact through the dot product of their final embeddings.
