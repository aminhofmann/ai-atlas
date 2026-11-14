---
id: probing
title: Probing Classifiers
category: interpretability
level: 2
year: 2016
summary: >-
  Training a small, simple classifier on a frozen model's hidden states to test
  whether some property (part of speech, syntax depth, cell type) can be read
  out of them.
requires: [embeddings, transformer-block]
links:
  - { to: bert, type: applied-in }
  - { to: gpt, type: applied-in }
  - { to: residual-connections, type: related }
  - { to: logit-lens, type: related }
  - { to: attention-visualization, type: related }
  - { to: mechanistic-interpretability, type: related }
  - { to: sparse-autoencoders, type: related }
  - { to: biological-foundation-models, type: applied-in }
  - { to: geneformer, type: applied-in }
  - { to: esm2, type: applied-in }
sources:
  - { title: "Understanding intermediate layers using linear classifier probes (Alain & Bengio, 2016)", url: "https://arxiv.org/abs/1610.01644" }
  - { title: "A Structural Probe for Finding Syntax in Word Representations (Hewitt & Manning, 2019)", url: "https://aclanthology.org/N19-1419/" }
  - { title: "Designing and Interpreting Probes with Control Tasks (Hewitt & Liang, 2019)", url: "https://arxiv.org/abs/1909.03368" }
  - { title: "BERT Rediscovers the Classical NLP Pipeline (Tenney et al., 2019)", url: "https://arxiv.org/abs/1905.05950" }
  - { title: "Probing Classifiers: Promises, Shortcomings, and Advances (Belinkov, 2022)", url: "https://arxiv.org/abs/2102.12452" }
---

## Intuition
Imagine you suspect a colleague secretly knows French. You can't open their
head, but you can hand them a short French sentence and see if they translate
it. Probing does this to a neural network: freeze the model, take its internal
vectors, and train a tiny "translator" (usually a linear classifier) to predict
some label from them. If a weak probe succeeds, the information is plausibly
*encoded* there in an easy-to-read form.

The important word is *encoded*, not *used*. A probe can find information the
model never relies on, and a powerful probe can learn the task itself rather
than reading anything from the model.

## Mechanics
Take hidden states $h^{(\ell)}_i \in \mathbb{R}^{d}$ from layer $\ell$ at token
$i$ of a frozen model, and labels $y_i$ (e.g. POS tags). Train

$$\hat{y}_i = \mathrm{softmax}(W h^{(\ell)}_i + b), \quad W \in \mathbb{R}^{C \times d},$$

with cross-entropy, and report held-out accuracy per layer. Variants:

- **Layer-wise probing.** Alain & Bengio (2016) introduced linear probes on
  each layer of image classifiers. Tenney et al. (2019) found that in
  BERT, lower-level tasks (POS) peak earlier than higher-level ones
  (coreference), echoing a "classical NLP pipeline".
- **Structural probe.** Hewitt & Manning (2019) learn a matrix $B$ so that
  squared distance $\lVert B(h_i - h_j)\rVert^2$ approximates the distance
  between words $i, j$ in the parse tree — showing syntax trees are embedded
  geometrically.
- **Control tasks.** Hewitt & Liang (2019) assign random but consistent labels
  to word types; a probe that also does well on these is memorising. *Selectivity*
  = real accuracy − control accuracy.
- **Information-theoretic probes** (MDL probing) measure how *cheaply* labels
  can be decoded, rather than accuracy alone.
- **Causal follow-ups.** To show the model *uses* a direction, remove it
  (e.g. amnesic probing / iterative nullspace projection) or steer along it and
  watch behaviour change.

In biology, linear probes on frozen Geneformer, scGPT or
ESM-2 embeddings ("linear evaluation") are the standard way to test
whether cell type or protein properties are captured.

## Code
```python
import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer, AutoModel

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased", output_hidden_states=True).eval()

def features(sentences, layer):
    X = []
    for s in sentences:
        enc = tok(s, return_tensors="pt")
        with torch.no_grad():
            hs = model(**enc).hidden_states[layer][0]   # (n_tokens, 768)
        X.append(hs[1:-1].mean(0).numpy())              # mean over word tokens
    return X

train_s = ["the dogs run", "a dog runs", "cats sleep", "the cat sleeps"]
train_y = [1, 0, 1, 0]                                  # plural subject?
for layer in [1, 6, 12]:
    probe = LogisticRegression(max_iter=1000).fit(features(train_s, layer), train_y)
    print(layer, probe.predict(features(["the birds sing"], layer)))
```
(Toy data — a real probe needs a held-out set and a control task.)

## Why it matters
Probing was the main interpretability tool of the "BERTology" era (~2018–2021)
and is still the quickest way to ask "is X represented here?". Its limits —
correlational, sensitive to probe capacity — motivated control tasks, causal
interventions, and the shift toward mechanistic interpretability
and sparse autoencoders, which look for features without
choosing labels in advance.

## Self-test
Q: Why are probes usually kept linear?
A: A weak probe can only succeed if the information is already easily decodable in the representation; a powerful probe could learn the task itself.

Q: What does a control task measure?
A: How well the probe does on random-but-consistent labels; high control accuracy means the probe memorises, so selectivity (real minus control) is reported.

Q: What did Hewitt & Manning's structural probe show?
A: That a learned linear transform of BERT/ELMo vectors makes squared distances approximate parse-tree distances, so syntax is embedded geometrically.

Q: What can a successful probe NOT tell you?
A: That the model actually uses that information for its predictions — that needs a causal intervention.
