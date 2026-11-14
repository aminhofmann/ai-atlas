---
id: molecular-transformers
title: Molecular Transformers (Chemical Language Models)
category: domains
level: 2
year: 2019
summary: >-
  Transformers that treat molecules as strings (SMILES or SELFIES) — for
  reaction prediction (Molecular Transformer), property prediction (ChemBERTa,
  MolFormer) and molecule generation.
requires: [tokenization, transformer, masked-language-modeling]
links:
  - { to: bert, type: based-on }
  - { to: encoder-decoder, type: uses }
  - { to: rotary-position-embedding, type: uses }
  - { to: pretraining-and-fine-tuning, type: uses }
  - { to: gpt, type: related }
  - { to: esm2, type: related }
  - { to: biological-foundation-models, type: related }
  - { to: attention-visualization, type: related }
  - { to: transformers-in-other-domains, type: part-of }
sources:
  - { title: "Molecular Transformer: A Model for Uncertainty-Calibrated Chemical Reaction Prediction (Schwaller et al., 2019)", url: "https://arxiv.org/abs/1811.02633" }
  - { title: "ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction (Chithrananda et al., 2020)", url: "https://arxiv.org/abs/2010.09885" }
  - { title: "Large-scale chemical language representations capture molecular structure and properties (MoLFormer, Ross et al., 2022)", url: "https://arxiv.org/abs/2106.09553" }
  - { title: "Self-Referencing Embedded Strings (SELFIES) (Krenn et al., 2020)", url: "https://arxiv.org/abs/1905.13741" }
  - { title: "SMILES Enumeration as Data Augmentation for Neural Network Modeling of Molecules (Bjerrum, 2017)", url: "https://arxiv.org/abs/1703.07076" }
---

## Intuition
Chemists already have a language for molecules: **SMILES**, which writes a
molecular graph as a line of text — `CCO` is ethanol, `c1ccccc1` is benzene,
parentheses mark branches and digits mark ring closures. If a molecule is a
sentence, then a transformer trained on millions of them can learn chemical
"grammar" — valence, functional groups, which fragments go together — the same
way a language model learns English.

That opens two families. **Encoder models** (ChemBERTa, MolFormer) are
pretrained with masked-token prediction and fine-tuned to predict properties
such as solubility or toxicity. **Sequence-to-sequence models** (the Molecular
Transformer) translate reactant SMILES into product SMILES, treating chemistry
literally as machine translation.

## Mechanics
**Tokenisation.** Usually a regex that splits SMILES into atoms and symbols
(`Cl`, `Br`, `[nH]`, `=`, `(`, `1`), or BPE learned on SMILES. **SELFIES** is an
alternative grammar in which *every* string decodes to a valid molecule — useful
for generation.

**Non-uniqueness.** One molecule has many valid SMILES (different starting atoms
and traversal orders). *Canonical* SMILES pick one; *randomised* SMILES are used
as data augmentation (Bjerrum, 2017).

**Models.**
- **Molecular Transformer** (Schwaller et al., 2019): an
  encoder–decoder transformer from reactants (+ reagents)
  to products; trained on patent reaction data (USPTO) with SMILES augmentation;
  its likelihood serves as a calibrated confidence. Also used reversed for
  retrosynthesis.
- **ChemBERTa** (Chithrananda et al., 2020): RoBERTa with masked-token
  pretraining on up to ~10M PubChem SMILES, evaluated on MoleculeNet; they
  reported performance improving with pretraining set size. ChemBERTa-2
  followed.
- **MoLFormer** (Ross et al., IBM; Nature Machine Intelligence 2022): an encoder
  with **linear attention** and **rotary position embeddings**, masked-token
  pretraining on ~1.1B SMILES from PubChem and ZINC; competitive with graph neural
  networks on MoleculeNet; attention analysis suggested it captures
  inter-atomic spatial relations.
- **Generative decoders** (GPT-style on SMILES/SELFIES) sample new molecules,
  often steered by property conditioning or reinforcement learning.

**Limits.** A string hides 3D geometry and stereochemistry is easy to get wrong;
graph and 3D-aware models (GNNs, Graphormer, equivariant networks) remain strong,
and many molecular datasets are small enough that pretraining helps only modestly.

## Code
```python
import re, torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Standard SMILES regex tokenizer (atoms, bracket atoms, bonds, ring digits)
PATTERN = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])"
smiles = "CC(=O)Oc1ccccc1C(=O)O"                      # aspirin
print(re.findall(PATTERN, smiles))                    # ['C','C','(','=','O',')','O','c','1',...]

# Masked-token prediction with a pretrained ChemBERTa checkpoint
name = "seyonec/ChemBERTa-zinc-base-v1"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForMaskedLM.from_pretrained(name).eval()
enc = tok(f"CC(=O)Oc1ccccc1C(={tok.mask_token})O", return_tensors="pt")
with torch.no_grad():
    logits = model(**enc).logits
pos = (enc.input_ids[0] == tok.mask_token_id).nonzero()[0, 0]
print(tok.decode(logits[0, pos].argmax()))            # hopefully 'O'
# Property prediction: AutoModelForSequenceClassification + MoleculeNet labels.
```
(Checkpoint name is a community upload; treat as illustrative.)

## Why it matters
Chemical language models brought NLP's pretraining playbook to chemistry:
reaction prediction became translation, and property prediction became
fine-tuning a pretrained encoder. The Molecular Transformer was a landmark for
forward reaction prediction, and SMILES/SELFIES language models are now standard
baselines in drug discovery alongside graph models. They are the small-molecule
counterpart of protein language models like ESM-2.

## Self-test
Q: What is SMILES?
A: A line notation that writes a molecular graph as a text string, with symbols for atoms, bonds, branches (parentheses) and ring closures (digits).

Q: Why is SMILES enumeration useful for training?
A: A molecule has many valid SMILES strings, so random variants act as data augmentation and teach invariance to the string order.

Q: What task did the Molecular Transformer frame as translation?
A: Chemical reaction prediction: translating reactant SMILES into product SMILES with an encoder–decoder transformer.

Q: What two architectural choices distinguish MoLFormer?
A: Linear attention (for efficiency over ~1.1B molecules) and rotary position embeddings.

Q: What advantage does SELFIES have over SMILES?
A: Every SELFIES string decodes to a valid molecule, which helps generative models.
