# AI Atlas

A personal learning environment for transformer concepts — from tokenization and
attention, through BERT/GPT/Llama, MoE, MLA and Mamba, to biology models
(Geneformer, scGPT, ESM, AlphaFold, Evo, …), interpretability, and other domains.

## Run it

```bash
npm install
cp .env.example .env.local   # add ANTHROPIC_API_KEY for the tutor
npm run dev                  # http://localhost:3000
```

## What's in it

Built as a study session, not a documentation site: one concept, one next action, depth on demand.

| Page | What it's for |
|---|---|
| **Today** (`/`) | Today's queue — due reviews, then practise what you recognise, implement what you can explain, then one new concept whose prerequisites are done. Each item says *why now*. One button: Start session. |
| **Concept** (`/c/<id>`) | A study workspace with four modes. **Learn** — one question per screen: what it is + why it exists + diagram → intuition → mechanics → why it matters. **Practice** — one question at a time; write your answer, reveal, grade yourself; the tutor can critique your answer. **Implement** — write it from scratch, then compare with the reference. **Graph** — the concept's one-hop neighbourhood. Prerequisites, related concepts and sources sit in the side rail. |
| **Tutor** (drawer, <kbd>E</kbd>) | Claude, grounded in the page and its graph neighbours. **Go deeper** drafts a new concept page for a sub-topic; you review it, then add it. |
| **Path** (`/path`) | All concepts in prerequisite order, by track, with "up next". |
| **Review** (`/review`) | Spaced repetition (FSRS) across all concepts you've started. |
| **Map** (`/graph`) | Focus (one concept + neighbours; click to move), List, or the Whole map. |
| **Rebuild from memory** | In a concept's Graph mode, or "Rebuild from memory" on the Map. Its neighbours plus two unrelated concepts arrive shuffled; sort each into *it builds on* / *built on it* / *alternative* / *not connected*. Checking moves every card to where it really belongs and says where you'd put it. "Try a new shuffle" redraws the distractors. |

Mastery is what you can *do*: Recognize → Explain → Apply → Teach. "Explain" counts a prerequisite as done.

Keyboard: <kbd>L</kbd>/<kbd>P</kbd>/<kbd>I</kbd>/<kbd>G</kbd> switch mode · <kbd>N</kbd> next step · <kbd>Space</kbd> reveal (<kbd>⌘↵</kbd> while typing) · <kbd>1</kbd>–<kbd>4</kbd> grade · <kbd>E</kbd> tutor.

**Diagrams** live in `components/diagrams/<concept-id>.tsx` (rules in that folder's README); 20 mechanism-heavy concepts have one. Add a file, run `npm run diagrams`, and check it at `/diagrams`.

## How it's built

- **Content is plain Markdown** in `content/concepts/<id>.md` — frontmatter for the
  graph (`requires`, typed `links`, `sources`), five fixed sections for the layers.
  The format is specified in [`content/SCHEMA.md`](content/SCHEMA.md). Edit a file,
  reload, done. `npm run check` validates every file (ids, links, sections, cards).
- **Progress** (mastery + FSRS card state) lives in `data/progress.json`, git-ignored.
- **Stack:** Next.js 16 (App Router, Cache Components), React Flow, KaTeX,
  ts-fsrs, Anthropic SDK (`claude-opus-5-5`, streamed, with server-side refusal fallback).

## Accuracy

The 68 seed pages were drafted by AI with web verification on recent models, and
told to omit what they couldn't verify. A few claims are still worth a check
against the papers — they're listed in [`content/TO-VERIFY.md`](content/TO-VERIFY.md).
Verify before you memorise.
