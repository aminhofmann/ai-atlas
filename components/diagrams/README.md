# Concept diagrams

One file per concept: `components/diagrams/<concept-id>.tsx`. Each file:

```tsx
// No "use client", no hooks, no state — a pure SVG server component.
export const caption = "One sentence saying what to look at."; // shown under the diagram

export default function Diagram() {
  return (
    <svg viewBox="0 0 640 300" role="img" aria-labelledby="d-title" className="w-full h-auto">
      <title id="d-title">Short accessible description</title>
      {/* ... */}
    </svg>
  );
}
```

## Visual rules (the app is a calm study tool — the diagram must read in 5 seconds)

- **Width 640** in the viewBox; height whatever the content needs (typically 220–360).
- **Colours only via CSS variables**, so light and dark mode both work. Never hex.
  - text: `var(--ink)` (primary), `var(--ink-2)` (secondary), `var(--ink-3)` (annotations)
  - strokes/borders: `var(--line)`; fills for boxes: `var(--panel)` / `var(--panel-2)`
  - ONE emphasis colour: `var(--accent)` — for the thing the diagram is about (e.g. the attention weights, the selected experts). Use it on 1–3 elements, not more.
  - A second, rarely: `var(--warn)` (amber) for "masked / discarded / cost".
- **Text:** `fontFamily="inherit"`, sizes 11–14, `fill` set explicitly. Monospace (`fontFamily="var(--font-geist-mono), monospace"`) only for tokens, ids, tensor shapes.
- **Labels sit next to what they label** (spatial contiguity). No legends that force eye-jumping.
- Strokes 1–1.5px, corner radius 6–8, generous spacing. No gradients, no shadows, no animation, no emoji.
- Show tensor shapes where they help (`n × d`).
- Arrows: define one `<marker>` in `<defs>` with an id prefixed by the concept id (ids must be unique per page), fill `var(--ink-3)`.
- Must be correct. Draw the mechanism exactly as the concept page in `content/concepts/<id>.md` describes it — read that file first.
