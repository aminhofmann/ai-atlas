// Vision Transformer: patchify -> flatten + linear projection -> [CLS] + position embeddings -> encoder -> class from [CLS].
export const caption =
  "The image becomes a sequence of 196 patch tokens plus a learnable [CLS] token; after the encoder, only the [CLS] state is read out for the class.";

const MONO = "var(--font-geist-mono), monospace";

const STRIPS = [
  { x: 156, label: "1" },
  { x: 178, label: "2" },
  { x: 200, label: "3" },
  { x: 262, label: "196" },
];
const PATCH_TOKENS = [
  { x: 74, w: 30, label: "1" },
  { x: 110, w: 30, label: "2" },
  { x: 146, w: 30, label: "3" },
  { x: 200, w: 38, label: "196" },
];

export default function Diagram() {
  return (
    <svg viewBox="0 0 640 286" role="img" aria-labelledby="vision-transformer-title" className="w-full h-auto">
      <title id="vision-transformer-title">
        A 224 by 224 image is cut into 16 by 16 patches; each patch is flattened to a 768-value vector and linearly
        projected. A learnable [CLS] token is prepended, position embeddings are added, the 197 tokens pass through a
        transformer encoder, and the final [CLS] state goes to an MLP head that outputs the class.
      </title>
      <defs>
        <marker id="vision-transformer-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="var(--ink-3)" />
        </marker>
        <marker id="vision-transformer-arrow-accent" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="var(--accent)" />
        </marker>
      </defs>

      {/* Image cut into patches (grid drawn 4 × 4; ViT-B/16 at 224 px gives 14 × 14) */}
      <rect x={20} y={40} width={96} height={96} rx={4} fill="var(--panel-2)" stroke="var(--ink-3)" strokeWidth={1} />
      {[1, 2, 3].map((k) => (
        <g key={k}>
          <line x1={20 + k * 24} y1={40} x2={20 + k * 24} y2={136} stroke="var(--ink-3)" strokeWidth={1} />
          <line x1={20} y1={40 + k * 24} x2={116} y2={40 + k * 24} stroke="var(--ink-3)" strokeWidth={1} />
        </g>
      ))}
      <text x={68} y={30} textAnchor="middle" fontFamily="inherit" fontSize={12} fill="var(--ink)">image</text>
      <text x={68} y={152} textAnchor="middle" fontFamily={MONO} fontSize={11} fill="var(--ink-2)">224 × 224 × 3</text>
      <text x={68} y={166} textAnchor="middle" fontFamily="inherit" fontSize={11} fill="var(--ink-3)">P = 16 → 14 × 14</text>

      <line x1={124} y1={88} x2={148} y2={88} stroke="var(--ink-3)" strokeWidth={1.25} markerEnd="url(#vision-transformer-arrow)" />

      {/* Flattened patches */}
      <text x={216} y={30} textAnchor="middle" fontFamily="inherit" fontSize={12} fill="var(--ink)">flatten each patch</text>
      {STRIPS.map((s) => (
        <g key={s.label}>
          <rect x={s.x} y={44} width={14} height={80} rx={3} fill="var(--panel)" stroke="var(--ink-3)" strokeWidth={1} />
          <text x={s.x + 7} y={138} textAnchor="middle" fontFamily={MONO} fontSize={11} fill="var(--ink-3)">{s.label}</text>
        </g>
      ))}
      <text x={238} y={88} textAnchor="middle" fontFamily={MONO} fontSize={12} fill="var(--ink-3)">…</text>
      <text x={216} y={160} textAnchor="middle" fontFamily={MONO} fontSize={11} fill="var(--ink-2)">196 × 768</text>
      <text x={216} y={174} textAnchor="middle" fontFamily="inherit" fontSize={11} fill="var(--ink-3)">768 = 16 · 16 · 3</text>

      <line x1={284} y1={88} x2={312} y2={88} stroke="var(--ink-3)" strokeWidth={1.25} markerEnd="url(#vision-transformer-arrow)" />

      {/* Linear projection */}
      <rect x={314} y={56} width={132} height={64} rx={8} fill="var(--panel-2)" stroke="var(--line)" strokeWidth={1} />
      <text x={380} y={80} textAnchor="middle" fontFamily="inherit" fontSize={12} fill="var(--ink)">linear projection E</text>
      <text x={380} y={96} textAnchor="middle" fontFamily={MONO} fontSize={11} fill="var(--ink-2)">768 → D</text>
      <text x={380} y={111} textAnchor="middle" fontFamily="inherit" fontSize={11} fill="var(--ink-3)">(= stride-16 conv)</text>

      {/* Down to the token row */}
      <path d="M380,122 V194 H157 V212" fill="none" stroke="var(--ink-3)" strokeWidth={1.25} markerEnd="url(#vision-transformer-arrow)" />
      <text x={388} y={160} fontFamily={MONO} fontSize={11} fill="var(--ink-2)">196 × D</text>

      {/* Token sequence */}
      <text x={43} y={208} textAnchor="middle" fontFamily="inherit" fontSize={11} fill="var(--ink-3)">learnable</text>
      <rect x={20} y={216} width={46} height={26} rx={5} fill="var(--panel)" stroke="var(--accent)" strokeWidth={1.5} />
      <text x={43} y={233} textAnchor="middle" fontFamily={MONO} fontSize={11} fill="var(--accent)">[CLS]</text>
      {PATCH_TOKENS.map((t) => (
        <g key={t.label}>
          <rect x={t.x} y={216} width={t.w} height={26} rx={5} fill="var(--panel-2)" stroke="var(--line)" strokeWidth={1} />
          <text x={t.x + t.w / 2} y={233} textAnchor="middle" fontFamily={MONO} fontSize={11} fill="var(--ink)">{t.label}</text>
        </g>
      ))}
      <text x={188} y={233} textAnchor="middle" fontFamily={MONO} fontSize={12} fill="var(--ink-3)">…</text>
      <text x={20} y={260} fontFamily="inherit" fontSize={11} fill="var(--ink-3)">+ learned 1-D position embeddings</text>
      <text x={20} y={274} fontFamily={MONO} fontSize={11} fill="var(--ink-3)">E_pos: 197 × D</text>

      <line x1={244} y1={229} x2={270} y2={229} stroke="var(--ink-3)" strokeWidth={1.25} markerEnd="url(#vision-transformer-arrow)" />

      {/* Encoder */}
      <rect x={274} y={204} width={146} height={52} rx={8} fill="var(--panel-2)" stroke="var(--line)" strokeWidth={1} />
      <text x={347} y={225} textAnchor="middle" fontFamily="inherit" fontSize={12} fill="var(--ink)">Transformer encoder</text>
      <text x={347} y={243} textAnchor="middle" fontFamily={MONO} fontSize={11} fill="var(--ink-3)">ViT-B: L 12, D 768</text>

      {/* Read out [CLS] */}
      <line x1={422} y1={229} x2={470} y2={229} stroke="var(--accent)" strokeWidth={1.5} markerEnd="url(#vision-transformer-arrow-accent)" />
      <text x={446} y={220} textAnchor="middle" fontFamily={MONO} fontSize={11} fill="var(--ink-2)">[CLS]</text>

      <rect x={474} y={208} width={74} height={42} rx={8} fill="var(--panel-2)" stroke="var(--line)" strokeWidth={1} />
      <text x={511} y={233} textAnchor="middle" fontFamily="inherit" fontSize={12} fill="var(--ink)">MLP head</text>

      <line x1={550} y1={229} x2={570} y2={229} stroke="var(--ink-3)" strokeWidth={1.25} markerEnd="url(#vision-transformer-arrow)" />
      <text x={576} y={227} fontFamily="inherit" fontSize={12} fill="var(--ink)">class</text>
      <text x={576} y={242} fontFamily="inherit" fontSize={11} fill="var(--ink-3)">logits</text>
    </svg>
  );
}
