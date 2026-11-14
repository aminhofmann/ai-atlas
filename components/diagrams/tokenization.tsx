// Tokenization: raw text -> subword tokens -> integer ids (plus a special <bos>).
export const caption =
  "A rare word is assembled from frequent subword pieces; each piece, and the special <bos> token, becomes one integer id.";

const MONO = "var(--font-geist-mono), monospace";

const TOKENS = [
  { text: "<bos>", id: "1", special: true },
  { text: "un", id: "403", special: false },
  { text: "believ", id: "6893", special: false },
  { text: "ably", id: "1801", special: false },
  { text: "␣fast", id: "3049", special: false },
];

const BOX_W = 92;
const GAP = 16;
const X0 = 58;

export default function Diagram() {
  return (
    <svg viewBox="0 0 640 300" role="img" aria-labelledby="tokenization-title" className="w-full h-auto">
      <title id="tokenization-title">
        The text &quot;unbelievably fast&quot; is split into the tokens &lt;bos&gt;, un, believ, ably and
        &quot; fast&quot;, each mapped to an integer id.
      </title>
      <defs>
        <marker id="tokenization-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="var(--ink-3)" />
        </marker>
      </defs>

      {/* Raw text */}
      <text x={40} y={34} fontFamily="inherit" fontSize={12} fill="var(--ink-3)">
        raw text
      </text>
      <rect x={190} y={16} width={260} height={34} rx={7} fill="var(--panel)" stroke="var(--line)" strokeWidth={1} />
      <text x={320} y={38} textAnchor="middle" fontFamily={MONO} fontSize={14} fill="var(--ink)">
        &quot;unbelievably fast&quot;
      </text>

      <line x1={320} y1={54} x2={320} y2={88} stroke="var(--ink-3)" strokeWidth={1.25} markerEnd="url(#tokenization-arrow)" />
      <text x={332} y={76} fontFamily="inherit" fontSize={12} fill="var(--ink-2)">
        split into subwords (learned vocabulary, e.g. BPE)
      </text>

      {/* Tokens */}
      <text x={50} y={119} textAnchor="end" fontFamily="inherit" fontSize={11} fill="var(--ink-3)">
        tokens
      </text>
      <text x={50} y={202} textAnchor="end" fontFamily="inherit" fontSize={11} fill="var(--ink-3)">
        ids
      </text>
      {TOKENS.map((t, i) => {
        const x = X0 + i * (BOX_W + GAP);
        const cx = x + BOX_W / 2;
        return (
          <g key={t.text}>
            <rect
              x={x}
              y={96}
              width={BOX_W}
              height={36}
              rx={7}
              fill={t.special ? "none" : "var(--panel-2)"}
              stroke={t.special ? "var(--accent)" : "var(--line)"}
              strokeWidth={t.special ? 1.5 : 1}
            />
            <text x={cx} y={119} textAnchor="middle" fontFamily={MONO} fontSize={13} fill={t.special ? "var(--accent)" : "var(--ink)"}>
              {t.text}
            </text>
            <line x1={cx} y1={136} x2={cx} y2={176} stroke="var(--ink-3)" strokeWidth={1.25} markerEnd="url(#tokenization-arrow)" />
            <rect x={x + 16} y={182} width={BOX_W - 32} height={30} rx={6} fill="var(--panel)" stroke="var(--line)" strokeWidth={1} />
            <text x={cx} y={202} textAnchor="middle" fontFamily={MONO} fontSize={13} fill="var(--ink)">
              {t.id}
            </text>
          </g>
        );
      })}
      <text x={50} y={160} textAnchor="end" fontFamily="inherit" fontSize={11} fill="var(--ink-3)">
        lookup
      </text>

      {/* Annotations beside what they describe */}
      <text x={X0 + BOX_W / 2} y={234} textAnchor="middle" fontFamily="inherit" fontSize={11} fill="var(--accent)">
        special token:
      </text>
      <text x={X0 + BOX_W / 2} y={248} textAnchor="middle" fontFamily="inherit" fontSize={11} fill="var(--accent)">
        start of sequence
      </text>
      <path d="M182 220 v6 H458 v-6" fill="none" stroke="var(--ink-3)" strokeWidth={1} />
      <text x={320} y={242} textAnchor="middle" fontFamily="inherit" fontSize={11} fill="var(--ink-3)">
        one rare word → three frequent pieces
      </text>
      <text x={X0 + 4 * (BOX_W + GAP) + BOX_W / 2} y={234} textAnchor="middle" fontFamily="inherit" fontSize={11} fill="var(--ink-3)">
        ␣ = leading space
      </text>

      <text x={320} y={278} textAnchor="middle" fontFamily={MONO} fontSize={12} fill="var(--ink-2)">
        x ∈ {"{"}0, …, V−1{"}"}ⁿ, n = 5   (id values are illustrative; they depend on the tokenizer)
      </text>
    </svg>
  );
}
