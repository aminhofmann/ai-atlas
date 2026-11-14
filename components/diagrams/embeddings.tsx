// Embeddings: each token id selects one row of the embedding matrix E (V × d_model).
export const caption =
  "Each token id picks out one row of the learned matrix E; the gathered rows form the sequence matrix X, and training places similar tokens near each other.";

const MONO = "var(--font-geist-mono), monospace";

// Illustrative ids, listed in sequence order. Their rows sit in E in index order.
const IDS = [
  { tok: "the", id: 262, y: 104, rowY: 70 },
  { tok: "cat", id: 3797, y: 164, rowY: 150 },
  { tok: "sat", id: 7543, y: 224, rowY: 230 },
];

const E_X = 200;
const E_W = 130;
const E_TOP = 44;
const E_BOTTOM = 296;
const ROW_H = 16;

const X_X = 404;
const X_W = 150;
const X_TOP = 112;

export default function Diagram() {
  const stripes: number[] = [];
  for (let y = E_TOP + 10; y < E_BOTTOM; y += 10) stripes.push(y);

  return (
    <svg viewBox="0 0 640 340" role="img" aria-labelledby="embeddings-title" className="w-full h-auto">
      <title id="embeddings-title">
        Three token ids select three rows of the embedding matrix E, which are stacked into the matrix X of shape n by
        d_model. An inset shows similar tokens close together.
      </title>
      <defs>
        <marker id="embeddings-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="var(--ink-3)" />
        </marker>
      </defs>

      {/* Token ids */}
      <text x={74} y={78} textAnchor="middle" fontFamily="inherit" fontSize={12} fill="var(--ink-3)">
        token ids
      </text>
      {IDS.map((t) => (
        <g key={t.id}>
          <rect x={24} y={t.y - 15} width={100} height={30} rx={6} fill="var(--panel)" stroke="var(--line)" strokeWidth={1} />
          <text x={36} y={t.y + 4} fontFamily={MONO} fontSize={12} fill="var(--ink-3)">
            {t.tok}
          </text>
          <text x={114} y={t.y + 4} textAnchor="end" fontFamily={MONO} fontSize={13} fill="var(--ink)">
            {t.id}
          </text>
          <line x1={128} y1={t.y} x2={E_X - 4} y2={t.rowY + ROW_H / 2} stroke="var(--ink-3)" strokeWidth={1.25} markerEnd="url(#embeddings-arrow)" />
        </g>
      ))}

      {/* Embedding matrix */}
      <text x={E_X + E_W / 2} y={20} textAnchor="middle" fontFamily={MONO} fontSize={13} fill="var(--ink)">
        E   V × d_model
      </text>
      <text x={E_X + E_W / 2} y={35} textAnchor="middle" fontFamily="inherit" fontSize={11} fill="var(--ink-3)">
        e.g. 50,257 × 768, learned
      </text>
      <rect x={E_X} y={E_TOP} width={E_W} height={E_BOTTOM - E_TOP} rx={6} fill="var(--panel-2)" stroke="var(--line)" strokeWidth={1} />
      {stripes.map((y) => (
        <line key={y} x1={E_X + 1} y1={y} x2={E_X + E_W - 1} y2={y} stroke="var(--line)" strokeWidth={0.75} />
      ))}
      {IDS.map((t) => (
        <g key={`row-${t.id}`}>
          <rect x={E_X} y={t.rowY} width={E_W} height={ROW_H} rx={3} fill="var(--panel)" stroke="var(--accent)" strokeWidth={1.5} />
          <text x={E_X + E_W / 2} y={t.rowY + 12} textAnchor="middle" fontFamily={MONO} fontSize={11} fill="var(--accent)">
            row {t.id}
          </text>
        </g>
      ))}
      <text x={E_X + E_W / 2} y={316} textAnchor="middle" fontFamily="inherit" fontSize={11} fill="var(--ink-3)">
        row t = one-hot(t) · E, done as an index gather
      </text>

      {/* Gathered rows → X */}
      {IDS.map((t, i) => (
        <line
          key={`g-${t.id}`}
          x1={E_X + E_W + 4}
          y1={t.rowY + ROW_H / 2}
          x2={X_X - 4}
          y2={X_TOP + i * 20 + 9}
          stroke="var(--ink-3)"
          strokeWidth={1.25}
          markerEnd="url(#embeddings-arrow)"
        />
      ))}
      <text x={X_X} y={84} fontFamily={MONO} fontSize={13} fill="var(--ink)">
        X   n × d_model
      </text>
      <text x={X_X} y={99} fontFamily="inherit" fontSize={11} fill="var(--ink-3)">
        one row per token (here 3 × 768)
      </text>
      {IDS.map((t, i) => (
        <g key={`x-${t.id}`}>
          <rect x={X_X} y={X_TOP + i * 20} width={X_W} height={18} rx={3} fill="var(--panel)" stroke="var(--line)" strokeWidth={1} />
          <text x={X_X + X_W + 8} y={X_TOP + i * 20 + 13} fontFamily={MONO} fontSize={11} fill="var(--ink-3)">
            {t.tok}
          </text>
        </g>
      ))}

      {/* Inset: geometry of the embedding space */}
      <rect x={386} y={196} width={238} height={124} rx={8} fill="var(--panel)" stroke="var(--line)" strokeWidth={1} />
      <text x={398} y={214} fontFamily="inherit" fontSize={11} fill="var(--ink-2)">
        similar tokens sit close (2-D sketch of ℝ^d)
      </text>
      {[
        { label: "cat", x: 430, y: 246 },
        { label: "dog", x: 458, y: 262 },
        { label: "kitten", x: 424, y: 272 },
        { label: "car", x: 556, y: 244 },
        { label: "truck", x: 574, y: 266 },
        { label: "the", x: 506, y: 302 },
      ].map((p) => (
        <g key={p.label}>
          <circle cx={p.x} cy={p.y} r={3} fill="var(--ink-2)" />
          <text x={p.x + 6} y={p.y + 4} fontFamily={MONO} fontSize={11} fill="var(--ink)">
            {p.label}
          </text>
        </g>
      ))}
    </svg>
  );
}
