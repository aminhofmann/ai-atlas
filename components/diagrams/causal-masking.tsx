// Causal masking: scores for keys after the query (j > i) are set to −∞ before softmax.
export const caption =
  "Row i is the query, column j the key: every cell right of the diagonal is set to −∞, so after softmax each token's weights cover only itself and earlier tokens.";

const MONO = "var(--font-geist-mono), monospace";

const TOKENS = ["The", "cat", "sat", "on", "the"];
// Softmax weights over the visible keys of each row; each row sums to 1.
const WEIGHTS = [
  [1.0],
  [0.4, 0.6],
  [0.2, 0.3, 0.5],
  [0.1, 0.2, 0.3, 0.4],
  [0.1, 0.1, 0.2, 0.25, 0.35],
];

const C = 44;
const GX = 200;
const GY = 70;

export default function Diagram() {
  // Staircase outline around the visible (j ≤ i) region.
  let stair = `M${GX} ${GY}`;
  for (let i = 0; i < 5; i++) {
    stair += ` H${GX + (i + 1) * C} V${GY + (i + 1) * C}`;
  }
  stair += ` H${GX} Z`;

  return (
    <svg viewBox="0 0 640 330" role="img" aria-labelledby="causal-masking-title" className="w-full h-auto">
      <title id="causal-masking-title">
        A five by five grid of attention scores. Rows are queries and columns are keys. Cells above the diagonal, where
        the key comes after the query, are masked with minus infinity; the lower triangle keeps its weights.
      </title>

      {/* Axis labels */}
      <text x={GX + (5 * C) / 2} y={22} textAnchor="middle" fontFamily="inherit" fontSize={12} fill="var(--ink-2)">
        keys j →
      </text>
      <text
        x={0}
        y={0}
        transform={`translate(${GX - 66} ${GY + (5 * C) / 2}) rotate(-90)`}
        textAnchor="middle"
        fontFamily="inherit"
        fontSize={12}
        fill="var(--ink-2)"
      >
        ← queries i
      </text>

      {TOKENS.map((t, k) => (
        <g key={k}>
          <text x={GX + k * C + C / 2} y={GY - 10} textAnchor="middle" fontFamily={MONO} fontSize={12} fill="var(--ink-2)">
            {t}
          </text>
          <text x={GX - 10} y={GY + k * C + C / 2 + 4} textAnchor="end" fontFamily={MONO} fontSize={12} fill="var(--ink-2)">
            {t}
          </text>
        </g>
      ))}

      {/* Cells */}
      {TOKENS.map((_, i) =>
        TOKENS.map((__, j) => {
          const x = GX + j * C;
          const y = GY + i * C;
          const masked = j > i;
          return (
            <g key={`${i}-${j}`}>
              <rect x={x} y={y} width={C} height={C} fill={masked ? "var(--panel-2)" : "var(--panel)"} stroke="var(--line)" strokeWidth={1} />
              <text
                x={x + C / 2}
                y={y + C / 2 + 4}
                textAnchor="middle"
                fontFamily={MONO}
                fontSize={11}
                fill={masked ? "var(--warn)" : "var(--ink)"}
              >
                {masked ? "−∞" : WEIGHTS[i][j].toFixed(2).replace(/^0/, "").replace(/^1\.00$/, "1")}
              </text>
            </g>
          );
        }),
      )}
      <path d={stair} fill="none" stroke="var(--accent)" strokeWidth={1.5} />

      {/* Annotations beside the regions they describe */}
      <text x={GX + 5 * C + 18} y={GY + 26} fontFamily="inherit" fontSize={12} fill="var(--warn)">
        masked: j &gt; i (the future)
      </text>
      <text x={GX + 5 * C + 18} y={GY + 42} fontFamily="inherit" fontSize={11} fill="var(--ink-3)">
        score + (−∞) → exp = 0 → weight 0
      </text>

      <text x={GX + 5 * C + 18} y={GY + 4 * C + 18} fontFamily="inherit" fontSize={12} fill="var(--accent)">
        visible: j ≤ i (past and self)
      </text>
      <text x={GX + 5 * C + 18} y={GY + 4 * C + 34} fontFamily="inherit" fontSize={11} fill="var(--ink-3)">
        weights in each row sum to 1
      </text>

      <text x={320} y={GY + 5 * C + 32} textAnchor="middle" fontFamily={MONO} fontSize={12} fill="var(--ink-2)">
        softmax(QKᵀ/√dₖ + M) V,   Mᵢⱼ = 0 if j ≤ i, −∞ if j &gt; i
      </text>
    </svg>
  );
}
