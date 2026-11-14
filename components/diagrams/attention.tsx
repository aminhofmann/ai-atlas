// Scaled dot-product attention: softmax(QKᵀ/√d_k) V, followed for one query row.
export const caption =
  "Follow query 2: its row of scaled scores QKᵀ/√dₖ goes through softmax into weights that sum to 1, and its output is the weighted sum of the rows of V.";

const MONO = "var(--font-geist-mono), monospace";

const CELL = 32;
const GX = 190; // score grid
const GY = 60;
const ROW = 1; // highlighted query row (0-based → q₂)
const SCORES = ["0.3", "2.1", "1.0", "−0.2"];
const WEIGHTS = [0.1, 0.63, 0.21, 0.06]; // softmax of SCORES, rounded; sums to 1
const SUB = ["₁", "₂", "₃", "₄"];

function Matrix({ x, y, w, label, shape }: { x: number; y: number; w: number; label: string; shape: string }) {
  return (
    <g>
      <text x={x} y={y - 8} fontFamily={MONO} fontSize={13} fill="var(--ink)">
        {label}
        <tspan fontSize={11} fill="var(--ink-3)" dx={8}>
          {shape}
        </tspan>
      </text>
      <rect x={x} y={y} width={w} height={64} rx={6} fill="var(--panel-2)" stroke="var(--line)" strokeWidth={1} />
      {[1, 2, 3].map((r) => (
        <line key={r} x1={x + 1} y1={y + r * 16} x2={x + w - 1} y2={y + r * 16} stroke="var(--line)" strokeWidth={0.75} />
      ))}
    </g>
  );
}

export default function Diagram() {
  const wx = 372; // weights row
  const wy = GY + ROW * CELL; // aligned with the highlighted score row
  const vx = 404; // V rows on the right
  const vy = 236;
  return (
    <svg viewBox="0 0 640 380" role="img" aria-labelledby="attention-title" className="w-full h-auto">
      <title id="attention-title">
        Q and K produce an n by n score matrix QK transpose over root d k. Softmax turns one row into weights that sum
        to one, and the output for that query is the weighted sum of the rows of V.
      </title>
      <defs>
        <marker id="attention-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="var(--ink-3)" />
        </marker>
      </defs>

      {/* Inputs */}
      <Matrix x={30} y={50} w={90} label="Q" shape="n × dₖ" />
      <Matrix x={30} y={156} w={90} label="K" shape="n × dₖ" />
      <Matrix x={30} y={262} w={112} label="V" shape="n × dᵥ" />

      <line x1={124} y1={82} x2={162} y2={100} stroke="var(--ink-3)" strokeWidth={1.25} markerEnd="url(#attention-arrow)" />
      <line x1={124} y1={188} x2={162} y2={156} stroke="var(--ink-3)" strokeWidth={1.25} markerEnd="url(#attention-arrow)" />

      {/* Score matrix */}
      <text x={GX} y={GY - 26} fontFamily={MONO} fontSize={13} fill="var(--ink)">
        QKᵀ / √dₖ
      </text>
      <text x={GX} y={GY - 10} fontFamily="inherit" fontSize={11} fill="var(--ink-3)">
        scores, n × n
      </text>
      {[0, 1, 2, 3].map((r) =>
        [0, 1, 2, 3].map((c) => (
          <rect
            key={`${r}-${c}`}
            x={GX + c * CELL}
            y={GY + r * CELL}
            width={CELL}
            height={CELL}
            fill={r === ROW ? "var(--panel)" : "var(--panel-2)"}
            stroke="var(--line)"
            strokeWidth={1}
          />
        )),
      )}
      <rect x={GX} y={GY + ROW * CELL} width={4 * CELL} height={CELL} fill="none" stroke="var(--ink)" strokeWidth={1.5} />
      {SCORES.map((s, c) => (
        <text key={c} x={GX + c * CELL + CELL / 2} y={GY + ROW * CELL + 20} textAnchor="middle" fontFamily={MONO} fontSize={11} fill="var(--ink)">
          {s}
        </text>
      ))}
      {SUB.map((s, i) => (
        <g key={s}>
          <text x={GX - 6} y={GY + i * CELL + 20} textAnchor="end" fontFamily={MONO} fontSize={11} fill={i === ROW ? "var(--ink)" : "var(--ink-3)"}>
            q{s}
          </text>
          <text x={GX + i * CELL + CELL / 2} y={GY + 4 * CELL + 16} textAnchor="middle" fontFamily={MONO} fontSize={11} fill="var(--ink-3)">
            k{s}
          </text>
        </g>
      ))}

      {/* Softmax over the highlighted row */}
      <line x1={GX + 4 * CELL + 6} y1={wy + CELL / 2} x2={wx - 6} y2={wy + CELL / 2} stroke="var(--ink-3)" strokeWidth={1.25} markerEnd="url(#attention-arrow)" />
      <text x={(GX + 4 * CELL + wx) / 2} y={wy + CELL / 2 - 8} textAnchor="middle" fontFamily="inherit" fontSize={11} fill="var(--ink-2)">
        softmax
      </text>
      <text x={wx} y={GY - 10} fontFamily="inherit" fontSize={11} fill="var(--accent)">
        weights α for q₂ — sum to 1
      </text>
      {WEIGHTS.map((w, c) => {
        const h = w * 48;
        return (
          <g key={c}>
            <rect x={wx + c * CELL + 6} y={wy - 4 - h} width={CELL - 12} height={h} rx={2} fill="var(--accent)" />
            <rect x={wx + c * CELL} y={wy} width={CELL} height={CELL} fill="none" stroke="var(--accent)" strokeWidth={1.25} />
            <text x={wx + c * CELL + CELL / 2} y={wy + 20} textAnchor="middle" fontFamily={MONO} fontSize={11} fill="var(--accent)">
              {w.toFixed(2).replace(/^0/, "")}
            </text>
          </g>
        );
      })}

      {/* Weighted sum of V rows */}
      <line x1={wx + 2 * CELL} y1={wy + CELL + 6} x2={wx + 2 * CELL} y2={vy - 22} stroke="var(--ink-3)" strokeWidth={1.25} markerEnd="url(#attention-arrow)" />
      <text x={wx + 2 * CELL + 8} y={vy - 34} fontFamily="inherit" fontSize={11} fill="var(--ink-2)">
        scale each value row
      </text>
      <line x1={146} y1={294} x2={330} y2={vy + 28} stroke="var(--ink-3)" strokeWidth={1.25} markerEnd="url(#attention-arrow)" />
      {WEIGHTS.map((w, i) => (
        <g key={i}>
          <text x={vx - 8} y={vy + i * 20 + 12} textAnchor="end" fontFamily={MONO} fontSize={11} fill="var(--accent)">
            {w.toFixed(2)}·v{SUB[i]}
          </text>
          <rect x={vx} y={vy + i * 20} width={120} height={16} rx={3} fill="var(--panel-2)" stroke="var(--line)" strokeWidth={1} />
          <rect x={vx} y={vy + i * 20} width={120 * w} height={16} rx={3} fill="var(--ink-3)" />
        </g>
      ))}
      <line x1={vx + 60} y1={vy + 84} x2={vx + 60} y2={vy + 106} stroke="var(--ink-3)" strokeWidth={1.25} markerEnd="url(#attention-arrow)" />
      <text x={vx + 70} y={vy + 100} fontFamily="inherit" fontSize={11} fill="var(--ink-2)">
        sum
      </text>
      <rect x={vx} y={vy + 112} width={120} height={18} rx={3} fill="var(--panel)" stroke="var(--ink)" strokeWidth={1.5} />
      <text x={vx - 8} y={vy + 125} textAnchor="end" fontFamily={MONO} fontSize={12} fill="var(--ink)">
        out₂ = Σⱼ αⱼ vⱼ
      </text>
      <text x={vx + 128} y={vy + 125} fontFamily={MONO} fontSize={11} fill="var(--ink-3)">
        1 × dᵥ
      </text>
    </svg>
  );
}
