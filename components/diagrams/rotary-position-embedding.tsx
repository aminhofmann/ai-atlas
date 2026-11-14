// RoPE: q and k are split into 2-D pairs; pair i is rotated by (position × θ_i).
export const caption =
  "Each 2-D pair is rotated by position × θᵢ; a query at m and a key at n end up separated by an extra (n − m)·θᵢ, so the score depends only on the offset.";

const MONO = "var(--font-geist-mono), monospace";
const R = 58;
const CY = 238;

// Point on a circle at math angle `deg` (counter-clockwise, y up).
function pt(cx: number, deg: number, r: number) {
  const a = (deg * Math.PI) / 180;
  return { x: +(cx + r * Math.cos(a)).toFixed(1), y: +(CY - r * Math.sin(a)).toFixed(1) };
}

// Counter-clockwise arc from deg a to deg b (b > a, < 180° apart).
function arc(cx: number, a: number, b: number, r: number) {
  const p = pt(cx, a, r);
  const q = pt(cx, b, r);
  return `M${p.x} ${p.y} A${r} ${r} 0 0 0 ${q.x} ${q.y}`;
}

function Plane({ cx }: { cx: number }) {
  return (
    <g>
      <circle cx={cx} cy={CY} r={R} fill="none" stroke="var(--line)" strokeWidth={1} strokeDasharray="3 4" />
      <line x1={cx - R - 8} y1={CY} x2={cx + R + 8} y2={CY} stroke="var(--line)" strokeWidth={1} />
      <line x1={cx} y1={CY - R - 8} x2={cx} y2={CY + R + 8} stroke="var(--line)" strokeWidth={1} />
    </g>
  );
}

function Vec({ cx, deg, dashed, label, dx = 6, dy = 0 }: { cx: number; deg: number; dashed?: boolean; label: string; dx?: number; dy?: number }) {
  const tip = pt(cx, deg, R);
  return (
    <g>
      <line
        x1={cx}
        y1={CY}
        x2={tip.x}
        y2={tip.y}
        stroke={dashed ? "var(--ink-3)" : "var(--ink)"}
        strokeWidth={1.5}
        strokeDasharray={dashed ? "4 3" : undefined}
        markerEnd="url(#rotary-position-embedding-arrow)"
      />
      <text x={tip.x + dx} y={tip.y + dy} fontFamily={MONO} fontSize={11} fill={dashed ? "var(--ink-3)" : "var(--ink)"}>
        {label}
      </text>
    </g>
  );
}

const PAIRS = [
  { text: "q₀ q₁", note: "θ₀ fast" },
  { text: "q₂ q₃", note: "θ₁" },
  { text: "…", note: "" },
  { text: "q_d−2 q_d−1", note: "θ_d/2−1 slow" },
];

export default function Diagram() {
  // Plane A: query pair i, original angle α = 15°, rotated by mθᵢ = 60°.
  // Plane B: key pair i, original angle β = 35°, rotated by nθᵢ = 100°.
  // Plane C: both rotated; angle between = (β − α) + (n − m)θᵢ = 20° + 40° = 60°.
  const A = 100;
  const B = 280;
  const C = 460;
  return (
    <svg viewBox="0 0 640 380" role="img" aria-labelledby="rotary-position-embedding-title" className="w-full h-auto">
      <title id="rotary-position-embedding-title">
        A query or key vector is split into two-dimensional pairs. Pair i of the query at position m is rotated by m
        theta i, the key at position n by n theta i, so the angle between them shifts by (n minus m) theta i.
      </title>
      <defs>
        <marker id="rotary-position-embedding-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="var(--ink-3)" />
        </marker>
      </defs>

      {/* Step 1: split into pairs */}
      <text x={24} y={28} fontFamily="inherit" fontSize={13} fill="var(--ink)">
        1. split q (and k) ∈ ℝ^d_head into d/2 two-dimensional pairs
      </text>
      {PAIRS.map((p, i) => {
        const x = 24 + i * 150;
        const w = i === 2 ? 60 : 130;
        const bx = i === 3 ? x - 70 : x;
        return (
          <g key={i}>
            {i !== 2 ? (
              <rect x={bx} y={42} width={w} height={30} rx={6} fill="var(--panel-2)" stroke="var(--line)" strokeWidth={1} />
            ) : null}
            <text x={bx + w / 2} y={62} textAnchor="middle" fontFamily={MONO} fontSize={12} fill="var(--ink)">
              {p.text}
            </text>
            <text x={bx + w / 2} y={88} textAnchor="middle" fontFamily="inherit" fontSize={11} fill="var(--ink-3)">
              {p.note}
            </text>
          </g>
        );
      })}
      <text x={24} y={108} fontFamily={MONO} fontSize={11} fill="var(--ink-3)">
        θᵢ = 10000^(−2i/d): early pairs turn fast (local order), late pairs slowly (long range)
      </text>

      {/* Step 2: rotate */}
      <text x={24} y={130} fontFamily="inherit" fontSize={13} fill="var(--ink)">
        2. rotate pair i by its position × θᵢ (applied to q and k, not v)
      </text>

      <Plane cx={A} />
      <Vec cx={A} deg={15} dashed label="q" dy={4} />
      <Vec cx={A} deg={75} label="R_m q" dx={4} dy={-4} />
      <path d={arc(A, 15, 75, 28)} fill="none" stroke="var(--ink-2)" strokeWidth={1.25} />
      <text x={pt(A, 45, 40).x} y={pt(A, 45, 40).y} fontFamily={MONO} fontSize={11} fill="var(--ink-2)">
        mθᵢ
      </text>
      <text x={A} y={CY + R + 28} textAnchor="middle" fontFamily="inherit" fontSize={12} fill="var(--ink-2)">
        query at position m
      </text>

      <Plane cx={B} />
      <Vec cx={B} deg={35} dashed label="k" dy={4} />
      <Vec cx={B} deg={135} label="R_n k" dx={-40} dy={-4} />
      <path d={arc(B, 35, 135, 28)} fill="none" stroke="var(--ink-2)" strokeWidth={1.25} />
      <text x={pt(B, 85, 40).x} y={pt(B, 85, 40).y} fontFamily={MONO} fontSize={11} fill="var(--ink-2)">
        nθᵢ
      </text>
      <text x={B} y={CY + R + 28} textAnchor="middle" fontFamily="inherit" fontSize={12} fill="var(--ink-2)">
        key at position n
      </text>

      <Plane cx={C} />
      <Vec cx={C} deg={75} label="R_m q" dx={4} dy={-4} />
      <Vec cx={C} deg={135} label="R_n k" dx={-40} dy={-4} />
      <path d={arc(C, 75, 135, 32)} fill="none" stroke="var(--accent)" strokeWidth={2} />
      <text x={C + R + 22} y={CY - 32} fontFamily="inherit" fontSize={11} fill="var(--accent)">
        angle between =
      </text>
      <text x={C + R + 22} y={CY - 17} fontFamily={MONO} fontSize={12} fill="var(--accent)">
        (n−m)θᵢ
      </text>
      <text x={C + R + 22} y={CY - 2} fontFamily={MONO} fontSize={11} fill="var(--ink-3)">
        + (β−α)
      </text>
      <text x={C + R + 22} y={CY + 13} fontFamily="inherit" fontSize={11} fill="var(--ink-3)">
        from content
      </text>
      <text x={C} y={CY + R + 28} textAnchor="middle" fontFamily="inherit" fontSize={12} fill="var(--ink-2)">
        compared in attention
      </text>

      <text x={320} y={364} textAnchor="middle" fontFamily={MONO} fontSize={13} fill="var(--ink)">
        ⟨R_m q, R_n k⟩ = qᵀ R_(n−m) k   — content × offset only
      </text>
    </svg>
  );
}
