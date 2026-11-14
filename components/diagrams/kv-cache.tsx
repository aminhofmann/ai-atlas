// KV cache: each decoding step computes only the new token's q, k, v and attends to the stored K, V.
export const caption =
  "At step t only the new token's q, k, v are computed; q attends to the cached keys and values of tokens 1…t−1 plus its own, and the cache grows by one column per step.";

const MONO = "var(--font-geist-mono), monospace";
const SUB = ["₁", "₂", "₃", "₄", "₅"];
const CX = 160; // first cache column
const STEP = 32;
const CW = 30;

function Row({ t, y }: { t: number; y: number }) {
  const right = CX + 5 * STEP + 20;
  return (
    <g>
      <text x={20} y={y + 30} fontFamily="inherit" fontSize={12} fill="var(--ink-2)">
        step t = {t}
      </text>
      <rect x={96} y={y} width={40} height={22} rx={5} fill="var(--panel)" stroke="var(--ink)" strokeWidth={1.25} />
      <text x={116} y={y + 15} textAnchor="middle" fontFamily={MONO} fontSize={12} fill="var(--ink)">
        q{SUB[t - 1]}
      </text>
      <line x1={138} y1={y + 11} x2={CX - 4} y2={y + 11} stroke="var(--ink-3)" strokeWidth={1.25} markerEnd="url(#kv-cache-arrow)" />

      {Array.from({ length: t }, (_, j) => {
        const isNew = j === t - 1;
        const x = CX + j * STEP;
        return (
          <g key={j}>
            {["k", "v"].map((kind, r) => (
              <g key={kind}>
                <rect
                  x={x}
                  y={y + r * 26}
                  width={CW}
                  height={22}
                  rx={4}
                  fill={isNew ? "var(--panel)" : "var(--panel-2)"}
                  stroke={isNew ? "var(--accent)" : "var(--line)"}
                  strokeWidth={isNew ? 1.5 : 1}
                />
                <text
                  x={x + CW / 2}
                  y={y + r * 26 + 15}
                  textAnchor="middle"
                  fontFamily={MONO}
                  fontSize={11}
                  fill={isNew ? "var(--accent)" : "var(--ink-2)"}
                >
                  {kind}
                  {SUB[j]}
                </text>
              </g>
            ))}
          </g>
        );
      })}

      <text x={right} y={y + 15} fontFamily={MONO} fontSize={11} fill="var(--ink-2)">
        scores: 1 × {t} → softmax
      </text>
      <text x={right} y={y + 41} fontFamily={MONO} fontSize={11} fill="var(--ink-2)">
        o{SUB[t - 1]} = Σⱼ αⱼ vⱼ over {t} values
      </text>
    </g>
  );
}

export default function Diagram() {
  const rows = [
    { t: 3, y: 68 },
    { t: 4, y: 148 },
    { t: 5, y: 228 },
  ];
  return (
    <svg viewBox="0 0 640 390" role="img" aria-labelledby="kv-cache-title" className="w-full h-auto">
      <title id="kv-cache-title">
        Three decoding steps. At each step the new query attends to the cached keys and values of earlier tokens plus
        its own new key and value, and the cache grows by one column. Below, the cache memory formula.
      </title>
      <defs>
        <marker id="kv-cache-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="var(--ink-3)" />
        </marker>
      </defs>

      <text x={CX} y={36} fontFamily="inherit" fontSize={11} fill="var(--ink-3)">
        cached kⱼ, vⱼ for j &lt; t: computed once, reused
      </text>
      <text x={CX + 2 * STEP} y={56} fontFamily="inherit" fontSize={11} fill="var(--accent)">
        new kₜ, vₜ: computed this step, appended
      </text>

      {rows.map((r) => (
        <Row key={r.t} t={r.t} y={r.y} />
      ))}
      {rows.slice(0, 2).map((r) => (
        <text key={`g-${r.t}`} x={CX} y={r.y + 68} fontFamily="inherit" fontSize={11} fill="var(--ink-3)">
          ↓ cache grows by one column
        </text>
      ))}

      {/* Memory cost */}
      <rect x={20} y={308} width={600} height={64} rx={8} fill="var(--panel)" stroke="var(--warn)" strokeWidth={1.25} />
      <text x={320} y={332} textAnchor="middle" fontFamily={MONO} fontSize={12} fill="var(--ink)">
        bytes = 2 × layers × kv_heads × d_head × seq_len × bytes/elem
      </text>
      <text x={320} y={354} textAnchor="middle" fontFamily="inherit" fontSize={11} fill="var(--ink-3)">
        2 = one K and one V · e.g. 32 layers, 32 heads, d_head 128, fp16 → 512 KiB per token, 2 GiB at 4,096 tokens
      </text>
    </svg>
  );
}
