// MHA vs GQA vs MQA: 8 query heads mapped onto 8, 2 or 1 key/value heads.
export const caption =
  "Every panel keeps 8 query heads; what changes is how many key/value heads they read from — and the KV cache shrinks with it.";

const MONO = "var(--font-geist-mono), monospace";

const QW = 16; // query head box width
const QG = 6; // gap
const QY = 64;
const QH = 18;
const KY = 150;
const KH = 22;
const SPAN = 8 * QW + 7 * QG; // 170

type Panel = { x0: number; title: string; g: number; cache: string };
const PANELS: Panel[] = [
  { x0: 10, title: "Multi-head (MHA)", g: 8, cache: "2 · 8 · d_h   (1×)" },
  { x0: 220, title: "Grouped-query (GQA)", g: 2, cache: "2 · 2 · d_h   (¼)" },
  { x0: 430, title: "Multi-query (MQA)", g: 1, cache: "2 · 1 · d_h   (⅛)" },
];

export default function Diagram() {
  return (
    <svg viewBox="0 0 640 270" role="img" aria-labelledby="grouped-query-attention-title" className="w-full h-auto">
      <title id="grouped-query-attention-title">
        Three panels: multi-head attention with 8 query heads and 8 KV heads, grouped-query attention with 8 query heads sharing 2 KV heads, and multi-query attention with 8 query heads sharing 1 KV head, each with its KV-cache size.
      </title>

      <line x1="215" y1="12" x2="215" y2="258" stroke="var(--line)" strokeWidth="1" />
      <line x1="425" y1="12" x2="425" y2="258" stroke="var(--line)" strokeWidth="1" />

      {PANELS.map(({ x0, title, g, cache }) => {
        const left = x0 + 15;
        const cx = left + SPAN / 2;
        const qx = (i: number) => left + i * (QW + QG);
        const perGroup = 8 / g;
        // a KV head spans the query heads of its group
        const kvX = (j: number) => qx(j * perGroup);
        const kvW = perGroup * QW + (perGroup - 1) * QG;
        return (
          <g key={title}>
            <text x={cx} y="24" textAnchor="middle" fontSize="13" fontFamily="inherit" fill="var(--ink)" fontWeight="600">
              {title}
            </text>
            <text x={cx} y="40" textAnchor="middle" fontSize="11" fontFamily={MONO} fill="var(--ink-3)">
              G = {g}
            </text>

            <text x={cx} y={QY - 8} textAnchor="middle" fontSize="11" fontFamily="inherit" fill="var(--ink-2)">
              8 query heads
            </text>

            {/* which query head reads which KV head */}
            {Array.from({ length: 8 }, (_, i) => {
              const j = Math.floor(i / perGroup);
              return (
                <line
                  key={i}
                  x1={qx(i) + QW / 2}
                  y1={QY + QH}
                  x2={kvX(j) + kvW / 2}
                  y2={KY}
                  stroke="var(--ink-3)"
                  strokeWidth="1"
                />
              );
            })}

            {Array.from({ length: 8 }, (_, i) => (
              <rect key={i} x={qx(i)} y={QY} width={QW} height={QH} rx="4" fill="var(--panel)" stroke="var(--line)" strokeWidth="1.5" />
            ))}

            {Array.from({ length: g }, (_, j) => (
              <g key={j}>
                <rect x={kvX(j)} y={KY} width={kvW} height={KH} rx="5" fill="var(--panel-2)" stroke="var(--line)" strokeWidth="1.5" />
                {kvW > 40 && (
                  <text x={kvX(j) + kvW / 2} y={KY + 15} textAnchor="middle" fontSize="11" fontFamily={MONO} fill="var(--ink-2)">
                    K, V
                  </text>
                )}
              </g>
            ))}
            <text x={cx} y={KY + KH + 16} textAnchor="middle" fontSize="11" fontFamily="inherit" fill="var(--ink-2)">
              {g} KV head{g > 1 ? "s" : ""}
            </text>

            {/* KV cache size */}
            <text x={left} y="216" fontSize="11" fontFamily="inherit" fill="var(--ink-3)">
              KV cache per token, per layer
            </text>
            <rect x={left} y="224" width={SPAN} height="10" rx="3" fill="none" stroke="var(--line)" strokeWidth="1" />
            <rect x={left} y="224" width={(SPAN * g) / 8} height="10" rx="3" fill="var(--accent)" />
            <text x={left} y="252" fontSize="11" fontFamily={MONO} fill="var(--ink-2)">
              {cache}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
