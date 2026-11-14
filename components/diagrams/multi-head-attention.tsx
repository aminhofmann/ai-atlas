// Multi-head attention: h parallel attention heads in their own subspaces, concatenated, then W_O.
export const caption =
  "Each head projects X into its own d_head-wide subspace and runs attention there; the h outputs are concatenated and mixed back to d_model by W_O.";

const MONO = "var(--font-geist-mono), monospace";

const LANES = [
  { cx: 130, name: "head 1", i: "1", sup: "¹" },
  { cx: 290, name: "head 2", i: "2", sup: "²" },
  { cx: 510, name: "head h", i: "h", sup: "ʰ" },
];
const LANE_W = 136;

// Segments of the concatenated output, in the same order as the lanes.
const SEGMENTS = [
  { x: 62, w: 136, label: "head₁" },
  { x: 222, w: 136, label: "head₂" },
  { x: 358, w: 84, label: "…" },
  { x: 442, w: 136, label: "head_h" },
];

export default function Diagram() {
  return (
    <svg viewBox="0 0 640 410" role="img" aria-labelledby="multi-head-attention-title" className="w-full h-auto">
      <title id="multi-head-attention-title">
        The input X is projected separately for each of h heads. Each head runs attention in its own subspace of width
        d_head; the head outputs are concatenated and multiplied by W_O to give an output of shape n by d_model.
      </title>
      <defs>
        <marker id="multi-head-attention-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="var(--ink-3)" />
        </marker>
      </defs>

      {/* Input */}
      <rect x={200} y={16} width={240} height={28} rx={6} fill="var(--panel)" stroke="var(--line)" strokeWidth={1} />
      <text x={320} y={35} textAnchor="middle" fontFamily={MONO} fontSize={13} fill="var(--ink)">
        X   n × d_model
      </text>
      <text x={452} y={35} fontFamily={MONO} fontSize={11} fill="var(--ink-3)">
        d_head = d_model / h
      </text>

      {LANES.map((l) => {
        const x = l.cx - LANE_W / 2;
        return (
          <g key={l.name}>
            <line x1={320} y1={46} x2={l.cx} y2={80} stroke="var(--ink-3)" strokeWidth={1.25} markerEnd="url(#multi-head-attention-arrow)" />

            {/* Per-head projections */}
            <rect x={x} y={86} width={LANE_W} height={44} rx={7} fill="var(--panel-2)" stroke="var(--line)" strokeWidth={1} />
            <text x={l.cx} y={104} textAnchor="middle" fontFamily={MONO} fontSize={11} fill="var(--ink)">
              X · W_Q,K,V⁽{l.sup}⁾
            </text>
            <text x={l.cx} y={121} textAnchor="middle" fontFamily="inherit" fontSize={11} fill="var(--ink-3)">
              own Q, K, V: n × d_head
            </text>
            <line x1={l.cx} y1={132} x2={l.cx} y2={150} stroke="var(--ink-3)" strokeWidth={1.25} markerEnd="url(#multi-head-attention-arrow)" />

            {/* Attention in the head's subspace */}
            <rect x={x} y={154} width={LANE_W} height={44} rx={7} fill="var(--panel)" stroke="var(--accent)" strokeWidth={1.5} />
            <text x={l.cx} y={173} textAnchor="middle" fontFamily="inherit" fontSize={13} fill="var(--accent)">
              Attention
            </text>
            <text x={l.cx} y={189} textAnchor="middle" fontFamily="inherit" fontSize={11} fill="var(--ink-3)">
              its own n × n weights
            </text>
            <line x1={l.cx} y1={200} x2={l.cx} y2={218} stroke="var(--ink-3)" strokeWidth={1.25} markerEnd="url(#multi-head-attention-arrow)" />
            <text x={l.cx} y={234} textAnchor="middle" fontFamily={MONO} fontSize={11} fill="var(--ink-2)">
              head{l.i === "h" ? "_h" : l.i === "1" ? "₁" : "₂"}: n × d_head
            </text>
            <line x1={l.cx} y1={240} x2={l.cx} y2={260} stroke="var(--ink-3)" strokeWidth={1.25} markerEnd="url(#multi-head-attention-arrow)" />
          </g>
        );
      })}
      <text x={400} y={176} textAnchor="middle" fontFamily="inherit" fontSize={14} fill="var(--ink-3)">
        …
      </text>
      <text x={400} y={120} textAnchor="middle" fontFamily="inherit" fontSize={11} fill="var(--ink-3)">
        h heads
      </text>
      <text x={400} y={134} textAnchor="middle" fontFamily="inherit" fontSize={11} fill="var(--ink-3)">
        in parallel
      </text>

      {/* Concatenation */}
      {SEGMENTS.map((s) => (
        <g key={s.label}>
          <rect x={s.x} y={264} width={s.w} height={26} fill="var(--panel-2)" stroke="var(--line)" strokeWidth={1} />
          <text x={s.x + s.w / 2} y={281} textAnchor="middle" fontFamily={MONO} fontSize={11} fill="var(--ink)">
            {s.label}
          </text>
        </g>
      ))}
      <text x={62} y={306} fontFamily="inherit" fontSize={11} fill="var(--ink-3)">
        Concat   n × (h · d_head) = n × d_model
      </text>

      <line x1={320} y1={292} x2={320} y2={316} stroke="var(--ink-3)" strokeWidth={1.25} markerEnd="url(#multi-head-attention-arrow)" />
      <rect x={250} y={320} width={140} height={28} rx={6} fill="var(--panel)" stroke="var(--line)" strokeWidth={1} />
      <text x={320} y={339} textAnchor="middle" fontFamily={MONO} fontSize={13} fill="var(--ink)">
        · W_O
      </text>
      <text x={402} y={339} fontFamily={MONO} fontSize={11} fill="var(--ink-3)">
        (h·d_head) × d_model
      </text>
      <text x={238} y={339} textAnchor="end" fontFamily="inherit" fontSize={11} fill="var(--ink-3)">
        mixes the heads
      </text>

      <line x1={320} y1={350} x2={320} y2={370} stroke="var(--ink-3)" strokeWidth={1.25} markerEnd="url(#multi-head-attention-arrow)" />
      <text x={320} y={392} textAnchor="middle" fontFamily={MONO} fontSize={13} fill="var(--ink)">
        MHA(X)   n × d_model
      </text>
    </svg>
  );
}
