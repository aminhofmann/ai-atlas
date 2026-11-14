// Pre-norm transformer block: two branches read the residual stream through LayerNorm and add back.
export const caption =
  "The residual stream runs straight through the block; each branch reads it through LayerNorm, transforms it (attention, then MLP) and adds the result back.";

const MONO = "var(--font-geist-mono), monospace";
const SX = 170; // residual stream x

function Branch({ tapY, addY, sub, note }: { tapY: number; addY: number; sub: string; note: string }) {
  return (
    <g>
      {/* tap off the stream into LayerNorm */}
      <line x1={SX} y1={tapY} x2={236} y2={tapY} stroke="var(--ink-3)" strokeWidth={1.25} markerEnd="url(#transformer-block-arrow)" />
      <rect x={240} y={tapY - 14} width={74} height={28} rx={6} fill="var(--panel-2)" stroke="var(--line)" strokeWidth={1} />
      <text x={277} y={tapY + 4} textAnchor="middle" fontFamily="inherit" fontSize={12} fill="var(--ink)">
        LayerNorm
      </text>
      <line x1={316} y1={tapY} x2={336} y2={tapY} stroke="var(--ink-3)" strokeWidth={1.25} markerEnd="url(#transformer-block-arrow)" />
      <rect x={340} y={tapY - 18} width={136} height={36} rx={7} fill="var(--panel)" stroke="var(--line)" strokeWidth={1} />
      <text x={408} y={tapY + 4} textAnchor="middle" fontFamily="inherit" fontSize={13} fill="var(--ink)">
        {sub}
      </text>
      <text x={408} y={tapY + 34} textAnchor="middle" fontFamily="inherit" fontSize={11} fill="var(--ink-3)">
        {note}
      </text>
      {/* back into the stream at the add node */}
      <path
        d={`M476 ${tapY} H530 V${addY} H${SX + 14}`}
        fill="none"
        stroke="var(--ink-3)"
        strokeWidth={1.25}
        markerEnd="url(#transformer-block-arrow)"
      />
      <circle cx={SX} cy={addY} r={10} fill="var(--panel)" stroke="var(--accent)" strokeWidth={1.5} />
      <path d={`M${SX - 5} ${addY} H${SX + 5} M${SX} ${addY - 5} V${addY + 5}`} stroke="var(--accent)" strokeWidth={1.5} />
    </g>
  );
}

export default function Diagram() {
  return (
    <svg viewBox="0 0 640 400" role="img" aria-labelledby="transformer-block-title" className="w-full h-auto">
      <title id="transformer-block-title">
        A pre-norm transformer block. The residual stream is a vertical line. One branch applies LayerNorm then
        multi-head attention and adds the result back; a second branch applies LayerNorm then an MLP and adds back.
      </title>
      <defs>
        <marker id="transformer-block-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="var(--ink-3)" />
        </marker>
        <marker id="transformer-block-arrow-accent" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="var(--accent)" />
        </marker>
      </defs>

      {/* Residual stream */}
      <line x1={SX} y1={372} x2={SX} y2={46} stroke="var(--accent)" strokeWidth={2} markerEnd="url(#transformer-block-arrow-accent)" />
      <text x={SX} y={390} textAnchor="middle" fontFamily={MONO} fontSize={13} fill="var(--ink)">
        X   n × d_model
      </text>
      <text x={SX} y={30} textAnchor="middle" fontFamily={MONO} fontSize={13} fill="var(--ink)">
        Y   n × d_model
      </text>
      <text x={SX + 70} y={30} fontFamily="inherit" fontSize={11} fill="var(--ink-3)">
        → next block (stacked L times)
      </text>
      <text x={SX - 14} y={346} textAnchor="end" fontFamily="inherit" fontSize={12} fill="var(--accent)">
        residual stream
      </text>
      <text x={SX - 14} y={361} textAnchor="end" fontFamily="inherit" fontSize={11} fill="var(--ink-3)">
        never normalized
      </text>

      <Branch tapY={310} addY={244} sub="Multi-head attention" note="mixes information across tokens" />
      <Branch tapY={176} addY={110} sub="MLP (FFN)" note="each token on its own" />

      <text x={SX - 16} y={248} textAnchor="end" fontFamily={MONO} fontSize={12} fill="var(--ink)">
        H = X + MHA(LN(X))
      </text>
      <text x={SX - 16} y={114} textAnchor="end" fontFamily={MONO} fontSize={12} fill="var(--ink)">
        Y = H + MLP(LN(H))
      </text>
    </svg>
  );
}
