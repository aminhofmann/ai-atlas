// Mixture of Experts: router scores 8 experts, only the top-2 run, outputs are mixed.
export const caption =
  "The router scores all 8 experts, but only the top 2 (highlighted) run for this token; their outputs are added, weighted by the renormalised gates.";

const MONO = "var(--font-geist-mono), monospace";

// softmax(x W_r) for one token — sums to 1.00
const SCORES = [0.05, 0.08, 0.41, 0.04, 0.06, 0.32, 0.03, 0.01];
const TOP = new Set([2, 5]);
const EX = 290; // expert column x
const EW = 120;
const EH = 26;
const ey = (i: number) => 28 + i * 34;
const RX = 210; // router right edge
const RY = 160;
const SX = 480; // sum node
const SY = 160;

export default function Diagram() {
  return (
    <svg viewBox="0 0 640 320" role="img" aria-labelledby="mixture-of-experts-title" className="w-full h-auto">
      <title id="mixture-of-experts-title">
        A token vector goes to a router that scores eight expert feed-forward networks; the two highest-scoring experts run and their outputs are combined in a weighted sum.
      </title>
      <defs>
        <marker id="mixture-of-experts-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0 0 L10 5 L0 10 z" fill="var(--ink-3)" />
        </marker>
      </defs>

      {/* token */}
      <rect x="20" y="140" width="56" height="40" rx="7" fill="var(--panel)" stroke="var(--line)" strokeWidth="1.5" />
      <text x="48" y="165" textAnchor="middle" fontSize="14" fontFamily={MONO} fill="var(--ink)">x</text>
      <text x="48" y="198" textAnchor="middle" fontSize="11" fontFamily="inherit" fill="var(--ink-3)">token</text>
      <text x="48" y="212" textAnchor="middle" fontSize="11" fontFamily={MONO} fill="var(--ink-3)">1 × d</text>
      <line x1="76" y1="160" x2="106" y2="160" stroke="var(--ink-3)" strokeWidth="1.5" markerEnd="url(#mixture-of-experts-arrow)" />

      {/* router */}
      <rect x="110" y="130" width="100" height="60" rx="7" fill="var(--panel-2)" stroke="var(--line)" strokeWidth="1.5" />
      <text x="160" y="155" textAnchor="middle" fontSize="13" fontFamily="inherit" fill="var(--ink)">Router</text>
      <text x="160" y="174" textAnchor="middle" fontSize="11" fontFamily={MONO} fill="var(--ink-2)">softmax(xW_r)</text>
      <text x="160" y="210" textAnchor="middle" fontSize="11" fontFamily="inherit" fill="var(--ink-3)">8 scores, keep top-2</text>

      {/* router -> experts, and experts */}
      {SCORES.map((s, i) => {
        const on = TOP.has(i);
        const y = ey(i);
        return (
          <g key={i} opacity={on ? 1 : 0.45}>
            <line
              x1={RX}
              y1={RY}
              x2={EX - 2}
              y2={y + EH / 2}
              stroke={on ? "var(--accent)" : "var(--line)"}
              strokeWidth={on ? 1.5 : 1}
              markerEnd={on ? "url(#mixture-of-experts-arrow)" : undefined}
            />
            <rect
              x={EX}
              y={y}
              width={EW}
              height={EH}
              rx="6"
              fill="var(--panel)"
              stroke={on ? "var(--accent)" : "var(--line)"}
              strokeWidth="1.5"
            />
            <text x={EX + 10} y={y + 17} fontSize="12" fontFamily="inherit" fill="var(--ink)">
              FFN {i + 1}
            </text>
            <text x={EX + EW - 10} y={y + 17} textAnchor="end" fontSize="11" fontFamily={MONO} fill={on ? "var(--ink)" : "var(--ink-3)"}>
              {s.toFixed(2)}
            </text>
          </g>
        );
      })}

      {/* selected experts -> weighted sum */}
      <line x1={EX + EW} y1={ey(2) + EH / 2} x2={SX - 16} y2={SY - 8} stroke="var(--accent)" strokeWidth="1.5" markerEnd="url(#mixture-of-experts-arrow)" />
      <line x1={EX + EW} y1={ey(5) + EH / 2} x2={SX - 16} y2={SY + 8} stroke="var(--accent)" strokeWidth="1.5" markerEnd="url(#mixture-of-experts-arrow)" />
      <text x="446" y="114" textAnchor="middle" fontSize="11" fontFamily={MONO} fill="var(--ink-2)">g̃ = 0.56</text>
      <text x="446" y="222" textAnchor="middle" fontSize="11" fontFamily={MONO} fill="var(--ink-2)">g̃ = 0.44</text>

      <circle cx={SX} cy={SY} r="16" fill="var(--panel)" stroke="var(--line)" strokeWidth="1.5" />
      <text x={SX} y={SY + 5} textAnchor="middle" fontSize="15" fontFamily="inherit" fill="var(--ink)">Σ</text>
      <text x={SX} y={SY + 34} textAnchor="middle" fontSize="11" fontFamily="inherit" fill="var(--ink-3)">weighted sum</text>

      <line x1={SX + 16} y1={SY} x2="538" y2={SY} stroke="var(--ink-3)" strokeWidth="1.5" markerEnd="url(#mixture-of-experts-arrow)" />
      <rect x="542" y="140" width="56" height="40" rx="7" fill="var(--panel)" stroke="var(--line)" strokeWidth="1.5" />
      <text x="570" y="165" textAnchor="middle" fontSize="14" fontFamily={MONO} fill="var(--ink)">y</text>
      <text x="570" y="198" textAnchor="middle" fontSize="11" fontFamily={MONO} fill="var(--ink-3)">1 × d</text>

      <text x={EX + EW / 2} y="310" textAnchor="middle" fontSize="11" fontFamily="inherit" fill="var(--ink-3)">
        6 experts idle for this token: compute grows with k = 2, parameters with E = 8
      </text>
    </svg>
  );
}
