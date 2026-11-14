// FlashAttention vs standard attention: who touches slow HBM, and how much.
import type { ReactNode } from "react";

export const caption =
  "Standard attention writes the full n×n matrices S and P to slow HBM and reads them back; FlashAttention loads small tiles into SRAM, runs an online softmax there, and writes back only O.";

const MONO = "var(--font-geist-mono), monospace";
const A = "url(#flash-attention-arrow)";

function Sub({ children }: { children: ReactNode }) {
  return (
    <>
      <tspan dy="3" fontSize="0.75em">{children}</tspan>
      <tspan dy="-3">{"​"}</tspan>
    </>
  );
}

// a column of row-blocks stored in HBM, with one block highlighted as "loaded"
function Blocks({ x, y, hi, label }: { x: number; y: number; hi: number; label: string }) {
  return (
    <g>
      {[0, 1, 2, 3].map((i) => (
        <rect
          key={i}
          x={x}
          y={y + i * 12}
          width="24"
          height="12"
          fill={i === hi ? "var(--panel-2)" : "var(--panel)"}
          stroke={i === hi ? "var(--ink-2)" : "var(--line)"}
          strokeWidth="1"
        />
      ))}
      <text x={x + 12} y={y + 64} textAnchor="middle" fontSize="11" fontFamily={MONO} fill="var(--ink-2)">
        {label}
      </text>
    </g>
  );
}

function Tile({ x, y, children }: { x: number; y: number; children: ReactNode }) {
  return (
    <g>
      <rect x={x} y={y} width="30" height="22" rx="4" fill="var(--panel-2)" stroke="var(--ink-2)" strokeWidth="1" />
      <text x={x + 15} y={y + 15} textAnchor="middle" fontSize="11" fontFamily={MONO} fill="var(--ink)">
        {children}
      </text>
    </g>
  );
}

export default function Diagram() {
  return (
    <svg viewBox="0 0 640 344" role="img" aria-labelledby="flash-attention-title" className="w-full h-auto">
      <title id="flash-attention-title">
        Left: standard attention moves the n by n score and probability matrices between SRAM and HBM. Right: FlashAttention loads query, key and value tiles into SRAM, computes one score tile at a time with an online softmax, and writes only the output back to HBM.
      </title>
      <defs>
        <marker id="flash-attention-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0 0 L10 5 L0 10 z" fill="var(--ink-3)" />
        </marker>
      </defs>

      <line x1="320" y1="10" x2="320" y2="334" stroke="var(--line)" strokeWidth="1" />

      {/* ---------- standard attention ---------- */}
      <text x="162" y="24" textAnchor="middle" fontSize="13" fontWeight="600" fontFamily="inherit" fill="var(--ink)">
        Standard attention
      </text>

      <rect x="20" y="40" width="285" height="115" rx="8" fill="var(--panel)" stroke="var(--line)" strokeWidth="1.5" />
      <text x="32" y="58" fontSize="11" fontFamily="inherit" fill="var(--ink-2)">SRAM (on-chip: fast, small)</text>
      <text x="162" y="104" textAnchor="middle" fontSize="11" fontFamily={MONO} fill="var(--ink-2)">
        S = QKᵀ/√d → P = softmax(S) → O = PV
      </text>
      <text x="162" y="124" textAnchor="middle" fontSize="11" fontFamily="inherit" fill="var(--ink-3)">
        three separate steps, each round-trips HBM
      </text>

      <rect x="20" y="205" width="285" height="95" rx="8" fill="var(--panel)" stroke="var(--line)" strokeWidth="1.5" />
      <text x="32" y="223" fontSize="11" fontFamily="inherit" fill="var(--ink-2)">HBM (off-chip: slow, large)</text>

      {/* Q, K, V */}
      {["Q", "K", "V"].map((l, i) => (
        <g key={l}>
          <rect x={34 + i * 22} y="234" width="16" height="48" rx="2" fill="var(--panel)" stroke="var(--line)" strokeWidth="1" />
          <text x={42 + i * 22} y="295" textAnchor="middle" fontSize="11" fontFamily={MONO} fill="var(--ink-2)">{l}</text>
        </g>
      ))}
      {/* S and P: n x n */}
      {[
        { x: 116, l: "S" },
        { x: 196, l: "P" },
      ].map(({ x, l }) => (
        <g key={l}>
          <rect x={x} y="230" width="56" height="56" rx="3" fill="var(--panel-2)" stroke="var(--warn)" strokeWidth="1.5" />
          <text x={x + 28} y="256" textAnchor="middle" fontSize="12" fontFamily={MONO} fill="var(--ink)">{l}</text>
          <text x={x + 28} y="272" textAnchor="middle" fontSize="10" fontFamily={MONO} fill="var(--ink-3)">n × n</text>
          <line x1={x + 18} y1="157" x2={x + 18} y2="227" stroke="var(--warn)" strokeWidth="1.5" markerEnd={A} />
          <line x1={x + 38} y1="227" x2={x + 38} y2="159" stroke="var(--warn)" strokeWidth="1.5" markerEnd={A} />
        </g>
      ))}
      <text x="124" y="186" textAnchor="end" fontSize="10" fontFamily="inherit" fill="var(--ink-3)">write</text>
      <text x="164" y="186" fontSize="10" fontFamily="inherit" fill="var(--ink-3)">read</text>
      {/* O */}
      <rect x="276" y="234" width="16" height="48" rx="2" fill="var(--panel)" stroke="var(--line)" strokeWidth="1" />
      <text x="284" y="295" textAnchor="middle" fontSize="11" fontFamily={MONO} fill="var(--ink-2)">O</text>
      <line x1="64" y1="232" x2="64" y2="159" stroke="var(--ink-3)" strokeWidth="1" markerEnd={A} />
      <line x1="284" y1="157" x2="284" y2="230" stroke="var(--ink-3)" strokeWidth="1" markerEnd={A} />

      <text x="162" y="324" textAnchor="middle" fontSize="11" fontFamily="inherit" fill="var(--warn)">
        O(n²) memory and O(n²) HBM traffic
      </text>

      {/* ---------- FlashAttention ---------- */}
      <text x="477" y="24" textAnchor="middle" fontSize="13" fontWeight="600" fontFamily="inherit" fill="var(--ink)">
        FlashAttention
      </text>

      <rect x="335" y="40" width="285" height="115" rx="8" fill="var(--panel)" stroke="var(--line)" strokeWidth="1.5" />
      <text x="347" y="58" fontSize="11" fontFamily="inherit" fill="var(--ink-2)">SRAM (on-chip)</text>
      <Tile x={347} y={70}>Q<Sub>i</Sub></Tile>
      <Tile x={381} y={70}>K<Sub>j</Sub></Tile>
      <Tile x={415} y={70}>V<Sub>j</Sub></Tile>
      <text x="347" y="118" fontSize="11" fontFamily="inherit" fill="var(--ink-2)">online softmax keeps</text>
      <text x="347" y="134" fontSize="11" fontFamily={MONO} fill="var(--ink-2)">
        m, ℓ, O<Sub>i</Sub> per row
      </text>

      {/* the n x n score matrix, only ever seen tile by tile */}
      {Array.from({ length: 16 }, (_, k) => {
        const r = Math.floor(k / 4);
        const c = k % 4;
        const cur = r === 1 && c === 2;
        return (
          <rect
            key={k}
            x={500 + c * 14}
            y={64 + r * 14}
            width="14"
            height="14"
            fill={cur ? "var(--accent)" : "none"}
            stroke={cur ? "var(--accent)" : "var(--line)"}
            strokeWidth="1"
            strokeDasharray={cur ? undefined : "2 2"}
          />
        );
      })}
      <text x="474" y="96" textAnchor="end" fontSize="11" fontFamily={MONO} fill="var(--ink)">
        S<Sub>ij</Sub>
      </text>
      <line x1="478" y1="92" x2="532" y2="85" stroke="var(--ink-3)" strokeWidth="1" />
      <text x="528" y="136" textAnchor="middle" fontSize="11" fontFamily="inherit" fill="var(--ink-3)">n × n scores,</text>
      <text x="528" y="149" textAnchor="middle" fontSize="11" fontFamily="inherit" fill="var(--ink-3)">one tile at a time</text>

      <rect x="335" y="205" width="285" height="95" rx="8" fill="var(--panel)" stroke="var(--line)" strokeWidth="1.5" />
      <text x="347" y="223" fontSize="11" fontFamily="inherit" fill="var(--ink-2)">HBM</text>
      <Blocks x={350} y={232} hi={1} label="Q" />
      <Blocks x={384} y={232} hi={2} label="K" />
      <Blocks x={418} y={232} hi={2} label="V" />
      <g>
        {[0, 1, 2, 3].map((i) => (
          <rect key={i} x="562" y={232 + i * 12} width="24" height="12" fill="var(--panel)" stroke={i === 1 ? "var(--accent)" : "var(--line)"} strokeWidth="1" />
        ))}
        <text x="574" y="296" textAnchor="middle" fontSize="11" fontFamily={MONO} fill="var(--ink-2)">O</text>
      </g>

      <line x1="396" y1="230" x2="396" y2="159" stroke="var(--ink-3)" strokeWidth="1.5" markerEnd={A} />
      <text x="404" y="186" fontSize="11" fontFamily="inherit" fill="var(--ink-2)">load tiles</text>
      <line x1="574" y1="157" x2="574" y2="228" stroke="var(--ink-3)" strokeWidth="1.5" markerEnd={A} />
      <text x="566" y="186" textAnchor="end" fontSize="11" fontFamily="inherit" fill="var(--ink-2)">
        write only O<Sub>i</Sub>
      </text>

      <text x="477" y="324" textAnchor="middle" fontSize="11" fontFamily="inherit" fill="var(--ink-3)">
        O(n) extra memory, exactly the same output
      </text>
    </svg>
  );
}
