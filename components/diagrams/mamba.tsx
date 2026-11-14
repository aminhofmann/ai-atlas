// Mamba: a recurrent state whose update rules (Δ, B, C) are computed from each input token.
import type { ReactNode } from "react";

export const caption =
  "A fixed-size state h is carried along the sequence; at every step the token itself sets Δ, B and C — how fast to forget, what to write and what to read out.";

const MONO = "var(--font-geist-mono), monospace";
const A = "url(#mamba-arrow)";

function Sub({ children }: { children: ReactNode }) {
  return (
    <>
      <tspan dy="3" fontSize="0.75em">{children}</tspan>
      <tspan dy="-3">{"​"}</tspan>
    </>
  );
}

const STEPS = [
  { cx: 150, s: "1" },
  { cx: 260, s: "2" },
  { cx: 370, s: "3" },
  { cx: 540, s: "t" },
];
const HY = 74;
const HW = 56;
const HH = 32;

function Selected({ y, label }: { y: number; label: string }) {
  return (
    <g>
      <rect x="140" y={y} width="56" height="26" rx="6" fill="var(--panel)" stroke="var(--accent)" strokeWidth="1.5" />
      <text x="168" y={y + 17} textAnchor="middle" fontSize="12" fontFamily={MONO} fill="var(--ink)">
        {label}
        <Sub>t</Sub>
      </text>
    </g>
  );
}

export default function Diagram() {
  return (
    <svg viewBox="0 0 640 340" role="img" aria-labelledby="mamba-title" className="w-full h-auto">
      <title id="mamba-title">
        Inputs x1 to xt update a recurrent state h step by step and each step emits an output y. Inside one step, Delta, B and C are computed from the input token, giving the selective state update.
      </title>
      <defs>
        <marker id="mamba-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0 0 L10 5 L0 10 z" fill="var(--ink-3)" />
        </marker>
      </defs>

      {/* ---------- the recurrence over the sequence ---------- */}
      <text x="50" y={HY + 21} textAnchor="middle" fontSize="13" fontFamily={MONO} fill="var(--ink-2)">
        h<Sub>0</Sub>
      </text>
      <line x1="66" y1={HY + HH / 2} x2={STEPS[0].cx - HW / 2 - 2} y2={HY + HH / 2} stroke="var(--ink-3)" strokeWidth="1.5" markerEnd={A} />

      {STEPS.map(({ cx, s }, i) => (
        <g key={s}>
          <rect x={cx - HW / 2} y={HY} width={HW} height={HH} rx="7" fill="var(--panel-2)" stroke="var(--line)" strokeWidth="1.5" />
          <text x={cx} y={HY + 21} textAnchor="middle" fontSize="13" fontFamily={MONO} fill="var(--ink)">
            h<Sub>{s}</Sub>
          </text>
          {/* input up into the state */}
          <text x={cx} y="160" textAnchor="middle" fontSize="13" fontFamily={MONO} fill="var(--ink)">
            x<Sub>{s}</Sub>
          </text>
          <line x1={cx} y1="146" x2={cx} y2={HY + HH + 3} stroke="var(--ink-3)" strokeWidth="1.5" markerEnd={A} />
          {/* output */}
          <line x1={cx} y1={HY} x2={cx} y2="40" stroke="var(--ink-3)" strokeWidth="1.5" markerEnd={A} />
          <text x={cx} y="30" textAnchor="middle" fontSize="13" fontFamily={MONO} fill="var(--ink)">
            y<Sub>{s}</Sub>
          </text>
          {i < 2 && (
            <line
              x1={cx + HW / 2}
              y1={HY + HH / 2}
              x2={STEPS[i + 1].cx - HW / 2 - 2}
              y2={HY + HH / 2}
              stroke="var(--ink-3)"
              strokeWidth="1.5"
              markerEnd={A}
            />
          )}
        </g>
      ))}
      <line x1="398" y1={HY + HH / 2} x2="432" y2={HY + HH / 2} stroke="var(--ink-3)" strokeWidth="1.5" />
      <text x="455" y={HY + 21} textAnchor="middle" fontSize="14" fontFamily="inherit" fill="var(--ink-3)">…</text>
      <line x1="476" y1={HY + HH / 2} x2="510" y2={HY + HH / 2} stroke="var(--ink-3)" strokeWidth="1.5" markerEnd={A} />
      <line x1="568" y1={HY + HH / 2} x2="612" y2={HY + HH / 2} stroke="var(--ink-3)" strokeWidth="1.5" markerEnd={A} />
      <text x="205" y={HY - 6} textAnchor="middle" fontSize="11" fontFamily="inherit" fill="var(--ink-3)">fixed-size state</text>

      {/* step t is zoomed below */}
      <rect x="500" y="10" width="80" height="160" rx="8" fill="none" stroke="var(--ink-3)" strokeWidth="1" strokeDasharray="4 3" />
      <line x1="540" y1="170" x2="540" y2="188" stroke="var(--ink-3)" strokeWidth="1" strokeDasharray="4 3" />

      {/* ---------- one step, zoomed ---------- */}
      <rect x="20" y="190" width="600" height="140" rx="8" fill="var(--panel)" stroke="var(--line)" strokeWidth="1.5" />
      <text x="34" y="208" fontSize="11" fontFamily="inherit" fill="var(--ink-3)">inside step t</text>

      <rect x="36" y="238" width="52" height="32" rx="7" fill="var(--panel-2)" stroke="var(--line)" strokeWidth="1.5" />
      <text x="62" y="259" textAnchor="middle" fontSize="13" fontFamily={MONO} fill="var(--ink)">
        x<Sub>t</Sub>
      </text>
      <text x="62" y="288" textAnchor="middle" fontSize="11" fontFamily="inherit" fill="var(--ink-3)">Linear</text>

      {[217, 254, 291].map((y) => (
        <line key={y} x1="88" y1="254" x2="137" y2={y} stroke="var(--ink-3)" strokeWidth="1" markerEnd={A} />
      ))}
      <Selected y={204} label="Δ" />
      <Selected y={241} label="B" />
      <Selected y={278} label="C" />

      {/* discretisation */}
      <text x="214" y="222" fontSize="12" fontFamily={MONO} fill="var(--ink)">
        Ā<Sub>t</Sub> = exp(Δ<Sub>t</Sub> A)
      </text>
      <text x="214" y="236" fontSize="11" fontFamily="inherit" fill="var(--ink-3)">A itself is fixed, not from x</text>
      <text x="214" y="259" fontSize="12" fontFamily={MONO} fill="var(--ink)">
        B̄<Sub>t</Sub> ≈ Δ<Sub>t</Sub> B<Sub>t</Sub>
      </text>

      {/* the update */}
      <line x1="340" y1="236" x2="384" y2="244" stroke="var(--ink-3)" strokeWidth="1" markerEnd={A} />
      <line x1="320" y1="256" x2="384" y2="250" stroke="var(--ink-3)" strokeWidth="1" markerEnd={A} />
      <line x1="196" y1="291" x2="384" y2="291" stroke="var(--ink-3)" strokeWidth="1" markerEnd={A} />
      <text x="390" y="252" fontSize="13" fontFamily={MONO} fill="var(--ink)">
        h<Sub>t</Sub> = Ā<Sub>t</Sub> ⊙ h<Sub>t−1</Sub> + B̄<Sub>t</Sub> x<Sub>t</Sub>
      </text>
      <text x="390" y="295" fontSize="13" fontFamily={MONO} fill="var(--ink)">
        y<Sub>t</Sub> = C<Sub>t</Sub> h<Sub>t</Sub>
      </text>

      <text x="36" y="322" fontSize="11" fontFamily="inherit" fill="var(--ink-2)">
        Δ, B, C come from the token itself (selective): large Δ resets the state, small Δ keeps it
      </text>
    </svg>
  );
}
