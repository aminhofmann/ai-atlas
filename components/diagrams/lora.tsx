// LoRA: frozen W0 plus a trainable low-rank bypass B·A, outputs summed.
import type { ReactNode } from "react";

export const caption =
  "The pretrained weight W₀ stays frozen; only the thin matrices A and B are trained, and their low-rank product is added to W₀'s output.";

const MONO = "var(--font-geist-mono), monospace";
const A = "url(#lora-arrow)";

function Sub({ children }: { children: ReactNode }) {
  return (
    <>
      <tspan dy="3" fontSize="0.75em">{children}</tspan>
      <tspan dy="-3">{"​"}</tspan>
    </>
  );
}

export default function Diagram() {
  return (
    <svg viewBox="0 0 640 352" role="img" aria-labelledby="lora-title" className="w-full h-auto">
      <title id="lora-title">
        Input x goes through a frozen d by k weight W0 and, in parallel, through a trainable low-rank path A (r by k) then B (d by r); the two outputs are added to give h.
      </title>
      <defs>
        <marker id="lora-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0 0 L10 5 L0 10 z" fill="var(--ink-3)" />
        </marker>
      </defs>

      {/* input */}
      <rect x="20" y="146" width="50" height="40" rx="7" fill="var(--panel)" stroke="var(--line)" strokeWidth="1.5" />
      <text x="45" y="171" textAnchor="middle" fontSize="14" fontFamily={MONO} fill="var(--ink)">x</text>
      <text x="45" y="204" textAnchor="middle" fontSize="11" fontFamily={MONO} fill="var(--ink-3)">k</text>
      <path d="M70 166 L100 166 M100 90 L100 243" fill="none" stroke="var(--ink-3)" strokeWidth="1.5" />
      <line x1="100" y1="90" x2="156" y2="90" stroke="var(--ink-3)" strokeWidth="1.5" markerEnd={A} />
      <line x1="100" y1="243" x2="136" y2="243" stroke="var(--ink-3)" strokeWidth="1.5" markerEnd={A} />

      {/* frozen W0: d rows x k columns */}
      <rect x="160" y="30" width="120" height="120" rx="6" fill="var(--panel-2)" stroke="var(--line)" strokeWidth="1.5" />
      <text x="220" y="84" textAnchor="middle" fontSize="15" fontFamily={MONO} fill="var(--ink-2)">
        W<Sub>0</Sub>
      </text>
      <text x="220" y="104" textAnchor="middle" fontSize="11" fontFamily={MONO} fill="var(--ink-3)">d × k</text>
      <text x="220" y="22" textAnchor="middle" fontSize="11" fontFamily="inherit" fill="var(--ink-3)">frozen</text>
      <text x="220" y="168" textAnchor="middle" fontSize="11" fontFamily={MONO} fill="var(--ink-3)">d·k = 16.8M params</text>

      {/* A: r x k (short and wide) */}
      <rect x="140" y="236" width="120" height="14" rx="3" fill="var(--panel)" stroke="var(--accent)" strokeWidth="1.5" />
      <text x="200" y="270" textAnchor="middle" fontSize="12" fontFamily={MONO} fill="var(--ink)">A</text>
      <text x="200" y="285" textAnchor="middle" fontSize="11" fontFamily={MONO} fill="var(--ink-3)">r × k</text>

      <line x1="260" y1="243" x2="316" y2="243" stroke="var(--ink-3)" strokeWidth="1.5" markerEnd={A} />
      <text x="288" y="236" textAnchor="middle" fontSize="11" fontFamily={MONO} fill="var(--ink-2)">r</text>

      {/* B: d x r (tall and thin) */}
      <rect x="320" y="184" width="14" height="120" rx="3" fill="var(--panel)" stroke="var(--accent)" strokeWidth="1.5" />
      <text x="342" y="203" fontSize="12" fontFamily={MONO} fill="var(--ink)">B</text>
      <text x="342" y="218" fontSize="11" fontFamily={MONO} fill="var(--ink-3)">d × r</text>

      {/* the two paths meet */}
      <path d="M280 90 L470 90 L470 134" fill="none" stroke="var(--ink-3)" strokeWidth="1.5" markerEnd={A} />
      <path d="M334 243 L470 243 L470 166" fill="none" stroke="var(--ink-3)" strokeWidth="1.5" markerEnd={A} />
      <text x="375" y="82" textAnchor="middle" fontSize="11" fontFamily={MONO} fill="var(--ink-2)">
        W<Sub>0</Sub>x
      </text>
      <text x="404" y="235" textAnchor="middle" fontSize="11" fontFamily={MONO} fill="var(--ink-2)">(α/r)·BAx</text>

      <circle cx="470" cy="150" r="14" fill="var(--panel)" stroke="var(--line)" strokeWidth="1.5" />
      <text x="470" y="155" textAnchor="middle" fontSize="15" fontFamily="inherit" fill="var(--ink)">+</text>
      <line x1="484" y1="150" x2="526" y2="150" stroke="var(--ink-3)" strokeWidth="1.5" markerEnd={A} />
      <rect x="530" y="130" width="60" height="40" rx="7" fill="var(--panel)" stroke="var(--line)" strokeWidth="1.5" />
      <text x="560" y="155" textAnchor="middle" fontSize="14" fontFamily={MONO} fill="var(--ink)">h</text>
      <text x="560" y="188" textAnchor="middle" fontSize="11" fontFamily={MONO} fill="var(--ink-3)">d</text>

      {/* parameter counts */}
      <text x="20" y="330" fontSize="11" fontFamily="inherit" fill="var(--ink-2)">
        trainable: <tspan fontFamily={MONO}>r(d + k) = 65,536</tspan> vs <tspan fontFamily={MONO}>d·k = 16.8M</tspan> (≈0.4%) for{" "}
        <tspan fontFamily={MONO}>d = k = 4096, r = 8</tspan>
      </text>
      <text x="20" y="345" fontSize="11" fontFamily="inherit" fill="var(--ink-3)">
        B starts at zero, so training begins exactly at the pretrained model
      </text>
    </svg>
  );
}
