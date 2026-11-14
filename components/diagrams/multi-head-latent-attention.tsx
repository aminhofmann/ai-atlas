// MLA: cache one small latent (plus a tiny shared RoPE key) instead of per-head K and V.
import type { ReactNode } from "react";

export const caption =
  "MLA caches only the small latent c_KV and one shared RoPE key per token; per-head keys and values are rebuilt from the latent by up-projections.";

const MONO = "var(--font-geist-mono), monospace";

// subscript / superscript inside an SVG <text>, reset with an opposite dy
function Sub({ children }: { children: ReactNode }) {
  return (
    <>
      <tspan dy="3" fontSize="0.75em">{children}</tspan>
      <tspan dy="-3">{"​"}</tspan>
    </>
  );
}
function Sup({ children }: { children: ReactNode }) {
  return (
    <>
      <tspan dy="-5" fontSize="0.75em">{children}</tspan>
      <tspan dy="5">{"​"}</tspan>
    </>
  );
}

function HeadSlices({ x, y, w, h }: { x: number; y: number; w: number; h: number }) {
  const n = 8;
  return (
    <>
      {Array.from({ length: n - 1 }, (_, i) => (
        <line key={i} x1={x + ((i + 1) * w) / n} y1={y} x2={x + ((i + 1) * w) / n} y2={y + h} stroke="var(--line)" strokeWidth="1" />
      ))}
    </>
  );
}

const A = "url(#multi-head-latent-attention-arrow)";

export default function Diagram() {
  return (
    <svg viewBox="0 0 640 340" role="img" aria-labelledby="multi-head-latent-attention-title" className="w-full h-auto">
      <title id="multi-head-latent-attention-title">
        The hidden state is down-projected to a small latent vector, which is cached and up-projected to per-head keys and values; a separate small RoPE key shared by all heads is also cached. The cache is far smaller than multi-head attention&apos;s.
      </title>
      <defs>
        <marker id="multi-head-latent-attention-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0 0 L10 5 L0 10 z" fill="var(--ink-3)" />
        </marker>
      </defs>

      {/* hidden state */}
      <rect x="20" y="92" width="60" height="44" rx="7" fill="var(--panel)" stroke="var(--line)" strokeWidth="1.5" />
      <text x="50" y="118" textAnchor="middle" fontSize="14" fontFamily={MONO} fill="var(--ink)">
        h<Sub>t</Sub>
      </text>
      <text x="50" y="80" textAnchor="middle" fontSize="11" fontFamily={MONO} fill="var(--ink-3)">d = 5120</text>

      {/* down-projection */}
      <line x1="80" y1="114" x2="178" y2="114" stroke="var(--ink-3)" strokeWidth="1.5" markerEnd={A} />
      <text x="129" y="106" textAnchor="middle" fontSize="11" fontFamily={MONO} fill="var(--ink-2)">
        W<Sup>DKV</Sup>
      </text>
      <text x="129" y="132" textAnchor="middle" fontSize="11" fontFamily="inherit" fill="var(--ink-3)">down</text>

      {/* latent: the cached thing */}
      <rect x="182" y="94" width="80" height="40" rx="7" fill="var(--panel)" stroke="var(--accent)" strokeWidth="1.5" />
      <text x="222" y="119" textAnchor="middle" fontSize="13" fontFamily={MONO} fill="var(--ink)">
        c<Sub>KV</Sub>
      </text>
      <text x="222" y="84" textAnchor="middle" fontSize="11" fontFamily="inherit" fill="var(--ink-2)">cached</text>
      <text x="222" y="150" textAnchor="middle" fontSize="11" fontFamily={MONO} fill="var(--ink-3)">
        d<Sub>c</Sub> = 512
      </text>

      {/* up-projections */}
      <line x1="262" y1="106" x2="386" y2="60" stroke="var(--ink-3)" strokeWidth="1.5" markerEnd={A} />
      <line x1="262" y1="122" x2="386" y2="128" stroke="var(--ink-3)" strokeWidth="1.5" markerEnd={A} />
      <text x="318" y="70" textAnchor="middle" fontSize="11" fontFamily={MONO} fill="var(--ink-2)">
        W<Sup>UK</Sup>
      </text>
      <text x="318" y="144" textAnchor="middle" fontSize="11" fontFamily={MONO} fill="var(--ink-2)">
        W<Sup>UV</Sup>
      </text>

      <rect x="390" y="40" width="128" height="36" rx="7" fill="var(--panel-2)" stroke="var(--line)" strokeWidth="1.5" />
      <HeadSlices x={390} y={40} w={128} h={36} />
      <text x="454" y="32" textAnchor="middle" fontSize="11" fontFamily="inherit" fill="var(--ink-2)">
        keys, per head (no RoPE)
      </text>
      <text x="530" y="63" fontSize="12" fontFamily={MONO} fill="var(--ink)">
        k<Sup>C</Sup>
      </text>

      <rect x="390" y="110" width="128" height="36" rx="7" fill="var(--panel-2)" stroke="var(--line)" strokeWidth="1.5" />
      <HeadSlices x={390} y={110} w={128} h={36} />
      <text x="530" y="133" fontSize="12" fontFamily={MONO} fill="var(--ink)">
        v<Sup>C</Sup>
      </text>
      <text x="454" y="162" textAnchor="middle" fontSize="11" fontFamily={MONO} fill="var(--ink-3)">
        n<Sub>h</Sub> × d<Sub>h</Sub> = 128 × 128 each
      </text>

      {/* decoupled RoPE key */}
      <path d="M50 136 L50 204 L178 204" fill="none" stroke="var(--ink-3)" strokeWidth="1.5" markerEnd={A} />
      <text x="114" y="196" textAnchor="middle" fontSize="11" fontFamily={MONO} fill="var(--ink-2)">
        W<Sup>KR</Sup>
      </text>
      <rect x="182" y="186" width="80" height="36" rx="7" fill="var(--panel)" stroke="var(--line)" strokeWidth="1.5" />
      <text x="222" y="209" textAnchor="middle" fontSize="12" fontFamily="inherit" fill="var(--ink)">RoPE</text>
      <line x1="262" y1="204" x2="386" y2="204" stroke="var(--ink-3)" strokeWidth="1.5" markerEnd={A} />
      <rect x="390" y="188" width="56" height="32" rx="7" fill="var(--panel)" stroke="var(--accent)" strokeWidth="1.5" />
      <text x="418" y="209" textAnchor="middle" fontSize="12" fontFamily={MONO} fill="var(--ink)">
        k<Sup>R</Sup>
      </text>
      <text x="456" y="202" fontSize="11" fontFamily="inherit" fill="var(--ink-2)">cached, shared by all heads</text>
      <text x="456" y="216" fontSize="11" fontFamily={MONO} fill="var(--ink-3)">
        d<Sub>h</Sub><Sup>R</Sup> = 64
      </text>

      {/* cache comparison */}
      <line x1="20" y1="244" x2="620" y2="244" stroke="var(--line)" strokeWidth="1" />
      <text x="20" y="264" fontSize="11" fontFamily="inherit" fill="var(--ink-3)">
        KV cache per token, per layer (DeepSeek-V2 sizes)
      </text>
      <text x="20" y="289" fontSize="12" fontFamily="inherit" fill="var(--ink-2)">MHA</text>
      <rect x="64" y="280" width="400" height="12" rx="3" fill="var(--panel-2)" stroke="var(--line)" strokeWidth="1" />
      <text x="474" y="290" fontSize="11" fontFamily={MONO} fill="var(--ink-2)">
        2·n<Sub>h</Sub>·d<Sub>h</Sub> = 32,768
      </text>
      <text x="20" y="317" fontSize="12" fontFamily="inherit" fill="var(--ink-2)">MLA</text>
      <rect x="64" y="308" width="7" height="12" rx="2" fill="var(--accent)" />
      <text x="80" y="318" fontSize="11" fontFamily={MONO} fill="var(--ink-2)">
        d<Sub>c</Sub> + d<Sub>h</Sub><Sup>R</Sup> = 512 + 64 = 576   (≈57× smaller)
      </text>
    </svg>
  );
}
