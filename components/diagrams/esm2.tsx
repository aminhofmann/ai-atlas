// ESM-2: one token per amino acid -> masked-LM transformer encoder -> per-residue embeddings; attention ≈ contact map.
export const caption =
  "Trained only to fill in masked amino acids, ESM-2's attention maps — read out by a linear probe — recover which residues touch in 3-D.";

const MONO = "var(--font-geist-mono), monospace";

const TOKENS = ["<cls>", "M", "K", "T", "<mask>", "Y", "I", "A", "K", "<eos>"];
const tokenW = (t: string) => (t.startsWith("<") ? 44 : 22);
const POS: { t: string; x: number; w: number }[] = [];
{
  let x = 16;
  for (const t of TOKENS) {
    const w = tokenW(t);
    POS.push({ t, x, w });
    x += w + 4;
  }
}

// Contact map over the 8 residues of MKTAYIAK (illustrative hairpin-like pattern).
const RESIDUES = ["M", "K", "T", "A", "Y", "I", "A", "K"];
const CONTACTS = new Set(["0-7", "1-6", "2-5"]);
const CELL = 19;
const MAP_X = 440;
const MAP_Y = 70;

export default function Diagram() {
  return (
    <svg viewBox="0 0 640 290" role="img" aria-labelledby="esm2-title" className="w-full h-auto">
      <title id="esm2-title">
        A protein sequence is tokenized one amino acid per token, with one position masked; a transformer encoder
        predicts the masked residue and outputs one embedding per residue. Its attention maps approximate the
        residue–residue contact map.
      </title>
      <defs>
        <marker id="esm2-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="var(--ink-3)" />
        </marker>
      </defs>

      {/* Tokens */}
      <text x={16} y={26} fontFamily="inherit" fontSize={12} fill="var(--ink)">
        sequence → tokens (one per amino acid)
      </text>
      {POS.map((p, i) => {
        const masked = p.t === "<mask>";
        return (
          <g key={i}>
            <rect
              x={p.x}
              y={36}
              width={p.w}
              height={22}
              rx={5}
              fill="var(--panel-2)"
              stroke={masked ? "var(--warn)" : "var(--line)"}
              strokeWidth={1}
            />
            <text x={p.x + p.w / 2} y={51} textAnchor="middle" fontFamily={MONO} fontSize={11} fill={masked ? "var(--warn)" : "var(--ink)"}>
              {p.t}
            </text>
          </g>
        );
      })}

      <line x1={176} y1={62} x2={176} y2={78} stroke="var(--ink-3)" strokeWidth={1.25} markerEnd="url(#esm2-arrow)" />

      {/* Encoder */}
      <rect x={16} y={82} width={326} height={58} rx={8} fill="var(--panel-2)" stroke="var(--line)" strokeWidth={1} />
      <text x={179} y={106} textAnchor="middle" fontFamily="inherit" fontSize={13} fill="var(--ink)">
        Transformer encoder (masked LM)
      </text>
      <text x={179} y={124} textAnchor="middle" fontFamily="inherit" fontSize={11} fill="var(--ink-3)">
        pre-LayerNorm · rotary positions · 6–48 layers
      </text>

      {/* Per-residue embeddings */}
      {POS.map((p, i) => {
        const cx = p.x + p.w / 2;
        return (
          <g key={i}>
            <line x1={cx} y1={142} x2={cx} y2={154} stroke="var(--ink-3)" strokeWidth={1} />
            {[0, 1, 2].map((k) => (
              <rect key={k} x={cx - 5} y={156 + k * 10} width={10} height={10} fill="var(--panel)" stroke="var(--ink-3)" strokeWidth={1} />
            ))}
          </g>
        );
      })}
      {(() => {
        const m = POS[4];
        const cx = m.x + m.w / 2;
        return (
          <text x={cx} y={202} textAnchor="middle" fontFamily={MONO} fontSize={11} fill="var(--warn)">
            → A?
          </text>
        );
      })()}
      <text x={16} y={226} fontFamily="inherit" fontSize={12} fill="var(--ink)">per-residue embeddings</text>
      <text x={16} y={242} fontFamily={MONO} fontSize={11} fill="var(--ink-3)">(L + 2) × d, e.g. d = 1280 (650M)</text>
      <text x={16} y={262} fontFamily="inherit" fontSize={11} fill="var(--ink-3)">
        mean-pool over residues → protein embedding
      </text>

      {/* Attention -> contacts */}
      <text x={390} y={104} textAnchor="middle" fontFamily="inherit" fontSize={11} fill="var(--ink-3)">attention</text>
      <text x={390} y={117} textAnchor="middle" fontFamily="inherit" fontSize={11} fill="var(--ink-3)">+ probe</text>
      <line x1={346} y1={126} x2={424} y2={126} stroke="var(--ink-3)" strokeWidth={1.25} markerEnd="url(#esm2-arrow)" />

      <text x={MAP_X + (CELL * 8) / 2} y={30} textAnchor="middle" fontFamily="inherit" fontSize={12} fill="var(--accent)">
        attention ≈ contact map (L × L)
      </text>
      {RESIDUES.map((r, i) => (
        <g key={i}>
          <text x={MAP_X + i * CELL + CELL / 2} y={MAP_Y - 6} textAnchor="middle" fontFamily={MONO} fontSize={11} fill="var(--ink-3)">
            {r}
          </text>
          <text x={MAP_X - 6} y={MAP_Y + i * CELL + CELL / 2 + 4} textAnchor="end" fontFamily={MONO} fontSize={11} fill="var(--ink-3)">
            {r}
          </text>
        </g>
      ))}
      {RESIDUES.map((_, i) =>
        RESIDUES.map((__, j) => {
          const contact = CONTACTS.has(`${Math.min(i, j)}-${Math.max(i, j)}`);
          const near = Math.abs(i - j) <= 1;
          return (
            <rect
              key={`${i}-${j}`}
              x={MAP_X + j * CELL}
              y={MAP_Y + i * CELL}
              width={CELL}
              height={CELL}
              fill={contact ? "var(--accent)" : near ? "var(--panel-2)" : "var(--panel)"}
              stroke="var(--line)"
              strokeWidth={1}
            />
          );
        }),
      )}
      <text x={MAP_X + (CELL * 8) / 2} y={MAP_Y + CELL * 8 + 20} textAnchor="middle" fontFamily="inherit" fontSize={11} fill="var(--ink-2)">
        filled: residues i, j close in 3-D
      </text>
      <text x={MAP_X + (CELL * 8) / 2} y={MAP_Y + CELL * 8 + 36} textAnchor="middle" fontFamily="inherit" fontSize={11} fill="var(--ink-3)">
        no structures seen in training
      </text>
    </svg>
  );
}
