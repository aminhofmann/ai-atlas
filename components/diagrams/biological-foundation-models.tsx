// Biological foundation models: three modalities, three tokenizations, one pretrain-then-reuse recipe.
export const caption =
  "Cells, proteins and DNA are each turned into a token sequence (outlined) — the hard part differs per modality — then pretrained and reused through their embeddings.";

const MONO = "var(--font-geist-mono), monospace";
const ROW_Y = [44, 136, 228];

type Chip = { text: string; w: number };

const ROWS: {
  model: [string, string];
  tasks: [string, string];
  chips: Chip[];
  tokenNote: string;
}[] = [
  {
    model: ["Transformer", "encoder"],
    tasks: ["cell-type annotation", "in-silico perturbation"],
    chips: [
      { text: "NANOG", w: 44 },
      { text: "SOX2", w: 44 },
      { text: "ACTB", w: 44 },
    ],
    tokenNote: "genes ranked, or gene + bin",
  },
  {
    model: ["Transformer", "enc. or dec."],
    tasks: ["structure (ESMFold)", "variant effects"],
    chips: ["M", "K", "T", "A", "Y", "I", "A"].map((t) => ({ text: t, w: 18 })),
    tokenNote: "1 token per amino acid",
  },
  {
    model: ["Transformer", "or SSM"],
    tasks: ["regulatory elements", "variant scoring"],
    chips: [
      { text: "ATGCGT", w: 46 },
      { text: "TGCGTA", w: 46 },
      { text: "GCGTAC", w: 46 },
    ],
    tokenNote: "k-mers · BPE · single bases",
  },
];

function Arrow({ x1, x2 }: { x1: number; x2: number }) {
  return (
    <line
      x1={x1}
      y1={40}
      x2={x2}
      y2={40}
      stroke="var(--ink-3)"
      strokeWidth={1.25}
      markerEnd="url(#biological-foundation-models-arrow)"
    />
  );
}

export default function Diagram() {
  return (
    <svg viewBox="0 0 640 320" role="img" aria-labelledby="biological-foundation-models-title" className="w-full h-auto">
      <title id="biological-foundation-models-title">
        Three rows: a single cell becomes a ranked list of gene tokens, a protein becomes one token per amino acid,
        and DNA becomes k-mer, BPE or single-base tokens; each sequence feeds a transformer or state-space model whose
        embedding is reused for downstream tasks.
      </title>
      <defs>
        <marker id="biological-foundation-models-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="var(--ink-3)" />
        </marker>
      </defs>

      {/* Column headers */}
      <g fontFamily="inherit" fontSize={11} fill="var(--ink-3)">
        <text x={70} y={30} textAnchor="middle">data</text>
        <text x={240} y={30} textAnchor="middle">tokens</text>
        <text x={396} y={30} textAnchor="middle">pretrained model</text>
        <text x={468} y={30} textAnchor="middle">emb.</text>
        <text x={500} y={30}>reused for</text>
      </g>

      {ROWS.map((row, r) => {
        let cx = 158;
        return (
          <g key={r} transform={`translate(0, ${ROW_Y[r]})`}>
            {/* Data */}
            <rect x={10} y={8} width={120} height={64} rx={8} fill="var(--panel)" stroke="var(--line)" strokeWidth={1} />
            {r === 0 && (
              <g>
                <circle cx={36} cy={40} r={17} fill="var(--panel-2)" stroke="var(--ink-3)" strokeWidth={1} />
                <circle cx={31} cy={35} r={2} fill="var(--ink-3)" />
                <circle cx={41} cy={38} r={2} fill="var(--ink-3)" />
                <circle cx={34} cy={46} r={2} fill="var(--ink-3)" />
                <circle cx={43} cy={47} r={2} fill="var(--ink-3)" />
                <text x={60} y={33} fontFamily="inherit" fontSize={12} fill="var(--ink)">a cell</text>
                <text x={60} y={48} fontFamily="inherit" fontSize={11} fill="var(--ink-3)">~20k genes,</text>
                <text x={60} y={61} fontFamily="inherit" fontSize={11} fill="var(--ink-3)">no order</text>
              </g>
            )}
            {r === 1 && (
              <g>
                <text x={70} y={37} textAnchor="middle" fontFamily={MONO} fontSize={12} fill="var(--ink)">MKTAYIAKQR</text>
                <text x={70} y={56} textAnchor="middle" fontFamily="inherit" fontSize={11} fill="var(--ink-3)">protein residues</text>
              </g>
            )}
            {r === 2 && (
              <g>
                <text x={70} y={37} textAnchor="middle" fontFamily={MONO} fontSize={12} fill="var(--ink)">ATGCGTACGT</text>
                <text x={70} y={56} textAnchor="middle" fontFamily="inherit" fontSize={11} fill="var(--ink-3)">DNA nucleotides</text>
              </g>
            )}

            <Arrow x1={132} x2={148} />

            {/* Tokens — the part that differs per modality */}
            <rect x={150} y={8} width={180} height={64} rx={8} fill="none" stroke="var(--accent)" strokeWidth={1.5} />
            {row.chips.map((c, i) => {
              const x = cx;
              cx += c.w + (c.w < 30 ? 4 : 6);
              return (
                <g key={i}>
                  <rect x={x} y={16} width={c.w} height={22} rx={5} fill="var(--panel-2)" stroke="var(--line)" strokeWidth={1} />
                  <text x={x + c.w / 2} y={31} textAnchor="middle" fontFamily={MONO} fontSize={11} fill="var(--ink)">
                    {c.text}
                  </text>
                </g>
              );
            })}
            <text x={cx + 4} y={31} fontFamily={MONO} fontSize={11} fill="var(--ink-3)">…</text>
            <text x={240} y={60} textAnchor="middle" fontFamily="inherit" fontSize={11} fill="var(--ink-2)">
              {row.tokenNote}
            </text>

            <Arrow x1={332} x2={348} />

            {/* Model */}
            <rect x={350} y={16} width={92} height={48} rx={8} fill="var(--panel-2)" stroke="var(--line)" strokeWidth={1} />
            <text x={396} y={36} textAnchor="middle" fontFamily="inherit" fontSize={12} fill="var(--ink)">{row.model[0]}</text>
            <text x={396} y={52} textAnchor="middle" fontFamily="inherit" fontSize={11} fill="var(--ink-2)">{row.model[1]}</text>

            <Arrow x1={444} x2={460} />

            {/* Embedding vector */}
            {[0, 1, 2, 3].map((k) => (
              <rect key={k} x={463} y={20 + k * 10} width={10} height={10} fill="var(--panel)" stroke="var(--ink-3)" strokeWidth={1} />
            ))}

            <Arrow x1={478} x2={494} />

            {/* Downstream */}
            <text x={500} y={36} fontFamily="inherit" fontSize={11} fill="var(--ink-2)">{row.tasks[0]}</text>
            <text x={500} y={52} fontFamily="inherit" fontSize={11} fill="var(--ink-2)">{row.tasks[1]}</text>
          </g>
        );
      })}
    </svg>
  );
}
