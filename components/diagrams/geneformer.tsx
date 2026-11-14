// Geneformer: rank-value encoding of a cell -> masked-gene BERT encoder -> embeddings; in-silico perturbation below.
export const caption =
  "Dividing by each gene's corpus median lifts distinctive genes (NANOG, SOX2) above housekeeping ones; the ranked gene ids, not the values, are the cell's sentence.";

const MONO = "var(--font-geist-mono), monospace";
const ROW_Y = [50, 74, 98, 122];

// Illustrative numbers (cf. the page's code example): raw counts vs. counts ÷ corpus median.
const RAW = [
  { g: "GAPDH", w: 64 },
  { g: "NANOG", w: 3 },
  { g: "ACTB", w: 57 },
  { g: "SOX2", w: 4 },
];
const RANKED = [
  { g: "NANOG", w: 60 },
  { g: "SOX2", w: 50 },
  { g: "GAPDH", w: 18 },
  { g: "ACTB", w: 18 },
];
const TOKENS = ["NANOG", "SOX2", "[MASK]", "ACTB"];

function Chip({ x, y, text, kind }: { x: number; y: number; text: string; kind?: "mask" | "deleted" }) {
  const warn = kind === "mask" || kind === "deleted";
  return (
    <g>
      <rect
        x={x}
        y={y}
        width={50}
        height={18}
        rx={5}
        fill="var(--panel-2)"
        stroke={warn ? "var(--warn)" : "var(--line)"}
        strokeWidth={1}
        strokeDasharray={kind === "deleted" ? "3 2" : undefined}
      />
      <text x={x + 25} y={y + 13} textAnchor="middle" fontFamily={MONO} fontSize={11} fill={warn ? "var(--warn)" : "var(--ink)"}>
        {text}
      </text>
      {kind === "deleted" && <line x1={x + 6} y1={y + 9} x2={x + 44} y2={y + 9} stroke="var(--warn)" strokeWidth={1.25} />}
    </g>
  );
}

export default function Diagram() {
  return (
    <svg viewBox="0 0 640 330" role="img" aria-labelledby="geneformer-title" className="w-full h-auto">
      <title id="geneformer-title">
        A cell&apos;s gene counts are normalised, divided by each gene&apos;s corpus median and sorted; the gene ids in
        rank order, with about 15 percent masked, feed a BERT-style encoder that predicts the masked genes and yields
        cell and gene embeddings. Below, an in-silico knockout deletes a gene token and measures how the cell embedding
        shifts.
      </title>
      <defs>
        <marker id="geneformer-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="var(--ink-3)" />
        </marker>
      </defs>

      {/* 1. Expression profile */}
      <text x={10} y={32} fontFamily="inherit" fontSize={12} fill="var(--ink)">counts ÷ total</text>
      <rect x={10} y={40} width={124} height={108} rx={8} fill="var(--panel)" stroke="var(--line)" strokeWidth={1} />
      {RAW.map((d, i) => (
        <g key={d.g}>
          <text x={16} y={ROW_Y[i] + 10} fontFamily={MONO} fontSize={11} fill="var(--ink-2)">{d.g}</text>
          <rect x={62} y={ROW_Y[i]} width={d.w} height={12} rx={2} fill="var(--ink-3)" />
        </g>
      ))}
      <text x={72} y={166} textAnchor="middle" fontFamily="inherit" fontSize={11} fill="var(--ink-3)">no natural order</text>

      <line x1={136} y1={94} x2={150} y2={94} stroke="var(--ink-3)" strokeWidth={1.25} markerEnd="url(#geneformer-arrow)" />

      {/* 2. Rank-value encoding (the idea) */}
      <text x={152} y={32} fontFamily="inherit" fontSize={12} fill="var(--accent)">÷ median m_g, sort ↓</text>
      <rect x={152} y={40} width={124} height={108} rx={8} fill="var(--panel)" stroke="var(--accent)" strokeWidth={1.5} />
      {RANKED.map((d, i) => (
        <g key={d.g}>
          <text x={158} y={ROW_Y[i] + 10} fontFamily={MONO} fontSize={11} fill="var(--ink-2)">{d.g}</text>
          <rect x={204} y={ROW_Y[i]} width={d.w} height={12} rx={2} fill="var(--ink-3)" />
        </g>
      ))}
      <text x={214} y={166} textAnchor="middle" fontFamily="inherit" fontSize={11} fill="var(--ink-3)">m_g: corpus median</text>

      <line x1={278} y1={94} x2={292} y2={94} stroke="var(--ink-3)" strokeWidth={1.25} markerEnd="url(#geneformer-arrow)" />

      {/* 3. Gene tokens in rank order */}
      <text x={294} y={32} fontFamily="inherit" fontSize={12} fill="var(--ink)">gene tokens</text>
      {TOKENS.map((t, i) => (
        <Chip key={t} x={296} y={ROW_Y[i] - 3} text={t} kind={t === "[MASK]" ? "mask" : undefined} />
      ))}
      <text x={321} y={150} textAnchor="middle" fontFamily={MONO} fontSize={11} fill="var(--ink-3)">… ≤ 2,048</text>
      <text x={321} y={166} textAnchor="middle" fontFamily="inherit" fontSize={11} fill="var(--ink-3)">rank = position</text>

      <line x1={350} y1={94} x2={366} y2={94} stroke="var(--ink-3)" strokeWidth={1.25} markerEnd="url(#geneformer-arrow)" />

      {/* 4. Encoder */}
      <rect x={368} y={48} width={104} height={92} rx={8} fill="var(--panel-2)" stroke="var(--line)" strokeWidth={1} />
      <text x={420} y={80} textAnchor="middle" fontFamily="inherit" fontSize={12} fill="var(--ink)">BERT-style</text>
      <text x={420} y={96} textAnchor="middle" fontFamily="inherit" fontSize={12} fill="var(--ink)">encoder</text>
      <text x={420} y={116} textAnchor="middle" fontFamily="inherit" fontSize={11} fill="var(--ink-3)">6 layers · d 256</text>

      <line x1={474} y1={94} x2={490} y2={94} stroke="var(--ink-3)" strokeWidth={1.25} markerEnd="url(#geneformer-arrow)" />

      {/* 5. Outputs */}
      <text x={494} y={58} fontFamily="inherit" fontSize={11} fill="var(--ink-3)">pretraining target</text>
      <text x={494} y={74} fontFamily={MONO} fontSize={11} fill="var(--warn)">[MASK]</text>
      <text x={540} y={74} fontFamily={MONO} fontSize={11} fill="var(--ink)">= GAPDH</text>
      <text x={494} y={102} fontFamily="inherit" fontSize={12} fill="var(--ink)">cell embedding</text>
      {[0, 1, 2, 3, 4].map((k) => (
        <rect key={k} x={494 + k * 11} y={108} width={11} height={10} fill="var(--panel)" stroke="var(--ink-3)" strokeWidth={1} />
      ))}
      <text x={556} y={117} fontFamily="inherit" fontSize={11} fill="var(--ink-3)">pooled</text>
      <text x={494} y={140} fontFamily="inherit" fontSize={12} fill="var(--ink)">gene embeddings</text>
      <text x={494} y={154} fontFamily="inherit" fontSize={11} fill="var(--ink-3)">one per token</text>

      {/* In-silico perturbation */}
      <rect x={10} y={190} width={620} height={128} rx={8} fill="var(--panel)" stroke="var(--line)" strokeWidth={1} />
      <text x={24} y={212} fontFamily="inherit" fontSize={13} fill="var(--ink)">In-silico perturbation</text>

      <text x={24} y={245} fontFamily="inherit" fontSize={11} fill="var(--ink-3)">original</text>
      {["NANOG", "SOX2", "GAPDH", "ACTB"].map((g, i) => (
        <Chip key={g} x={90 + i * 56} y={232} text={g} />
      ))}
      <text x={24} y={277} fontFamily="inherit" fontSize={11} fill="var(--ink-3)">knockout</text>
      {["NANOG", "SOX2", "GAPDH", "ACTB"].map((g, i) => (
        <Chip key={g} x={90 + i * 56} y={264} text={g} kind={g === "SOX2" ? "deleted" : undefined} />
      ))}

      <line x1={316} y1={241} x2={352} y2={241} stroke="var(--ink-3)" strokeWidth={1.25} markerEnd="url(#geneformer-arrow)" />
      <line x1={316} y1={273} x2={352} y2={273} stroke="var(--ink-3)" strokeWidth={1.25} markerEnd="url(#geneformer-arrow)" />
      <text x={334} y={259} textAnchor="middle" fontFamily="inherit" fontSize={11} fill="var(--ink-3)">encode</text>
      <text x={358} y={245} fontFamily={MONO} fontSize={12} fill="var(--ink)">e</text>
      <text x={358} y={277} fontFamily={MONO} fontSize={12} fill="var(--ink)">e′</text>

      <path d="M374,241 H384 V273 H374" fill="none" stroke="var(--ink-3)" strokeWidth={1} />
      <line x1={384} y1={257} x2={400} y2={257} stroke="var(--ink-3)" strokeWidth={1.25} markerEnd="url(#geneformer-arrow)" />
      <text x={406} y={253} fontFamily="inherit" fontSize={12} fill="var(--ink)">embedding shift (cosine)</text>
      <text x={406} y={269} fontFamily="inherit" fontSize={11} fill="var(--ink-3)">toward or away from disease state</text>

      <text x={24} y={304} fontFamily="inherit" fontSize={11} fill="var(--ink-3)">
        overexpression instead: move the gene token to the front, re-embed, compare
      </text>
    </svg>
  );
}
