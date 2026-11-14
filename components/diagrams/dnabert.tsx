// DNABERT vs DNABERT-2: the same 14 bp tokenized as overlapping 6-mers vs non-overlapping BPE tokens.
export const caption =
  "Overlapping 6-mers put every base into up to six tokens (9 tokens for 14 bp); DNABERT-2's learned BPE tokens tile the sequence without overlap, so it needs far fewer.";

const MONO = "var(--font-geist-mono), monospace";

const SEQ = "ATGCGTACGTTAGC";
const K = 6;
const KMERS = Array.from({ length: SEQ.length - K + 1 }, (_, i) => SEQ.slice(i, i + K));
// Illustrative BPE split — real DNABERT-2 vocabularies are learned from genomes.
const BPE = ["ATGCG", "TAC", "GTTAGC"];

const X0 = 120;
const BASE = 24;
const DNA_Y = 36;
const KMER_Y = 76;
const KMER_STEP = 15;
const BPE_Y = 232;

// Start offset (in bases) of each BPE token: the running sum of the lengths before it.
const BPE_STARTS = BPE.map((_, i) => BPE.slice(0, i).reduce((s, t) => s + t.length, 0));

export default function Diagram() {
  return (
    <svg viewBox="0 0 640 308" role="img" aria-labelledby="dnabert-title" className="w-full h-auto">
      <title id="dnabert-title">
        The 14-base sequence ATGCGTACGTTAGC tokenized two ways: DNABERT&apos;s overlapping 6-mers with stride 1 give 9
        tokens, while DNABERT-2&apos;s byte-pair encoding gives a few variable-length, non-overlapping tokens.
      </title>

      {/* DNA */}
      <text x={10} y={DNA_Y + 15} fontFamily="inherit" fontSize={12} fill="var(--ink)">DNA, 14 bp</text>
      {SEQ.split("").map((b, i) => (
        <g key={i}>
          <rect x={X0 + i * BASE} y={DNA_Y} width={BASE} height={22} fill="var(--panel-2)" stroke="var(--line)" strokeWidth={1} />
          <text x={X0 + i * BASE + BASE / 2} y={DNA_Y + 15} textAnchor="middle" fontFamily={MONO} fontSize={12} fill="var(--ink)">
            {b}
          </text>
        </g>
      ))}

      {/* DNABERT: overlapping k-mers */}
      <text x={10} y={KMER_Y + 50} fontFamily="inherit" fontSize={12} fill="var(--ink)">DNABERT</text>
      <text x={10} y={KMER_Y + 66} fontFamily="inherit" fontSize={11} fill="var(--ink-3)">6-mers,</text>
      <text x={10} y={KMER_Y + 80} fontFamily="inherit" fontSize={11} fill="var(--ink-3)">stride 1</text>
      {KMERS.map((km, i) => {
        const x = X0 + i * BASE;
        const y = KMER_Y + i * KMER_STEP;
        return (
          <g key={i}>
            <rect x={x + 1} y={y} width={K * BASE - 2} height={13} rx={4} fill="var(--panel)" stroke="var(--line)" strokeWidth={1} />
            <text x={x + (K * BASE) / 2} y={y + 10} textAnchor="middle" fontFamily={MONO} fontSize={11} fill="var(--ink-2)">
              {km}
            </text>
          </g>
        );
      })}
      <text x={476} y={KMER_Y + 52} fontFamily="inherit" fontSize={14} fill="var(--accent)">
        {KMERS.length} tokens
      </text>
      <text x={476} y={KMER_Y + 70} fontFamily={MONO} fontSize={11} fill="var(--ink-3)">14 − 6 + 1</text>
      <text x={476} y={KMER_Y + 88} fontFamily="inherit" fontSize={11} fill="var(--ink-3)">each base in ≤ 6 tokens;</text>
      <text x={476} y={KMER_Y + 102} fontFamily="inherit" fontSize={11} fill="var(--ink-3)">masked in 6-token spans</text>

      <line x1={10} y1={BPE_Y - 14} x2={630} y2={BPE_Y - 14} stroke="var(--line)" strokeWidth={1} />

      {/* DNABERT-2: BPE */}
      <text x={10} y={BPE_Y + 10} fontFamily="inherit" fontSize={12} fill="var(--ink)">DNABERT-2</text>
      <text x={10} y={BPE_Y + 25} fontFamily="inherit" fontSize={11} fill="var(--ink-3)">BPE, learned</text>
      {BPE.map((t, i) => {
        const x = X0 + BPE_STARTS[i] * BASE;
        return (
          <g key={t}>
            <rect x={x + 1} y={BPE_Y} width={t.length * BASE - 2} height={22} rx={5} fill="var(--panel)" stroke="var(--ink-2)" strokeWidth={1} />
            <text x={x + (t.length * BASE) / 2} y={BPE_Y + 15} textAnchor="middle" fontFamily={MONO} fontSize={12} fill="var(--ink)">
              {t}
            </text>
          </g>
        );
      })}
      <text x={476} y={BPE_Y + 15} fontFamily="inherit" fontSize={14} fill="var(--accent)">
        {BPE.length} tokens
      </text>
      <text x={476} y={BPE_Y + 31} fontFamily="inherit" fontSize={11} fill="var(--ink-3)">variable length,</text>
      <text x={476} y={BPE_Y + 45} fontFamily="inherit" fontSize={11} fill="var(--ink-3)">no overlap</text>

      <text x={10} y={298} fontFamily="inherit" fontSize={11} fill="var(--ink-3)">
        BPE split shown is illustrative; the real vocabulary is learned from multi-species genomes.
      </text>
    </svg>
  );
}
