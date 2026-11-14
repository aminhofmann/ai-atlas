// AlphaFold 2: MSA and pair representations exchange information in the Evoformer; the structure module emits 3-D coordinates.
export const caption =
  "Inside each of the 48 Evoformer blocks the MSA feeds the pair table (outer product mean) and the pair table biases the MSA's row attention; the structure module then places residues in 3-D, and the result is recycled.";

const MONO = "var(--font-geist-mono), monospace";

const MSA_X = 40;
const MSA_Y = 92;
const MSA_CELL = 15;
const S = 5; // sequences shown
const R = 8; // residues shown

const PAIR_X = 290;
const PAIR_Y = 78;
const PAIR_CELL = 13;

// A small backbone trace for the output.
const TRACE: [number, number][] = [
  [566, 150],
  [584, 132],
  [604, 140],
  [612, 162],
  [596, 178],
  [576, 172],
  [570, 194],
  [590, 206],
];

export default function Diagram() {
  return (
    <svg viewBox="0 0 640 312" role="img" aria-labelledby="alphafold2-title" className="w-full h-auto">
      <title id="alphafold2-title">
        The Evoformer holds an MSA representation of sequences by residues and a pair representation of residues by
        residues. The MSA updates the pair through an outer product mean, and the pair biases the MSA row attention.
        The structure module turns the result into 3-D coordinates, which are recycled back as input.
      </title>
      <defs>
        <marker id="alphafold2-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="var(--ink-3)" />
        </marker>
        <marker id="alphafold2-arrow-accent" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="var(--accent)" />
        </marker>
      </defs>

      <text x={10} y={20} fontFamily="inherit" fontSize={11} fill="var(--ink-3)">
        inputs: target sequence + MSA from database search (+ templates)
      </text>

      {/* Evoformer */}
      <rect x={10} y={32} width={404} height={222} rx={8} fill="var(--panel)" stroke="var(--line)" strokeWidth={1} />
      <text x={24} y={54} fontFamily="inherit" fontSize={13} fill="var(--ink)">Evoformer</text>
      <text x={96} y={54} fontFamily={MONO} fontSize={11} fill="var(--ink-3)">× 48 blocks</text>

      {/* MSA grid */}
      <text x={MSA_X} y={MSA_Y - 6} fontFamily="inherit" fontSize={11} fill="var(--ink-3)">residues →</text>
      <text
        x={MSA_X - 8}
        y={MSA_Y + (S * MSA_CELL) / 2}
        textAnchor="middle"
        fontFamily="inherit"
        fontSize={11}
        fill="var(--ink-3)"
        transform={`rotate(-90 ${MSA_X - 8} ${MSA_Y + (S * MSA_CELL) / 2})`}
      >
        seqs
      </text>
      {Array.from({ length: S }).map((_, i) =>
        Array.from({ length: R }).map((__, j) => (
          <rect
            key={`m${i}-${j}`}
            x={MSA_X + j * MSA_CELL}
            y={MSA_Y + i * MSA_CELL}
            width={MSA_CELL}
            height={MSA_CELL}
            fill={i === 0 ? "var(--panel-2)" : "var(--panel)"}
            stroke="var(--line)"
            strokeWidth={1}
          />
        )),
      )}
      <text x={MSA_X + (R * MSA_CELL) / 2} y={194} textAnchor="middle" fontFamily="inherit" fontSize={12} fill="var(--ink)">
        MSA representation
      </text>
      <text x={MSA_X + (R * MSA_CELL) / 2} y={209} textAnchor="middle" fontFamily={MONO} fontSize={11} fill="var(--ink-3)">
        s × r × c_m
      </text>
      <text x={MSA_X + (R * MSA_CELL) / 2} y={224} textAnchor="middle" fontFamily="inherit" fontSize={11} fill="var(--ink-3)">
        row + column attention
      </text>

      {/* Pair grid */}
      {Array.from({ length: R }).map((_, i) =>
        Array.from({ length: R }).map((__, j) => (
          <rect
            key={`p${i}-${j}`}
            x={PAIR_X + j * PAIR_CELL}
            y={PAIR_Y + i * PAIR_CELL}
            width={PAIR_CELL}
            height={PAIR_CELL}
            fill={i === j ? "var(--panel-2)" : "var(--panel)"}
            stroke="var(--line)"
            strokeWidth={1}
          />
        )),
      )}
      <text x={PAIR_X + (R * PAIR_CELL) / 2} y={PAIR_Y - 6} textAnchor="middle" fontFamily="inherit" fontSize={11} fill="var(--ink-3)">
        residue j →
      </text>
      <text x={PAIR_X + (R * PAIR_CELL) / 2} y={194} textAnchor="middle" fontFamily="inherit" fontSize={12} fill="var(--ink)">
        pair representation
      </text>
      <text x={PAIR_X + (R * PAIR_CELL) / 2} y={209} textAnchor="middle" fontFamily={MONO} fontSize={11} fill="var(--ink-3)">
        r × r × c_z
      </text>
      <text x={PAIR_X + (R * PAIR_CELL) / 2} y={224} textAnchor="middle" fontFamily="inherit" fontSize={11} fill="var(--ink-3)">
        triangle updates
      </text>
      <text x={PAIR_X + (R * PAIR_CELL) / 2} y={238} textAnchor="middle" fontFamily="inherit" fontSize={11} fill="var(--ink-3)">
        + triangle attention
      </text>

      {/* The exchange — what the diagram is about */}
      <line x1={168} y1={110} x2={282} y2={110} stroke="var(--accent)" strokeWidth={1.5} markerEnd="url(#alphafold2-arrow-accent)" />
      <text x={225} y={102} textAnchor="middle" fontFamily="inherit" fontSize={11} fill="var(--ink-2)">
        outer product mean
      </text>
      <line x1={282} y1={150} x2={168} y2={150} stroke="var(--accent)" strokeWidth={1.5} markerEnd="url(#alphafold2-arrow-accent)" />
      <text x={225} y={166} textAnchor="middle" fontFamily="inherit" fontSize={11} fill="var(--ink-2)">
        biases row attention
      </text>

      {/* Structure module */}
      <line x1={416} y1={150} x2={436} y2={150} stroke="var(--ink-3)" strokeWidth={1.25} markerEnd="url(#alphafold2-arrow)" />
      <rect x={438} y={108} width={104} height={84} rx={8} fill="var(--panel-2)" stroke="var(--line)" strokeWidth={1} />
      <text x={490} y={134} textAnchor="middle" fontFamily="inherit" fontSize={12} fill="var(--ink)">Structure</text>
      <text x={490} y={150} textAnchor="middle" fontFamily="inherit" fontSize={12} fill="var(--ink)">module</text>
      <text x={490} y={168} textAnchor="middle" fontFamily="inherit" fontSize={11} fill="var(--ink-3)">8 layers · IPA</text>
      <text x={490} y={182} textAnchor="middle" fontFamily="inherit" fontSize={11} fill="var(--ink-3)">rigid frames</text>
      <line x1={544} y1={150} x2={558} y2={150} stroke="var(--ink-3)" strokeWidth={1.25} markerEnd="url(#alphafold2-arrow)" />

      {/* 3-D output */}
      <polyline
        points={TRACE.map(([x, y]) => `${x},${y}`).join(" ")}
        fill="none"
        stroke="var(--ink-2)"
        strokeWidth={1.5}
        strokeLinejoin="round"
      />
      {TRACE.map(([x, y], i) => (
        <circle key={i} cx={x} cy={y} r={3} fill="var(--panel)" stroke="var(--ink-2)" strokeWidth={1.25} />
      ))}
      <text x={590} y={228} textAnchor="middle" fontFamily="inherit" fontSize={11} fill="var(--ink)">all-atom</text>
      <text x={590} y={242} textAnchor="middle" fontFamily="inherit" fontSize={11} fill="var(--ink)">coordinates</text>

      {/* Recycling */}
      <path
        d="M590,250 V284 H212 V256"
        fill="none"
        stroke="var(--ink-3)"
        strokeWidth={1.25}
        strokeDasharray="4 3"
        markerEnd="url(#alphafold2-arrow)"
      />
      <text x={400} y={300} textAnchor="middle" fontFamily="inherit" fontSize={11} fill="var(--ink-3)">
        recycling: outputs fed back as inputs (3× by default)
      </text>
    </svg>
  );
}
