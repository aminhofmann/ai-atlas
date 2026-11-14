"use client";

import { useMemo, useState } from "react";
import { neighbourhood, ROLE_LEFT, ROLE_MIDDLE, ROLE_RIGHT } from "@/lib/neighbourhood";
import type { ConceptMeta, Edge, LinkType } from "@/lib/types";
import { categoryColor } from "./category";

type Zone = "left" | "right" | "middle" | "none";

const ZONES: { key: Zone; label: string; hint: string }[] = [
  { key: "left", label: "It builds on", hint: "prerequisites, foundations, what it's a variant of" },
  { key: "right", label: "Built on it", hint: "what needs it, uses it, extends it" },
  { key: "middle", label: "Alternative / related", hint: "side by side, neither builds on the other" },
  { key: "none", label: "Not directly connected", hint: "no direct link in the atlas" },
];

/** What the neighbour is to the centre, in the words of the zone it belongs in. */
const roleOf = (item: Item) =>
  !item.type
    ? "not directly connected"
    : ((item.answer === "left" ? ROLE_LEFT : item.answer === "right" ? ROLE_RIGHT : ROLE_MIDDLE)[item.type] ?? item.type);

/** Deterministic shuffle (seeded by concept + round), so server and client render the same order. */
function seeded(seed: string) {
  let h = 2166136261;
  for (const ch of seed) h = Math.imul(h ^ ch.charCodeAt(0), 16777619);
  return () => {
    h = Math.imul(h ^ (h >>> 15), 2246822507);
    h = Math.imul(h ^ (h >>> 13), 3266489909);
    return ((h ^= h >>> 16) >>> 0) / 4294967296;
  };
}
function shuffle<T>(xs: T[], rand: () => number) {
  const a = [...xs];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

type Item = { id: string; answer: Zone; type?: LinkType };

function buildRound(center: string, nodes: ConceptMeta[], edges: Edge[], round: number): Item[] {
  const rand = seeded(`${center}:${round}`);
  const hood = neighbourhood(center, edges);
  const items: Item[] = [
    ...hood.left.map((n) => ({ id: n.id, answer: "left" as Zone, type: n.type })),
    ...hood.right.map((n) => ({ id: n.id, answer: "right" as Zone, type: n.type })),
    ...hood.middle.map((n) => ({ id: n.id, answer: "middle" as Zone, type: n.type })),
  ];
  // Two distractors with NO direct edge to the centre — same topic if possible, so they're plausible.
  // At least three hops away: "not connected" must not be a near-miss trap (e.g. Attention for MHA).
  const adj = new Map<string, Set<string>>();
  for (const e of edges) {
    if (!adj.has(e.from)) adj.set(e.from, new Set());
    if (!adj.has(e.to)) adj.set(e.to, new Set());
    adj.get(e.from)!.add(e.to);
    adj.get(e.to)!.add(e.from);
  }
  const linked = new Set<string>([center]);
  let frontier = [center];
  for (let hop = 0; hop < 2; hop++) {
    frontier = frontier.flatMap((id) => [...(adj.get(id) ?? [])].filter((n) => !linked.has(n)));
    frontier.forEach((n) => linked.add(n));
  }
  const cat = nodes.find((n) => n.id === center)?.category;
  const pool = nodes.filter((n) => !linked.has(n.id));
  const sameTopic = shuffle(pool.filter((n) => n.category === cat), rand);
  const other = shuffle(pool.filter((n) => n.category !== cat), rand);
  for (const d of [...sameTopic, ...other].slice(0, 2)) items.push({ id: d.id, answer: "none" });
  return shuffle(items, rand);
}

/**
 * "Rebuild from memory": the centre's neighbours (plus two unconnected
 * distractors) arrive shuffled and unlabelled; place each where it belongs,
 * then check. Retrieval of the STRUCTURE, not just the definitions.
 */
export function RebuildExercise({
  center,
  nodes,
  edges,
}: {
  center: string;
  nodes: ConceptMeta[];
  edges: Edge[];
}) {
  const [round, setRound] = useState(0);
  const items = useMemo(() => buildRound(center, nodes, edges, round), [center, nodes, edges, round]);
  const byId = useMemo(() => new Map(nodes.map((n) => [n.id, n])), [nodes]);
  const [placed, setPlaced] = useState<Record<string, Zone>>({});
  const [selected, setSelected] = useState<string | null>(null);
  const [checked, setChecked] = useState(false);

  const centerMeta = byId.get(center);
  const unplaced = items.filter((i) => !placed[i.id]);
  const score = items.filter((i) => placed[i.id] === i.answer).length;
  const distractors = items.filter((i) => i.answer === "none").length;

  function place(id: string, zone: Zone) {
    if (checked) return;
    setPlaced((p) => ({ ...p, [id]: zone }));
    setSelected(null);
  }
  function reset() {
    setRound((r) => r + 1);
    setPlaced({});
    setChecked(false);
    setSelected(null);
  }

  if (items.length <= 2) {
    return <p className="text-sm text-ink-2">This concept has too few links for the exercise yet.</p>;
  }

  const card = (item: Item) => {
    const meta = byId.get(item.id)!;
    const right = placed[item.id] === item.answer;
    const yourZone = ZONES.find((z) => z.key === placed[item.id]);
    return (
      <button
        draggable={!checked}
        onDragStart={(e) => e.dataTransfer.setData("text/plain", item.id)}
        onClick={() => !checked && setSelected(selected === item.id ? null : item.id)}
        aria-pressed={selected === item.id}
        className={`w-full rounded-xl border px-3 py-2 text-left text-[13px] transition-colors ${
          checked
            ? right
              ? "border-[var(--m-explain)] bg-panel"
              : "border-due bg-due-soft"
            : selected === item.id
              ? "border-accent bg-accent-soft"
              : "border-line bg-panel hover:border-ink-3"
        } ${checked ? "cursor-default" : "cursor-grab"}`}
      >
        <span className="flex items-center gap-1.5">
          <span className="size-1.5 shrink-0 rounded-full" style={{ background: categoryColor(meta.category) }} />
          <span className="font-medium">{meta.title}</span>
          {checked && <span className="ml-auto text-xs">{right ? "✓" : "✗"}</span>}
        </span>
        {checked && (
          <span className="mt-1 block text-xs text-ink-3">
            {right ? roleOf(item) : `${roleOf(item)} · you put it in “${yourZone?.label.toLowerCase()}”`}
          </span>
        )}
      </button>
    );
  };

  const zoneBox = (zone: (typeof ZONES)[number], className = "") => {
    // After checking, every card moves to where it truly belongs, so the board shows the real structure.
    const here = items.filter((i) => (checked ? i.answer : placed[i.id]) === zone.key);
    const target = selected !== null && !checked;
    return (
      <div
        onDragOver={(e) => !checked && e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          place(e.dataTransfer.getData("text/plain"), zone.key);
        }}
        onClick={(e) => {
          if (selected && e.target === e.currentTarget) place(selected, zone.key);
        }}
        className={`flex min-h-32 flex-col gap-2 rounded-2xl border border-dashed p-3 transition-colors ${
          target ? "cursor-pointer border-accent bg-accent-soft/40" : "border-line"
        } ${className}`}
      >
        <button
          type="button"
          disabled={!target}
          onClick={() => selected && place(selected, zone.key)}
          className="text-left disabled:cursor-default"
        >
          <span className="block text-xs font-medium uppercase tracking-wider text-ink-2">{zone.label}</span>
          <span className="block text-[11px] text-ink-3">{zone.hint}</span>
        </button>
        {here.map((i) => (
          <div key={i.id}>{card(i)}</div>
        ))}
      </div>
    );
  };

  return (
    <div className="fade-in">
      <p className="text-sm text-ink-2">
        Where does each concept sit relative to <strong className="text-ink">{centerMeta?.title}</strong>? Drag it, or
        click a card and then a box.{" "}
        {distractors === 1 ? "One of them isn’t" : distractors > 1 ? `${distractors === 2 ? "Two" : distractors} of them aren’t` : ""}{distractors ? " connected at all." : ""}
      </p>

      {/* Tray */}
      {!checked && (
        <div className="mt-5 flex min-h-12 flex-wrap gap-2">
          {unplaced.length ? (
            unplaced.map((i) => (
              <div key={i.id} className="w-56">
                {card(i)}
              </div>
            ))
          ) : (
            <p className="self-center text-sm text-ink-3">All placed — check when you&apos;re ready.</p>
          )}
        </div>
      )}

      {/* Board, mirroring the local graph: left → centre → right */}
      <div className="mt-5 grid gap-3 md:grid-cols-[1fr_minmax(0,0.8fr)_1fr]">
        {zoneBox(ZONES[0])}
        <div className="flex flex-col gap-3">
          <div className="grid min-h-20 place-items-center rounded-2xl border border-accent bg-accent-soft px-3 text-center font-medium">
            {centerMeta?.title}
          </div>
          {zoneBox(ZONES[2], "flex-1")}
        </div>
        {zoneBox(ZONES[1])}
      </div>
      {zoneBox(ZONES[3], "mt-3 min-h-20")}

      <div className="mt-6 flex items-center gap-4">
        {!checked ? (
          <button
            disabled={unplaced.length > 0}
            onClick={() => setChecked(true)}
            className="h-11 rounded-xl bg-accent px-6 font-medium text-accent-ink disabled:opacity-40"
          >
            Check
          </button>
        ) : (
          <>
            <p className="font-serif text-xl">
              {score} / {items.length} placed right
            </p>
            <button onClick={reset} className="h-11 rounded-xl border border-line px-5 text-ink-2 hover:text-ink">
              Try a new shuffle
            </button>
          </>
        )}
        {!checked && Object.keys(placed).length > 0 && (
          <button onClick={() => setPlaced({})} className="text-sm text-ink-3 hover:text-ink">
            Clear
          </button>
        )}
      </div>
    </div>
  );
}
