"use client";

import { useMemo, useState, useCallback } from "react";
import Link from "next/link";
import {
  ReactFlow,
  Background,
  Controls,
  MarkerType,
  Handle,
  Position,
  type Edge as FlowEdge,
  type Node as FlowNode,
  type NodeProps,
} from "@xyflow/react";
import {
  CATEGORIES,
  CATEGORY_LABEL,
  type Category,
  type ConceptMeta,
  type GraphData,
  type LinkType,
  type Mastery,
} from "@/lib/types";
import { categoryColor, LEVEL_LABEL } from "./category";
import { relationsOf } from "@/lib/relations";

const NODE_W = 190;
const NODE_H = 46;

/** Edge types that shape the layout and are drawn by default; the rest appear on selection. */
const STRUCTURAL: LinkType[] = ["requires", "based-on"];

const EDGE_STYLE: Record<LinkType, { color: string; dash?: string }> = {
  requires: { color: "var(--ink-3)" },
  "part-of": { color: "var(--c-foundations)", dash: "2 4" },
  uses: { color: "var(--c-training)", dash: "6 4" },
  "based-on": { color: "var(--c-architectures)" },
  "variant-of": { color: "var(--c-innovations)", dash: "6 4" },
  "alternative-to": { color: "var(--c-interpretability)", dash: "2 4" },
  improves: { color: "var(--c-biology)", dash: "6 4" },
  "applied-in": { color: "var(--c-domains)", dash: "6 4" },
  related: { color: "var(--ink-3)", dash: "1 5" },
};

type NodeData = { meta: ConceptMeta; mastery: Mastery; dim: boolean; selected: boolean };

function ConceptNode({ data }: NodeProps<FlowNode<NodeData>>) {
  const { meta, mastery, dim, selected } = data;
  return (
    <div
      className="flex h-full items-center gap-2 rounded-lg border bg-panel px-2.5 text-[13px] leading-tight shadow-sm transition-opacity"
      style={{
        borderTopColor: selected ? categoryColor(meta.category) : "var(--line)",
        borderRightColor: selected ? categoryColor(meta.category) : "var(--line)",
        borderBottomColor: selected ? categoryColor(meta.category) : "var(--line)",
        borderLeftColor: categoryColor(meta.category),
        borderLeftWidth: 4,
        opacity: dim ? 0.25 : 1,
        boxShadow: selected ? `0 0 0 2px ${categoryColor(meta.category)}` : undefined,
      }}
    >
      <Handle type="target" position={Position.Left} className="!size-1 !border-0 !bg-transparent" />
      <span className="line-clamp-2 flex-1 font-medium text-ink">{meta.title}</span>
      <span
        aria-label={`mastery: ${mastery}`}
        className="size-2.5 shrink-0 rounded-full"
        style={{ background: `var(--m-${mastery})` }}
      />
      <Handle type="source" position={Position.Right} className="!size-1 !border-0 !bg-transparent" />
    </div>
  );
}


const COL_W = NODE_W + 70;
const ROW_H = NODE_H + 14;
const LANE_GAP = 36;
const LANE_LABEL_W = 150;

/**
 * Swimlane layout: one horizontal lane per category (top → bottom in CATEGORIES
 * order), and columns by prerequisite DEPTH (left = needs nothing, right = needs
 * the most). So "where" a node sits says what it is AND how far in it is.
 */
function layout(nodes: ConceptMeta[], edges: GraphData["edges"]) {
  const deps = new Map<string, string[]>();
  for (const e of edges) if (STRUCTURAL.includes(e.type)) deps.set(e.from, [...(deps.get(e.from) ?? []), e.to]);
  const depth = new Map<string, number>();
  const visiting = new Set<string>();
  const depthOf = (id: string): number => {
    if (depth.has(id)) return depth.get(id)!;
    if (visiting.has(id)) return 0; // cycle guard
    visiting.add(id);
    const d = Math.max(-1, ...(deps.get(id) ?? []).map(depthOf)) + 1;
    visiting.delete(id);
    depth.set(id, d);
    return d;
  };
  nodes.forEach((n) => depthOf(n.id));

  const pos = new Map<string, { x: number; y: number }>();
  const lanes: { category: Category; y: number; h: number }[] = [];
  let y = 0;
  for (const cat of CATEGORIES) {
    const inLane = nodes.filter((n) => n.category === cat);
    if (!inLane.length) continue;
    const cols = new Map<number, ConceptMeta[]>();
    for (const n of inLane) {
      const d = depth.get(n.id)!;
      cols.set(d, [...(cols.get(d) ?? []), n]);
    }
    let rows = 1;
    for (const [d, list] of cols) {
      list.sort((a, b) => a.level - b.level || a.title.localeCompare(b.title));
      list.forEach((n, i) => pos.set(n.id, { x: LANE_LABEL_W + d * COL_W, y: y + i * ROW_H }));
      rows = Math.max(rows, list.length);
    }
    const h = rows * ROW_H - 14;
    lanes.push({ category: cat, y, h });
    y += h + LANE_GAP;
  }
  const maxDepth = Math.max(0, ...depth.values());
  return { pos, lanes, width: LANE_LABEL_W + (maxDepth + 1) * COL_W };
}

const LANE_PAD = 14;

type LaneData = { label: string; color: string; width: number; height: number };
function LaneNode({ data }: NodeProps<FlowNode<LaneData>>) {
  return (
    <div
      className="pointer-events-none rounded-2xl border border-line"
      style={{ width: data.width, height: data.height, background: `color-mix(in srgb, ${data.color} 5%, transparent)` }}
    >
      <span className="absolute left-4 top-3 text-xs font-semibold uppercase tracking-wider" style={{ color: data.color }}>
        {data.label}
      </span>
    </div>
  );
}

const nodeTypes = { concept: ConceptNode, lane: LaneNode };

export function GraphExplorer({
  data,
  mastery,
}: {
  data: GraphData;
  mastery: Record<string, Mastery>;
}) {
  const [selected, setSelected] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [cats, setCats] = useState<Set<Category>>(new Set(CATEGORIES));
  const [showAll, setShowAll] = useState(false);

  const { pos: positions, lanes, width: graphWidth } = useMemo(() => layout(data.nodes, data.edges), [data]);
  const byId = useMemo(() => new Map(data.nodes.map((n) => [n.id, n])), [data]);

  const neighbours = useMemo(() => {
    if (!selected) return null;
    const s = new Set([selected]);
    for (const e of data.edges) {
      if (e.from === selected) s.add(e.to);
      if (e.to === selected) s.add(e.from);
    }
    return s;
  }, [selected, data.edges]);

  const q = query.trim().toLowerCase();
  const matches = useCallback(
    (n: ConceptMeta) =>
      cats.has(n.category) &&
      (!q || n.title.toLowerCase().includes(q) || n.summary.toLowerCase().includes(q) || n.id.includes(q)),
    [cats, q],
  );

  const flowNodes: FlowNode[] = useMemo(
    () => [
      ...lanes.map(
        (l): FlowNode<LaneData> => ({
          id: `lane-${l.category}`,
          type: "lane",
          position: { x: 0, y: l.y - LANE_PAD },
          data: {
            label: CATEGORY_LABEL[l.category],
            color: categoryColor(l.category),
            width: graphWidth,
            height: l.h + LANE_PAD * 2,
          },
          selectable: false,
          draggable: false,
          focusable: false,
          // Lanes are backdrop only: clicks must fall through to the pane.
          style: { pointerEvents: "none" },
          className: "pointer-events-none",
          zIndex: -1,
        }),
      ),
      ...data.nodes.map((n): FlowNode<NodeData> => ({
        id: n.id,
        type: "concept",
        position: positions.get(n.id)!,
        width: NODE_W,
        height: NODE_H,
        data: {
          meta: n,
          mastery: mastery[n.id] ?? "new",
          selected: n.id === selected,
          dim: !matches(n) || (neighbours !== null && !neighbours.has(n.id)),
        },
      })),
    ],
    [data.nodes, positions, lanes, graphWidth, mastery, selected, matches, neighbours],
  );

  const flowEdges: FlowEdge[] = useMemo(
    () =>
      data.edges
        .filter((e) => {
          const touches = selected && (e.from === selected || e.to === selected);
          if (selected) return touches;
          return showAll || STRUCTURAL.includes(e.type);
        })
        .map((e) => {
          const st = EDGE_STYLE[e.type];
          // Draw requires/based-on from prerequisite to dependent; other types as written.
          const flip = STRUCTURAL.includes(e.type);
          return {
            id: `${e.from}-${e.type}-${e.to}`,
            source: flip ? e.to : e.from,
            target: flip ? e.from : e.to,
            label: selected ? e.type : undefined,
            labelStyle: { fill: "var(--ink-2)", fontSize: 10 },
            labelBgStyle: { fill: "var(--bg)" },
            style: {
              stroke: st.color,
              strokeWidth: selected ? 1.6 : 1,
              strokeDasharray: st.dash,
              // Unselected, the edges are texture; selected, they are the point.
              opacity: selected ? 1 : 0.28,
            },
            markerEnd: { type: MarkerType.ArrowClosed, color: st.color, width: 14, height: 14 },
            animated: false,
          };
        }),
    [data.edges, selected, showAll],
  );

  const sel = selected ? byId.get(selected) : null;
  const selEdges = selected
    ? relationsOf(selected, data.edges, { includeRequires: true }).map((r) => ({
        phrase: r.phrase,
        type: r.type,
        other: byId.get(r.otherId)!,
      }))
    : [];

  return (
    <div className="relative min-h-0 flex-1">
      {/* Filters */}
      <aside className="absolute bottom-3 left-3 z-10 w-64 space-y-3 rounded-xl border border-line bg-panel/95 p-3 shadow-sm backdrop-blur">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search concepts…"
          className="w-full rounded-md border border-line bg-bg px-2.5 py-1.5 text-sm outline-none focus:border-accent"
        />
        <div className="flex flex-wrap gap-1.5">
          {CATEGORIES.map((c) => {
            const on = cats.has(c);
            return (
              <button
                key={c}
                onClick={() => {
                  const next = new Set(cats);
                  if (on) next.delete(c);
                  else next.add(c);
                  setCats(next.size ? next : new Set(CATEGORIES));
                }}
                onDoubleClick={() => setCats(new Set([c]))}
                title="Click to toggle, double-click to show only this"
                className={`flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-xs transition-opacity ${
                  on ? "border-line text-ink" : "border-transparent text-ink-3 opacity-60"
                }`}
              >
                <span className="size-2 rounded-full" style={{ background: categoryColor(c) }} />
                {CATEGORY_LABEL[c]}
              </button>
            );
          })}
        </div>
        <label className="flex items-center gap-2 text-xs text-ink-2">
          <input type="checkbox" checked={showAll} onChange={(e) => setShowAll(e.target.checked)} />
          Show all relationship types
        </label>
      </aside>

      <div className="absolute inset-0">
        <ReactFlow
          nodes={flowNodes}
          edges={flowEdges}
          nodeTypes={nodeTypes}
          onNodeClick={(_, n) => {
            if (n.type !== "concept") return setSelected(null);
            setSelected(n.id === selected ? null : n.id);
          }}
          onPaneClick={() => setSelected(null)}
          nodesDraggable={false}
          nodesConnectable={false}
          fitView
          fitViewOptions={{ padding: 0.1 }}
          minZoom={0.15}
          proOptions={{ hideAttribution: true }}
        >
          <Background gap={24} size={1} color="var(--line)" />
          <Controls showInteractive={false} position="top-left" />
        </ReactFlow>
      </div>

      {/* Selection panel */}
      {sel && (
        <aside className="absolute right-3 top-3 bottom-3 z-10 flex w-80 flex-col overflow-hidden rounded-xl border border-line bg-panel shadow-sm">
          <div className="space-y-2 border-b border-line p-4">
            <div className="flex items-center gap-2 text-xs text-ink-3">
              <span className="size-2 rounded-full" style={{ background: categoryColor(sel.category) }} />
              {CATEGORY_LABEL[sel.category]} · {LEVEL_LABEL[sel.level] ?? `Level ${sel.level}`}
              {sel.year ? ` · ${sel.year}` : ""}
            </div>
            <h2 className="text-lg font-semibold leading-snug">{sel.title}</h2>
            <p className="text-sm leading-relaxed text-ink-2">{sel.summary}</p>
            <Link
              href={`/c/${sel.id}`}
              className="inline-block rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-accent-ink"
            >
              Study this concept →
            </Link>
          </div>
          <div className="flex-1 space-y-1 overflow-y-auto p-3 text-sm">
            <p className="px-1 pb-1 text-xs font-medium uppercase tracking-wide text-ink-3">Relationships</p>
            {selEdges.length === 0 && <p className="px-1 text-ink-3">No links yet.</p>}
            {selEdges.map((e, i) => (
              <button
                key={i}
                onClick={() => setSelected(e.other.id)}
                className="flex w-full items-baseline gap-2 rounded-md px-1 py-1 text-left hover:bg-panel-2"
              >
                <span className="w-28 shrink-0 text-xs text-ink-3">{e.phrase}</span>
                <span className="flex items-center gap-1.5 text-ink">
                  <span className="size-1.5 rounded-full" style={{ background: categoryColor(e.other.category) }} />
                  {e.other.title}
                </span>
              </button>
            ))}
          </div>
        </aside>
      )}
    </div>
  );
}
