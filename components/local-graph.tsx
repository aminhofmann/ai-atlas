"use client";

import { useMemo } from "react";
import {
  ReactFlow,
  Handle,
  Position,
  MarkerType,
  type Edge as FlowEdge,
  type Node as FlowNode,
  type NodeProps,
} from "@xyflow/react";
import { neighbourhood, ROLE_LEFT, ROLE_RIGHT } from "@/lib/neighbourhood";
import type { ConceptMeta, Edge, LinkType, Mastery } from "@/lib/types";
import { categoryColor, MasteryMeter } from "./category";

const W = 200;
const H = 52;
const COL = 290;
const ROW = 70;

type Data = { meta: ConceptMeta; mastery: Mastery; center: boolean };

function Node({ data }: NodeProps<FlowNode<Data>>) {
  const { meta, mastery, center } = data;
  return (
    <div
      className={`flex h-full cursor-pointer flex-col justify-center gap-1 rounded-xl border px-3 transition-colors ${
        center ? "border-accent bg-accent-soft" : "border-line bg-panel hover:border-ink-3"
      }`}
    >
      <Handle type="target" position={Position.Left} className="!size-1 !border-0 !bg-transparent" />
      <span className="flex items-center gap-1.5">
        <span className="size-1.5 shrink-0 rounded-full" style={{ background: categoryColor(meta.category) }} />
        <span className={`line-clamp-1 text-[13px] ${center ? "font-semibold" : "font-medium"}`}>{meta.title}</span>
      </span>
      <MasteryMeter mastery={mastery} />
      <Handle type="source" position={Position.Right} className="!size-1 !border-0 !bg-transparent" />
      <Handle id="t" type="target" position={Position.Top} className="!size-1 !border-0 !bg-transparent" />
      <Handle id="b" type="source" position={Position.Bottom} className="!size-1 !border-0 !bg-transparent" />
    </div>
  );
}
const nodeTypes = { c: Node };


const edgeColor =(t: LinkType) => (t === "requires" ? "var(--prereq)" : "var(--ink-3)");

export function LocalGraph({
  center,
  nodes,
  edges,
  mastery,
  onSelect,
  height = 460,
}: {
  center: string;
  nodes: ConceptMeta[];
  edges: Edge[];
  mastery: Record<string, Mastery>;
  onSelect: (id: string) => void;
  height?: number;
}) {
  const byId = useMemo(() => new Map(nodes.map((n) => [n.id, n])), [nodes]);
  const hood = useMemo(() => neighbourhood(center, edges), [center, edges]);

  const { flowNodes, flowEdges } = useMemo(() => {
    const fn: FlowNode<Data>[] = [];
    const fe: FlowEdge[] = [];
    const add = (id: string, x: number, y: number) => {
      const meta = byId.get(id);
      if (meta) fn.push({ id, type: "c", position: { x, y }, width: W, height: H, data: { meta, mastery: mastery[id] ?? "new", center: id === center } });
    };
    const column = (list: { id: string }[], x: number) =>
      list.forEach((n, i) => add(n.id, x, (i - (list.length - 1) / 2) * ROW));
    add(center, COL, 0);
    column(hood.left, 0);
    column(hood.right, COL * 2);
    hood.middle.forEach((n, i) => add(n.id, COL, (i % 2 === 0 ? -1 : 1) * (Math.floor(i / 2) + 1) * (ROW + 30)));

    const edge = (source: string, target: string, type: LinkType, label: string, vertical = false) =>
      fe.push({
        id: `${source}-${target}-${type}`,
        source,
        target,
        sourceHandle: vertical ? "b" : undefined,
        targetHandle: vertical ? "t" : undefined,
        label,
        labelStyle: { fill: "var(--ink-2)", fontSize: 11 },
        labelBgStyle: { fill: "var(--bg)" },
        labelBgPadding: [4, 2],
        style: { stroke: edgeColor(type), strokeWidth: 1.3, strokeDasharray: type === "alternative-to" || type === "related" ? "4 4" : undefined },
        markerEnd: { type: MarkerType.ArrowClosed, color: edgeColor(type), width: 14, height: 14 },
      });
    for (const n of hood.left) edge(n.id, center, n.type, ROLE_LEFT[n.type] ?? n.type);
    for (const n of hood.right) edge(center, n.id, n.type, ROLE_RIGHT[n.type] ?? n.type);
    for (const n of hood.middle) {
      const above = hood.middle.indexOf(n) % 2 === 0;
      edge(above ? n.id : center, above ? center : n.id, n.type, n.type === "alternative-to" ? "alternative" : "related", true);
    }
    return { flowNodes: fn, flowEdges: fe };
  }, [byId, hood, center, mastery]);

  return (
    <div style={{ height }} className="relative">
      <div className="pointer-events-none absolute inset-x-0 top-0 z-10 flex justify-between px-1 text-xs text-ink-3">
        <span>{hood.left.length ? "← builds on" : ""}</span>
        <span>{hood.right.length ? "built on it →" : ""}</span>
      </div>
      <ReactFlow
        key={center}
        nodes={flowNodes}
        edges={flowEdges}
        nodeTypes={nodeTypes}
        onNodeClick={(_, n) => onSelect(n.id)}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={false}
        fitView
        fitViewOptions={{ padding: 0.15, maxZoom: 1.1 }}
        minZoom={0.4}
        panOnScroll={false}
        zoomOnScroll={false}
        preventScrolling={false}
        proOptions={{ hideAttribution: true }}
      />
      {hood.hidden > 0 && (
        <p className="absolute bottom-1 right-1 text-xs text-ink-3">+{hood.hidden} more in the list</p>
      )}
    </div>
  );
}
