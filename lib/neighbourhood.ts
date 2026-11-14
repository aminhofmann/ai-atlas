import { relationsOf } from "./relations";
import type { Edge, LinkType } from "./types";

/** Which side of the centre a neighbour sits on, read as "the centre ___ it". */
const BUILDS_ON: LinkType[] = ["requires", "based-on", "uses", "variant-of", "improves"];
const BUILT_BY: LinkType[] = ["part-of", "applied-in"]; // centre is part of / applied in → the other builds on it
const ALONGSIDE: LinkType[] = ["alternative-to", "related"];

/** What the NEIGHBOUR is to the centre, by side — shared by the local graph's edge labels and the rebuild exercise. */
export const ROLE_LEFT: Partial<Record<LinkType, string>> = {
  requires: "prerequisite",
  "based-on": "basis",
  uses: "used by it",
  "variant-of": "the original",
  improves: "what it improves on",
  "part-of": "component",
  "applied-in": "applied in it",
};
export const ROLE_RIGHT: Partial<Record<LinkType, string>> = {
  requires: "needs it",
  "based-on": "builds on it",
  uses: "uses it",
  "variant-of": "a variant",
  improves: "an improvement",
  "part-of": "contains it",
  "applied-in": "an application",
};
export const ROLE_MIDDLE: Partial<Record<LinkType, string>> = {
  "alternative-to": "alternative",
  related: "related",
};

export type Neighbour ={ id: string; type: LinkType; phrase: string };
export type Neighbourhood = {
  left: Neighbour[]; // what the centre builds on
  right: Neighbour[]; // what builds on the centre
  middle: Neighbour[]; // alternatives / related
  hidden: number;
};

/** One-hop neighbourhood, capped so the picture stays readable (≤ ~15 nodes). */
export function neighbourhood(center: string, edges: Edge[], perColumn = 5): Neighbourhood {
  const left: Neighbour[] = [];
  const right: Neighbour[] = [];
  const middle: Neighbour[] = [];
  const placed = new Set<string>();
  for (const r of relationsOf(center, edges, { includeRequires: true })) {
    if (placed.has(r.otherId)) continue;
    placed.add(r.otherId);
    const outgoing = edges.some((e) => e.from === center && e.to === r.otherId && e.type === r.type);
    const n = { id: r.otherId, type: r.type, phrase: r.phrase };
    if (ALONGSIDE.includes(r.type)) middle.push(n);
    else if (BUILDS_ON.includes(r.type) ? outgoing : BUILT_BY.includes(r.type) && !outgoing) left.push(n);
    else right.push(n);
  }
  const cap = (xs: Neighbour[], k: number) => xs.slice(0, k);
  // Middle holds one neighbour above and one below the centre; a third would sit on another's edge.
  const shown = cap(left, perColumn).length + cap(right, perColumn).length + cap(middle, 2).length;
  return {
    left: cap(left, perColumn),
    right: cap(right, perColumn),
    middle: cap(middle, 2),
    hidden: left.length + right.length + middle.length - shown,
  };
}
