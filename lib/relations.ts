import { LINK_PHRASE, LINK_TYPES, type Edge, type LinkType } from "./types";

export type Relation = { otherId: string; type: LinkType; phrase: string };

const SYMMETRIC: LinkType[] = ["related", "alternative-to"];

/**
 * Every relationship of one concept, readable from its side, without duplicates:
 * - the same pair written in both files collapses to one entry
 * - a vague "related" is dropped when the pair already has a specific relationship
 * Ordered by LINK_TYPES, so the strongest relationships come first.
 */
export function relationsOf(id: string, edges: Edge[], opts: { includeRequires?: boolean } = {}): Relation[] {
  const out: Relation[] = [];
  const seen = new Set<string>();
  // Any specific tie (prerequisites included) makes a "related" to the same concept redundant.
  const specific = new Set<string>();
  for (const e of edges) {
    if (e.type === "related") continue;
    if (e.from === id) specific.add(e.to);
    if (e.to === id) specific.add(e.from);
  }
  for (const e of edges) {
    if (e.type === "requires" && !opts.includeRequires) continue;
    let r: Relation | null = null;
    if (e.from === id) r = { otherId: e.to, type: e.type, phrase: LINK_PHRASE[e.type].out };
    else if (e.to === id) r = { otherId: e.from, type: e.type, phrase: LINK_PHRASE[e.type].in };
    if (!r) continue;
    const key = SYMMETRIC.includes(r.type) ? `${r.type}|${r.otherId}` : `${r.phrase}|${r.otherId}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(r);
  }
  return out
    .filter((r) => r.type !== "related" || !specific.has(r.otherId))
    .sort((a, b) => LINK_TYPES.indexOf(a.type) - LINK_TYPES.indexOf(b.type));
}
