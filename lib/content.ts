import "server-only";
import fs from "node:fs";
import path from "node:path";
import matter from "gray-matter";
import {
  CATEGORIES,
  LINK_TYPES,
  type Card,
  type Category,
  type Concept,
  type ConceptMeta,
  type Edge,
  type GraphData,
  type LinkType,
  type Source,
} from "./types";

export const CONCEPT_DIR = path.join(process.cwd(), "content", "concepts");
export const ID_PATTERN = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

function asStringArray(v: unknown): string[] {
  return Array.isArray(v) ? v.filter((x): x is string => typeof x === "string") : [];
}

/** Split the body on H2 headings into a lowercase-keyed map. */
function splitSections(body: string): Record<string, string> {
  const out: Record<string, string> = {};
  let key = "_intro";
  let buf: string[] = [];
  let inFence = false;
  for (const line of body.split("\n")) {
    if (line.trimStart().startsWith("```")) inFence = !inFence;
    const m = !inFence && /^##\s+(.+?)\s*$/.exec(line);
    if (m) {
      out[key] = buf.join("\n").trim();
      key = m[1].toLowerCase();
      buf = [];
    } else buf.push(line);
  }
  out[key] = buf.join("\n").trim();
  return out;
}

function parseCards(id: string, text: string): Card[] {
  // Each card starts at a line beginning "Q:"; its answer starts at the next "A:" line.
  const cards: Card[] = [];
  for (const block of text.split(/^(?=Q:)/m)) {
    const m = /^Q:\s*([\s\S]*?)\n\s*A:\s*([\s\S]*)$/.exec(block.trim());
    if (m && m[1].trim() && m[2].trim()) {
      cards.push({ id: `${id}#${cards.length}`, q: m[1].trim(), a: m[2].trim() });
    }
  }
  return cards;
}

function parseFile(file: string): Concept | null {
  const raw = fs.readFileSync(path.join(CONCEPT_DIR, file), "utf8");
  const { data, content } = matter(raw);
  const id = file.replace(/\.md$/, "");
  const category = (CATEGORIES as readonly string[]).includes(data.category)
    ? (data.category as Category)
    : "foundations";
  const links = (Array.isArray(data.links) ? data.links : [])
    .filter(
      (l: unknown): l is { to: string; type: LinkType } =>
        !!l &&
        typeof (l as { to?: unknown }).to === "string" &&
        (LINK_TYPES as readonly string[]).includes((l as { type?: string }).type ?? ""),
    )
    .map((l: { to: string; type: LinkType }) => ({ to: l.to, type: l.type }));
  const sources: Source[] = (Array.isArray(data.sources) ? data.sources : [])
    .filter((s: unknown) => !!s && typeof (s as Source).title === "string")
    .map((s: Source) => ({ title: s.title, url: typeof s.url === "string" ? s.url : undefined }));
  const s = splitSections(content);
  return {
    id,
    title: typeof data.title === "string" ? data.title : id,
    category,
    level: typeof data.level === "number" ? data.level : 2,
    year: typeof data.year === "number" ? data.year : undefined,
    summary: typeof data.summary === "string" ? data.summary.trim() : "",
    requires: [...new Set(asStringArray(data.requires))].filter((r) => r !== id),
    links,
    sources,
    sections: {
      intuition: s["intuition"] ?? "",
      mechanics: s["mechanics"] ?? "",
      code: s["code"] ?? "",
      why: s["why it matters"] ?? "",
    },
    cards: parseCards(id, s["self-test"] ?? ""),
    raw,
  };
}

/** Read every concept file. Cheap enough to do per request for a few hundred files. */
export function getAllConcepts(): Concept[] {
  if (!fs.existsSync(CONCEPT_DIR)) return [];
  return fs
    .readdirSync(CONCEPT_DIR)
    .filter((f) => f.endsWith(".md") && ID_PATTERN.test(f.replace(/\.md$/, "")))
    .map(parseFile)
    .filter((c): c is Concept => c !== null)
    .sort((a, b) => a.title.localeCompare(b.title));
}

export function getConcept(id: string): Concept | null {
  if (!ID_PATTERN.test(id)) return null;
  const file = `${id}.md`;
  if (!fs.existsSync(path.join(CONCEPT_DIR, file))) return null;
  return parseFile(file);
}

export function toMeta(c: Concept): ConceptMeta {
  const { id, title, category, level, year, summary, requires, links, sources } = c;
  return { id, title, category, level, year, summary, requires, links, sources };
}

/** Nodes plus edges; edges pointing at ids that have no file yet are dropped. */
export function getGraph(concepts = getAllConcepts()): GraphData {
  const ids = new Set(concepts.map((c) => c.id));
  const edges: Edge[] = [];
  const seen = new Set<string>();
  const add = (e: Edge) => {
    const k = `${e.from}|${e.to}|${e.type}`;
    if (e.from === e.to || !ids.has(e.to) || seen.has(k)) return;
    seen.add(k);
    edges.push(e);
  };
  for (const c of concepts) {
    for (const r of c.requires) add({ from: c.id, to: r, type: "requires" });
    for (const l of c.links) add({ from: c.id, to: l.to, type: l.type });
  }
  return { nodes: concepts.map(toMeta), edges };
}

/** Prerequisite-first order (Kahn's algorithm on `requires`), level then title as tiebreak. */
export function learningPath(concepts = getAllConcepts()): Concept[] {
  const byId = new Map(concepts.map((c) => [c.id, c]));
  const indeg = new Map<string, number>();
  const dependents = new Map<string, string[]>();
  for (const c of concepts) {
    const reqs = c.requires.filter((r) => byId.has(r) && r !== c.id);
    indeg.set(c.id, reqs.length);
    for (const r of reqs) dependents.set(r, [...(dependents.get(r) ?? []), c.id]);
  }
  const cmp = (a: string, b: string) => {
    const A = byId.get(a)!, B = byId.get(b)!;
    return A.level - B.level || A.title.localeCompare(B.title);
  };
  const ready = concepts.filter((c) => indeg.get(c.id) === 0).map((c) => c.id).sort(cmp);
  const out: Concept[] = [];
  while (ready.length) {
    const id = ready.shift()!;
    out.push(byId.get(id)!);
    for (const d of dependents.get(id) ?? []) {
      indeg.set(d, indeg.get(d)! - 1);
      if (indeg.get(d) === 0) {
        ready.push(d);
        ready.sort(cmp);
      }
    }
  }
  // Anything left sits in a requires-cycle; append rather than lose it.
  for (const c of concepts) if (!out.includes(c)) out.push(c);
  return out;
}
