export const CATEGORIES = [
  "foundations",
  "training",
  "architectures",
  "innovations",
  "biology",
  "interpretability",
  "domains",
] as const;
export type Category = (typeof CATEGORIES)[number];

export const CATEGORY_LABEL: Record<Category, string> = {
  foundations: "Foundations",
  training: "Training",
  architectures: "Architectures",
  innovations: "Innovations",
  biology: "Biology",
  interpretability: "Interpretability",
  domains: "Other domains",
};

export const LINK_TYPES = [
  "requires",
  "part-of",
  "uses",
  "based-on",
  "variant-of",
  "alternative-to",
  "improves",
  "applied-in",
  "related",
] as const;
export type LinkType = (typeof LINK_TYPES)[number];

/** How an edge reads from each end: "A <out> B" / "B <in> A". */
export const LINK_PHRASE: Record<LinkType, { out: string; in: string }> = {
  requires: { out: "requires", in: "is required by" },
  "part-of": { out: "is part of", in: "contains" },
  uses: { out: "uses", in: "is used by" },
  "based-on": { out: "is based on", in: "is the basis of" },
  "variant-of": { out: "is a variant of", in: "has variant" },
  "alternative-to": { out: "is an alternative to", in: "is an alternative to" },
  improves: { out: "improves on", in: "is improved by" },
  "applied-in": { out: "is applied in", in: "makes use of" },
  related: { out: "is related to", in: "is related to" },
};

/**
 * What you can DO with a concept, not how often you've seen it.
 * recognize: you know what it is · explain: you can explain it unaided ·
 * apply: you can use/implement it · teach: you can teach it, edge cases included.
 */
export const MASTERY_LEVELS = ["new", "recognize", "explain", "apply", "teach"] as const;
export type Mastery = (typeof MASTERY_LEVELS)[number];
export const MASTERY_LABEL: Record<Mastery, string> = {
  new: "New",
  recognize: "Recognize",
  explain: "Explain",
  apply: "Apply",
  teach: "Teach",
};
export const MASTERY_HINT: Record<Mastery, string> = {
  new: "Haven't studied it yet",
  recognize: "I know what it is",
  explain: "I can explain it without notes",
  apply: "I can use or implement it",
  teach: "I could teach it, edge cases included",
};
/** "Explain" is the bar for counting a prerequisite as done. */
export const isSolid = (m: Mastery | undefined) =>
  m !== undefined && MASTERY_LEVELS.indexOf(m) >= MASTERY_LEVELS.indexOf("explain");

export type Source ={ title: string; url?: string };

export type Card = { id: string; q: string; a: string };

export type ConceptMeta = {
  id: string;
  title: string;
  category: Category;
  level: number;
  year?: number;
  summary: string;
  requires: string[];
  links: { to: string; type: LinkType }[];
  sources: Source[];
};

export type Concept = ConceptMeta & {
  sections: { intuition: string; mechanics: string; code: string; why: string };
  cards: Card[];
  raw: string;
};

export type Edge = { from: string; to: string; type: LinkType };

export type GraphData = { nodes: ConceptMeta[]; edges: Edge[] };
