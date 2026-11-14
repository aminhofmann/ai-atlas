import type { Progress } from "./progress";
import { isSolid, type Category, type Mastery } from "./types";

/**
 * The slice of a concept the planner needs — computed on the server at build
 * time (see `toPlanConcept` in lib/content.ts), then planned in the browser
 * against the visitor's own progress. Pure functions only: no I/O here.
 */
export type PlanConcept = {
  id: string;
  title: string;
  summary: string;
  category: Category;
  requires: string[];
  cardIds: string[];
  hasCode: boolean;
  coreMinutes: number;
};

export type SessionItem = {
  kind: "review" | "learn" | "practice" | "implement";
  title: string;
  why: string;
  href: string;
  minutes: number;
};

const mastery = (p: Progress, id: string): Mastery => p.concepts[id]?.mastery ?? "new";

/** First concept, in path order, that's new and whose prerequisites you can explain. */
export function nextConcept(path: PlanConcept[], p: Progress): PlanConcept | undefined {
  const ids = new Set(path.map((c) => c.id));
  return path.find(
    (c) => mastery(p, c.id) === "new" && c.requires.every((r) => !ids.has(r) || isSolid(p.concepts[r]?.mastery)),
  );
}

/**
 * Today's queue, in the order it should be done: what's due first (forgetting
 * is the expensive failure), then deepen what's in progress, then one new thing.
 */
export function buildSession(path: PlanConcept[], p: Progress, due: string[]) {
  const byId = new Map(path.map((c) => [c.id, c]));
  const items: SessionItem[] = [];

  if (due.length) {
    const titles = [...new Set(due.map((d) => d.split("#")[0]))].map((id) => byId.get(id)?.title).filter(Boolean);
    items.push({
      kind: "review",
      title: titles.length === 1 ? titles[0]! : `${titles[0]} and ${titles.length - 1} more`,
      why: `${due.length} card${due.length === 1 ? "" : "s"} due — reviewing now is cheaper than relearning later.`,
      href: "/review",
      minutes: Math.max(2, Math.round(due.length * 0.6)),
    });
  }

  const inProgress = path
    .filter((c) => mastery(p, c.id) === "recognize")
    .sort((a, b) => (p.concepts[b.id]?.updatedAt ?? "").localeCompare(p.concepts[a.id]?.updatedAt ?? ""));
  if (inProgress[0]) {
    const c = inProgress[0];
    items.push({
      kind: "practice",
      title: c.title,
      why: "You recognise it — explaining it from memory is what makes it stick.",
      href: `/c/${c.id}?mode=practice`,
      minutes: Math.max(3, Math.round(c.cardIds.length * 1.5)),
    });
  }

  const implementable = path.find((c) => mastery(p, c.id) === "explain" && c.hasCode);
  if (implementable) {
    items.push({
      kind: "implement",
      title: implementable.title,
      why: "You can explain it. Writing it from scratch is the next level.",
      href: `/c/${implementable.id}?mode=implement`,
      minutes: 15,
    });
  }

  const next = nextConcept(path, p);
  if (next) {
    const reqs = next.requires.map((r) => byId.get(r)?.title).filter(Boolean);
    items.push({
      kind: "learn",
      title: next.title,
      why: reqs.length
        ? `Its prerequisites (${reqs.slice(0, 2).join(", ")}${reqs.length > 2 ? "…" : ""}) are done.`
        : "A starting point — it needs nothing else first.",
      href: `/c/${next.id}?mode=learn`,
      minutes: next.coreMinutes + 5,
    });
  }

  const track: Category | null = next?.category ?? inProgress[0]?.category ?? null;
  const inTrack = track ? path.filter((c) => c.category === track) : [];
  return {
    items,
    totalMinutes: items.reduce((s, i) => s + i.minutes, 0),
    track,
    trackProgress: { done: inTrack.filter((c) => isSolid(p.concepts[c.id]?.mastery)).length, total: inTrack.length },
  };
}
