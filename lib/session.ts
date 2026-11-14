import "server-only";
import { learningPath } from "./content";
import { dueCardIds, readProgress, type Progress } from "./progress";
import { isSolid, type Category, type Concept, type Mastery } from "./types";

const WPM = 200;
const words = (s: string) => s.split(/\s+/).filter(Boolean).length;
/** Paragraph(s) of the "why it matters" section that state the problem — the first one. */
export const firstParagraph = (s: string) => s.trim().split(/\n\s*\n/)[0] ?? "";

/** Minutes for the core read (definition + why + intuition) and the full deep pass. */
export function studyTime(c: Concept) {
  const core = words(c.summary) + words(firstParagraph(c.sections.why)) + words(c.sections.intuition);
  const deep = core + words(c.sections.mechanics) + words(c.sections.why) + words(c.sections.code) * 0.5;
  return {
    core: Math.max(2, Math.round(core / WPM + 1)),
    deep: Math.max(5, Math.round(deep / WPM + c.cards.length * 1.5 + 8)),
  };
}

export type SessionItem = {
  kind: "review" | "learn" | "practice" | "implement";
  title: string;
  why: string;
  href: string;
  minutes: number;
};

export function masteryOf(p: Progress, id: string): Mastery {
  return p.concepts[id]?.mastery ?? "new";
}

/**
 * Today's queue, in the order it should be done: what's due first (forgetting
 * is the expensive failure), then deepen what's in progress, then one new thing.
 */
export function buildSession() {
  const progress = readProgress();
  const path = learningPath();
  const byId = new Map(path.map((c) => [c.id, c]));
  const items: SessionItem[] = [];

  const due = dueCardIds(path.flatMap((c) => c.cards.map((k) => k.id)));
  if (due.length) {
    const concepts = [...new Set(due.map((d) => d.split("#")[0]))].map((id) => byId.get(id)?.title).filter(Boolean);
    items.push({
      kind: "review",
      title: concepts.length === 1 ? concepts[0]! : `${concepts[0]} and ${concepts.length - 1} more`,
      why: `${due.length} card${due.length === 1 ? "" : "s"} due — reviewing now is cheaper than relearning later.`,
      href: "/review",
      minutes: Math.max(2, Math.round(due.length * 0.6)),
    });
  }

  // In progress: recognised but not yet explainable → practice it.
  const inProgress = path
    .filter((c) => masteryOf(progress, c.id) === "recognize")
    .sort((a, b) => (progress.concepts[b.id]?.updatedAt ?? "").localeCompare(progress.concepts[a.id]?.updatedAt ?? ""));
  if (inProgress[0]) {
    const c = inProgress[0];
    items.push({
      kind: "practice",
      title: c.title,
      why: "You recognise it — explaining it from memory is what makes it stick.",
      href: `/c/${c.id}?mode=practice`,
      minutes: Math.max(3, Math.round(c.cards.length * 1.5)),
    });
  }

  // Can explain but not yet apply, and has code → implement it.
  const implementable = path.find((c) => masteryOf(progress, c.id) === "explain" && c.sections.code.includes("```"));
  if (implementable) {
    items.push({
      kind: "implement",
      title: implementable.title,
      why: "You can explain it. Writing it from scratch is the next level.",
      href: `/c/${implementable.id}?mode=implement`,
      minutes: 15,
    });
  }

  // Next new concept whose prerequisites are solid.
  const next = nextConcept(path, progress);
  if (next) {
    const reqs = next.requires.map((r) => byId.get(r)?.title).filter(Boolean);
    items.push({
      kind: "learn",
      title: next.title,
      why: reqs.length
        ? `Its prerequisites (${reqs.slice(0, 2).join(", ")}${reqs.length > 2 ? "…" : ""}) are done.`
        : "A starting point — it needs nothing else first.",
      href: `/c/${next.id}?mode=learn`,
      minutes: studyTime(next).core + 5,
    });
  }

  // Current track = topic of the next concept; milestone = that topic at "explain" or better.
  const track: Category | null = next?.category ?? inProgress[0]?.category ?? null;
  const inTrack = track ? path.filter((c) => c.category === track) : [];
  const solidInTrack = inTrack.filter((c) => isSolid(progress.concepts[c.id]?.mastery)).length;

  return {
    items,
    totalMinutes: items.reduce((s, i) => s + i.minutes, 0),
    dueCount: due.length,
    track,
    trackProgress: { done: solidInTrack, total: inTrack.length },
    next,
  };
}

export function nextConcept(path: Concept[], progress: Progress): Concept | undefined {
  const ids = new Set(path.map((c) => c.id));
  return path.find(
    (c) =>
      masteryOf(progress, c.id) === "new" &&
      c.requires.every((r) => !ids.has(r) || isSolid(progress.concepts[r]?.mastery)),
  );
}
