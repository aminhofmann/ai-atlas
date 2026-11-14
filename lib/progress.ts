import "server-only";
import fs from "node:fs";
import path from "node:path";
import { createEmptyCard, fsrs, Rating, type Card as FsrsCard, type Grade } from "ts-fsrs";

/**
 * Personal progress, stored as one JSON file next to the content.
 * A local single-user app: a file is simpler than a database and survives
 * clearing the browser. `data/` is git-ignored.
 */
const FILE = path.join(process.cwd(), "data", "progress.json");

import type { Mastery } from "./types";
export type { Mastery };

export type Progress = {
  cards: Record<string, FsrsCard>;
  concepts: Record<string, { mastery: Mastery; updatedAt: string }>;
};

const scheduler = fsrs({ enable_fuzz: true });

export function readProgress(): Progress {
  try {
    const p = JSON.parse(fs.readFileSync(FILE, "utf8")) as Progress;
    for (const c of Object.values(p.cards ?? {})) {
      c.due = new Date(c.due);
      if (c.last_review) c.last_review = new Date(c.last_review);
    }
    // The first version had three levels; map them onto the five.
    const legacy: Record<string, Mastery> = { learning: "recognize", known: "explain" };
    for (const c of Object.values(p.concepts ?? {})) {
      if (legacy[c.mastery as string]) c.mastery = legacy[c.mastery as string];
    }
    return { cards: p.cards ?? {}, concepts: p.concepts ?? {} };
  } catch {
    return { cards: {}, concepts: {} };
  }
}

function write(p: Progress) {
  fs.mkdirSync(path.dirname(FILE), { recursive: true });
  const tmp = `${FILE}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(p, null, 2));
  fs.renameSync(tmp, FILE);
}

export const GRADES = {
  again: Rating.Again,
  hard: Rating.Hard,
  good: Rating.Good,
  easy: Rating.Easy,
} as const satisfies Record<string, Grade>;
export type GradeName = keyof typeof GRADES;

export function reviewCard(cardId: string, grade: GradeName, now = new Date()) {
  const p = readProgress();
  const current = p.cards[cardId] ?? createEmptyCard(now);
  const { card } = scheduler.next(current, now, GRADES[grade]);
  p.cards[cardId] = card;
  // Reviewing a concept's cards means you are at least learning it.
  const conceptId = cardId.split("#")[0];
  if (!p.concepts[conceptId] || p.concepts[conceptId].mastery === "new") {
    p.concepts[conceptId] = { mastery: "recognize", updatedAt: now.toISOString() };
  }
  write(p);
  return card;
}

export function setMastery(conceptId: string, mastery: Mastery) {
  const p = readProgress();
  p.concepts[conceptId] = { mastery, updatedAt: new Date().toISOString() };
  write(p);
}

/** Card ids due now: reviewed cards past their due date, plus unseen cards of concepts you are learning. */
export function dueCardIds(allCardIds: string[], now = new Date()): string[] {
  const p = readProgress();
  const due: { id: string; due: number }[] = [];
  for (const id of allCardIds) {
    const c = p.cards[id];
    const conceptId = id.split("#")[0];
    if (c) {
      if (c.due.getTime() <= now.getTime()) due.push({ id, due: c.due.getTime() });
    } else if (p.concepts[conceptId] && p.concepts[conceptId].mastery !== "new") {
      due.push({ id, due: Number.MAX_SAFE_INTEGER });
    }
  }
  return due.sort((a, b) => a.due - b.due).map((d) => d.id);
}
