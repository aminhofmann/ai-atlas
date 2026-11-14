"use client";

import { useSyncExternalStore } from "react";
import { createEmptyCard, fsrs, Rating, type Card as FsrsCard, type Grade } from "ts-fsrs";
import type { Mastery } from "./types";

/**
 * Personal progress lives in the VISITOR'S BROWSER (localStorage): the app is
 * hosted for anyone to use, so each person keeps their own mastery and review
 * schedule, with no account and no database. Reads never throw — private mode
 * or blocked storage just means an empty, in-memory session.
 */
const KEY = "ai-atlas.progress.v1";

export type { Mastery };
export type Progress = {
  cards: Record<string, FsrsCard>;
  concepts: Record<string, { mastery: Mastery; updatedAt: string }>;
};

const EMPTY: Progress = { cards: {}, concepts: {} };
const scheduler = fsrs({ enable_fuzz: true });

let cache: Progress | null = null;
let memoryFallback: Progress = EMPTY;
const listeners = new Set<() => void>();

function revive(p: Partial<Progress>): Progress {
  const cards: Record<string, FsrsCard> = {};
  for (const [id, c] of Object.entries(p.cards ?? {})) {
    cards[id] = { ...c, due: new Date(c.due), last_review: c.last_review ? new Date(c.last_review) : undefined };
  }
  return { cards, concepts: p.concepts ?? {} };
}

function read(): Progress {
  if (cache) return cache;
  try {
    const raw = window.localStorage.getItem(KEY);
    cache = raw ? revive(JSON.parse(raw)) : EMPTY;
  } catch {
    cache = memoryFallback;
  }
  return cache;
}

function write(p: Progress) {
  cache = p;
  memoryFallback = p;
  try {
    window.localStorage.setItem(KEY, JSON.stringify(p));
  } catch {
    /* storage unavailable — keep the in-memory copy for this tab */
  }
  listeners.forEach((l) => l());
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  const onStorage = (e: StorageEvent) => {
    if (e.key === KEY) {
      cache = null; // another tab changed it
      listener();
    }
  };
  window.addEventListener("storage", onStorage);
  return () => {
    listeners.delete(listener);
    window.removeEventListener("storage", onStorage);
  };
}

/**
 * The visitor's progress. On the server and during hydration it is EMPTY
 * (there is no browser storage there); it fills in right after.
 */
export function useProgress(): Progress {
  return useSyncExternalStore(subscribe, read, () => EMPTY);
}

/** True once the browser's progress has been read (false during server render / hydration). */
export function useProgressReady(): boolean {
  return useSyncExternalStore(subscribe, () => true, () => false);
}

export const masteryOf = (p: Progress, id: string): Mastery => p.concepts[id]?.mastery ?? "new";

export function masteryMap(p: Progress): Record<string, Mastery> {
  return Object.fromEntries(Object.entries(p.concepts).map(([id, c]) => [id, c.mastery]));
}

export const GRADES = {
  again: Rating.Again,
  hard: Rating.Hard,
  good: Rating.Good,
  easy: Rating.Easy,
} as const satisfies Record<string, Grade>;
export type GradeName = keyof typeof GRADES;

export function reviewCard(cardId: string, grade: GradeName, now = new Date()) {
  const p = read();
  const { card } = scheduler.next(p.cards[cardId] ?? createEmptyCard(now), now, GRADES[grade]);
  const conceptId = cardId.split("#")[0];
  const concepts = { ...p.concepts };
  // Reviewing a concept's questions means you at least recognise it.
  if (!concepts[conceptId] || concepts[conceptId].mastery === "new") {
    concepts[conceptId] = { mastery: "recognize", updatedAt: now.toISOString() };
  }
  write({ cards: { ...p.cards, [cardId]: card }, concepts });
  return card;
}

export function setMastery(conceptId: string, mastery: Mastery) {
  const p = read();
  write({ ...p, concepts: { ...p.concepts, [conceptId]: { mastery, updatedAt: new Date().toISOString() } } });
}

/** Card ids due now: reviewed cards past due, plus unseen cards of concepts you've started. */
export function dueCardIds(p: Progress, allCardIds: string[], now = new Date()): string[] {
  const due: { id: string; due: number }[] = [];
  for (const id of allCardIds) {
    const c = p.cards[id];
    const conceptId = id.split("#")[0];
    if (c) {
      if (c.due.getTime() <= now.getTime()) due.push({ id, due: c.due.getTime() });
    } else if (masteryOf(p, conceptId) !== "new") {
      due.push({ id, due: Number.MAX_SAFE_INTEGER });
    }
  }
  return due.sort((a, b) => a.due - b.due).map((d) => d.id);
}
