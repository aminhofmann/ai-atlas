"use client";

import { useState } from "react";
import Link from "next/link";
import { dueCardIds, useProgress, useProgressReady, type Progress } from "@/lib/progress";
import { PracticeSession, type PracticeCard } from "./practice-session";

export function ReviewView({ cards }: { cards: PracticeCard[] }) {
  const ready = useProgressReady();
  const progress = useProgress();
  if (!ready) return <p className="text-ink-3">Loading your reviews…</p>;
  // Mounted only once progress is loaded, so the queue is computed from real data.
  return <Queue cards={cards} progress={progress} />;
}

function Queue({ cards, progress }: { cards: PracticeCard[]; progress: Progress }) {
  // Frozen at the start of the session: grading a card must not reshuffle the rest.
  const [queue] = useState(() => {
    const byId = new Map(cards.map((c) => [c.id, c]));
    return dueCardIds(progress, [...byId.keys()]).map((id) => byId.get(id)!);
  });

  if (queue.length === 0) {
    return (
      <div className="py-10 text-center">
        <p className="font-serif text-2xl">Nothing due.</p>
        <p className="mt-2 text-sm text-ink-3">
          Questions join this queue once you start a concept.{" "}
          <Link className="text-accent hover:underline" href="/">
            Back to today →
          </Link>
        </p>
      </div>
    );
  }
  return <PracticeSession cards={queue} showConcept done={{ label: "Back to today", href: "/" }} />;
}
