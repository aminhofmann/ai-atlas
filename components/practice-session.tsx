"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import type { Card } from "@/lib/types";
import { Markdown } from "./markdown-client";
import { reviewCard, type GradeName } from "@/lib/progress";

export type { GradeName };

export type PracticeCard = Card & { conceptId?: string; conceptTitle?: string };

const GRADES: { grade: GradeName; label: string; key: string }[] = [
  { grade: "again", label: "I was wrong", key: "1" },
  { grade: "hard", label: "Hard", key: "2" },
  { grade: "good", label: "Good", key: "3" },
  { grade: "easy", label: "Easy", key: "4" },
];


const isTyping = (t: EventTarget | null) =>
  t instanceof HTMLElement && (t.tagName === "TEXTAREA" || t.tagName === "INPUT" || t.isContentEditable);

/**
 * One question per screen: attempt first (write it down), then reveal, then
 * grade yourself. Feedback appears only after the attempt.
 */
export function PracticeSession({
  cards,
  showConcept = false,
  onAskTutor,
  done,
}: {
  cards: PracticeCard[];
  showConcept?: boolean;
  onAskTutor?: (prompt: string) => void;
  done?: { label: string; href?: string; onClick?: () => void };
}) {
  const [index, setIndex] = useState(0);
  const [busy, setBusy] = useState(false);
  const [missed, setMissed] = useState(0);
  const card = cards[index];

  const grade = useCallback(
    async (g: GradeName) => {
      if (!card || busy) return;
      setBusy(true);
      try {
        reviewCard(card.id, g);
        if (g === "again") setMissed((n) => n + 1);
        setIndex((i) => i + 1);
      } finally {
        setBusy(false);
      }
    },
    [card, busy],
  );

  if (cards.length === 0) {
    return <p className="text-ink-2">No self-test questions for this concept yet.</p>;
  }

  if (!card) {
    return (
      <div className="fade-in py-10 text-center">
        <p className="font-serif text-2xl">
          Done — {cards.length} question{cards.length === 1 ? "" : "s"}.
        </p>
        <p className="mt-2 text-sm text-ink-3">
          {missed
            ? `${missed} to revisit soon; they'll come back in your review queue.`
            : "All recalled. They'll come back just before you'd forget them."}
        </p>
        {done &&
          (done.href ? (
            <Link href={done.href} className="mt-8 inline-flex h-11 items-center rounded-xl bg-accent px-6 font-medium text-accent-ink">
              {done.label}
            </Link>
          ) : (
            <button onClick={done.onClick} className="mt-8 inline-flex h-11 items-center rounded-xl bg-accent px-6 font-medium text-accent-ink">
              {done.label}
            </button>
          ))}
      </div>
    );
  }

  return (
    <div>
      <div className="flex items-baseline justify-between text-sm text-ink-3">
        <span className="tabular-nums">
          Question {index + 1} / {cards.length}
        </span>
        {showConcept && card.conceptId && (
          <Link href={`/c/${card.conceptId}`} className="hover:text-ink hover:underline">
            {card.conceptTitle}
          </Link>
        )}
      </div>
      <div className="mt-1 h-0.5 overflow-hidden rounded-full bg-panel-2">
        <div className="h-full bg-accent transition-[width] duration-300" style={{ width: `${(index / cards.length) * 100}%` }} />
      </div>
      {/* Keyed per card, so the attempt and reveal state start fresh for each question. */}
      <Question key={card.id} card={card} busy={busy} onGrade={grade} onAskTutor={onAskTutor} />
    </div>
  );
}

function Question({
  card,
  busy,
  onGrade,
  onAskTutor,
}: {
  card: PracticeCard;
  busy: boolean;
  onGrade: (g: GradeName) => void;
  onAskTutor?: (prompt: string) => void;
}) {
  const [attempt, setAttempt] = useState("");
  const [revealed, setRevealed] = useState(false);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (!revealed && e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setRevealed(true);
        return;
      }
      if (isTyping(e.target) || e.metaKey || e.ctrlKey || e.altKey) return;
      if (!revealed && e.key === " ") {
        e.preventDefault();
        setRevealed(true);
      } else if (revealed) {
        const g = GRADES.find((x) => x.key === e.key);
        if (g) onGrade(g.grade);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [revealed, onGrade]);

  const mine = attempt.trim();

  return (
    <div className="fade-in">
      <div className="mt-10 font-serif text-2xl leading-snug [&_.prose-atlas]:text-[inherit] [&_.prose-atlas]:leading-snug">
        <Markdown>{card.q}</Markdown>
      </div>

      {!revealed ? (
        <>
          <textarea
            autoFocus
            value={attempt}
            onChange={(e) => setAttempt(e.target.value)}
            placeholder="Write your explanation — even a rough one. Retrieval is the exercise."
            rows={5}
            className="mt-8 w-full resize-y rounded-xl border border-line bg-panel px-4 py-3 text-[15px] leading-relaxed outline-none placeholder:text-ink-3 focus:border-accent"
          />
          <div className="mt-4 flex items-center gap-3">
            <button onClick={() => setRevealed(true)} className="h-11 rounded-xl bg-accent px-5 font-medium text-accent-ink">
              Reveal answer
            </button>
            <span className="text-xs text-ink-3">
              <kbd>⌘</kbd> <kbd>↵</kbd> while typing · <kbd>Space</kbd> otherwise
            </span>
          </div>
        </>
      ) : (
        <div className="mt-8 space-y-5 fade-in">
          {mine && (
            <section>
              <h3 className="text-xs font-medium uppercase tracking-wider text-ink-3">Your answer</h3>
              <p className="mt-1.5 whitespace-pre-wrap border-l-2 border-line pl-4 text-[15px] leading-relaxed text-ink-2">{mine}</p>
            </section>
          )}
          <section>
            <h3 className="text-xs font-medium uppercase tracking-wider text-ink-3">Reference</h3>
            <div className="mt-1.5 border-l-2 border-accent pl-4">
              <Markdown>{card.a}</Markdown>
            </div>
          </section>

          {onAskTutor && (
            <button
              onClick={() =>
                onAskTutor(
                  mine
                    ? `Question: ${card.q}\n\nMy answer: ${mine}\n\nReference: ${card.a}\n\nWhat did I get right, what did I miss, and what follow-up question would test whether I really understand it?`
                    : `Question: ${card.q}\n\nReference answer: ${card.a}\n\nExplain why this is the answer, then ask me one follow-up question.`,
                )
              }
              className="text-sm text-accent hover:underline"
            >
              {mine ? "Get feedback on my answer →" : "Ask the tutor why →"}
            </button>
          )}

          <div className="border-t border-line pt-5">
            <p className="mb-3 text-sm text-ink-2">How well did you recall it?</p>
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
              {GRADES.map((g) => (
                <button
                  key={g.grade}
                  disabled={busy}
                  onClick={() => onGrade(g.grade)}
                  className="flex h-12 items-center justify-center gap-2 rounded-xl border border-line bg-panel text-sm transition-colors hover:border-accent disabled:opacity-50"
                >
                  {g.label} <kbd>{g.key}</kbd>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
