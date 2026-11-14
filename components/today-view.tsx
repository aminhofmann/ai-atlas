"use client";

import Link from "next/link";
import { buildSession, type PlanConcept, type SessionItem } from "@/lib/session";
import { dueCardIds, useProgress, useProgressReady } from "@/lib/progress";
import { CATEGORY_LABEL } from "@/lib/types";

const KIND: Record<SessionItem["kind"], { label: string; className: string }> = {
  review: { label: "Review", className: "text-due" },
  practice: { label: "Practice", className: "text-accent" },
  implement: { label: "Implement", className: "text-accent" },
  learn: { label: "Learn", className: "text-ink-2" },
};

export function TodayView({ path }: { path: PlanConcept[] }) {
  const progress = useProgress();
  const ready = useProgressReady();
  if (!ready) return <p className="text-ink-3">Planning today…</p>;

  const due = dueCardIds(progress, path.flatMap((c) => c.cardIds));
  const s = buildSession(path, progress, due);
  const first = s.items[0];
  const isFirstVisit = Object.keys(progress.concepts).length === 0;

  return (
    <div className="fade-in">
      <p className="text-sm text-ink-3">Today&apos;s session · {s.totalMinutes} min</p>

      {first ? (
        <>
          <h1 className="mt-2 font-serif text-3xl leading-tight tracking-tight">
            {first.kind === "review" ? "Start with your reviews." : `Start with ${first.title}.`}
          </h1>
          {isFirstVisit && (
            <p className="mt-3 text-sm leading-relaxed text-ink-2">
              Your progress is saved in this browser only — no account needed.
            </p>
          )}

          <ol className="mt-8 divide-y divide-line border-y border-line">
            {s.items.map((item, i) => (
              <li key={item.href} className="flex gap-4 py-4">
                <span className="w-5 pt-0.5 text-sm tabular-nums text-ink-3">{i + 1}</span>
                <div className="min-w-0 flex-1">
                  <p className="flex items-baseline justify-between gap-3">
                    <span>
                      <span className={`text-sm font-medium ${KIND[item.kind].className}`}>{KIND[item.kind].label}</span>
                      <span className="text-ink-3"> · </span>
                      <Link href={item.href} className="font-medium hover:underline">
                        {item.title}
                      </Link>
                    </span>
                    <span className="shrink-0 text-xs tabular-nums text-ink-3">{item.minutes} min</span>
                  </p>
                  <p className="mt-0.5 text-sm text-ink-2">{item.why}</p>
                </div>
              </li>
            ))}
          </ol>

          <Link
            href={first.href}
            className="mt-8 flex h-12 items-center justify-center rounded-xl bg-accent text-[15px] font-medium text-accent-ink transition-opacity hover:opacity-90"
          >
            Start session
          </Link>
          {s.items.length > 1 && (
            <p className="mt-3 text-center text-xs text-ink-3">Each step links to the next when you finish it.</p>
          )}
        </>
      ) : (
        <h1 className="mt-2 font-serif text-3xl leading-tight">Nothing queued — every concept is solid.</h1>
      )}

      {s.track && (
        <div className="mt-14 flex items-center gap-4 text-sm">
          <div className="min-w-0 flex-1">
            <p className="text-ink-2">
              Track: <span className="text-ink">{CATEGORY_LABEL[s.track]}</span>
            </p>
            <div className="mt-2 h-1 overflow-hidden rounded-full bg-panel-2">
              <div
                className="h-full rounded-full bg-accent"
                style={{ width: `${(s.trackProgress.done / Math.max(1, s.trackProgress.total)) * 100}%` }}
              />
            </div>
          </div>
          <p className="shrink-0 text-xs tabular-nums text-ink-3">
            {s.trackProgress.done} / {s.trackProgress.total} explainable
          </p>
        </div>
      )}
    </div>
  );
}
