"use client";

import Link from "next/link";
import { masteryOf, useProgress } from "@/lib/progress";
import { nextConcept, type PlanConcept } from "@/lib/session";
import { isSolid } from "@/lib/types";
import { MasteryMeter } from "./category";

export function PathView({ list }: { list: PlanConcept[] }) {
  const progress = useProgress();
  const solid = list.filter((c) => isSolid(progress.concepts[c.id]?.mastery)).length;
  const next = nextConcept(list, progress);

  return (
    <>
      <p className="mt-2 text-ink-2">
        Prerequisites always come first. {solid} of {list.length} you can explain.
      </p>
      <ol className="mt-8">
        {list.map((c, i) => {
          const m = masteryOf(progress, c.id);
          const isNext = next?.id === c.id;
          return (
            <li key={c.id}>
              <Link
                href={`/c/${c.id}`}
                className={`group flex items-baseline gap-4 border-b border-line py-3 ${isNext ? "-mx-4 rounded-xl border-transparent bg-accent-soft px-4" : ""}`}
              >
                <span className="w-6 shrink-0 text-right text-xs tabular-nums text-ink-3">{i + 1}</span>
                <span className="min-w-0 flex-1">
                  <span className="flex items-center gap-2">
                    <span className={`truncate ${m === "new" && !isNext ? "text-ink-2" : "text-ink"} group-hover:underline`}>{c.title}</span>
                    {isNext && <span className="shrink-0 text-xs font-medium text-accent">Up next</span>}
                  </span>
                  {isNext && <span className="mt-1 block text-sm leading-snug text-ink-2">{c.summary}</span>}
                </span>
                <MasteryMeter mastery={m} />
              </Link>
            </li>
          );
        })}
      </ol>
    </>
  );
}
