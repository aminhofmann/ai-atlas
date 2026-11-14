"use client";

import { MASTERY_HINT, MASTERY_LABEL, MASTERY_LEVELS, type Mastery } from "@/lib/types";
import { masteryOf, setMastery, useProgress } from "@/lib/progress";

/** Self-assessment on what you can DO with the concept. */
export function MasterySelect({ id }: { id: string }) {
  const value = masteryOf(useProgress(), id);
  const level = MASTERY_LEVELS.indexOf(value);
  const set = (m: Mastery) => setMastery(id, m);

  return (
    <div>
      <div className="flex gap-1" role="radiogroup" aria-label="How well do you know this?">
        {MASTERY_LEVELS.slice(1).map((m, i) => (
          <button
            key={m}
            role="radio"
            aria-checked={value === m}
            title={MASTERY_HINT[m]}
            onClick={() => set(value === m ? MASTERY_LEVELS[i] : m)}
            className="group flex-1"
          >
            <span
              className="block h-1.5 rounded-full transition-colors"
              style={{ background: i < level ? `var(--m-${value})` : "var(--m-new)" }}
            />
            <span className={`mt-1.5 block text-[11px] ${value === m ? "font-medium text-ink" : "text-ink-3 group-hover:text-ink-2"}`}>
              {MASTERY_LABEL[m]}
            </span>
          </button>
        ))}
      </div>
      <p className="mt-2 text-xs text-ink-3">{MASTERY_HINT[value]}</p>
    </div>
  );
}
