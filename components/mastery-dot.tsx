"use client";

import { masteryOf, useProgress } from "@/lib/progress";
import { MasteryMeter } from "./category";

/** Small mastery dot for a concept, read from the visitor's browser progress. Nothing while "new". */
export function MasteryDot({ id }: { id: string }) {
  const m = masteryOf(useProgress(), id);
  if (m === "new") return null;
  return <span className="size-1.5 shrink-0 rounded-full" style={{ background: `var(--m-${m})` }} title={m} />;
}

/** Five-step meter for a concept, from browser progress. */
export function MasteryMeterLive({ id }: { id: string }) {
  return <MasteryMeter mastery={masteryOf(useProgress(), id)} />;
}
