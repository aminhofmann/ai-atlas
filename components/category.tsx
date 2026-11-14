import { CATEGORY_LABEL, MASTERY_LABEL, MASTERY_LEVELS, type Category, type Mastery } from "@/lib/types";

export const categoryColor = (c: Category) => `var(--c-${c})`;

export function CategoryTag({ category }: { category: Category }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-ink-3">
      <span aria-hidden className="size-1.5 rounded-full" style={{ background: categoryColor(category) }} />
      {CATEGORY_LABEL[category]}
    </span>
  );
}

export const LEVEL_LABEL: Record<number, string> = { 1: "Core", 2: "Intermediate", 3: "Advanced" };

/** Five-step bar, filled up to the current level. */
export function MasteryMeter({ mastery, size = "sm" }: { mastery: Mastery; size?: "sm" | "md" }) {
  const level = MASTERY_LEVELS.indexOf(mastery);
  const w = size === "sm" ? "w-2.5" : "w-4";
  return (
    <span className="inline-flex items-center gap-0.5" title={MASTERY_LABEL[mastery]} aria-label={`Mastery: ${MASTERY_LABEL[mastery]}`}>
      {MASTERY_LEVELS.slice(1).map((m, i) => (
        <span
          key={m}
          className={`h-1.5 ${w} rounded-full`}
          style={{ background: i < level ? `var(--m-${mastery})` : "var(--m-new)" }}
        />
      ))}
    </span>
  );
}
