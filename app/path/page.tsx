import { Suspense } from "react";
import Link from "next/link";
import { connection } from "next/server";
import { learningPath } from "@/lib/content";
import { readProgress } from "@/lib/progress";
import { nextConcept } from "@/lib/session";
import { CATEGORIES, CATEGORY_LABEL, isSolid } from "@/lib/types";
import { categoryColor, MasteryMeter } from "@/components/category";

async function PathList({ searchParams }: { searchParams: Promise<{ track?: string }> }) {
  await connection();
  const { track } = await searchParams;
  const progress = readProgress();
  const all = learningPath();
  const valid = track && (CATEGORIES as readonly string[]).includes(track);
  const list = valid ? all.filter((c) => c.category === track) : all;
  const solid = list.filter((c) => isSolid(progress.concepts[c.id]?.mastery)).length;
  const next = nextConcept(list, progress);

  return (
    <div className="mx-auto w-full max-w-2xl px-5 py-12">
      <h1 className="font-serif text-3xl tracking-tight">Learning path</h1>
      <p className="mt-2 text-ink-2">Prerequisites always come first. {solid} of {list.length} you can explain.</p>

      <div className="mt-6 flex flex-wrap gap-x-4 gap-y-2 text-sm">
        <Link href="/path" className={!valid ? "text-ink" : "text-ink-3 hover:text-ink"}>
          All
        </Link>
        {CATEGORIES.map((c) => (
          <Link key={c} href={`/path?track=${c}`} className={`flex items-center gap-1.5 ${track === c ? "text-ink" : "text-ink-3 hover:text-ink"}`}>
            <span className="size-1.5 rounded-full" style={{ background: categoryColor(c) }} />
            {CATEGORY_LABEL[c]}
          </Link>
        ))}
      </div>

      <ol className="mt-8">
        {list.map((c, i) => {
          const m = progress.concepts[c.id]?.mastery ?? "new";
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
    </div>
  );
}

export default function Page(props: PageProps<"/path">) {
  return (
    <Suspense fallback={<p className="p-8 text-ink-3">Loading…</p>}>
      <PathList searchParams={props.searchParams as Promise<{ track?: string }>} />
    </Suspense>
  );
}
