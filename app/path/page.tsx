import { Suspense } from "react";
import Link from "next/link";
import { learningPath, toPlanConcept } from "@/lib/content";
import { CATEGORIES, CATEGORY_LABEL } from "@/lib/types";
import { categoryColor } from "@/components/category";
import { PathView } from "@/components/path-view";

async function PathList({ searchParams }: { searchParams: Promise<{ track?: string }> }) {
  const { track } = await searchParams;
  const all = learningPath().map(toPlanConcept);
  const valid = track && (CATEGORIES as readonly string[]).includes(track);
  const list = valid ? all.filter((c) => c.category === track) : all;

  return (
    <>
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
      <PathView list={list} />
    </>
  );
}

export default function Page(props: PageProps<"/path">) {
  return (
    <div className="mx-auto w-full max-w-2xl px-5 py-12">
      <h1 className="font-serif text-3xl tracking-tight">Learning path</h1>
      <Suspense fallback={<p className="mt-4 text-ink-3">Loading…</p>}>
        <PathList searchParams={props.searchParams as Promise<{ track?: string }>} />
      </Suspense>
    </div>
  );
}
