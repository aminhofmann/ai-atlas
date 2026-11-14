import { Suspense } from "react";
import { connection } from "next/server";
import { getGraph, learningPath } from "@/lib/content";
import { readProgress } from "@/lib/progress";
import { nextConcept } from "@/lib/session";
import { GraphBrowser } from "@/components/graph-browser";

async function Graph({ searchParams }: { searchParams: Promise<{ c?: string }> }) {
  await connection();
  const { c } = await searchParams;
  const data = getGraph();
  const progress = readProgress();
  const mastery = Object.fromEntries(Object.entries(progress.concepts).map(([id, v]) => [id, v.mastery]));
  // Open on the requested concept, else on what you'd study next.
  const start =
    (c && data.nodes.some((n) => n.id === c) && c) ||
    nextConcept(learningPath(), progress)?.id ||
    data.nodes[0]?.id;
  if (!start) return <p className="p-8 text-ink-3">No concepts yet.</p>;
  return <GraphBrowser data={data} mastery={mastery} start={start} />;
}

export default function Page(props: PageProps<"/graph">) {
  return (
    <Suspense fallback={<p className="p-8 text-ink-3">Loading the map…</p>}>
      <Graph searchParams={props.searchParams as Promise<{ c?: string }>} />
    </Suspense>
  );
}
