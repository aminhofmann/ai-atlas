import { Suspense } from "react";
import { getGraph, learningPath, toPlanConcept } from "@/lib/content";
import { GraphBrowser } from "@/components/graph-browser";

async function Graph({ searchParams }: { searchParams: Promise<{ c?: string }> }) {
  const { c } = await searchParams;
  const data = getGraph();
  if (!data.nodes.length) return <p className="p-8 text-ink-3">No concepts yet.</p>;
  const start = c && data.nodes.some((n) => n.id === c) ? c : null;
  return <GraphBrowser data={data} path={learningPath().map(toPlanConcept)} start={start} />;
}

export default function Page(props: PageProps<"/graph">) {
  return (
    <Suspense fallback={<p className="p-8 text-ink-3">Loading the map…</p>}>
      <Graph searchParams={props.searchParams as Promise<{ c?: string }>} />
    </Suspense>
  );
}
