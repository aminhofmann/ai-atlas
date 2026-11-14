import { Suspense } from "react";
import Link from "next/link";
import { notFound } from "next/navigation";
import { connection } from "next/server";
import { getAllConcepts, getGraph, learningPath } from "@/lib/content";
import { readProgress } from "@/lib/progress";
import { relationsOf } from "@/lib/relations";
import { firstParagraph, nextConcept, studyTime } from "@/lib/session";
import type { Concept, Mastery } from "@/lib/types";
import { Markdown } from "@/components/markdown";
import { CategoryTag, categoryColor, LEVEL_LABEL } from "@/components/category";
import { MasterySelect } from "@/components/mastery-select";
import { ConceptWorkspace, type LearnStep, type Mode } from "@/components/concept-workspace";
import { getDiagram } from "@/components/diagrams";

const MODES: Mode[] = ["learn", "practice", "implement", "graph"];

function splitCode(md: string) {
  const fences = md.match(/```[\s\S]*?```/g) ?? [];
  const brief = md.replace(/```[\s\S]*?```/g, "").trim();
  const code = fences.map((f) => f.replace(/^```\w*\n?/, "").replace(/```$/, "").trimEnd()).join("\n\n");
  return { brief, fences: fences.join("\n\n"), code };
}

function DiagramFigure({ id }: { id: string }) {
  const d = getDiagram(id);
  if (!d) return null;
  return (
    <figure className="rounded-2xl border border-line bg-panel p-5">
      <d.Diagram />
      <figcaption className="mt-3 text-sm text-ink-3">{d.caption}</figcaption>
    </figure>
  );
}

function RailLink({ c, mastery }: { c: Concept; mastery: Mastery }) {
  return (
    <Link href={`/c/${c.id}`} className="group flex items-center gap-2 py-1">
      <span className="size-1.5 shrink-0 rounded-full" style={{ background: categoryColor(c.category) }} />
      <span className="min-w-0 flex-1 truncate text-ink-2 group-hover:text-ink">{c.title}</span>
      {mastery !== "new" && <span className="size-1.5 shrink-0 rounded-full" style={{ background: `var(--m-${mastery})` }} />}
    </Link>
  );
}

async function ConceptPage({ params, searchParams }: { params: Promise<{ id: string }>; searchParams: Promise<{ mode?: string }> }) {
  await connection();
  const [{ id }, { mode }] = await Promise.all([params, searchParams]);
  const all = getAllConcepts();
  const byId = new Map(all.map((c) => [c.id, c]));
  const c = byId.get(id);
  if (!c) notFound();

  const progress = readProgress();
  const masteryMap: Record<string, Mastery> = Object.fromEntries(
    Object.entries(progress.concepts).map(([k, v]) => [k, v.mastery]),
  );
  const m = (cid: string) => masteryMap[cid] ?? "new";
  const graph = getGraph(all);
  const time = studyTime(c);
  const hasDiagram = !!getDiagram(id);

  // --- Learn: one question per screen, visual first ---
  const whyParas = c.sections.why.trim().split(/\n\s*\n/);
  const problem = firstParagraph(c.sections.why);
  const steps: LearnStep[] = [
    {
      key: "what",
      label: "What it is",
      question: `What is ${c.title}, and what problem does it solve?`,
      content: (
        <div className="space-y-8">
          <p className="max-w-[60ch] font-serif text-2xl leading-snug">{c.summary}</p>
          {problem && (
            <div className="max-w-[68ch] rounded-2xl bg-accent-soft px-6 py-5">
              <p className="text-xs font-medium uppercase tracking-wider text-accent">Why it exists</p>
              <Markdown className="mt-2 prose-compact">{problem}</Markdown>
            </div>
          )}
          <DiagramFigure id={id} />
        </div>
      ),
    },
    c.sections.intuition && {
      key: "intuition",
      label: "Intuition",
      question: "How should you picture it?",
      content: <Markdown>{c.sections.intuition}</Markdown>,
    },
    c.sections.mechanics && {
      key: "mechanics",
      label: "Mechanics",
      question: "How does it actually work?",
      content: (
        <div className="space-y-8">
          {hasDiagram && <DiagramFigure id={id} />}
          <Markdown>{c.sections.mechanics}</Markdown>
        </div>
      ),
    },
    whyParas.length > 1 && {
      key: "why",
      label: "Why it matters",
      question: "Where does it sit in the field today?",
      content: <Markdown>{whyParas.slice(1).join("\n\n")}</Markdown>,
    },
  ].filter(Boolean) as LearnStep[];

  // --- Implement ---
  const code = splitCode(c.sections.code);
  const implement = code.code
    ? {
        brief: code.brief ? <Markdown className="prose-compact">{code.brief}</Markdown> : null,
        solution: <Markdown>{code.fences}</Markdown>,
        code: code.code,
      }
    : null;

  // --- Context rail ---
  const prereqs = c.requires.map((r) => byId.get(r)).filter((x): x is Concept => !!x);
  const unlocks = all.filter((o) => o.requires.includes(id));
  const related = new Map<string, Concept[]>();
  for (const r of relationsOf(id, graph.edges)) {
    const other = byId.get(r.otherId);
    const list = related.get(r.phrase) ?? [];
    if (other && !list.includes(other)) related.set(r.phrase, [...list, other]);
  }
  const relatedCount = [...related.values()].reduce((s, l) => s + l.length, 0);

  const next = nextConcept(learningPath(all), progress);
  const nextHref = next && next.id !== id ? `/c/${next.id}?mode=learn` : "/";

  const header = (
    <header>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-ink-3">
        <CategoryTag category={c.category} />
        <span>{LEVEL_LABEL[c.level] ?? `Level ${c.level}`}</span>
        <span>
          {time.core} min core · {time.deep} min deep
        </span>
        {c.year && <span>{c.year}</span>}
      </div>
      <h1 className="mt-2 font-serif text-4xl leading-tight tracking-tight">{c.title}</h1>
    </header>
  );

  const rail = (
    <div className="space-y-7">
      <section>
        <h3 className="mb-3 text-xs font-medium uppercase tracking-wider text-ink-3">I can…</h3>
        <MasterySelect id={id} initial={m(id)} />
      </section>
      <section>
        <h3 className="mb-1 text-xs font-medium uppercase tracking-wider text-prereq">Learn first</h3>
        {prereqs.length ? prereqs.map((p) => <RailLink key={p.id} c={p} mastery={m(p.id)} />) : <p className="py-1 text-ink-3">Nothing — a starting point.</p>}
      </section>
      {unlocks.length > 0 && (
        <section>
          <h3 className="mb-1 text-xs font-medium uppercase tracking-wider text-ink-3">Unlocks</h3>
          {unlocks.slice(0, 5).map((p) => <RailLink key={p.id} c={p} mastery={m(p.id)} />)}
          {unlocks.length > 5 && <p className="py-1 text-xs text-ink-3">+{unlocks.length - 5} more in Graph</p>}
        </section>
      )}
      {relatedCount > 0 && (
        <details className="group">
          <summary className="cursor-pointer list-none text-xs font-medium uppercase tracking-wider text-ink-3 hover:text-ink">
            Related ({relatedCount}) <span className="group-open:hidden">+</span>
          </summary>
          <div className="mt-2 space-y-3">
            {[...related.entries()].map(([phrase, items]) => (
              <div key={phrase}>
                <p className="text-xs text-ink-3">{phrase}</p>
                {items.map((p) => <RailLink key={p.id} c={p} mastery={m(p.id)} />)}
              </div>
            ))}
          </div>
        </details>
      )}
      {c.sources.length > 0 && (
        <details className="group">
          <summary className="cursor-pointer list-none text-xs font-medium uppercase tracking-wider text-ink-3 hover:text-ink">
            Sources ({c.sources.length}) <span className="group-open:hidden">+</span>
          </summary>
          <ul className="mt-2 space-y-2">
            {c.sources.map((s, i) => (
              <li key={i} className="leading-snug">
                {s.url ? (
                  <a href={s.url} target="_blank" rel="noreferrer" className="text-ink-2 underline decoration-line underline-offset-2 hover:text-ink">
                    {s.title}
                  </a>
                ) : (
                  <span className="text-ink-2">{s.title}</span>
                )}
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );

  return (
    <ConceptWorkspace
      key={id}
      id={id}
      title={c.title}
      header={header}
      rail={rail}
      initialMode={MODES.includes(mode as Mode) ? (mode as Mode) : "learn"}
      steps={steps}
      cards={c.cards}
      implement={implement}
      graph={{ nodes: graph.nodes, edges: graph.edges, mastery: masteryMap }}
      nextHref={nextHref}
    />
  );
}

export default function Page(props: PageProps<"/c/[id]">) {
  return (
    <Suspense fallback={<div className="mx-auto w-full max-w-6xl px-5 py-8 text-ink-3">Loading…</div>}>
      <ConceptPage params={props.params} searchParams={props.searchParams as Promise<{ mode?: string }>} />
    </Suspense>
  );
}
