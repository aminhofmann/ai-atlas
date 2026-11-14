"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { CATEGORIES, CATEGORY_LABEL, type GraphData, type Mastery } from "@/lib/types";
import { LocalGraph } from "./local-graph";
import { RebuildExercise } from "./rebuild-exercise";
import { GraphExplorer } from "./graph-explorer";
import { categoryColor, MasteryMeter } from "./category";

type View = "focus" | "list" | "map";

/**
 * Three ways to see the atlas. Focus (default) answers "how does this fit, and
 * what next?" with one concept and its neighbours; List is for scanning; Map
 * is the whole network for when you want the overview.
 */
export function GraphBrowser({
  data,
  mastery,
  start,
}: {
  data: GraphData;
  mastery: Record<string, Mastery>;
  start: string;
}) {
  const [view, setView] = useState<View>("focus");
  const [trail, setTrail] = useState<string[]>([start]);
  const [query, setQuery] = useState("");
  const [testing, setTesting] = useState(false);
  const center = trail[trail.length - 1];
  const byId = useMemo(() => new Map(data.nodes.map((n) => [n.id, n])), [data.nodes]);
  const c = byId.get(center);

  const go = (id: string) => setTrail((t) => (t.includes(id) ? t.slice(0, t.indexOf(id) + 1) : [...t, id].slice(-6)));

  useEffect(() => {
    window.history.replaceState(null, "", `/graph?c=${center}`);
  }, [center]);

  const q = query.trim().toLowerCase();
  const hits = q
    ? data.nodes.filter((n) => n.title.toLowerCase().includes(q) || n.id.includes(q)).slice(0, 8)
    : [];

  return (
    <div className="flex flex-1 flex-col">
      <div className="mx-auto flex w-full max-w-6xl flex-wrap items-center gap-4 px-5 pt-6">
        <div role="tablist" className="flex rounded-lg border border-line p-0.5 text-sm">
          {(["focus", "list", "map"] as const).map((v) => (
            <button
              key={v}
              role="tab"
              aria-selected={view === v}
              onClick={() => setView(v)}
              className={`rounded-md px-3 py-1 capitalize ${view === v ? "bg-panel-2 text-ink" : "text-ink-3 hover:text-ink"}`}
            >
              {v === "map" ? "Whole map" : v}
            </button>
          ))}
        </div>
        {view === "focus" && (
          <div className="relative w-full max-w-xs">
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Jump to a concept…"
              className="w-full rounded-lg border border-line bg-panel px-3 py-1.5 text-sm outline-none focus:border-accent"
            />
            {hits.length > 0 && (
              <ul className="absolute z-20 mt-1 w-full overflow-hidden rounded-lg border border-line bg-panel shadow-lg">
                {hits.map((h) => (
                  <li key={h.id}>
                    <button
                      onClick={() => {
                        go(h.id);
                        setQuery("");
                      }}
                      className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm hover:bg-panel-2"
                    >
                      <span className="size-1.5 rounded-full" style={{ background: categoryColor(h.category) }} />
                      {h.title}
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>

      {view === "focus" && c && (
        <div className="mx-auto w-full max-w-6xl px-5 py-6 fade-in">
          <nav className="flex flex-wrap items-center gap-1 text-sm text-ink-3" aria-label="Path through the map">
            {trail.map((id, i) => (
              <span key={id} className="flex items-center gap-1">
                {i > 0 && <span>›</span>}
                <button onClick={() => go(id)} className={i === trail.length - 1 ? "text-ink" : "hover:text-ink"}>
                  {byId.get(id)?.title}
                </button>
              </span>
            ))}
          </nav>
          {testing ? (
            <div className="mt-4">
              <RebuildExercise key={center} center={center} nodes={data.nodes} edges={data.edges} />
            </div>
          ) : (
            <div className="mt-4 rounded-2xl border border-line bg-panel">
              <LocalGraph center={center} nodes={data.nodes} edges={data.edges} mastery={mastery} onSelect={go} height={520} />
            </div>
          )}
          <div className="mt-5 flex flex-wrap items-start justify-between gap-4">
            <div className="max-w-2xl">
              <p className="font-serif text-xl">{c.title}</p>
              <p className="mt-1 text-sm leading-relaxed text-ink-2">{c.summary}</p>
            </div>
            <div className="flex shrink-0 gap-2">
              <button
                onClick={() => setTesting((t) => !t)}
                className="h-10 rounded-xl border border-line px-4 text-ink-2 hover:text-ink"
              >
                {testing ? "Back to the map" : "Rebuild from memory"}
              </button>
              <Link href={`/c/${c.id}`} className="h-10 rounded-xl bg-accent px-5 leading-10 font-medium text-accent-ink">
                Study {c.title.length > 28 ? "this" : c.title}
              </Link>
            </div>
          </div>
          <p className="mt-6 text-xs text-ink-3">Click any neighbour to move the focus there.</p>
        </div>
      )}

      {view === "list" && (
        <div className="mx-auto w-full max-w-6xl columns-1 gap-10 px-5 py-8 md:columns-2 lg:columns-3">
          {CATEGORIES.map((cat) => {
            const items = data.nodes.filter((n) => n.category === cat).sort((a, b) => a.level - b.level || a.title.localeCompare(b.title));
            if (!items.length) return null;
            return (
              <section key={cat} className="mb-8 break-inside-avoid">
                <h2 className="mb-2 flex items-center gap-2 text-xs font-medium uppercase tracking-wider text-ink-3">
                  <span className="size-1.5 rounded-full" style={{ background: categoryColor(cat) }} />
                  {CATEGORY_LABEL[cat]}
                </h2>
                <ul>
                  {items.map((n) => (
                    <li key={n.id}>
                      <Link href={`/c/${n.id}`} className="flex items-center justify-between gap-3 border-b border-line py-2 text-sm hover:text-accent">
                        <span className="truncate">{n.title}</span>
                        <MasteryMeter mastery={mastery[n.id] ?? "new"} />
                      </Link>
                    </li>
                  ))}
                </ul>
              </section>
            );
          })}
        </div>
      )}

      {view === "map" && (
        <div className="flex h-[calc(100dvh-7.5rem)] flex-col">
          <GraphExplorer data={data} mastery={mastery} />
        </div>
      )}
    </div>
  );
}
