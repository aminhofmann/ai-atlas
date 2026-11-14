"use client";

import { useCallback, useEffect, useState, type ReactNode } from "react";
import { useRouter } from "next/navigation";
import type { ConceptMeta, Edge } from "@/lib/types";
import { PracticeSession, type PracticeCard } from "./practice-session";
import { LocalGraph } from "./local-graph";
import { RebuildExercise } from "./rebuild-exercise";
import { masteryMap, useProgress } from "@/lib/progress";
import { TutorPanel } from "./tutor-panel";

export type Mode = "learn" | "practice" | "implement" | "graph";
const MODES: { key: Mode; label: string; shortcut: string }[] = [
  { key: "learn", label: "Learn", shortcut: "L" },
  { key: "practice", label: "Practice", shortcut: "P" },
  { key: "implement", label: "Implement", shortcut: "I" },
  { key: "graph", label: "Graph", shortcut: "G" },
];

const isTyping = (t: EventTarget | null) =>
  t instanceof HTMLElement && (t.tagName === "TEXTAREA" || t.tagName === "INPUT" || t.isContentEditable);

export type LearnStep = { key: string; label: string; question: string; content: ReactNode };

export function ConceptWorkspace(props: {
  id: string;
  title: string;
  header: ReactNode;
  rail: ReactNode;
  initialMode: Mode;
  steps: LearnStep[];
  cards: PracticeCard[];
  implement: { brief: ReactNode; solution: ReactNode; code: string } | null;
  graph: { nodes: ConceptMeta[]; edges: Edge[] };
  nextHref: string;
  /** Show the Claude tutor (drawer, E key, answer feedback, hints). Off when no API key is configured. */
  tutor: boolean;
}) {
  const [mode, setModeState] = useState<Mode>(props.initialMode);
  const [step, setStep] = useState(0);
  const [tutorOpen, setTutorOpen] = useState(false);
  const [graphView, setGraphView] = useState<"explore" | "rebuild">("explore");
  const [tutorPrompt, setTutorPrompt] = useState<{ text: string; nonce: number } | null>(null);
  const router = useRouter();
  const mastery = masteryMap(useProgress());

  const setMode = useCallback(
    (m: Mode) => {
      setModeState(m);
      window.history.replaceState(null, "", `/c/${props.id}?mode=${m}`);
      window.scrollTo({ top: 0 });
    },
    [props.id],
  );

  const askTutor = useCallback((text: string) => {
    setTutorPrompt({ text, nonce: Date.now() });
    setTutorOpen(true);
  }, []);

  const lastStep = props.steps.length - 1;
  const next = useCallback(() => {
    if (step < lastStep) {
      setStep((s) => s + 1);
      window.scrollTo({ top: 0, behavior: "smooth" });
    } else setMode("practice");
  }, [step, lastStep, setMode]);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") return setTutorOpen(false);
      if (isTyping(e.target) || e.metaKey || e.ctrlKey || e.altKey) return;
      const k = e.key.toLowerCase();
      const m = MODES.find((x) => x.shortcut.toLowerCase() === k);
      if (m) return setMode(m.key);
      if (k === "e" && props.tutor) return setTutorOpen((o) => !o);
      if (mode === "learn" && (k === "n" || e.key === "ArrowRight")) return next();
      if (mode === "learn" && e.key === "ArrowLeft") return setStep((s) => Math.max(0, s - 1));
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [mode, next, setMode, props.tutor]);

  const current = props.steps[step];

  return (
    <div className="mx-auto w-full max-w-6xl px-5 pb-24 pt-8">
      {props.header}

      {/* Mode switch */}
      <div className="mt-6 flex items-center justify-between gap-4 border-b border-line">
        <div role="tablist" className="-mb-px flex gap-6">
          {MODES.map((m) => (
            <button
              key={m.key}
              role="tab"
              aria-selected={mode === m.key}
              onClick={() => setMode(m.key)}
              className={`border-b-2 pb-3 text-[15px] transition-colors ${
                mode === m.key ? "border-accent font-medium text-ink" : "border-transparent text-ink-3 hover:text-ink"
              }`}
            >
              {m.label}
              {m.key === "practice" && props.cards.length > 0 && (
                <span className="ml-1.5 text-xs tabular-nums text-ink-3">{props.cards.length}</span>
              )}
            </button>
          ))}
        </div>
        {props.tutor && (
          <button
            onClick={() => setTutorOpen((o) => !o)}
            className="mb-2 flex items-center gap-2 rounded-lg border border-line px-3 py-1.5 text-sm text-ink-2 hover:text-ink"
          >
            Ask tutor <kbd>E</kbd>
          </button>
        )}
      </div>

      <div className="mt-10 grid gap-12 lg:grid-cols-[minmax(0,1fr)_240px]">
        <div className="min-w-0">
          {mode === "learn" && current && (
            <div key={current.key} className="fade-in">
              {/* Step dots */}
              <div className="flex items-center gap-3 text-sm text-ink-3">
                <div className="flex gap-1.5">
                  {props.steps.map((s, i) => (
                    <button
                      key={s.key}
                      onClick={() => setStep(i)}
                      aria-label={`Step ${i + 1}: ${s.label}`}
                      className={`h-1.5 rounded-full transition-all ${i === step ? "w-6 bg-accent" : i < step ? "w-1.5 bg-ink-3" : "w-1.5 bg-line"}`}
                    />
                  ))}
                </div>
                <span>
                  {step + 1}/{props.steps.length} · {current.label}
                </span>
              </div>
              <h2 className="mt-4 text-sm font-medium text-ink-2">{current.question}</h2>
              <div className="mt-5">{current.content}</div>

              <div className="mt-12 flex items-center gap-4">
                {step > 0 && (
                  <button onClick={() => setStep((s) => s - 1)} className="h-11 rounded-xl px-4 text-ink-2 hover:text-ink">
                    Back
                  </button>
                )}
                <button onClick={next} className="h-11 rounded-xl bg-accent px-6 font-medium text-accent-ink hover:opacity-90">
                  {step < lastStep ? `Continue: ${props.steps[step + 1].label}` : "Practice it"}
                </button>
                <span className="text-xs text-ink-3">
                  <kbd>N</kbd> next
                </span>
              </div>
            </div>
          )}

          {mode === "practice" && (
            <div className="max-w-2xl">
              <PracticeSession
                cards={props.cards}
                onAskTutor={props.tutor ? askTutor : undefined}
                done={props.implement ? { label: "Implement it", onClick: () => setMode("implement") } : { label: "Next in today's session", href: props.nextHref }}
              />
            </div>
          )}

          {mode === "implement" && (
            <Implement data={props.implement} onAskTutor={props.tutor ? askTutor : undefined} title={props.title} nextHref={props.nextHref} />
          )}

          {mode === "graph" && (
            <div className="fade-in">
              <div className="flex rounded-lg border border-line p-0.5 text-sm w-fit" role="tablist">
                {(
                  [
                    ["explore", "Explore"],
                    ["rebuild", "Rebuild from memory"],
                  ] as const
                ).map(([k, label]) => (
                  <button
                    key={k}
                    role="tab"
                    aria-selected={graphView === k}
                    onClick={() => setGraphView(k)}
                    className={`rounded-md px-3 py-1 ${graphView === k ? "bg-panel-2 text-ink" : "text-ink-3 hover:text-ink"}`}
                  >
                    {label}
                  </button>
                ))}
              </div>
              {graphView === "explore" ? (
                <>
                  <p className="mt-4 text-sm text-ink-2">How {props.title} fits — one step out in each direction. Click a concept to go there.</p>
                  <div className="mt-4 rounded-2xl border border-line bg-panel">
                    <LocalGraph
                      center={props.id}
                      nodes={props.graph.nodes}
                      edges={props.graph.edges}
                      mastery={mastery}
                      onSelect={(id) => id !== props.id && router.push(`/c/${id}?mode=graph`)}
                    />
                  </div>
                </>
              ) : (
                <div className="mt-4">
                  <RebuildExercise center={props.id} nodes={props.graph.nodes} edges={props.graph.edges} />
                </div>
              )}
            </div>
          )}
        </div>

        <aside className="text-sm lg:sticky lg:top-24 lg:self-start">{props.rail}</aside>
      </div>

      {/* Tutor drawer — only when an Anthropic API key is configured */}
      {props.tutor && (
        <div
          className={`fixed inset-y-0 right-0 z-40 w-full max-w-md border-l border-line bg-panel shadow-2xl transition-transform duration-200 ${
            tutorOpen ? "translate-x-0" : "translate-x-full"
          }`}
          aria-hidden={!tutorOpen}
        >
          <TutorPanel conceptId={props.id} title={props.title} prompt={tutorPrompt} onClose={() => setTutorOpen(false)} />
        </div>
      )}
    </div>
  );
}

function Implement({
  data,
  onAskTutor,
  title,
  nextHref,
}: {
  data: { brief: ReactNode; solution: ReactNode; code: string } | null;
  onAskTutor?: (t: string) => void;
  title: string;
  nextHref: string;
}) {
  const [show, setShow] = useState(false);
  const [copied, setCopied] = useState(false);
  if (!data) return <p className="text-ink-2">No implementation exercise for this concept.</p>;
  return (
    <div className="fade-in max-w-2xl">
      <h2 className="font-serif text-2xl leading-snug">Write {title} from scratch.</h2>
      <p className="mt-2 text-sm text-ink-2">
        In a notebook, without looking. Then compare with the reference — the gaps are what you didn&apos;t understand yet.
      </p>
      <div className="mt-6">{data.brief}</div>

      <div className="mt-8 flex flex-wrap items-center gap-3">
        <button onClick={() => setShow((s) => !s)} className="h-11 rounded-xl bg-accent px-5 font-medium text-accent-ink">
          {show ? "Hide reference solution" : "Show reference solution"}
        </button>
        {onAskTutor && (
          <button
            onClick={() => onAskTutor(`I'm implementing ${title} from scratch. Give me a step-by-step plan with hints — but don't write the code for me.`)}
            className="h-11 rounded-xl border border-line px-5 text-ink-2 hover:text-ink"
          >
            Get hints instead
          </button>
        )}
      </div>

      {show && (
        <div className="mt-6 fade-in">
          <div className="flex justify-end">
            <button
              onClick={() => {
                navigator.clipboard.writeText(data.code);
                setCopied(true);
                setTimeout(() => setCopied(false), 1500);
              }}
              className="mb-2 text-xs text-ink-3 hover:text-ink"
            >
              {copied ? "Copied" : "Copy code"}
            </button>
          </div>
          {data.solution}
          <a href={nextHref} className="mt-8 inline-block text-sm text-accent hover:underline">
            Next in today&apos;s session →
          </a>
        </div>
      )}
    </div>
  );
}
