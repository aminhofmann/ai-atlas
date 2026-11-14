"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { Markdown } from "./markdown-client";

type Msg = { role: "user" | "assistant"; content: string };

const QUICK = [
  "Explain this as if I'm new to it",
  "Walk me through the math step by step",
  "What's the most common misconception here?",
  "Quiz me with one harder question",
];

async function streamText(url: string, body: unknown, onChunk: (full: string) => void, signal?: AbortSignal) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) {
    const msg = await res.text().catch(() => "");
    throw new Error(msg || `Request failed (${res.status})`);
  }
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let full = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    full += dec.decode(value, { stream: true });
    onChunk(full);
  }
  return full;
}

export function TutorPanel({
  conceptId,
  title,
  prompt,
  onClose,
}: {
  conceptId: string;
  title: string;
  /** A question handed over from elsewhere (e.g. "get feedback on my answer"); sent once per nonce. */
  prompt?: { text: string; nonce: number } | null;
  onClose?: () => void;
}) {
  const [mode, setMode] = useState<"chat" | "deeper">("chat");
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);

  // go-deeper state
  const [topic, setTopic] = useState("");
  const [draft, setDraft] = useState("");
  const [saved, setSaved] = useState<string | null>(null);
  const router = useRouter();
  const lastNonce = useRef<number | null>(null);

  useEffect(() => {
    if (!prompt || prompt.nonce === lastNonce.current) return;
    lastNonce.current = prompt.nonce;
    setMode("chat");
    send(prompt.text);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- fire once per handed-over prompt
  }, [prompt]);

  async function send(text: string) {
    if (!text.trim() || busy) return;
    setError(null);
    const next: Msg[] = [...messages, { role: "user", content: text.trim() }];
    setMessages([...next, { role: "assistant", content: "" }]);
    setInput("");
    setBusy(true);
    abort.current = new AbortController();
    try {
      await streamText(
        "/api/tutor",
        { conceptId, messages: next },
        (full) => setMessages([...next, { role: "assistant", content: full }]),
        abort.current.signal,
      );
    } catch (e) {
      if ((e as Error).name !== "AbortError") setError((e as Error).message);
      setMessages(next);
    } finally {
      setBusy(false);
    }
  }

  async function generate() {
    if (!topic.trim() || busy) return;
    setError(null);
    setDraft("");
    setSaved(null);
    setBusy(true);
    try {
      await streamText("/api/deeper", { conceptId, topic: topic.trim() }, setDraft);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function save() {
    setError(null);
    const res = await fetch("/api/concepts", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ markdown: draft }),
    });
    const json = await res.json();
    if (!res.ok) return setError(json.error ?? "Could not save");
    setSaved(json.id);
    router.refresh();
  }

  return (
    <div className="flex h-full flex-col overflow-hidden bg-panel">
      <div className="flex items-center gap-5 border-b border-line px-4 text-sm">
        {(
          [
            ["chat", "Tutor"],
            ["deeper", "Go deeper"],
          ] as const
        ).map(([k, label]) => (
          <button
            key={k}
            onClick={() => setMode(k)}
            className={`-mb-px border-b-2 py-3.5 ${mode === k ? "border-accent font-medium text-ink" : "border-transparent text-ink-3 hover:text-ink"}`}
          >
            {label}
          </button>
        ))}
        {onClose && (
          <button onClick={onClose} className="ml-auto text-ink-3 hover:text-ink" aria-label="Close tutor">
            Close <kbd>Esc</kbd>
          </button>
        )}
      </div>

      {mode === "chat" ? (
        <>
          <div className="flex-1 space-y-4 overflow-y-auto p-4 text-sm">
            {messages.length === 0 && (
              <div className="space-y-3">
                <p className="text-ink-2">
                  Ask anything about <strong className="text-ink">{title}</strong>. The tutor sees this page and
                  its neighbours in the graph.
                </p>
                <div className="flex flex-col gap-1.5">
                  {QUICK.map((q) => (
                    <button
                      key={q}
                      onClick={() => send(q)}
                      className="rounded-md border border-line px-3 py-1.5 text-left text-ink-2 hover:border-accent hover:text-ink"
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            )}
            {messages.map((m, i) =>
              m.role === "user" ? (
                <p key={i} className="ml-6 whitespace-pre-wrap rounded-lg bg-panel-2 px-3 py-2">
                  {m.content}
                </p>
              ) : (
                <div key={i} className="text-[0.92rem]">
                  {m.content ? <Markdown className="prose-compact">{m.content}</Markdown> : <p className="text-ink-3">Thinking…</p>}
                </div>
              ),
            )}
            {error && <p className="text-[var(--c-interpretability)]">{error}</p>}
          </div>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              send(input);
            }}
            className="flex gap-2 border-t border-line p-3"
          >
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question…"
              className="flex-1 rounded-md border border-line bg-bg px-3 py-2 text-sm outline-none focus:border-accent"
            />
            {busy ? (
              <button type="button" onClick={() => abort.current?.abort()} className="rounded-md border border-line px-3 text-sm">
                Stop
              </button>
            ) : (
              <button className="rounded-md bg-accent px-3 text-sm font-medium text-accent-ink">Send</button>
            )}
          </form>
        </>
      ) : (
        <div className="flex flex-1 flex-col gap-3 overflow-y-auto p-4 text-sm">
          <p className="text-ink-2">
            Name a sub-topic of <strong className="text-ink">{title}</strong> you want its own page for. The
            tutor drafts a new concept, linked into the graph — review it, then add it.
          </p>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              generate();
            }}
            className="flex gap-2"
          >
            <input
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              placeholder="e.g. ALiBi, gating in MoE, triangle attention…"
              className="flex-1 rounded-md border border-line bg-bg px-3 py-2 outline-none focus:border-accent"
            />
            <button disabled={busy} className="rounded-md bg-accent px-3 font-medium text-accent-ink disabled:opacity-50">
              {busy ? "Drafting…" : "Draft"}
            </button>
          </form>
          {error && <p className="text-[var(--c-interpretability)]">{error}</p>}
          {draft && (
            <>
              <pre className="max-h-[50vh] overflow-auto whitespace-pre-wrap rounded-lg border border-line bg-bg p-3 font-mono text-xs leading-relaxed text-ink-2">
                {draft}
              </pre>
              {!busy &&
                (saved ? (
                  <a href={`/c/${saved}`} className="rounded-md border border-accent px-3 py-2 text-center text-accent">
                    Added — open the new page →
                  </a>
                ) : (
                  <div className="flex gap-2">
                    <button onClick={save} className="flex-1 rounded-md bg-accent px-3 py-2 font-medium text-accent-ink">
                      Add to atlas
                    </button>
                    <button onClick={() => setDraft("")} className="rounded-md border border-line px-3 py-2 text-ink-2">
                      Discard
                    </button>
                  </div>
                ))}
              <p className="text-xs text-ink-3">
                AI-drafted. Check the facts against the linked paper before you memorise its flashcards.
              </p>
            </>
          )}
        </div>
      )}
    </div>
  );
}
