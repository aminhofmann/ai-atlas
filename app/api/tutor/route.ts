import { getAllConcepts } from "@/lib/content";
import { neighbourContext, TUTOR_SYSTEM } from "@/lib/claude";
import { streamClaude } from "@/lib/stream";

type Msg = { role: "user" | "assistant"; content: string };

export async function POST(req: Request) {
  const body = (await req.json().catch(() => null)) as { conceptId?: string; messages?: Msg[] } | null;
  const all = getAllConcepts();
  const concept = all.find((c) => c.id === body?.conceptId);
  if (!concept) return new Response("Unknown concept", { status: 404 });

  const messages = (body?.messages ?? [])
    .filter((m) => (m.role === "user" || m.role === "assistant") && typeof m.content === "string" && m.content.trim())
    .slice(-30)
    .map((m) => ({ role: m.role, content: m.content.slice(0, 20_000) }));
  if (!messages.length || messages[0].role !== "user" || messages.at(-1)!.role !== "user") {
    return new Response("Conversation must start and end with a user message", { status: 400 });
  }

  // The concept page is stable per conversation, so it sits in the cached system prompt.
  const system = `${TUTOR_SYSTEM}

<concept_page id="${concept.id}">
${concept.raw}
</concept_page>

<graph_neighbours>
${neighbourContext(concept, all) || "(none yet)"}
</graph_neighbours>`;

  return streamClaude({ system, messages, maxTokens: 16_000, effort: "medium" });
}
