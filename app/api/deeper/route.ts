import { getAllConcepts } from "@/lib/content";
import { deeperSystem, tutorEnabled } from "@/lib/claude";
import { streamClaude } from "@/lib/stream";

/** Drafts a NEW concept page for a sub-topic of an existing one. Nothing is saved here. */
export async function POST(req: Request) {
  if (!tutorEnabled()) return new Response("The tutor is not configured (no ANTHROPIC_API_KEY).", { status: 404 });
  const body = (await req.json().catch(() => null)) as { conceptId?: string; topic?: string } | null;
  const all = getAllConcepts();
  const parent = all.find((c) => c.id === body?.conceptId);
  const topic = body?.topic?.trim().slice(0, 300);
  if (!parent || !topic) return new Response("Need an existing concept and a topic", { status: 400 });

  const prompt = `The learner is on the page "${parent.title}" (id: ${parent.id}) and wants to go deeper into: "${topic}".

Here is the parent page for context:
<parent>
${parent.raw}
</parent>

Write the new concept page for "${topic}". It must link back to "${parent.id}" (in \`requires\` if the parent is a genuine prerequisite, otherwise in \`links\` with the fitting type), and may link to other existing ids. Choose the most fitting category and level. Verify-or-omit: leave out any number, date or detail you are not sure of.`;

  return streamClaude({
    system: deeperSystem(all.map((c) => c.id)),
    messages: [{ role: "user", content: prompt }],
    maxTokens: 32_000,
    effort: "high",
  });
}
