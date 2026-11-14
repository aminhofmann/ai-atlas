import "server-only";
import fs from "node:fs";
import path from "node:path";
import Anthropic from "@anthropic-ai/sdk";
import type { Concept } from "./types";

export const MODEL = "claude-opus-5-5";

let client: Anthropic | null = null;
export function claude() {
  // Reads ANTHROPIC_API_KEY (or an `ant auth login` profile) from the environment.
  client ??= new Anthropic();
  return client;
}

export function neighbourContext(c: Concept, all: Concept[]) {
  const byId = new Map(all.map((x) => [x.id, x]));
  const lines: string[] = [];
  for (const r of c.requires) {
    const o = byId.get(r);
    if (o) lines.push(`- prerequisite: ${o.title} — ${o.summary}`);
  }
  for (const l of c.links) {
    const o = byId.get(l.to);
    if (o) lines.push(`- ${c.title} ${l.type} ${o.title} — ${o.summary}`);
  }
  for (const o of all) {
    for (const l of o.links) if (l.to === c.id) lines.push(`- ${o.title} ${l.type} ${c.title} — ${o.summary}`);
    if (o.requires.includes(c.id)) lines.push(`- unlocks: ${o.title} — ${o.summary}`);
  }
  return lines.join("\n");
}

export const TUTOR_SYSTEM = `You are a patient, rigorous tutor inside "AI Atlas", a personal learning app about transformers and their applications (including biology). The learner is studying one concept page at a time; the page and its neighbours in the concept graph are given below.

How to teach:
- Answer the question that was asked, at the depth it was asked. Prefer one clear explanation over a survey.
- Build on the concept page and point to neighbouring concepts by name when they help ("this is the same trick as in Rotary Position Embedding").
- Use LaTeX ($...$ inline, $$...$$ display) for math and short Python/PyTorch snippets when code makes it concrete.
- When asked to quiz, ask ONE question and wait for the answer; then give feedback.
- Be honest about uncertainty. If something is recent or you are not sure of a detail (a number, a date, an architecture choice), say so and point to the paper rather than guess — the learner memorises what you say.
- Write in Markdown. Keep answers reasonably short unless asked for depth.`;

export function deeperSystem(existingIds: string[]) {
  const schema = fs.readFileSync(path.join(process.cwd(), "content", "SCHEMA.md"), "utf8");
  return `You write new concept pages for "AI Atlas", a personal learning app about transformers. Each page is one Markdown file that must follow this specification exactly:

<schema>
${schema}
</schema>

Ids that already exist in the atlas (you may link to these, and only these, plus the parent concept; the new id must NOT be one of them):
${existingIds.join(", ")}

Output ONLY the file contents, starting with the --- frontmatter line. No code fence around the whole file, no commentary before or after.`;
}
