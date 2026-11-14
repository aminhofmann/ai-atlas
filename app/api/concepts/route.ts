import { tutorEnabled } from "@/lib/claude";
import fs from "node:fs";
import path from "node:path";
import matter from "gray-matter";
import { CONCEPT_DIR, ID_PATTERN } from "@/lib/content";
import { CATEGORIES } from "@/lib/types";

/** Saves a reviewed "go deeper" draft as a new concept file. Never overwrites. */
export async function POST(req: Request) {
  // Writing concept files only works on a local checkout, and only the tutor's "Go deeper" uses it.
  if (!tutorEnabled()) return Response.json({ error: "Not available without the tutor." }, { status: 404 });
  const body = (await req.json().catch(() => null)) as { markdown?: string } | null;
  let md = (body?.markdown ?? "").trim();
  // Tolerate a stray fence around the whole file.
  md = md.replace(/^```(?:markdown|md)?\s*\n/, "").replace(/\n```\s*$/, "").trim() + "\n";

  let data: Record<string, unknown>;
  try {
    data = matter(md).data;
  } catch {
    return Response.json({ error: "The draft's frontmatter is not valid YAML." }, { status: 400 });
  }
  const id = typeof data.id === "string" ? data.id : "";
  if (!ID_PATTERN.test(id)) return Response.json({ error: `Invalid id "${id}" (kebab-case required).` }, { status: 400 });
  if (typeof data.title !== "string" || !data.title.trim())
    return Response.json({ error: "The draft has no title." }, { status: 400 });
  if (!(CATEGORIES as readonly string[]).includes(String(data.category)))
    return Response.json({ error: `Unknown category "${String(data.category)}".` }, { status: 400 });

  const file = path.join(CONCEPT_DIR, `${id}.md`);
  if (fs.existsSync(file)) return Response.json({ error: `A concept with id "${id}" already exists.` }, { status: 409 });
  fs.mkdirSync(CONCEPT_DIR, { recursive: true });
  fs.writeFileSync(file, md, { flag: "wx" });
  return Response.json({ id });
}
