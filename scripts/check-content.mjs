import fs from "node:fs"; import path from "node:path"; import matter from "gray-matter";
const dir = process.argv[2] ?? "content/concepts";
const files = fs.readdirSync(dir).filter(f=>f.endsWith(".md"));
const ids = new Set(files.map(f=>f.slice(0,-3)));
let problems = 0;
for (const f of files) {
  const { data, content } = matter(fs.readFileSync(path.join(dir,f),"utf8"));
  const id = f.slice(0,-3); const errs = [];
  if (data.id !== id) errs.push(`id ${data.id}`);
  for (const r of data.requires ?? []) if (!ids.has(r)) errs.push(`requires missing ${r}`);
  for (const l of data.links ?? []) if (!ids.has(l.to)) errs.push(`link missing ${l.to}`);
  const h2 = [...content.matchAll(/^## (.+)$/gm)].map(m=>m[1]);
  if (h2.join("|") !== "Intuition|Mechanics|Code|Why it matters|Self-test") errs.push(`sections ${h2.join("|")}`);
  const qs = (content.split("## Self-test")[1] ?? "").match(/^Q:/gm)?.length ?? 0;
  if (qs < 3) errs.push(`only ${qs} cards`);
  if (!(data.sources?.length)) errs.push("no sources");
  if (errs.length) { problems++; console.log(id, "→", errs.join("; ")); }
}
console.log(`${files.length} files, ${problems} with problems`);
if (problems) process.exit(1);
