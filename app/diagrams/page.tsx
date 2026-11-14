import Link from "next/link";
import { getAllConcepts } from "@/lib/content";
import { getDiagram } from "@/components/diagrams";

/** Every concept diagram on one page — for checking them side by side. Not in the nav. */
export default function Page() {
  const withDiagram = getAllConcepts().filter((c) => getDiagram(c.id));
  return (
    <div className="mx-auto grid w-full max-w-6xl gap-8 px-5 py-10 lg:grid-cols-2">
      {withDiagram.map((c) => {
        const d = getDiagram(c.id)!;
        return (
          <figure key={c.id} id={c.id} className="rounded-2xl border border-line bg-panel p-5">
            <Link href={`/c/${c.id}`} className="text-sm font-medium hover:underline">
              {c.title}
            </Link>
            <div className="mt-3">
              <d.Diagram />
            </div>
            <figcaption className="mt-3 text-sm text-ink-3">{d.caption}</figcaption>
          </figure>
        );
      })}
    </div>
  );
}
