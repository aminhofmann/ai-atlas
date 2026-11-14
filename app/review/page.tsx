import { Suspense } from "react";
import Link from "next/link";
import { connection } from "next/server";
import { getAllConcepts } from "@/lib/content";
import { dueCardIds } from "@/lib/progress";
import { PracticeSession } from "@/components/practice-session";

async function Review() {
  await connection();
  const all = getAllConcepts();
  const cards = new Map(all.flatMap((c) => c.cards.map((k) => [k.id, { ...k, conceptId: c.id, conceptTitle: c.title }])));
  const due = dueCardIds([...cards.keys()]).map((id) => cards.get(id)!);

  if (due.length === 0) {
    return (
      <div className="py-10 text-center">
        <p className="font-serif text-2xl">Nothing due.</p>
        <p className="mt-2 text-sm text-ink-3">
          Questions join this queue once you start a concept. <Link className="text-accent hover:underline" href="/">Back to today →</Link>
        </p>
      </div>
    );
  }
  return <PracticeSession cards={due} showConcept done={{ label: "Back to today", href: "/" }} />;
}

export default function Page() {
  return (
    <div className="mx-auto w-full max-w-2xl px-5 py-12">
      <Suspense fallback={<p className="text-ink-3">Loading…</p>}>
        <Review />
      </Suspense>
    </div>
  );
}
