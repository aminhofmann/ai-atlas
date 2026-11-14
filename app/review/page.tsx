import { getAllConcepts } from "@/lib/content";
import { ReviewView } from "@/components/review-view";

export default function Page() {
  const cards = getAllConcepts().flatMap((c) => c.cards.map((k) => ({ ...k, conceptId: c.id, conceptTitle: c.title })));
  return (
    <div className="mx-auto w-full max-w-2xl px-5 py-12">
      <ReviewView cards={cards} />
    </div>
  );
}
