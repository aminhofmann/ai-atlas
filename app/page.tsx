import { learningPath, toPlanConcept } from "@/lib/content";
import { TodayView } from "@/components/today-view";

// Content is read at build time; the plan is made in the browser against the visitor's own progress.
export default function Home() {
  const path = learningPath().map(toPlanConcept);
  return (
    <div className="mx-auto w-full max-w-xl px-5 py-14">
      <TodayView path={path} />
    </div>
  );
}
