import { GRADES, reviewCard, type GradeName } from "@/lib/progress";

export async function POST(req: Request) {
  const body = (await req.json().catch(() => null)) as { cardId?: string; grade?: string } | null;
  const cardId = body?.cardId ?? "";
  if (!/^[a-z0-9-]+#\d+$/.test(cardId) || !(body?.grade && body.grade in GRADES)) {
    return Response.json({ error: "Need cardId and grade (again|hard|good|easy)" }, { status: 400 });
  }
  const card = reviewCard(cardId, body.grade as GradeName);
  return Response.json({ due: card.due.toISOString() });
}
