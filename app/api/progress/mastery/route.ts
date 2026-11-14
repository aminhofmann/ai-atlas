import { setMastery } from "@/lib/progress";
import { ID_PATTERN } from "@/lib/content";
import { MASTERY_LEVELS, type Mastery } from "@/lib/types";

const VALID: readonly Mastery[] = MASTERY_LEVELS;

export async function POST(req: Request) {
  const body = (await req.json().catch(() => null)) as { id?: string; mastery?: Mastery } | null;
  if (!body?.id || !ID_PATTERN.test(body.id) || !VALID.includes(body.mastery as Mastery)) {
    return Response.json({ error: `Need id and mastery (${MASTERY_LEVELS.join("|")})` }, { status: 400 });
  }
  setMastery(body.id, body.mastery as Mastery);
  return Response.json({ ok: true });
}
