import "server-only";
import Anthropic from "@anthropic-ai/sdk";
import { claude, MODEL } from "./claude";

type Params = {
  system: string;
  messages: Anthropic.Beta.BetaMessageParam[];
  maxTokens: number;
  effort: "low" | "medium" | "high";
};

/**
 * Streams Claude's text as a plain-text HTTP response.
 * Server-side fallbacks are on: if a request is declined by a safety
 * classifier, the API retries it on a fallback model inside the same call.
 */
export function streamClaude({ system, messages, maxTokens, effort }: Params): Response {
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    async start(controller) {
      try {
        const stream = claude().beta.messages.stream({
          model: MODEL,
          max_tokens: maxTokens,
          betas: ["server-side-fallback-2026-07-01"],
          fallbacks: "default",
          output_config: { effort },
          system: [{ type: "text", text: system, cache_control: { type: "ephemeral" } }],
          messages,
        });
        for await (const event of stream) {
          if (event.type === "content_block_delta" && event.delta.type === "text_delta") {
            controller.enqueue(encoder.encode(event.delta.text));
          }
        }
        const final = await stream.finalMessage();
        if (final.stop_reason === "refusal") {
          controller.enqueue(encoder.encode("\n\n_The model declined to answer this request._"));
        } else if (final.stop_reason === "max_tokens") {
          controller.enqueue(encoder.encode("\n\n_(cut off — answer hit the length limit)_"));
        }
      } catch (e) {
        controller.enqueue(encoder.encode(`\n\n**Error:** ${describeError(e)}`));
      } finally {
        controller.close();
      }
    },
  });
  return new Response(body, { headers: { "content-type": "text/plain; charset=utf-8", "cache-control": "no-store" } });
}

export function describeError(e: unknown): string {
  if (e instanceof Anthropic.AuthenticationError)
    return "No valid Anthropic API key. Put ANTHROPIC_API_KEY in .env.local and restart the dev server.";
  if (e instanceof Anthropic.RateLimitError) return "Rate limited — try again in a moment.";
  if (e instanceof Anthropic.APIError) return `API error ${e.status ?? ""}: ${e.message}`;
  if (e instanceof Error && /api key|apiKey|credentials/i.test(e.message))
    return "No Anthropic API key found. Put ANTHROPIC_API_KEY in .env.local and restart the dev server.";
  return e instanceof Error ? e.message : String(e);
}
