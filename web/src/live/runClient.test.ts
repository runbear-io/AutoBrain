import { describe, expect, test } from "bun:test";
import {
  PROJECTION_SCHEMA_VERSION,
  fetchRunOutcome,
  isTerminal,
  parseRunOutcome,
  summarize,
} from "./runClient";

const projection = {
  schema_version: 1,
  run_id: "RUN-A41F",
  status: "OK",
  verdict: "gbrain",
  rationale: "GBrain leads grounded recall.",
  corpus_hash: "a".repeat(64),
  benchmark_hash: "b".repeat(64),
  candidates: [
    {
      candidate: "gbrain",
      status: "OK",
      quality_score: 93.6,
      answer_success_rate: 0.93,
      source_support_rate: 0.79,
      contradiction_count: 1,
      scored_cases: 30,
      answered_cases: 28,
      cost_status: "COST_COMPLETE",
      total_cost_usd: 1.25,
      query_p50_ms: 820,
      query_p95_ms: 2460,
      operating_burden: 2,
    },
  ],
  warnings: [],
};

const succeeded = { status: "SUCCEEDED", projection, error: null };

describe("local run client contract", () => {
  test("parses a succeeded outcome and exposes the projection", () => {
    const outcome = parseRunOutcome(succeeded);
    expect(outcome.status).toBe("SUCCEEDED");
    expect(outcome.projection?.run_id).toBe("RUN-A41F");
    expect(outcome.projection?.candidates[0]?.candidate).toBe("gbrain");
  });

  test("parses failed and cancelled outcomes with no projection", () => {
    const failed = parseRunOutcome({ status: "FAILED", projection: null, error: "boom" });
    expect(failed.status).toBe("FAILED");
    expect(failed.projection).toBeNull();
    expect(failed.error).toBe("boom");

    const cancelled = parseRunOutcome({ status: "CANCELLED", projection: null, error: null });
    expect(cancelled.status).toBe("CANCELLED");
    expect(cancelled.projection).toBeNull();
  });

  test("rejects an unsupported projection schema version", () => {
    const drifted = {
      ...succeeded,
      projection: { ...projection, schema_version: PROJECTION_SCHEMA_VERSION + 1 },
    };
    expect(() => parseRunOutcome(drifted)).toThrow(/schema version/i);
  });

  test("rejects an unknown run status rather than guessing", () => {
    expect(() => parseRunOutcome({ status: "MAYBE", projection: null, error: null })).toThrow(
      /status/i,
    );
  });

  test("rejects a succeeded outcome that carries no projection", () => {
    expect(() => parseRunOutcome({ status: "SUCCEEDED", projection: null, error: null })).toThrow(
      /projection/i,
    );
  });

  test("all three terminal statuses are terminal", () => {
    expect(isTerminal("SUCCEEDED")).toBe(true);
    expect(isTerminal("FAILED")).toBe(true);
    expect(isTerminal("CANCELLED")).toBe(true);
  });

  test("summarize reports the engine status distinctly from the run status", () => {
    const noDecision = parseRunOutcome({
      status: "SUCCEEDED",
      projection: { ...projection, status: "NO_DECISION", verdict: "NO_DECISION" },
      error: null,
    });
    const summary = summarize(noDecision);
    expect(summary.runStatus).toBe("SUCCEEDED");
    expect(summary.engineStatus).toBe("NO_DECISION");
    expect(summary.headline).toMatch(/no decision/i);
  });

  test("summarize describes cancellation without implying failure", () => {
    const summary = summarize(parseRunOutcome({ status: "CANCELLED", projection: null, error: null }));
    expect(summary.runStatus).toBe("CANCELLED");
    expect(summary.engineStatus).toBeNull();
    expect(summary.headline).toMatch(/cancelled/i);
    expect(summary.headline).not.toMatch(/fail/i);
  });

  test("fetchRunOutcome reads the versioned endpoint from the injected transport", async () => {
    const seen: string[] = [];
    const outcome = await fetchRunOutcome("http://127.0.0.1:9999", async (url) => {
      seen.push(url);
      return succeeded;
    });
    expect(seen).toEqual(["http://127.0.0.1:9999/api/v1/run"]);
    expect(outcome.status).toBe("SUCCEEDED");
  });
});
