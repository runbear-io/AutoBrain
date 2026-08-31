/**
 * Reading a finished experiment's result from the local job boundary.
 *
 * The boundary owns lifecycle policy, so this client waits on the boundary's
 * own terminal status rather than on elapsed time, and refuses to render a
 * result the boundary has not actually finished producing.
 */

import { describe, expect, test } from "bun:test";
import {
  fetchExperimentResult,
  awaitExperimentResult,
  type ResultTransport,
} from "./resultsClient";

const CORPUS = "a".repeat(64);
const BENCHMARK = "b".repeat(64);

const PROJECTION = {
  schema_version: 1,
  run_id: "run-1",
  status: "OK",
  verdict: "gbrain",
  rationale: "GBrain led grounded retrieval recall.",
  corpus_hash: CORPUS,
  benchmark_hash: BENCHMARK,
  candidates: [
    {
      candidate: "gbrain",
      status: "OK",
      quality_score: 81.5,
      answer_success_rate: 0.9,
      source_support_rate: 0.8,
      contradiction_count: 0,
      scored_cases: 20,
      answered_cases: 18,
      cost_status: "COST_COMPLETE",
      total_cost_usd: 1.5,
      query_p50_ms: 100,
      query_p95_ms: 220,
      operating_burden: 2,
    },
  ],
  warnings: [],
};

/** Reply with a scripted sequence, recording every requested URL. */
function scripted(replies: { status: number; payload: unknown }[]) {
  const urls: string[] = [];
  let index = 0;
  const transport: ResultTransport = async (url) => {
    urls.push(url);
    const reply = replies[Math.min(index++, replies.length - 1)];
    if (reply === undefined) throw new Error(`unexpected request to ${url}`);
    return reply;
  };
  return { transport, urls };
}

describe("reading an experiment result", () => {
  test("requests the documented result route and parses the projection", async () => {
    const { transport, urls } = scripted([
      { status: 200, payload: { status: "SUCCEEDED", run_id: "run-1", projection: PROJECTION } },
    ]);

    const outcome = await fetchExperimentResult("http://127.0.0.1:8765", "exp-1", transport);

    expect(urls).toEqual(["http://127.0.0.1:8765/api/v1/experiments/exp-1/result"]);
    expect(outcome.status).toBe("SUCCEEDED");
    expect(outcome.projection?.benchmark_hash).toBe(BENCHMARK);
  });

  test("percent-encodes the experiment id so a hostile id cannot escape the path", async () => {
    const { transport, urls } = scripted([
      { status: 200, payload: { status: "CANCELLED", projection: null } },
    ]);

    await fetchExperimentResult("http://127.0.0.1:8765", "exp/../evil", transport);

    expect(urls[0]).toBe("http://127.0.0.1:8765/api/v1/experiments/exp%2F..%2Fevil/result");
  });

  test("a failed result keeps its detail and carries no projection", async () => {
    const { transport } = scripted([
      { status: 200, payload: { status: "FAILED", projection: null, error: "transport refused" } },
    ]);

    const outcome = await fetchExperimentResult("http://127.0.0.1:8765", "exp-1", transport);

    expect(outcome.status).toBe("FAILED");
    expect(outcome.projection).toBeNull();
    expect(outcome.error).toBe("transport refused");
  });

  test("a boundary rejection surfaces its stable code rather than an empty result", async () => {
    const { transport } = scripted([
      { status: 409, payload: { error: "NOT_READY", detail: "result is not ready" } },
    ]);

    const error = await fetchExperimentResult("http://127.0.0.1:8765", "exp-1", transport).then(
      () => null,
      (cause: unknown) => cause as Error & { code?: string },
    );

    expect(error?.code).toBe("NOT_READY");
    expect(error?.message).toContain("result is not ready");
  });

  test("a succeeded result that carries no projection is refused, not rendered", async () => {
    const { transport } = scripted([
      { status: 200, payload: { status: "SUCCEEDED", projection: null } },
    ]);

    expect(
      fetchExperimentResult("http://127.0.0.1:8765", "exp-1", transport),
    ).rejects.toThrow();
  });
});

describe("waiting for a terminal result", () => {
  test("polls the status route until the boundary reports a terminal state", async () => {
    const urls: string[] = [];
    const statuses = ["RUNNING", "RUNNING", "SUCCEEDED"];
    let index = 0;
    const transport: ResultTransport = async (url) => {
      urls.push(url);
      if (url.endsWith("/result")) {
        return {
          status: 200,
          payload: { status: "SUCCEEDED", run_id: "run-1", projection: PROJECTION },
        };
      }
      const status = statuses[Math.min(index++, statuses.length - 1)];
      return { status: 200, payload: { experiment_id: "exp-1", status } };
    };

    const outcome = await awaitExperimentResult("http://127.0.0.1:8765", "exp-1", {
      transport,
      attempts: 5,
      // No delay: the test asserts the polling contract, never a duration.
      delayMs: 0,
    });

    expect(outcome.status).toBe("SUCCEEDED");
    expect(urls.filter((url) => url.endsWith("/result"))).toHaveLength(1);
    expect(urls.filter((url) => url.endsWith("/exp-1"))).toHaveLength(3);
  });

  test("gives up with a clear error when the run never reaches a terminal state", async () => {
    const transport: ResultTransport = async () => ({
      status: 200,
      payload: { experiment_id: "exp-1", status: "RUNNING" },
    });

    expect(
      awaitExperimentResult("http://127.0.0.1:8765", "exp-1", {
        transport,
        attempts: 3,
        delayMs: 0,
      }),
    ).rejects.toThrow(/still running/i);
  });

  test("a run that ends CANCELLED resolves rather than throwing", async () => {
    const transport: ResultTransport = async (url) =>
      url.endsWith("/result")
        ? { status: 200, payload: { status: "CANCELLED", projection: null } }
        : { status: 200, payload: { experiment_id: "exp-1", status: "CANCELLED" } };

    const outcome = await awaitExperimentResult("http://127.0.0.1:8765", "exp-1", {
      transport,
      attempts: 3,
      delayMs: 0,
    });

    expect(outcome.status).toBe("CANCELLED");
    expect(outcome.projection).toBeNull();
  });
});
