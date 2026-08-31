/**
 * Client for the local experiment job boundary.
 *
 * The boundary is authoritative: this client only sequences create → validate →
 * start and translates the stable error codes into guidance a non-CLI user can
 * act on. It must never invent a lifecycle status or hide a rejection.
 */

import { describe, expect, test } from "bun:test";
import {
  EXPERIMENT_API_PATH,
  describeExperimentError,
  parseLifecycle,
  submitExperiment,
  type ExperimentTransport,
} from "./experimentClient";

const REQUEST = {
  schema_version: 1 as const,
  experiment_id: "exp-1",
  identity: {
    corpus: { sha256: "a".repeat(64), document_count: 2 },
    benchmark_sha256: "b".repeat(64),
    protocol: "retrieval-v1",
    evaluator: "retrieval",
    provider: null,
    model: null,
    configuration_hash: null,
    code_version: null,
  },
  candidates: ["gbrain"] as const,
  evaluation_mode: "retrieval_only" as const,
};

interface Call {
  url: string;
  method: string;
  body: string | null;
}

/** Record every request and reply with the scripted lifecycle statuses. */
function scriptedTransport(
  replies: { status: number; payload: unknown }[],
  calls: Call[] = [],
): { transport: ExperimentTransport; calls: Call[] } {
  let index = 0;
  const transport: ExperimentTransport = async (url, init) => {
    calls.push({ url, method: init.method, body: init.body ?? null });
    const reply = replies[index++];
    if (reply === undefined) throw new Error(`unexpected request to ${url}`);
    return reply;
  };
  return { transport, calls };
}

describe("experiment job client", () => {
  test("creates, validates, and starts through the documented job routes", async () => {
    const { transport, calls } = scriptedTransport([
      { status: 201, payload: { experiment_id: "exp-1", status: "CREATED", updated_at: null } },
      { status: 200, payload: { experiment_id: "exp-1", status: "READY", updated_at: null } },
      { status: 202, payload: { experiment_id: "exp-1", status: "RUNNING", updated_at: null } },
    ]);

    const lifecycle = await submitExperiment("http://127.0.0.1:8765", REQUEST, transport);

    expect(lifecycle.status).toBe("RUNNING");
    expect(calls.map((call) => `${call.method} ${call.url}`)).toEqual([
      `POST http://127.0.0.1:8765${EXPERIMENT_API_PATH}`,
      `POST http://127.0.0.1:8765${EXPERIMENT_API_PATH}/exp-1/validate`,
      `POST http://127.0.0.1:8765${EXPERIMENT_API_PATH}/exp-1/start`,
    ]);
    expect(JSON.parse(calls[0]?.body ?? "{}").schema_version).toBe(1);
  });

  test("stops at validation failure instead of starting a doomed run", async () => {
    const { transport, calls } = scriptedTransport([
      { status: 201, payload: { experiment_id: "exp-1", status: "CREATED", updated_at: null } },
      { status: 200, payload: { experiment_id: "exp-1", status: "FAILED", updated_at: null } },
    ]);

    const lifecycle = await submitExperiment("http://127.0.0.1:8765", REQUEST, transport);

    expect(lifecycle.status).toBe("FAILED");
    expect(calls).toHaveLength(2);
  });

  test("surfaces a stable rejection as a typed error, not a fake lifecycle", async () => {
    const { transport } = scriptedTransport([
      { status: 400, payload: { error: "INVALID_REQUEST", detail: "request failed validation" } },
    ]);

    const error = await submitExperiment("http://127.0.0.1:8765", REQUEST, transport).then(
      () => null,
      (cause: unknown) => cause as Error & { code?: string },
    );

    expect(error?.code).toBe("INVALID_REQUEST");
    expect(error?.message).toContain("request failed validation");
  });

  test("rejects an unknown lifecycle status rather than rendering it", () => {
    expect(() => parseLifecycle({ experiment_id: "exp-1", status: "ALMOST" })).toThrow(/status/);
    expect(() => parseLifecycle({ status: "READY" })).toThrow(/experiment_id/);
  });

  test("translates every stable error code into actionable guidance", () => {
    const codes = [
      "INVALID_TRANSITION",
      "INVALID_REQUEST",
      "NOT_READY",
      "NOT_FOUND",
      "RUN_FAILED",
      "CANCELLED",
    ] as const;
    for (const code of codes) {
      const described = describeExperimentError(code, "detail from the boundary");
      expect(described.title.length).toBeGreaterThan(0);
      expect(described.guidance.length).toBeGreaterThan(0);
      expect(described.guidance).not.toBe(code);
    }
  });

  test("an unrecognized code still yields guidance instead of a blank surface", () => {
    const described = describeExperimentError("SOMETHING_NEW", "unmapped");
    expect(described.title.length).toBeGreaterThan(0);
    expect(described.guidance.length).toBeGreaterThan(0);
  });
});
