/**
 * Malformed payload contract for the local run boundary.
 *
 * The client is the last line of defense between a loopback fixture and the
 * rendered UI. Every one of these payloads is rejected rather than coerced,
 * because rendering a plausible-but-wrong run summary is worse than showing
 * an error.
 */

import { describe, expect, test } from "bun:test";
import { parseRunOutcome } from "./runClient";

const HASH_A = "a".repeat(64);
const HASH_B = "b".repeat(64);

function candidate(overrides: Record<string, unknown> = {}) {
  return {
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
    ...overrides,
  };
}

function projection(overrides: Record<string, unknown> = {}) {
  return {
    schema_version: 1,
    run_id: "RUN-A41F",
    status: "OK",
    verdict: "gbrain",
    rationale: "GBrain leads grounded recall.",
    corpus_hash: HASH_A,
    benchmark_hash: HASH_B,
    candidates: [candidate()],
    warnings: [],
    ...overrides,
  };
}

function succeeded(projectionOverrides: Record<string, unknown> = {}) {
  return { status: "SUCCEEDED", projection: projection(projectionOverrides), error: null };
}

function withCandidate(overrides: Record<string, unknown>) {
  return succeeded({ candidates: [candidate(overrides)] });
}

describe("projection payload validation", () => {
  test("accepts a well-formed payload", () => {
    const outcome = parseRunOutcome(succeeded());
    expect(outcome.projection?.run_id).toBe("RUN-A41F");
    expect(outcome.projection?.candidates).toHaveLength(1);
  });

  test("accepts nullable numeric fields when explicitly null", () => {
    const outcome = parseRunOutcome(
      withCandidate({
        total_cost_usd: null,
        query_p50_ms: null,
        query_p95_ms: null,
        operating_burden: null,
      }),
    );
    expect(outcome.projection?.candidates[0]?.total_cost_usd).toBeNull();
  });

  test.each([
    "run_id",
    "status",
    "verdict",
    "rationale",
    "corpus_hash",
    "benchmark_hash",
    "candidates",
    "warnings",
  ])("rejects a projection missing required field %s", (field) => {
    const payload = succeeded();
    delete (payload.projection as Record<string, unknown>)[field];
    expect(() => parseRunOutcome(payload)).toThrow(new RegExp(field));
  });

  test.each([
    ["candidate", "quality_score"],
    ["status", "status"],
    ["cost_status", "cost_status"],
    ["quality_score", "quality_score"],
    ["answer_success_rate", "answer_success_rate"],
    ["contradiction_count", "contradiction_count"],
  ])("rejects a candidate missing required field %s", (field) => {
    const payload = withCandidate({});
    delete (payload.projection.candidates[0] as Record<string, unknown>)[field];
    expect(() => parseRunOutcome(payload)).toThrow(new RegExp(field));
  });

  test("rejects a non-object projection", () => {
    expect(() => parseRunOutcome({ status: "SUCCEEDED", projection: 42, error: null })).toThrow(
      /projection/i,
    );
    expect(() => parseRunOutcome({ status: "SUCCEEDED", projection: [], error: null })).toThrow(
      /projection/i,
    );
  });

  test("rejects an unknown engine status", () => {
    expect(() => parseRunOutcome(succeeded({ status: "DEFINITELY_NOT_A_STATUS" }))).toThrow(
      /status/i,
    );
  });

  test("rejects an unknown verdict", () => {
    expect(() => parseRunOutcome(succeeded({ verdict: "some-other-brain" }))).toThrow(/verdict/i);
  });

  test("rejects an unknown candidate id", () => {
    expect(() => parseRunOutcome(withCandidate({ candidate: "rogue-brain" }))).toThrow(
      /candidate/i,
    );
  });

  test("rejects an unknown cost status", () => {
    expect(() => parseRunOutcome(withCandidate({ cost_status: "COST_FREE" }))).toThrow(
      /cost_status/i,
    );
  });

  test.each([
    ["not-hex-at-all", "non-hex characters"],
    ["ABCDEF" + "a".repeat(58), "uppercase hex"],
    ["a".repeat(63), "too short"],
    ["a".repeat(65), "too long"],
  ])("rejects a corpus hash with %s", (value) => {
    expect(() => parseRunOutcome(succeeded({ corpus_hash: value }))).toThrow(/corpus_hash/i);
  });

  test("rejects a benchmark hash that is not a sha256", () => {
    expect(() => parseRunOutcome(succeeded({ benchmark_hash: "deadbeef" }))).toThrow(
      /benchmark_hash/i,
    );
  });

  test("rejects an empty run id", () => {
    expect(() => parseRunOutcome(succeeded({ run_id: "" }))).toThrow(/run_id/i);
  });

  test("rejects a non-string rationale", () => {
    expect(() => parseRunOutcome(succeeded({ rationale: 12 }))).toThrow(/rationale/i);
  });

  test("rejects candidates that are not an array", () => {
    expect(() => parseRunOutcome(succeeded({ candidates: { gbrain: 1 } }))).toThrow(/candidates/i);
  });

  test("rejects an empty candidate list", () => {
    expect(() => parseRunOutcome(succeeded({ candidates: [] }))).toThrow(/candidates/i);
  });

  test("rejects a candidate that is not an object", () => {
    expect(() => parseRunOutcome(succeeded({ candidates: ["gbrain"] }))).toThrow(/candidate/i);
  });

  test("rejects warnings that are not an array of strings", () => {
    expect(() => parseRunOutcome(succeeded({ warnings: "none" }))).toThrow(/warnings/i);
    expect(() => parseRunOutcome(succeeded({ warnings: [7] }))).toThrow(/warnings/i);
  });

  test.each([
    ["quality_score", -1],
    ["quality_score", 100.5],
    ["answer_success_rate", -0.1],
    ["answer_success_rate", 1.5],
    ["source_support_rate", 2],
    ["contradiction_count", -1],
    ["scored_cases", -3],
    ["answered_cases", -1],
    ["total_cost_usd", -0.01],
    ["query_p50_ms", -5],
    ["query_p95_ms", -1],
    ["operating_burden", -2],
  ])("rejects %s outside its allowed range (%p)", (field, value) => {
    expect(() => parseRunOutcome(withCandidate({ [field]: value }))).toThrow(new RegExp(field));
  });

  test.each(["contradiction_count", "scored_cases", "answered_cases"])(
    "rejects a non-integer count in %s",
    (field) => {
      expect(() => parseRunOutcome(withCandidate({ [field]: 1.5 }))).toThrow(new RegExp(field));
    },
  );

  test.each(["quality_score", "answer_success_rate", "total_cost_usd"])(
    "rejects a non-finite number in %s",
    (field) => {
      expect(() => parseRunOutcome(withCandidate({ [field]: Number.NaN }))).toThrow(
        new RegExp(field),
      );
      expect(() => parseRunOutcome(withCandidate({ [field]: Number.POSITIVE_INFINITY }))).toThrow(
        new RegExp(field),
      );
    },
  );

  test("rejects a numeric field delivered as a numeric string", () => {
    expect(() => parseRunOutcome(withCandidate({ quality_score: "93.6" }))).toThrow(
      /quality_score/i,
    );
  });

  test("rejects a non-nullable numeric field set to null", () => {
    expect(() => parseRunOutcome(withCandidate({ quality_score: null }))).toThrow(/quality_score/i);
  });

  test("reports the offending candidate by index", () => {
    const payload = succeeded({ candidates: [candidate(), candidate({ quality_score: 900 })] });
    expect(() => parseRunOutcome(payload)).toThrow(/candidates\[1\]/);
  });
});
