import { describe, expect, test } from "bun:test";
import {
  canTransitionExperiment,
  stableExperimentError,
  validateReadiness,
  validateRetrievalMetrics,
} from "./experimentContracts";

describe("experiment contract vocabulary", () => {
  test("allows only the canonical lifecycle transitions", () => {
    expect(canTransitionExperiment("CREATED", "VALIDATING")).toBe(true);
    expect(canTransitionExperiment("RUNNING", "SUCCEEDED")).toBe(true);
    expect(canTransitionExperiment("RUNNING", "CREATED")).toBe(false);
    expect(canTransitionExperiment("SUCCEEDED", "RUNNING")).toBe(false);
  });

  test("uses stable machine-readable errors", () => {
    expect(stableExperimentError("NOT_READY", "source").message).toBe("NOT_READY: source");
  });

  test("rejects readiness that contradicts its blockers", () => {
    expect(() => validateReadiness({ state: "READY", checks: {}, blockers: ["NO_SOURCE"] })).toThrow(
      /blockers/,
    );
    expect(() => validateReadiness({ state: "BLOCKED", checks: {}, blockers: [] })).toThrow(
      /requires/,
    );
  });

  test("rejects retrieval metrics that do not reconcile", () => {
    expect(() =>
      validateRetrievalMetrics({
        relevant_retrieved: 2,
        retrieved: 4,
        relevant_available: 3,
        missing_evidence: 0,
        noise: 2,
        latency_ms: null,
        cost_status: "COST_UNAVAILABLE",
      }),
    ).toThrow(/missing_evidence/);
  });
});
