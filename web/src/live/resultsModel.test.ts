/**
 * Retrieval results and comparison projection rules.
 *
 * These tests pin the honesty guarantees of the results surface: metrics are
 * derived from the engine projection rather than invented, evidence confidence
 * is always labeled, a leading candidate is never called a recommendation
 * unless the engine actually issued one, and every non-succeeded outcome
 * carries an actionable recovery action.
 */

import { describe, expect, test } from "bun:test";
import type { RunOutcome, RunProjection, CandidateProjection } from "./runClient";
import {
  compareRetrieval,
  describeRecovery,
  evidenceConfidence,
  resolveBenchmarkHash,
  retrievalRows,
} from "./resultsModel";

const CORPUS = "a".repeat(64);
const BENCHMARK = "b".repeat(64);
const OTHER_BENCHMARK = "c".repeat(64);

function candidate(overrides: Partial<CandidateProjection> = {}): CandidateProjection {
  return {
    candidate: "gbrain",
    status: "OK",
    quality_score: 82.5,
    answer_success_rate: 0.9,
    source_support_rate: 0.8,
    contradiction_count: 0,
    scored_cases: 20,
    answered_cases: 18,
    cost_status: "COST_COMPLETE",
    total_cost_usd: 1.25,
    query_p50_ms: 120,
    query_p95_ms: 240,
    operating_burden: 2,
    ...overrides,
  };
}

function projection(overrides: Partial<RunProjection> = {}): RunProjection {
  return {
    schema_version: 1,
    run_id: "run-1",
    status: "OK",
    verdict: "gbrain",
    rationale: "GBrain leads grounded retrieval recall.",
    corpus_hash: CORPUS,
    benchmark_hash: BENCHMARK,
    candidates: [candidate()],
    warnings: [],
    ...overrides,
  };
}

function succeeded(overrides: Partial<RunProjection> = {}): RunOutcome {
  return { status: "SUCCEEDED", projection: projection(overrides), error: null };
}

describe("per-Brain retrieval rows", () => {
  test("derive recall, precision, missing evidence and noise from scored cases", () => {
    const rows = retrievalRows(
      projection({
        candidates: [
          candidate({ candidate: "gbrain", scored_cases: 20, answered_cases: 15 }),
        ],
      }),
    );

    const row = rows[0];
    expect(row).toBeDefined();
    expect(row?.candidate).toBe("gbrain");
    // 15 of 20 scored cases produced grounded evidence.
    expect(row?.recall).toBeCloseTo(0.75, 5);
    expect(row?.missingEvidence).toBe(5);
    // Source support is the precision of the evidence that was returned.
    expect(row?.precision).toBeCloseTo(0.8, 5);
    expect(row?.noise).toBeCloseTo(0.2, 5);
  });

  test("rank by retrieval recall and mark exactly one leader", () => {
    const rows = retrievalRows(
      projection({
        candidates: [
          candidate({ candidate: "mem0", scored_cases: 10, answered_cases: 4 }),
          candidate({ candidate: "gbrain", scored_cases: 10, answered_cases: 9 }),
          candidate({ candidate: "llm-wiki", scored_cases: 10, answered_cases: 7 }),
        ],
      }),
    );

    expect(rows.map((row) => row.candidate)).toEqual(["gbrain", "llm-wiki", "mem0"]);
    expect(rows.filter((row) => row.leader)).toHaveLength(1);
    expect(rows[0]?.leader).toBe(true);
  });

  test("never fabricate a rate when no case was scored", () => {
    const rows = retrievalRows(
      projection({ candidates: [candidate({ scored_cases: 0, answered_cases: 0 })] }),
    );

    expect(rows[0]?.recall).toBeNull();
    expect(rows[0]?.missingEvidence).toBeNull();
    // A candidate with nothing scored cannot lead the comparison.
    expect(rows[0]?.leader).toBe(false);
  });

  test("carry latency and cost through, including unavailable cost", () => {
    const rows = retrievalRows(
      projection({
        candidates: [
          candidate({
            query_p50_ms: 90,
            query_p95_ms: 310,
            cost_status: "COST_UNAVAILABLE",
            total_cost_usd: null,
          }),
        ],
      }),
    );

    expect(rows[0]?.latencyP50Ms).toBe(90);
    expect(rows[0]?.latencyP95Ms).toBe(310);
    expect(rows[0]?.costStatus).toBe("COST_UNAVAILABLE");
    expect(rows[0]?.costUsd).toBeNull();
  });

  test("flag a candidate whose engine status is not OK as not comparable", () => {
    const rows = retrievalRows(
      projection({
        candidates: [
          candidate({ candidate: "gbrain", scored_cases: 10, answered_cases: 9 }),
          candidate({ candidate: "mem0", status: "ENV_UNAVAILABLE", scored_cases: 0, answered_cases: 0 }),
        ],
      }),
    );

    const degraded = rows.find((row) => row.candidate === "mem0");
    expect(degraded?.comparable).toBe(false);
    expect(degraded?.statusNote.length).toBeGreaterThan(10);
    expect(rows.find((row) => row.candidate === "gbrain")?.comparable).toBe(true);
  });
});

describe("evidence confidence is always stated", () => {
  test("a verdict naming a candidate is reported as an engine decision, not a recommendation", () => {
    const confidence = evidenceConfidence(projection({ verdict: "gbrain", status: "OK" }));

    expect(confidence.level).toBe("ENGINE_DECISION");
    expect(confidence.recommendationGrade).toBe(false);
    expect(confidence.label.toLowerCase()).not.toContain("recommended");
    expect(confidence.caveat.length).toBeGreaterThan(20);
  });

  test("NO_RECOMMENDATION is never dressed up as a winner", () => {
    const confidence = evidenceConfidence(
      projection({ verdict: "NO_RECOMMENDATION", status: "NO_RECOMMENDATION" }),
    );

    expect(confidence.level).toBe("NO_RECOMMENDATION");
    expect(confidence.recommendationGrade).toBe(false);
    expect(confidence.caveat.toLowerCase()).toContain("not");
  });

  test("an engine status that limits the run downgrades confidence and says why", () => {
    const confidence = evidenceConfidence(
      projection({ verdict: "gbrain", status: "INSUFFICIENT_BENCHMARK" }),
    );

    expect(confidence.level).toBe("LIMITED");
    expect(confidence.recommendationGrade).toBe(false);
    expect(confidence.caveat.toLowerCase()).toContain("benchmark");
  });

  test("warnings from the engine are surfaced verbatim on the confidence notice", () => {
    const confidence = evidenceConfidence(
      projection({ warnings: ["cost telemetry was incomplete for mem0"] }),
    );

    expect(confidence.warnings).toEqual(["cost telemetry was incomplete for mem0"]);
  });

  test("no projection can ever be recommendation grade", () => {
    for (const verdict of ["gbrain", "mem0", "llm-wiki", "NO_DECISION", "NO_RECOMMENDATION"] as const) {
      expect(evidenceConfidence(projection({ verdict })).recommendationGrade).toBe(false);
    }
  });
});

describe("before and after comparison", () => {
  test("pairs candidates across two runs and reports the signed delta", () => {
    const before = projection({
      run_id: "run-before",
      candidates: [candidate({ candidate: "gbrain", scored_cases: 10, answered_cases: 6 })],
    });
    const after = projection({
      run_id: "run-after",
      candidates: [candidate({ candidate: "gbrain", scored_cases: 10, answered_cases: 9 })],
    });

    const comparison = compareRetrieval(before, after);
    const row = comparison.rows[0];

    expect(comparison.comparable).toBe(true);
    expect(row?.candidate).toBe("gbrain");
    expect(row?.beforeRecall).toBeCloseTo(0.6, 5);
    expect(row?.afterRecall).toBeCloseTo(0.9, 5);
    expect(row?.recallDelta).toBeCloseTo(0.3, 5);
    expect(row?.direction).toBe("improved");
  });

  test("refuse to compare runs whose frozen corpus or benchmark differs", () => {
    const comparison = compareRetrieval(
      projection({ run_id: "left" }),
      projection({ run_id: "right", benchmark_hash: OTHER_BENCHMARK }),
    );

    expect(comparison.comparable).toBe(false);
    expect(comparison.blocker?.toLowerCase()).toContain("benchmark");
    expect(comparison.rows).toHaveLength(0);
  });

  test("a candidate present in only one run is reported rather than silently dropped", () => {
    const comparison = compareRetrieval(
      projection({ candidates: [candidate({ candidate: "gbrain" })] }),
      projection({
        candidates: [candidate({ candidate: "gbrain" }), candidate({ candidate: "mem0" })],
      }),
    );

    const added = comparison.rows.find((row) => row.candidate === "mem0");
    expect(added?.direction).toBe("added");
    expect(added?.beforeRecall).toBeNull();
  });

  test("an unchanged candidate reads as unchanged, not as an improvement", () => {
    const comparison = compareRetrieval(projection(), projection({ run_id: "run-2" }));

    expect(comparison.rows[0]?.direction).toBe("unchanged");
    expect(comparison.rows[0]?.recallDelta).toBe(0);
  });
});

describe("recovery actions for non-succeeded runs", () => {
  test("a failed run offers a retry and names the engine detail", () => {
    const recovery = describeRecovery({
      status: "FAILED",
      projection: null,
      error: "candidate transport was unavailable",
    });

    expect(recovery).not.toBeNull();
    expect(recovery?.tone).toBe("danger");
    expect(recovery?.detail).toContain("candidate transport was unavailable");
    expect(recovery?.actions.map((action) => action.id)).toContain("rerun");
    for (const action of recovery?.actions ?? []) {
      expect(action.label.length).toBeGreaterThan(3);
      expect(action.guidance.length).toBeGreaterThan(15);
    }
  });

  test("a cancelled run states nothing was scored and offers a rerun", () => {
    const recovery = describeRecovery({ status: "CANCELLED", projection: null, error: null });

    expect(recovery?.tone).toBe("neutral");
    expect(recovery?.detail.toLowerCase()).toContain("no");
    expect(recovery?.actions.map((action) => action.id)).toContain("rerun");
  });

  test("a failed run with no detail still gives an actionable next step", () => {
    const recovery = describeRecovery({ status: "FAILED", projection: null, error: null });

    expect(recovery?.actions.length).toBeGreaterThan(0);
    expect(recovery?.detail.length).toBeGreaterThan(15);
  });

  test("a succeeded run has no recovery banner", () => {
    expect(describeRecovery(succeeded())).toBeNull();
  });
});

describe("benchmark identity", () => {
  test("the engine benchmark hash replaces the locally derived placeholder", () => {
    const resolved = resolveBenchmarkHash(OTHER_BENCHMARK, succeeded());

    expect(resolved.sha256).toBe(BENCHMARK);
    expect(resolved.source).toBe("engine");
    expect(resolved.placeholder).toBe(false);
  });

  test("without an engine result the local derivation is labeled a placeholder", () => {
    const resolved = resolveBenchmarkHash(OTHER_BENCHMARK, null);

    expect(resolved.sha256).toBe(OTHER_BENCHMARK);
    expect(resolved.source).toBe("derived");
    expect(resolved.placeholder).toBe(true);
  });

  test("no derivation and no result yields no benchmark rather than a fake one", () => {
    const resolved = resolveBenchmarkHash(null, null);

    expect(resolved.sha256).toBeNull();
    expect(resolved.placeholder).toBe(true);
  });

  test("a failed run cannot promote a placeholder to an engine hash", () => {
    const resolved = resolveBenchmarkHash(OTHER_BENCHMARK, {
      status: "FAILED",
      projection: null,
      error: "boom",
    });

    expect(resolved.source).toBe("derived");
    expect(resolved.placeholder).toBe(true);
  });
});
