/**
 * Wizard readiness and request-building rules.
 *
 * The wizard must never let a user submit a configuration the local job
 * boundary would reject, and it must never claim READY while a blocker exists.
 * Every blocker is asserted to carry human-readable guidance, because a raw
 * machine code on screen is the failure mode this todo exists to remove.
 */

import { describe, expect, test } from "bun:test";
import {
  buildExperimentRequest,
  initialSetup,
  reduceSetup,
  setupReadiness,
  type SetupState,
} from "./setupModel";

const JSONL_TWO_RECORDS = [
  JSON.stringify({ source_id: "doc-1", title: "Refund policy", text: "Refunds within 30 days." }),
  JSON.stringify({ source_id: "doc-2", title: "Escalation", text: "Escalate after two replies." }),
].join("\n");

/** A configuration that satisfies every readiness check. */
function readySetup(): SetupState {
  let state = initialSetup();
  state = reduceSetup(state, { type: "source/select", source: "slack-export" });
  state = reduceSetup(state, { type: "candidate/toggle", candidate: "gbrain" });
  state = reduceSetup(state, { type: "subscription/select", subscription: "codex" });
  return state;
}

describe("experiment setup readiness", () => {
  test("starts unknown with no source, candidate, or subscription chosen", () => {
    const readiness = setupReadiness(initialSetup());
    expect(readiness.state).not.toBe("READY");
    expect(readiness.blockers.length).toBeGreaterThan(0);
  });

  test("every blocker is human readable, not a bare machine code", () => {
    const readiness = setupReadiness(initialSetup());
    for (const blocker of readiness.blockers) {
      expect(blocker.title.length).toBeGreaterThan(0);
      expect(blocker.guidance.length).toBeGreaterThan(0);
      // Guidance must be a sentence a non-CLI user can act on.
      expect(blocker.guidance).toMatch(/[a-z]{3,}\s+[a-z]{2,}/i);
      expect(blocker.guidance).not.toBe(blocker.code);
    }
  });

  test("an official Slack export with a candidate and subscription is ready", () => {
    const readiness = setupReadiness(readySetup());
    expect(readiness.state).toBe("READY");
    expect(readiness.blockers).toEqual([]);
    expect(readiness.checks.source).toBe("READY");
    expect(readiness.checks.candidates).toBe("READY");
    expect(readiness.checks.evaluator).toBe("READY");
  });

  test("a fixture source is blocked even if selected by an internal caller", () => {
    let state = readySetup();
    state = reduceSetup(state, { type: "source/select", source: "fixture" });
    const readiness = setupReadiness(state);
    expect(readiness.state).toBe("BLOCKED");
    expect(readiness.checks.source).toBe("BLOCKED");
    expect(readiness.blockers.find((item) => item.code === "SOURCE_NOT_SUPPORTED")?.guidance).toContain(
      "official",
    );
  });

  test("a gated source is blocked with remediation and can never be ready", () => {
    let state = readySetup();
    state = reduceSetup(state, { type: "source/select", source: "approved-read-only-connector" });
    const readiness = setupReadiness(state);
    expect(readiness.state).toBe("BLOCKED");
    expect(readiness.checks.source).toBe("BLOCKED");
    const blocker = readiness.blockers.find((item) => item.code === "SOURCE_GATED");
    expect(blocker?.guidance).toContain("approval");
  });

  test("JSONL selection requires a payload that parses into records", () => {
    let state = readySetup();
    state = reduceSetup(state, { type: "source/select", source: "slack-export" });
    state = reduceSetup(state, { type: "source/format", format: "JSONL" });
    expect(setupReadiness(state).state).toBe("BLOCKED");

    state = reduceSetup(state, { type: "source/payload", payload: JSONL_TWO_RECORDS });
    const readiness = setupReadiness(state);
    expect(readiness.state).toBe("READY");
    expect(state.parsed?.records.length).toBe(2);
  });

  test("malformed JSONL reports the offending line as guidance and stays blocked", () => {
    let state = readySetup();
    state = reduceSetup(state, { type: "source/format", format: "JSONL" });
    state = reduceSetup(state, { type: "source/payload", payload: '{"source_id":"a"}\nnot json' });
    const readiness = setupReadiness(state);
    expect(readiness.state).toBe("BLOCKED");
    const blocker = readiness.blockers.find((item) => item.code === "SOURCE_IMPORT_INVALID");
    expect(blocker?.guidance).toContain("line 2");
    expect(state.parsed).toBeNull();
  });

  test("credential-shaped fields in an import are refused with guidance", () => {
    let state = readySetup();
    state = reduceSetup(state, { type: "source/format", format: "JSONL" });
    state = reduceSetup(state, {
      type: "source/payload",
      payload: JSON.stringify({ source_id: "doc-1", title: "t", text: "x", api_key: "sk-live" }),
    });
    const readiness = setupReadiness(state);
    expect(readiness.state).toBe("BLOCKED");
    expect(
      readiness.blockers.some((item) => item.guidance.toLowerCase().includes("not allowed")),
    ).toBe(true);
    // The rejected secret value must never be echoed back to the operator.
    expect(JSON.stringify(readiness)).not.toContain("sk-live");
  });

  test("deselecting every candidate blocks the run", () => {
    let state = readySetup();
    state = reduceSetup(state, { type: "candidate/toggle", candidate: "gbrain" });
    const readiness = setupReadiness(state);
    expect(readiness.state).toBe("BLOCKED");
    expect(readiness.checks.candidates).toBe("NOT_CONFIGURED");
  });

  test("answer-aware evaluation is not offered as the retrieval-only default", () => {
    expect(initialSetup().evaluationMode).toBe("retrieval_only");
    const request = buildExperimentRequest(readySetup(), "exp-fixed");
    expect(request.evaluation_mode).toBe("retrieval_only");
  });
});

describe("experiment request building", () => {
  test("produces a schema-version-1 request the Python boundary accepts", () => {
    const request = buildExperimentRequest(readySetup(), "exp-fixed");
    expect(request.schema_version).toBe(1);
    expect(request.experiment_id).toBe("exp-fixed");
    expect(request.candidates).toEqual(["gbrain"]);
    expect(request.identity.protocol.length).toBeGreaterThan(0);
    expect(request.identity.evaluator).toBe("retrieval");
    expect(request.identity.corpus.sha256).toMatch(/^[0-9a-f]{64}$/);
    expect(request.identity.benchmark_sha256).toMatch(/^[0-9a-f]{64}$/);
  });

  test("refuses to build a request from a blocked configuration", () => {
    expect(() => buildExperimentRequest(initialSetup(), "exp-fixed")).toThrow(/NOT_READY/);
  });

  test("the request carries no payload text or credential-shaped field", () => {
    let state = readySetup();
    state = reduceSetup(state, { type: "source/format", format: "JSONL" });
    state = reduceSetup(state, { type: "source/payload", payload: JSONL_TWO_RECORDS });
    const serialized = JSON.stringify(buildExperimentRequest(state, "exp-fixed"));
    expect(serialized).not.toContain("Refunds within 30 days");
    expect(serialized.toLowerCase()).not.toContain("token");
  });

  test("the corpus fingerprint changes when the imported records change", () => {
    let base = readySetup();
    base = reduceSetup(base, { type: "source/format", format: "JSONL" });
    const first = reduceSetup(base, { type: "source/payload", payload: JSONL_TWO_RECORDS });
    const second = reduceSetup(base, {
      type: "source/payload",
      payload: JSON.stringify({ source_id: "doc-9", title: "Other", text: "Different corpus." }),
    });
    const left = buildExperimentRequest(first, "a").identity;
    const right = buildExperimentRequest(second, "b").identity;
    expect(left.corpus.sha256).not.toBe(right.corpus.sha256);
    expect(left.corpus.document_count).toBe(2);
    expect(right.corpus.document_count).toBe(1);
  });
});
