/**
 * Retrieval results, comparison, and recovery projections.
 *
 * This is a pure module. It reshapes what the Python engine already reported in
 * a `RunProjection` into rows a person can read; it never scores, ranks by an
 * invented heuristic, or fills a missing measurement with a plausible number.
 *
 * Two honesty rules are enforced here rather than in the view, so no future
 * component can quietly break them:
 *
 *   1. A missing measurement stays `null`. A candidate with nothing scored has
 *      no recall and cannot lead the comparison.
 *   2. Nothing produced here is recommendation-grade. The engine reports a
 *      `verdict` under a declared decision protocol; that is an engine decision
 *      about one frozen corpus, not advice about production. `recommendationGrade`
 *      is therefore a constant `false` and every projection carries the caveat
 *      that explains why.
 */

import type { CandidateId, CostStatus, RunStatus } from "../data/types";
import type { CandidateProjection, RunOutcome, RunProjection } from "./runClient";

/* -------------------------------------------------------------------- rows */

/** One Brain's retrieval behavior on the frozen corpus. */
export interface RetrievalRow {
  candidate: CandidateId;
  status: RunStatus;
  /** True only when the engine scored this candidate normally. */
  comparable: boolean;
  /** Why this candidate is or is not comparable, in a full sentence. */
  statusNote: string;
  scoredCases: number;
  /** Cases that produced grounded evidence, or null when nothing was scored. */
  groundedCases: number | null;
  /** groundedCases / scoredCases, or null when nothing was scored. */
  recall: number | null;
  /** Cases whose gold evidence was never retrieved, or null when unscored. */
  missingEvidence: number | null;
  /** Share of returned evidence that was actually supported by a source. */
  precision: number;
  /** The complement of precision: returned evidence with no source support. */
  noise: number;
  qualityScore: number;
  contradictionCount: number;
  latencyP50Ms: number | null;
  latencyP95Ms: number | null;
  costStatus: CostStatus;
  costUsd: number | null;
  operatingBurden: number | null;
  /** True for the single highest-recall comparable candidate, if any. */
  leader: boolean;
}

const STATUS_NOTES: Partial<Record<RunStatus, string>> = {
  ENV_UNAVAILABLE: "The local environment for this Brain was unavailable, so it was not scored.",
  MISSING_PROVIDER: "No provider was configured for this Brain, so it was not scored.",
  MCP_AUTH_UNAVAILABLE:
    "This Brain's read-only connector was not authorized, so it was not scored.",
  CAPABILITY_UNAVAILABLE: "This Brain does not support the capability this run required.",
  BUDGET_EXCEEDED: "The run stopped this Brain at the declared budget ceiling.",
  INSUFFICIENT_BENCHMARK: "The benchmark had too few grounded cases to score this Brain.",
  LEAKAGE_DETECTED: "Benchmark leakage was detected, so this Brain's score is not trustworthy.",
  FAILED: "This Brain failed during the run and produced no comparable measurement.",
  CANCELLED: "This Brain was cancelled before it finished, so its numbers are incomplete.",
  NO_DECISION: "The engine reached no decision for this Brain.",
  NO_RECOMMENDATION: "The engine declined to name this Brain under the decision protocol.",
};

const COMPARABLE_NOTE = "Scored normally against the frozen corpus.";

/**
 * Project each candidate onto a retrieval row, ranked by recall.
 *
 * `answered_cases` is the engine's count of cases that produced grounded
 * evidence, so recall is that count over the scored cases and missing evidence
 * is the remainder. Both stay null when nothing was scored rather than
 * collapsing into a flattering zero.
 */
export function retrievalRows(projection: RunProjection): RetrievalRow[] {
  const rows = projection.candidates.map(toRow);
  // Rank by recall descending. Unscored candidates sink to the bottom, and ties
  // fall back to the candidate id so the order is stable across renders.
  const ranked = [...rows].sort((left, right) => {
    const delta = (right.recall ?? -1) - (left.recall ?? -1);
    return delta !== 0 ? delta : left.candidate.localeCompare(right.candidate);
  });
  const leader = ranked.find((row) => row.comparable && row.recall !== null);
  return ranked.map((row) => ({ ...row, leader: row === leader }));
}

function toRow(candidate: CandidateProjection): RetrievalRow {
  const scored = candidate.scored_cases;
  const grounded = scored === 0 ? null : candidate.answered_cases;
  const comparable = candidate.status === "OK";
  return {
    candidate: candidate.candidate,
    status: candidate.status,
    comparable,
    statusNote: comparable
      ? COMPARABLE_NOTE
      : (STATUS_NOTES[candidate.status] ??
        "The engine reported a non-standard status for this Brain, so it is not comparable."),
    scoredCases: scored,
    groundedCases: grounded,
    recall: grounded === null ? null : grounded / scored,
    missingEvidence: grounded === null ? null : scored - grounded,
    precision: candidate.source_support_rate,
    noise: 1 - candidate.source_support_rate,
    qualityScore: candidate.quality_score,
    contradictionCount: candidate.contradiction_count,
    latencyP50Ms: candidate.query_p50_ms,
    latencyP95Ms: candidate.query_p95_ms,
    costStatus: candidate.cost_status,
    costUsd: candidate.total_cost_usd,
    operatingBurden: candidate.operating_burden,
    leader: false,
  };
}

/* -------------------------------------------------------------- confidence */

export type ConfidenceLevel = "ENGINE_DECISION" | "LIMITED" | "NO_DECISION" | "NO_RECOMMENDATION";

/**
 * How much weight this result can carry.
 *
 * `recommendationGrade` is read by the view and is always false: this surface
 * reports one measured run over one frozen corpus, which is evidence for a
 * decision rather than a production recommendation.
 */
export interface EvidenceConfidence {
  level: ConfidenceLevel;
  label: string;
  caveat: string;
  recommendationGrade: false;
  warnings: string[];
}

/** Engine statuses that mean the run finished but its evidence is limited. */
const LIMITING_STATUSES: Partial<Record<RunStatus, string>> = {
  INSUFFICIENT_BENCHMARK:
    "The benchmark held too few grounded cases, so this comparison settles nothing on its own.",
  LEAKAGE_DETECTED:
    "Benchmark leakage was detected, so these scores may reflect memorization rather than retrieval.",
  BUDGET_EXCEEDED:
    "The run stopped at its declared budget ceiling, so some Brains were scored on fewer cases.",
  ENV_UNAVAILABLE:
    "Part of the local environment was unavailable, so at least one Brain went unscored.",
  MISSING_PROVIDER: "A provider was missing, so at least one Brain went unscored.",
  MCP_AUTH_UNAVAILABLE:
    "A read-only connector was not authorized, so at least one Brain went unscored.",
  CAPABILITY_UNAVAILABLE:
    "A required capability was unavailable, so at least one Brain went unscored.",
};

const DECISION_CAVEAT =
  "This is one measured run over one frozen corpus under the retrieval protocol. " +
  "It is evidence for your decision, not a production recommendation.";

/**
 * Describe how far this projection can be trusted.
 *
 * The three cases are kept distinct on purpose: the engine naming a candidate,
 * the engine explicitly declining, and the engine finishing under a condition
 * that limits what the numbers mean.
 */
export function evidenceConfidence(projection: RunProjection): EvidenceConfidence {
  const warnings = [...projection.warnings];
  const limiting = LIMITING_STATUSES[projection.status];
  if (limiting !== undefined) {
    return {
      level: "LIMITED",
      label: "Limited evidence",
      caveat: `${limiting} ${DECISION_CAVEAT}`,
      recommendationGrade: false,
      warnings,
    };
  }
  if (projection.verdict === "NO_RECOMMENDATION") {
    return {
      level: "NO_RECOMMENDATION",
      label: "No candidate named",
      caveat:
        "The engine did not name a Brain: no candidate cleared the decision protocol on this " +
        `corpus. ${DECISION_CAVEAT}`,
      recommendationGrade: false,
      warnings,
    };
  }
  if (projection.verdict === "NO_DECISION") {
    return {
      level: "NO_DECISION",
      label: "No decision reached",
      caveat: `The engine reached no decision on this corpus. ${DECISION_CAVEAT}`,
      recommendationGrade: false,
      warnings,
    };
  }
  return {
    level: "ENGINE_DECISION",
    label: `Engine decision: ${projection.verdict}`,
    caveat: DECISION_CAVEAT,
    recommendationGrade: false,
    warnings,
  };
}

/* -------------------------------------------------------------- comparison */

export type ComparisonDirection = "improved" | "regressed" | "unchanged" | "added" | "removed";

export interface ComparisonRow {
  candidate: CandidateId;
  beforeRecall: number | null;
  afterRecall: number | null;
  /** after - before, or null when either side has no measurement. */
  recallDelta: number | null;
  direction: ComparisonDirection;
}

export interface RetrievalComparison {
  beforeRunId: string;
  afterRunId: string;
  /** False when the two runs did not share a frozen corpus and benchmark. */
  comparable: boolean;
  /** Why the comparison was refused, or null when it is valid. */
  blocker: string | null;
  rows: ComparisonRow[];
}

/**
 * Compare two runs candidate by candidate.
 *
 * Comparison is refused outright when the corpus or benchmark differs. Two runs
 * over different evidence produce numbers that look comparable and are not, and
 * a visible refusal is far safer than a delta nobody can interpret.
 */
export function compareRetrieval(
  before: RunProjection,
  after: RunProjection,
): RetrievalComparison {
  const identity = { beforeRunId: before.run_id, afterRunId: after.run_id };
  if (before.corpus_hash !== after.corpus_hash) {
    return {
      ...identity,
      comparable: false,
      blocker:
        "These runs used different frozen corpora, so their retrieval numbers are not comparable.",
      rows: [],
    };
  }
  if (before.benchmark_hash !== after.benchmark_hash) {
    return {
      ...identity,
      comparable: false,
      blocker:
        "These runs used different benchmarks, so their retrieval numbers are not comparable.",
      rows: [],
    };
  }

  const beforeRows = new Map(retrievalRows(before).map((row) => [row.candidate, row]));
  const afterRows = new Map(retrievalRows(after).map((row) => [row.candidate, row]));
  const candidates = [...new Set([...beforeRows.keys(), ...afterRows.keys()])].sort();

  return {
    ...identity,
    comparable: true,
    blocker: null,
    rows: candidates.map((candidate) =>
      toComparisonRow(candidate, beforeRows.get(candidate), afterRows.get(candidate)),
    ),
  };
}

function toComparisonRow(
  candidate: CandidateId,
  before: RetrievalRow | undefined,
  after: RetrievalRow | undefined,
): ComparisonRow {
  if (before === undefined) {
    return {
      candidate,
      beforeRecall: null,
      afterRecall: after?.recall ?? null,
      recallDelta: null,
      direction: "added",
    };
  }
  if (after === undefined) {
    return {
      candidate,
      beforeRecall: before.recall,
      afterRecall: null,
      recallDelta: null,
      direction: "removed",
    };
  }
  if (before.recall === null || after.recall === null) {
    // Present in both runs but unscored on at least one side: there is no delta
    // to report, and inventing one would imply a measurement that never existed.
    return {
      candidate,
      beforeRecall: before.recall,
      afterRecall: after.recall,
      recallDelta: null,
      direction: "unchanged",
    };
  }
  const recallDelta = after.recall - before.recall;
  return {
    candidate,
    beforeRecall: before.recall,
    afterRecall: after.recall,
    recallDelta,
    direction: recallDelta > 0 ? "improved" : recallDelta < 0 ? "regressed" : "unchanged",
  };
}

/* ---------------------------------------------------------------- recovery */

export interface RecoveryAction {
  id: "rerun" | "review-setup" | "inspect-runner";
  label: string;
  guidance: string;
}

export interface RecoveryNotice {
  tone: "danger" | "neutral";
  headline: string;
  detail: string;
  actions: RecoveryAction[];
}

const RERUN: RecoveryAction = {
  id: "rerun",
  label: "Run the Preview again",
  guidance: "Resubmit the same setup; the local runner assigns a fresh experiment identity.",
};

const REVIEW_SETUP: RecoveryAction = {
  id: "review-setup",
  label: "Review the setup",
  guidance: "Reopen the wizard to check the source, Brain candidates, and subscription choices.",
};

const INSPECT_RUNNER: RecoveryAction = {
  id: "inspect-runner",
  label: "Check the local runner",
  guidance: "Read the runner's own output to find the stage that failed before retrying.",
};

/**
 * Describe what a person can do about a run that did not produce a result.
 *
 * Returns null for a succeeded run so the view renders results rather than a
 * banner. Every other terminal state gets at least one concrete action, because
 * a dead end with no next step is not an acceptable failure surface.
 */
export function describeRecovery(outcome: RunOutcome): RecoveryNotice | null {
  if (outcome.status === "SUCCEEDED") return null;
  if (outcome.status === "CANCELLED") {
    return {
      tone: "neutral",
      headline: "This run was cancelled",
      detail: "No Brain was scored and no result was produced, so there is nothing to compare.",
      actions: [RERUN, REVIEW_SETUP],
    };
  }
  return {
    tone: "danger",
    headline: "This run failed before it produced a result",
    detail:
      outcome.error === null
        ? "The local runner reported a failure without a detail. Its own output has the failing stage."
        : `The local runner reported: ${outcome.error}`,
    actions: [RERUN, INSPECT_RUNNER, REVIEW_SETUP],
  };
}

/* ------------------------------------------------------------- benchmark id */

export interface BenchmarkIdentity {
  /** Null when no run has succeeded and no local derivation exists. */
  sha256: string | null;
  /** `derived` is the wizard's local placeholder; `engine` is authoritative. */
  source: "derived" | "engine";
  placeholder: boolean;
}

/**
 * Prefer the engine's benchmark hash over the wizard's local derivation.
 *
 * The setup wizard derives a benchmark fingerprint from the frozen corpus so an
 * experiment has an identity before it runs. That value is a placeholder: only
 * a succeeded run reports the benchmark the engine actually scored. Anything
 * short of a succeeded run leaves the placeholder in place and labeled.
 */
export function resolveBenchmarkHash(
  derived: string | null,
  outcome: RunOutcome | null,
): BenchmarkIdentity {
  const projection = outcome?.status === "SUCCEEDED" ? outcome.projection : null;
  if (projection === null) {
    return { sha256: derived, source: "derived", placeholder: true };
  }
  return { sha256: projection.benchmark_hash, source: "engine", placeholder: false };
}
