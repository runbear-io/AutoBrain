/**
 * Deterministic synthetic optimization data.
 *
 * Twelve fixed trials sweep retrieval mode, embedding model, top-k and reranker.
 * Trial 9 is the best feasible configuration; one trial is pruned and two
 * violate the declared constraints so the workspace has honest filter targets.
 */

import type {
  OptimizationRun,
  OptimizationStage,
  RegressionCase,
  Trial,
  TuningConfig,
} from "./types";

export const DEFAULT_TUNING: TuningConfig = {
  retrievalMode: "hybrid",
  embeddingModel: "text-embedding-3-small",
  topK: 8,
  reranker: "cross-encoder-mini",
  objective: "balanced",
  maxLatencyMs: 2500,
  maxCostUsd: 8,
  budgetUsd: 25,
  sealedHoldout: true,
};

export const OPTIMIZATION_STAGES: OptimizationStage[] = [
  { id: "warmup", label: "Warm up search space", detail: "Probe retrieval modes.", trials: 3 },
  {
    id: "embed",
    label: "Sweep embeddings",
    detail: "Compare embedding models at fixed top-k.",
    trials: 3,
  },
  {
    id: "depth",
    label: "Tune depth and rerank",
    detail: "Search top-k and reranker pairs.",
    trials: 4,
  },
  {
    id: "verify",
    label: "Verify on sealed holdout",
    detail: "Re-score the best configuration.",
    trials: 2,
  },
];

/** The unoptimized GBrain configuration carried over from the diagnosis run. */
export const BASELINE_TRIAL: Trial = {
  index: 0,
  retrievalMode: "keyword-only",
  embeddingModel: "text-embedding-3-small",
  topK: 5,
  reranker: "none",
  quality: 82.4,
  latencyP95Ms: 1810,
  costUsd: 4.87,
  state: "complete",
  violatesConstraint: false,
};

const trial = (
  index: number,
  retrievalMode: Trial["retrievalMode"],
  embeddingModel: Trial["embeddingModel"],
  topK: number,
  reranker: Trial["reranker"],
  quality: number,
  latencyP95Ms: number,
  costUsd: number,
  state: Trial["state"],
  violatesConstraint: boolean,
): Trial => ({
  index,
  retrievalMode,
  embeddingModel,
  topK,
  reranker,
  quality,
  latencyP95Ms,
  costUsd,
  state,
  violatesConstraint,
});

export const TRIALS: Trial[] = [
  trial(1, "keyword-only", "text-embedding-3-small", 5, "none", 82.4, 1780, 4.81, "complete", false),
  trial(2, "semantic", "text-embedding-3-small", 5, "none", 85.1, 1690, 5.04, "complete", false),
  trial(3, "hybrid", "text-embedding-3-small", 5, "none", 87.3, 1740, 5.22, "complete", false),
  trial(4, "hybrid", "text-embedding-3-large", 5, "none", 88.6, 1980, 6.41, "complete", false),
  trial(5, "hybrid", "voyage-4", 5, "none", 89.2, 1910, 6.05, "complete", false),
  trial(6, "hybrid", "nomic-embed-text", 5, "none", 79.8, 1520, 3.18, "pruned", false),
  trial(7, "hybrid", "voyage-4", 10, "none", 90.4, 2180, 6.88, "complete", false),
  trial(8, "hybrid", "voyage-4", 10, "cross-encoder-mini", 92.1, 2390, 7.42, "complete", false),
  trial(9, "hybrid", "voyage-4", 12, "cross-encoder-mini", 93.6, 2460, 7.81, "complete", false),
  trial(10, "hybrid", "voyage-4", 20, "cohere-rerank-3", 94.2, 3120, 9.64, "complete", true),
  trial(
    11,
    "semantic",
    "text-embedding-3-large",
    20,
    "cohere-rerank-3",
    91.0,
    2870,
    8.35,
    "complete",
    true,
  ),
  trial(12, "hybrid", "voyage-4", 12, "cohere-rerank-3", 93.1, 2480, 7.95, "complete", false),
];

/** Trial 9: highest quality among trials that satisfy every declared constraint. */
export const BEST_TRIAL: Trial = TRIALS[8] as Trial;

const REGRESSIONS: RegressionCase[] = [
  {
    id: "case-04",
    question: "What is the current SLA for enterprise support responses?",
    baselineOutcome: "pass",
    optimizedOutcome: "fail",
    note: "Reranker demoted the SLA table below a longer policy page.",
  },
  {
    id: "case-21",
    question: "Which vendor did we choose for log retention?",
    baselineOutcome: "pass",
    optimizedOutcome: "fail",
    note: "Wider top-k introduced a superseded vendor comparison doc.",
  },
];

export const OPTIMIZATION_RUN: OptimizationRun = {
  id: "opt-2026-08-24-b73c",
  label: "GBrain retrieval optimization",
  startedAt: "2026-08-24T10:41:00Z",
  status: "OK",
  baseline: BASELINE_TRIAL,
  best: BEST_TRIAL,
  trials: TRIALS,
  holdoutQuality: 92.8,
  holdoutDelta: -0.8,
  regressions: REGRESSIONS,
  recommendedConfig: {
    retrievalMode: "hybrid",
    embeddingModel: "voyage-4",
    topK: 12,
    reranker: "cross-encoder-mini",
    objective: "balanced",
    maxLatencyMs: 2500,
    maxCostUsd: 8,
    budgetUsd: 25,
    sealedHoldout: true,
  },
};

/**
 * Pareto frontier over (quality maximized, latency minimized).
 *
 * A trial is on the frontier when no other considered trial is at least as good
 * on both axes and strictly better on one. Constraint-violating and pruned
 * trials are excluded so the frontier only contains adoptable configurations.
 */
export function paretoFrontier(trials: Trial[]): Trial[] {
  const feasible = trials.filter((t) => t.state === "complete" && !t.violatesConstraint);
  const frontier = feasible.filter(
    (candidate) =>
      !feasible.some(
        (other) =>
          other !== candidate &&
          other.quality >= candidate.quality &&
          other.latencyP95Ms <= candidate.latencyP95Ms &&
          (other.quality > candidate.quality || other.latencyP95Ms < candidate.latencyP95Ms),
      ),
  );
  return frontier.sort((a, b) => a.latencyP95Ms - b.latencyP95Ms);
}

/** Best feasible trial by quality; ties break toward lower latency. */
export function bestFeasibleTrial(trials: Trial[]): Trial | null {
  const feasible = trials.filter((t) => t.state === "complete" && !t.violatesConstraint);
  return (
    feasible.reduce<Trial | null>((best, current) => {
      if (!best) return current;
      if (current.quality > best.quality) return current;
      if (current.quality === best.quality && current.latencyP95Ms < best.latencyP95Ms) {
        return current;
      }
      return best;
    }, null) ?? null
  );
}

export const QUALITY_DELTA = Number((BEST_TRIAL.quality - BASELINE_TRIAL.quality).toFixed(1));
export const LATENCY_DELTA_MS = BEST_TRIAL.latencyP95Ms - BASELINE_TRIAL.latencyP95Ms;
export const COST_DELTA_USD = Number((BEST_TRIAL.costUsd - BASELINE_TRIAL.costUsd).toFixed(2));
