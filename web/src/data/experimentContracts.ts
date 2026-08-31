/** Wire vocabulary for the canonical Python experiment contracts. */

export const EXPERIMENT_SCHEMA_VERSION = 1 as const;

export type ExperimentLifecycleStatus =
  | "CREATED"
  | "VALIDATING"
  | "READY"
  | "RUNNING"
  | "SUCCEEDED"
  | "FAILED"
  | "CANCELLED";

export type ExperimentReadinessState = "UNKNOWN" | "READY" | "BLOCKED";
export type ReadinessCheckState = "READY" | "CONFIGURED" | "NOT_CONFIGURED" | "UNSUPPORTED" | "BLOCKED";
export type ExperimentEvaluationMode = "retrieval_only" | "answer_aware";
export type StableExperimentErrorCode =
  | "INVALID_TRANSITION"
  | "INVALID_REQUEST"
  | "NOT_READY"
  | "NOT_FOUND"
  | "RUN_FAILED"
  | "CANCELLED";

export interface ExperimentIdentityContract {
  corpus: { sha256: string; document_count: number };
  benchmark_sha256: string;
  protocol: string;
  evaluator: string;
  provider: string | null;
  model: string | null;
  configuration_hash: string | null;
  code_version: string | null;
}

export interface ExperimentRequestContract {
  schema_version: typeof EXPERIMENT_SCHEMA_VERSION;
  experiment_id: string;
  identity: ExperimentIdentityContract;
  candidates: readonly CandidateId[];
  evaluation_mode: ExperimentEvaluationMode;
}

export type CandidateId = "llm-wiki" | "mem0" | "gbrain";

export interface ExperimentReadinessContract {
  state: ExperimentReadinessState;
  checks: Readonly<Record<string, ReadinessCheckState>>;
  blockers: readonly string[];
}

export interface RetrievalMetricsContract {
  relevant_retrieved: number;
  retrieved: number;
  relevant_available: number;
  missing_evidence: number;
  noise: number;
  latency_ms: number | null;
  cost_status: "COST_COMPLETE" | "COST_INCOMPLETE" | "COST_UNAVAILABLE";
  freshness_score?: number | null;
}

export interface RetrievalResultContract {
  candidate: CandidateId;
  case_id: string;
  retrieved_source_ids: readonly string[];
  metrics: RetrievalMetricsContract;
}

export interface StableExperimentErrorContract {
  code: StableExperimentErrorCode;
  detail: string;
}

const LIFECYCLE: Readonly<Record<ExperimentLifecycleStatus, readonly ExperimentLifecycleStatus[]>> = {
  CREATED: ["VALIDATING", "CANCELLED"],
  VALIDATING: ["READY", "FAILED", "CANCELLED"],
  READY: ["RUNNING", "CANCELLED"],
  RUNNING: ["SUCCEEDED", "FAILED", "CANCELLED"],
  SUCCEEDED: [],
  FAILED: [],
  CANCELLED: [],
};

export function canTransitionExperiment(
  from: ExperimentLifecycleStatus,
  to: ExperimentLifecycleStatus,
): boolean {
  return LIFECYCLE[from].includes(to);
}

export function stableExperimentError(code: StableExperimentErrorCode, detail: string): Error {
  return new Error(`${code}: ${detail}`);
}

export function validateReadiness(value: ExperimentReadinessContract): void {
  if (value.state === "READY" && value.blockers.length > 0) {
    throw new Error("invalid readiness: READY cannot contain blockers");
  }
  if (value.state === "BLOCKED" && value.blockers.length === 0) {
    throw new Error("invalid readiness: BLOCKED requires blockers");
  }
}

export function validateRetrievalMetrics(value: RetrievalMetricsContract): void {
  if (!Number.isInteger(value.relevant_retrieved) || value.relevant_retrieved < 0) {
    throw new Error("invalid retrieval metrics: relevant_retrieved");
  }
  if (value.relevant_retrieved > value.retrieved) {
    throw new Error("invalid retrieval metrics: relevant_retrieved exceeds retrieved");
  }
  if (value.relevant_retrieved > value.relevant_available) {
    throw new Error("invalid retrieval metrics: relevant_retrieved exceeds relevant_available");
  }
  if (value.missing_evidence !== value.relevant_available - value.relevant_retrieved) {
    throw new Error("invalid retrieval metrics: missing_evidence");
  }
  if (value.noise !== value.retrieved - value.relevant_retrieved) {
    throw new Error("invalid retrieval metrics: noise");
  }
}
