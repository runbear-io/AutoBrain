/**
 * Domain types for the AutoBrain demo UI.
 *
 * The string unions intentionally mirror the canonical machine vocabulary used
 * by the AutoBrain engine (`Status`, `Verdict`, `CandidateId`, `CostStatus`,
 * `ConnectionState`) so the demo renders the same tokens the real product does.
 */

export type CandidateId = "llm-wiki" | "mem0" | "gbrain";

export type Verdict = CandidateId | "NO_DECISION" | "NO_RECOMMENDATION";

export type RunStatus =
  | "OK"
  | "ENV_UNAVAILABLE"
  | "MISSING_PROVIDER"
  | "MCP_AUTH_UNAVAILABLE"
  | "CAPABILITY_UNAVAILABLE"
  | "BUDGET_EXCEEDED"
  | "INSUFFICIENT_BENCHMARK"
  | "LEAKAGE_DETECTED"
  | "FAILED"
  | "CANCELLED"
  | "NO_DECISION"
  | "NO_RECOMMENDATION";

export type CostStatus = "COST_COMPLETE" | "COST_INCOMPLETE" | "COST_UNAVAILABLE";

export type ConnectionState =
  | "DISCONNECTED"
  | "CONNECTED"
  | "EXPIRED"
  | "REAUTHORIZATION_REQUIRED";

/** Subscription providers used for grounded question generation and scoring. */
export type SubscriptionId = "codex" | "claude";

export interface Subscription {
  id: SubscriptionId;
  name: string;
  vendor: string;
  purpose: string;
  state: ConnectionState;
  plan: string;
  account: string | null;
  /** Model identifier used for benchmark generation and isolated evaluation. */
  model: string;
}

export type SourceId = "slack" | "notion";

/** Stable identifier shared with the source import/capability contract. */
export type SourceCapabilityId =
  | "fixture"
  | "slack-export"
  | "notion-snapshot"
  | "approved-read-only-connector";

export type SourceStage = "DISCONNECTED" | "CONNECTED" | "CONFIGURED" | "IMPORTED";

export interface SourceStats {
  documents: number;
  channelsOrPages: number;
  people: number;
  /** Byte size of the frozen corpus contribution. */
  bytes: number;
  oldest: string;
  newest: string;
}

export interface KnowledgeSource {
  id: SourceId;
  /** Capability/import contract used by setup and readiness surfaces. */
  capabilityId: Exclude<SourceCapabilityId, "fixture" | "approved-read-only-connector">;
  name: string;
  kind: string;
  stage: SourceStage;
  state: ConnectionState;
  detail: string;
  /** Populated once the source reaches the IMPORTED stage. */
  stats: SourceStats | null;
  mutability: "frozen_export" | "live_mcp_captured";
  coverage: "EXHAUSTIVE" | "SEARCH_DISCOVERED" | "UNKNOWN";
}

/* ------------------------------------------------------------------ scoring */

export interface QualityBreakdown {
  /** Grounded retrieval recall over gold source IDs, 0-100. */
  recall: number;
  answerSuccessRate: number;
  sourceSupportRate: number;
  contradictions: number;
}

export interface LatencyProfile {
  p50Ms: number;
  p95Ms: number;
  ingestSeconds: number;
}

export interface CostProfile {
  status: CostStatus;
  totalUsd: number;
  inputTokens: number;
  outputTokens: number;
}

export interface CandidateScore {
  id: CandidateId;
  name: string;
  blurb: string;
  status: RunStatus;
  eligible: boolean;
  quality: QualityBreakdown;
  latency: LatencyProfile;
  cost: CostProfile;
  /** Operational burden, 1 (lowest) to 5 (highest). */
  operationalBurden: number;
  weaknesses: string[];
  scoredCases: number;
}

export interface FailedCase {
  id: string;
  question: string;
  /** Candidates that failed to retrieve the gold evidence for this question. */
  failedFor: CandidateId[];
  goldSourceIds: string[];
  retrievedSourceIds: Record<CandidateId, string[]>;
  reason: string;
  sourceLabel: string;
}

export interface DiagnosisRun {
  id: string;
  label: string;
  startedAt: string;
  status: RunStatus;
  verdict: Verdict;
  corpusHash: string;
  benchmarkHash: string;
  scoredCases: number;
  candidates: CandidateScore[];
  failedCases: FailedCase[];
  recommendation: string;
  qualityLeadPoints: number;
}

/* ------------------------------------------------------- diagnosis progress */

export type StageState = "pending" | "active" | "complete";

export interface DiagnosisStage {
  id: string;
  label: string;
  detail: string;
  /** Relative weight used to derive deterministic progress percentages. */
  weight: number;
}

/* ------------------------------------------------------------- optimization */

export type RetrievalMode = "keyword-only" | "semantic" | "hybrid";

export type EmbeddingModel =
  | "text-embedding-3-small"
  | "text-embedding-3-large"
  | "voyage-4"
  | "gemini-embedding-001"
  | "nomic-embed-text";

export type Reranker = "none" | "cross-encoder-mini" | "cohere-rerank-3";

export type Objective = "max-quality" | "balanced" | "min-cost";

export interface TuningConfig {
  retrievalMode: RetrievalMode;
  embeddingModel: EmbeddingModel;
  topK: number;
  reranker: Reranker;
  objective: Objective;
  maxLatencyMs: number;
  maxCostUsd: number;
  budgetUsd: number;
  sealedHoldout: boolean;
}

export type TrialState = "queued" | "running" | "complete" | "pruned";

export interface Trial {
  index: number;
  retrievalMode: RetrievalMode;
  embeddingModel: EmbeddingModel;
  topK: number;
  reranker: Reranker;
  quality: number;
  latencyP95Ms: number;
  costUsd: number;
  state: TrialState;
  /** True when the trial violates a declared latency or cost constraint. */
  violatesConstraint: boolean;
}

export interface OptimizationStage {
  id: string;
  label: string;
  detail: string;
  trials: number;
}

export interface RegressionCase {
  id: string;
  question: string;
  baselineOutcome: "pass" | "fail";
  optimizedOutcome: "pass" | "fail";
  note: string;
}

export interface OptimizationRun {
  id: string;
  label: string;
  startedAt: string;
  status: RunStatus;
  baseline: Trial;
  best: Trial;
  trials: Trial[];
  holdoutQuality: number;
  holdoutDelta: number;
  regressions: RegressionCase[];
  recommendedConfig: TuningConfig;
}

/* ------------------------------------------------------------------ history */

export type HistoryKind = "diagnosis" | "optimization";

export interface HistoryEntry {
  id: string;
  kind: HistoryKind;
  label: string;
  startedAt: string;
  status: RunStatus;
  headline: string;
}
