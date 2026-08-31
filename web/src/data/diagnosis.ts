/**
 * Deterministic synthetic diagnosis data.
 *
 * Every value here is a fixed literal, never randomized, so screenshots, tests
 * and repeated demo runs always agree. GBrain wins on retrieval recall, which
 * is the primary decision dimension, while remaining honest about the fact that
 * it is not the cheapest or the fastest candidate.
 */

import type {
  CandidateScore,
  DiagnosisRun,
  DiagnosisStage,
  FailedCase,
  KnowledgeSource,
  Subscription,
} from "./types";

export const SUBSCRIPTIONS: Subscription[] = [
  {
    id: "codex",
    name: "Codex",
    vendor: "OpenAI",
    purpose: "Grounded benchmark generation and isolated answer scoring.",
    state: "CONNECTED",
    plan: "ChatGPT Plus",
    account: "kim@runbear.io",
    model: "gpt-5-codex",
  },
  {
    id: "claude",
    name: "Claude",
    vendor: "Anthropic",
    purpose: "Second-opinion evaluator used to cross-check judge bias.",
    state: "DISCONNECTED",
    plan: "Claude Max",
    account: null,
    model: "claude-sonnet-4-6",
  },
];

export const SOURCES: KnowledgeSource[] = [
  {
    id: "slack",
    capabilityId: "slack-export",
    name: "Slack",
    kind: "Workspace export ZIP",
    stage: "IMPORTED",
    state: "CONNECTED",
    detail: "slack-export-2026-08-02.zip verified by SHA-256 before every run.",
    mutability: "frozen_export",
    coverage: "EXHAUSTIVE",
    stats: {
      documents: 18432,
      channelsOrPages: 74,
      people: 128,
      bytes: 96337920,
      oldest: "2024-03-11",
      newest: "2026-08-02",
    },
  },
  {
    id: "notion",
    capabilityId: "notion-snapshot",
    name: "Notion",
    kind: "Read-only MCP snapshot",
    stage: "CONFIGURED",
    state: "REAUTHORIZATION_REQUIRED",
    detail: "Read-only MCP grant expired after 30 days. Reconnect to refresh coverage.",
    mutability: "live_mcp_captured",
    coverage: "SEARCH_DISCOVERED",
    stats: {
      documents: 2140,
      channelsOrPages: 312,
      people: 46,
      bytes: 14680064,
      oldest: "2023-09-05",
      newest: "2026-07-28",
    },
  },
];

/** Stages of a diagnosis run, in execution order. */
export const DIAGNOSIS_STAGES: DiagnosisStage[] = [
  {
    id: "freeze",
    label: "Freeze corpus",
    detail: "Normalize Slack and Notion into one immutable corpus.",
    weight: 18,
  },
  {
    id: "benchmark",
    label: "Build grounded benchmark",
    detail: "Generate questions anchored to real source documents.",
    weight: 20,
  },
  {
    id: "holdout",
    label: "Separate evaluator holdout",
    detail: "Withhold gold evidence from every candidate.",
    weight: 12,
  },
  {
    id: "llm-wiki",
    label: "Run LLM Wiki",
    detail: "Native ingest, retrieval, and answer lifecycle.",
    weight: 16,
  },
  { id: "mem0", label: "Run Mem0 OSS", detail: "Memory ingestion and recall lifecycle.", weight: 16 },
  { id: "gbrain", label: "Run GBrain", detail: "Init, import, sync, search, and query.", weight: 12 },
  { id: "score", label: "Score and decide", detail: "Apply eligibility gates and rank candidates.", weight: 6 },
];

export const TOTAL_STAGE_WEIGHT = DIAGNOSIS_STAGES.reduce((sum, stage) => sum + stage.weight, 0);

const CANDIDATES: CandidateScore[] = [
  {
    id: "gbrain",
    name: "GBrain",
    blurb: "Graph-backed retrieval with native sync over Slack threads and Notion pages.",
    status: "OK",
    eligible: true,
    quality: {
      recall: 82.4,
      answerSuccessRate: 98.0,
      sourceSupportRate: 79.2,
      contradictions: 1,
    },
    latency: { p50Ms: 940, p95Ms: 1810, ingestSeconds: 612 },
    cost: {
      status: "COST_COMPLETE",
      totalUsd: 4.87,
      inputTokens: 1284000,
      outputTokens: 96400,
    },
    operationalBurden: 3,
    weaknesses: [
      "Highest ingest time of the three candidates at 10m 12s.",
      "Thread-heavy channels still lose some reply context past depth 4.",
    ],
    scoredCases: 30,
  },
  {
    id: "llm-wiki",
    name: "LLM Wiki",
    blurb: "Chunked wiki index with a single-pass dense retriever.",
    status: "OK",
    eligible: true,
    quality: {
      recall: 71.6,
      answerSuccessRate: 96.7,
      sourceSupportRate: 68.4,
      contradictions: 3,
    },
    latency: { p50Ms: 610, p95Ms: 1180, ingestSeconds: 284 },
    cost: {
      status: "COST_COMPLETE",
      totalUsd: 3.12,
      inputTokens: 968000,
      outputTokens: 74200,
    },
    operationalBurden: 2,
    weaknesses: [
      "Loses multi-hop questions that span a Slack thread and a Notion spec.",
      "Chunk boundaries split decision records away from their rationale.",
    ],
    scoredCases: 30,
  },
  {
    id: "mem0",
    name: "Mem0 OSS",
    blurb: "Memory-extraction pipeline that stores distilled facts rather than passages.",
    status: "OK",
    eligible: true,
    quality: {
      recall: 63.9,
      answerSuccessRate: 93.3,
      sourceSupportRate: 57.1,
      contradictions: 6,
    },
    latency: { p50Ms: 520, p95Ms: 990, ingestSeconds: 356 },
    cost: {
      status: "COST_COMPLETE",
      totalUsd: 2.94,
      inputTokens: 742000,
      outputTokens: 58800,
    },
    operationalBurden: 2,
    weaknesses: [
      "Fact distillation drops the source IDs needed for grounded citations.",
      "Six contradictions between remembered facts and the frozen corpus.",
    ],
    scoredCases: 30,
  },
];

const FAILED_CASES: FailedCase[] = [
  {
    id: "case-07",
    question: "Why did we move the billing cutover from March to April?",
    failedFor: ["llm-wiki", "mem0"],
    goldSourceIds: ["slack:C04BILLING:1710182400", "notion:page:billing-cutover-rfc"],
    retrievedSourceIds: {
      gbrain: ["slack:C04BILLING:1710182400", "notion:page:billing-cutover-rfc"],
      "llm-wiki": ["notion:page:billing-cutover-rfc"],
      mem0: ["slack:C04BILLING:1709577600"],
    },
    reason: "Answer requires joining a Slack decision thread with the Notion RFC that supersedes it.",
    sourceLabel: "Slack #billing + Notion RFC",
  },
  {
    id: "case-12",
    question: "Which region did we pick for the EU data residency rollout?",
    failedFor: ["mem0"],
    goldSourceIds: ["notion:page:eu-residency-decision"],
    retrievedSourceIds: {
      gbrain: ["notion:page:eu-residency-decision"],
      "llm-wiki": ["notion:page:eu-residency-decision"],
      mem0: ["notion:page:gdpr-overview", "slack:C07LEGAL:1712448000"],
    },
    reason: "Distilled memory kept the topic but lost the specific region decision.",
    sourceLabel: "Notion decision log",
  },
  {
    id: "case-18",
    question: "What was the agreed p95 latency target for the search rewrite?",
    failedFor: ["llm-wiki", "mem0"],
    goldSourceIds: ["slack:C02SEARCH:1714003200", "slack:C02SEARCH:1714006800"],
    retrievedSourceIds: {
      gbrain: ["slack:C02SEARCH:1714003200", "slack:C02SEARCH:1714006800"],
      "llm-wiki": ["slack:C02SEARCH:1713916800"],
      mem0: [],
    },
    reason: "The target was set in a threaded reply, not in the parent message.",
    sourceLabel: "Slack #search thread",
  },
  {
    id: "case-23",
    question: "Who owns the on-call rotation for the ingestion pipeline?",
    failedFor: ["llm-wiki"],
    goldSourceIds: ["notion:page:oncall-rotation"],
    retrievedSourceIds: {
      gbrain: ["notion:page:oncall-rotation"],
      "llm-wiki": ["notion:page:team-directory"],
      mem0: ["notion:page:oncall-rotation"],
    },
    reason: "Ownership table lives in a Notion database view that chunking flattened.",
    sourceLabel: "Notion on-call database",
  },
  {
    id: "case-29",
    question: "Did we ship the retry budget change before or after the incident review?",
    failedFor: ["llm-wiki", "mem0"],
    goldSourceIds: ["slack:C09INCIDENT:1716595200", "notion:page:retry-budget-change"],
    retrievedSourceIds: {
      gbrain: ["slack:C09INCIDENT:1716595200", "notion:page:retry-budget-change"],
      "llm-wiki": ["notion:page:retry-budget-change"],
      mem0: ["slack:C09INCIDENT:1716508800"],
    },
    reason: "Ordering requires both the incident timeline and the change record.",
    sourceLabel: "Slack #incident + Notion change log",
  },
];

export const DIAGNOSIS_RUN: DiagnosisRun = {
  id: "run-2026-08-24-a41f",
  label: "Slack + Notion baseline diagnosis",
  startedAt: "2026-08-24T09:12:00Z",
  status: "OK",
  verdict: "gbrain",
  corpusHash: "sha256:7c1e94a0b3d2",
  benchmarkHash: "sha256:2f88ad61c904",
  scoredCases: 30,
  candidates: CANDIDATES,
  failedCases: FAILED_CASES,
  recommendation:
    "Adopt GBrain as the baseline Brain, then tune retrieval to close the remaining thread-depth gap.",
  qualityLeadPoints: 10.8,
};

/** Candidates ordered by the selection policy: recall first. */
export function rankedCandidates(run: DiagnosisRun): CandidateScore[] {
  return [...run.candidates].sort((a, b) => b.quality.recall - a.quality.recall);
}

export function winner(run: DiagnosisRun): CandidateScore {
  const [top] = rankedCandidates(run);
  if (!top) {
    throw new Error("diagnosis run has no candidates");
  }
  return top;
}
