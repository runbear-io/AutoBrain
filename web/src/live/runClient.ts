/**
 * Client for the local AutoBrain run boundary (`autobrain.local_server`).
 *
 * Scope, stated plainly: this talks to a loopback developer fixture on
 * 127.0.0.1. It is not a hosted API and there is no deployed endpoint behind
 * it. The synthetic demo under `src/data` is unaffected by anything here.
 *
 * Parsing is strict and fail-closed. An unknown run status, an unsupported
 * projection schema version, or a success that carries no projection all throw
 * rather than being coerced into something renderable, because a wrong run
 * summary is worse than a visible error.
 */

import type { CandidateId, CostStatus, RunStatus, Verdict } from "../data/types";

/** Projection schema version this client understands. Must match Python's. */
export const PROJECTION_SCHEMA_VERSION = 1;

export const PROJECTION_PATH = "/api/v1/run";

/**
 * Default origin for the local runner fixture.
 *
 * Loopback only. Nothing is deployed at this address; it is served by
 * `autobrain serve` when an operator starts it on their own machine. The port
 * is kept in sync with DEFAULT_LOCAL_PORT in autobrain/local_server.py.
 */
export const DEFAULT_LOCAL_RUNNER_URL = "http://127.0.0.1:8765";

/**
 * Resolve the local runner origin.
 *
 * Overridable at build time with VITE_LOCAL_RUNNER_URL so an operator running
 * `autobrain serve --port N` can point the client at their chosen port without
 * editing source. Non-loopback overrides are ignored: this client only ever
 * talks to a fixture on the operator's own machine.
 */
export function localRunnerUrl(override?: string | undefined): string {
  const candidate = override ?? readConfiguredRunnerUrl();
  if (candidate === undefined || candidate.length === 0) {
    return DEFAULT_LOCAL_RUNNER_URL;
  }
  return isLoopbackUrl(candidate) ? stripTrailingSlash(candidate) : DEFAULT_LOCAL_RUNNER_URL;
}

function readConfiguredRunnerUrl(): string | undefined {
  // import.meta.env is absent under plain node/bun test runs.
  const env = (import.meta as { env?: Record<string, string | undefined> }).env;
  return env?.VITE_LOCAL_RUNNER_URL;
}

function stripTrailingSlash(value: string): string {
  return value.endsWith("/") ? value.slice(0, -1) : value;
}

/** True only for http(s) origins whose host is loopback. */
export function isLoopbackUrl(value: string): boolean {
  let parsed: URL;
  try {
    parsed = new URL(value);
  } catch {
    return false;
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    return false;
  }
  if (
    parsed.username.length > 0 ||
    parsed.password.length > 0 ||
    parsed.pathname !== "/" ||
    parsed.search.length > 0 ||
    parsed.hash.length > 0
  ) {
    return false;
  }
  const host = parsed.hostname.toLowerCase();
  return host === "localhost" || host === "127.0.0.1" || host === "[::1]" || host === "::1";
}

/** Terminal status of a run attempt, distinct from the engine `RunStatus`. */
export type RunOutcomeStatus = "SUCCEEDED" | "FAILED" | "CANCELLED";

const TERMINAL_STATUSES: readonly RunOutcomeStatus[] = ["SUCCEEDED", "FAILED", "CANCELLED"];

export interface CandidateProjection {
  candidate: CandidateId;
  status: RunStatus;
  quality_score: number;
  answer_success_rate: number;
  source_support_rate: number;
  contradiction_count: number;
  scored_cases: number;
  answered_cases: number;
  cost_status: CostStatus;
  total_cost_usd: number | null;
  query_p50_ms: number | null;
  query_p95_ms: number | null;
  operating_burden: number | null;
}

export interface RunProjection {
  schema_version: number;
  run_id: string;
  status: RunStatus;
  verdict: Verdict;
  rationale: string;
  corpus_hash: string;
  benchmark_hash: string;
  candidates: CandidateProjection[];
  warnings: string[];
}

export interface RunOutcome {
  status: RunOutcomeStatus;
  projection: RunProjection | null;
  error: string | null;
}

export function isTerminal(status: string): status is RunOutcomeStatus {
  return (TERMINAL_STATUSES as readonly string[]).includes(status);
}

/* --------------------------------------------------------------- validation */

/*
 * Every value that crosses the boundary is validated structurally before it is
 * given a type. The enum members and numeric bounds below mirror the pydantic
 * models in `autobrain/projection.py`; if that schema changes, this must move
 * with it and the schema version must be bumped.
 *
 * Each helper takes the containing record, the key to read, and the path used
 * in error messages, so a failure names its exact location such as
 * `candidates[1].quality_score`.
 */

const SHA256 = /^[0-9a-f]{64}$/;

const RUN_STATUSES: readonly string[] = [
  "OK",
  "ENV_UNAVAILABLE",
  "MISSING_PROVIDER",
  "MCP_AUTH_UNAVAILABLE",
  "CAPABILITY_UNAVAILABLE",
  "BUDGET_EXCEEDED",
  "INSUFFICIENT_BENCHMARK",
  "LEAKAGE_DETECTED",
  "FAILED",
  "CANCELLED",
  "NO_DECISION",
  "NO_RECOMMENDATION",
];

const CANDIDATE_IDS: readonly string[] = ["llm-wiki", "mem0", "gbrain"];

const VERDICTS: readonly string[] = [...CANDIDATE_IDS, "NO_DECISION", "NO_RECOMMENDATION"];

const COST_STATUSES: readonly string[] = ["COST_COMPLETE", "COST_INCOMPLETE", "COST_UNAVAILABLE"];

interface Bounds {
  min: number;
  max?: number;
  integer?: boolean;
}

class ProjectionError extends Error {
  constructor(path: string, problem: string) {
    super(`local runner returned an invalid projection: ${path} ${problem}`);
    this.name = "ProjectionError";
  }
}

function asRecord(value: unknown, label: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`local runner returned a malformed ${label}`);
  }
  return value as Record<string, unknown>;
}

function read(raw: Record<string, unknown>, key: string, path: string): unknown {
  if (!(key in raw)) {
    throw new ProjectionError(path, "is required but missing");
  }
  return raw[key];
}

function readString(raw: Record<string, unknown>, key: string, path: string): string {
  const value = read(raw, key, path);
  if (typeof value !== "string") {
    throw new ProjectionError(path, "must be a string");
  }
  return value;
}

function readNonEmptyString(raw: Record<string, unknown>, key: string, path: string): string {
  const value = readString(raw, key, path);
  if (value.length === 0) {
    throw new ProjectionError(path, "must be a non-empty string");
  }
  return value;
}

function readSha256(raw: Record<string, unknown>, key: string, path: string): string {
  const value = readNonEmptyString(raw, key, path);
  if (!SHA256.test(value)) {
    throw new ProjectionError(path, "must be a lowercase sha256 hex digest");
  }
  return value;
}

function readEnum<T extends string>(
  raw: Record<string, unknown>,
  key: string,
  path: string,
  allowed: readonly string[],
): T {
  const value = read(raw, key, path);
  if (typeof value !== "string" || !allowed.includes(value)) {
    throw new ProjectionError(path, `must be one of ${allowed.join(", ")}`);
  }
  return value as T;
}

function readNumber(
  raw: Record<string, unknown>,
  key: string,
  path: string,
  bounds: Bounds,
): number {
  const value = read(raw, key, path);
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new ProjectionError(path, "must be a finite number");
  }
  if (bounds.integer === true && !Number.isInteger(value)) {
    throw new ProjectionError(path, "must be an integer");
  }
  const { min, max } = bounds;
  if (value < min || (max !== undefined && value > max)) {
    throw new ProjectionError(
      path,
      max === undefined ? `must be >= ${min}` : `must be between ${min} and ${max}`,
    );
  }
  return value;
}

function readNullableNumber(
  raw: Record<string, unknown>,
  key: string,
  path: string,
  bounds: Bounds,
): number | null {
  return read(raw, key, path) === null ? null : readNumber(raw, key, path, bounds);
}

function readStringArray(raw: Record<string, unknown>, key: string, path: string): string[] {
  const value = read(raw, key, path);
  if (!Array.isArray(value) || value.some((item) => typeof item !== "string")) {
    throw new ProjectionError(path, "must be an array of strings");
  }
  return value as string[];
}

function parseCandidate(value: unknown, index: number): CandidateProjection {
  const base = `candidates[${index}]`;
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new ProjectionError(base, "must be an object");
  }
  const raw = value as Record<string, unknown>;
  const at = (key: string) => `${base}.${key}`;
  const rate: Bounds = { min: 0, max: 1 };
  const count: Bounds = { min: 0, integer: true };
  const nonNegative: Bounds = { min: 0 };

  return {
    candidate: readEnum(raw, "candidate", at("candidate"), CANDIDATE_IDS),
    status: readEnum(raw, "status", at("status"), RUN_STATUSES),
    quality_score: readNumber(raw, "quality_score", at("quality_score"), { min: 0, max: 100 }),
    answer_success_rate: readNumber(raw, "answer_success_rate", at("answer_success_rate"), rate),
    source_support_rate: readNumber(raw, "source_support_rate", at("source_support_rate"), rate),
    contradiction_count: readNumber(raw, "contradiction_count", at("contradiction_count"), count),
    scored_cases: readNumber(raw, "scored_cases", at("scored_cases"), count),
    answered_cases: readNumber(raw, "answered_cases", at("answered_cases"), count),
    cost_status: readEnum(raw, "cost_status", at("cost_status"), COST_STATUSES),
    total_cost_usd: readNullableNumber(raw, "total_cost_usd", at("total_cost_usd"), nonNegative),
    query_p50_ms: readNullableNumber(raw, "query_p50_ms", at("query_p50_ms"), nonNegative),
    query_p95_ms: readNullableNumber(raw, "query_p95_ms", at("query_p95_ms"), nonNegative),
    operating_burden: readNullableNumber(
      raw,
      "operating_burden",
      at("operating_burden"),
      nonNegative,
    ),
  };
}

function parseProjection(value: unknown): RunProjection {
  const raw = asRecord(value, "projection");
  if (raw.schema_version !== PROJECTION_SCHEMA_VERSION) {
    throw new Error(
      `unsupported projection schema version ${String(raw.schema_version)}; ` +
        `this client understands ${PROJECTION_SCHEMA_VERSION}`,
    );
  }
  const candidates = read(raw, "candidates", "candidates");
  if (!Array.isArray(candidates) || candidates.length === 0) {
    throw new ProjectionError("candidates", "must be a non-empty array");
  }
  return {
    schema_version: PROJECTION_SCHEMA_VERSION,
    run_id: readNonEmptyString(raw, "run_id", "run_id"),
    status: readEnum(raw, "status", "status", RUN_STATUSES),
    verdict: readEnum(raw, "verdict", "verdict", VERDICTS),
    // Rationale may legitimately be empty, so emptiness is allowed here.
    rationale: readString(raw, "rationale", "rationale"),
    corpus_hash: readSha256(raw, "corpus_hash", "corpus_hash"),
    benchmark_hash: readSha256(raw, "benchmark_hash", "benchmark_hash"),
    candidates: candidates.map(parseCandidate),
    warnings: readStringArray(raw, "warnings", "warnings"),
  };
}

/** Parse a raw boundary payload into a typed outcome, failing closed. */
export function parseRunOutcome(payload: unknown): RunOutcome {
  const raw = asRecord(payload, "payload");
  const status = raw.status;
  if (typeof status !== "string" || !isTerminal(status)) {
    throw new Error(`local runner returned an unknown run status: ${String(status)}`);
  }
  if (status === "SUCCEEDED") {
    return {
      status,
      projection: parseProjection(raw.projection),
      error: null,
    };
  }
  if (raw.projection != null) {
    throw new Error(`a ${status} run must not carry a projection`);
  }
  return {
    status,
    projection: null,
    error: typeof raw.error === "string" ? raw.error : null,
  };
}

export type Transport = (url: string) => Promise<unknown>;

const defaultTransport: Transport = async (url) => {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`local runner responded ${response.status}`);
  }
  return response.json();
};

/** Read the current run outcome from a local runner base URL. */
export async function fetchRunOutcome(
  baseUrl: string,
  transport: Transport = defaultTransport,
): Promise<RunOutcome> {
  return parseRunOutcome(await transport(`${baseUrl}${PROJECTION_PATH}`));
}

export interface RunSummary {
  runStatus: RunOutcomeStatus;
  /** Engine status, present only when a projection was returned. */
  engineStatus: RunStatus | null;
  headline: string;
  detail: string;
  /** Semantic tone used to pick a design-system status token. */
  tone: "success" | "warning" | "danger" | "neutral";
}

const ENGINE_HEADLINES: Partial<Record<RunStatus, string>> = {
  NO_DECISION: "No decision",
  NO_RECOMMENDATION: "No recommendation",
  BUDGET_EXCEEDED: "Budget exceeded",
  INSUFFICIENT_BENCHMARK: "Insufficient benchmark",
  LEAKAGE_DETECTED: "Leakage detected",
};

/**
 * Describe an outcome for display.
 *
 * The run status and the engine status are reported separately on purpose: a
 * run can succeed as an operation while the engine declines to recommend.
 */
export function summarize(outcome: RunOutcome): RunSummary {
  if (outcome.status === "CANCELLED") {
    return {
      runStatus: outcome.status,
      engineStatus: null,
      headline: "Run cancelled",
      detail: "The operator stopped this run. No result was produced.",
      tone: "neutral",
    };
  }
  if (outcome.status === "FAILED") {
    return {
      runStatus: outcome.status,
      engineStatus: null,
      headline: "Run failed",
      detail: outcome.error ?? "The local runner reported a failure with no detail.",
      tone: "danger",
    };
  }
  const projection = outcome.projection;
  if (projection === null) {
    throw new Error("a succeeded run must carry a projection");
  }
  const engineHeadline = ENGINE_HEADLINES[projection.status];
  if (engineHeadline !== undefined) {
    return {
      runStatus: outcome.status,
      engineStatus: projection.status,
      headline: engineHeadline,
      detail: projection.rationale,
      tone: "warning",
    };
  }
  return {
    runStatus: outcome.status,
    engineStatus: projection.status,
    headline: `Recommended: ${projection.verdict}`,
    detail: projection.rationale,
    tone: "success",
  };
}
