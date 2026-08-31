/**
 * Client for the local experiment job boundary (`autobrain.experiment_job`).
 *
 * Scope, stated plainly: this talks to a loopback job boundary on 127.0.0.1
 * that the operator starts on their own machine. It is not a hosted API.
 *
 * The Python boundary owns lifecycle policy. This client only sequences the
 * documented create → validate → start routes and fails closed: an unknown
 * status or a malformed lifecycle throws rather than being rendered, because a
 * fabricated "RUNNING" is worse than a visible error.
 */

import type {
  ExperimentLifecycleStatus,
  ExperimentRequestContract,
  StableExperimentErrorCode,
} from "../data/experimentContracts";

/** Route prefix served by `ExperimentJobServer`. Must match Python's. */
export const EXPERIMENT_API_PATH = "/api/v1/experiments";

const LIFECYCLE_STATUSES: readonly string[] = [
  "CREATED",
  "VALIDATING",
  "READY",
  "RUNNING",
  "SUCCEEDED",
  "FAILED",
  "CANCELLED",
];

export interface ExperimentLifecycleView {
  experiment_id: string;
  status: ExperimentLifecycleStatus;
  updated_at: string | null;
}

/** Error carrying the boundary's stable machine code alongside its detail. */
export class ExperimentBoundaryError extends Error {
  readonly code: string;
  readonly detail: string;

  constructor(code: string, detail: string) {
    super(`${code}: ${detail}`);
    this.name = "ExperimentBoundaryError";
    this.code = code;
    this.detail = detail;
  }
}

/** Parse a lifecycle payload, refusing any status outside the contract. */
export function parseLifecycle(payload: unknown): ExperimentLifecycleView {
  if (typeof payload !== "object" || payload === null || Array.isArray(payload)) {
    throw new Error("local boundary returned a malformed lifecycle");
  }
  const raw = payload as Record<string, unknown>;
  const experimentId = raw.experiment_id;
  if (typeof experimentId !== "string" || experimentId.length === 0) {
    throw new Error("local boundary returned a lifecycle without an experiment_id");
  }
  const status = raw.status;
  if (typeof status !== "string" || !LIFECYCLE_STATUSES.includes(status)) {
    throw new Error(`local boundary returned an unknown lifecycle status: ${String(status)}`);
  }
  const updatedAt = raw.updated_at;
  return {
    experiment_id: experimentId,
    status: status as ExperimentLifecycleStatus,
    updated_at: typeof updatedAt === "string" ? updatedAt : null,
  };
}

export interface TransportReply {
  status: number;
  payload: unknown;
}

export type ExperimentTransport = (
  url: string,
  init: { method: string; body?: string },
) => Promise<TransportReply>;

const defaultTransport: ExperimentTransport = async (url, init) => {
  const response = await fetch(url, {
    method: init.method,
    ...(init.body === undefined
      ? {}
      : { headers: { "Content-Type": "application/json" }, body: init.body }),
  });
  const text = await response.text();
  return {
    status: response.status,
    payload: text.length === 0 ? null : (JSON.parse(text) as unknown),
  };
};

/** Raise the boundary's stable error for any non-success reply. */
function assertAccepted(reply: TransportReply, accepted: readonly number[]): void {
  if (accepted.includes(reply.status)) return;
  const raw =
    typeof reply.payload === "object" && reply.payload !== null
      ? (reply.payload as Record<string, unknown>)
      : {};
  const code = typeof raw.error === "string" ? raw.error : "RUN_FAILED";
  const detail =
    typeof raw.detail === "string" ? raw.detail : `local boundary responded ${reply.status}`;
  throw new ExperimentBoundaryError(code, detail);
}

/**
 * Create, validate, and start one experiment.
 *
 * Validation is the readiness authority: if it does not reach READY the run is
 * never started, and the terminal lifecycle is returned as-is so the caller can
 * show why.
 */
export async function submitExperiment(
  baseUrl: string,
  request: ExperimentRequestContract,
  transport: ExperimentTransport = defaultTransport,
): Promise<ExperimentLifecycleView> {
  const root = `${baseUrl}${EXPERIMENT_API_PATH}`;
  const created = await transport(root, { method: "POST", body: JSON.stringify(request) });
  assertAccepted(created, [201]);
  const id = parseLifecycle(created.payload).experiment_id;

  const validated = await transport(`${root}/${encodeURIComponent(id)}/validate`, {
    method: "POST",
  });
  assertAccepted(validated, [200]);
  const readiness = parseLifecycle(validated.payload);
  if (readiness.status !== "READY") {
    return readiness;
  }

  const started = await transport(`${root}/${encodeURIComponent(id)}/start`, { method: "POST" });
  assertAccepted(started, [202]);
  return parseLifecycle(started.payload);
}

export interface ErrorGuidance {
  title: string;
  guidance: string;
}

/*
 * Human-readable translations of the stable error codes.
 *
 * The detail from the boundary is already redacted on the Python side, so it is
 * safe to show; the caller renders it alongside this guidance so the operator
 * sees both what the engine said and what to do next.
 */
const ERROR_GUIDANCE: Record<StableExperimentErrorCode, ErrorGuidance> = {
  INVALID_REQUEST: {
    title: "The setup was rejected",
    guidance: "Review the source, candidates, and evaluator choices, then submit again.",
  },
  INVALID_TRANSITION: {
    title: "This experiment already moved on",
    guidance: "Start a new experiment instead of resubmitting this one.",
  },
  NOT_READY: {
    title: "The experiment is not ready to run",
    guidance: "Resolve the readiness warnings above, then submit the Preview again.",
  },
  NOT_FOUND: {
    title: "The experiment is no longer available",
    guidance: "The local runner restarted or dropped it. Submit the Preview again.",
  },
  RUN_FAILED: {
    title: "The run could not be started",
    guidance: "Check the local runner output for the failing stage, then retry the Preview.",
  },
  CANCELLED: {
    title: "The experiment was cancelled",
    guidance: "Nothing was scored. Submit the Preview again when you are ready.",
  },
};

const UNKNOWN_GUIDANCE: ErrorGuidance = {
  title: "The local runner refused this request",
  guidance: "Check that the local runner is up to date, then submit the Preview again.",
};

/**
 * Translate a boundary rejection into guidance a non-CLI user can act on.
 *
 * Unmapped codes still return guidance rather than a blank surface, so a newer
 * engine can never leave the operator staring at an unexplained failure.
 */
export function describeExperimentError(code: string, detail: string): ErrorGuidance {
  const known = (ERROR_GUIDANCE as Record<string, ErrorGuidance | undefined>)[code];
  const base = known ?? UNKNOWN_GUIDANCE;
  return {
    title: base.title,
    guidance: detail.length === 0 ? base.guidance : `${base.guidance} The runner reported: ${detail}`,
  };
}
