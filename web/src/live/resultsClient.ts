/**
 * Reads a finished experiment's result from the local job boundary.
 *
 * Scope: this talks to the same loopback boundary as `experimentClient`, on a
 * machine the operator started themselves. It is not a hosted API.
 *
 * Parsing is delegated to `parseRunOutcome`, which is already fail-closed, so a
 * malformed projection or a SUCCEEDED payload with no projection throws instead
 * of reaching the view. A wrong result is worse than a visible error.
 */

import { EXPERIMENT_API_PATH, ExperimentBoundaryError } from "./experimentClient";
import { isTerminal, parseRunOutcome, type RunOutcome } from "./runClient";

export interface ResultReply {
  status: number;
  payload: unknown;
}

export type ResultTransport = (url: string) => Promise<ResultReply>;

const defaultTransport: ResultTransport = async (url) => {
  const response = await fetch(url);
  const text = await response.text();
  return {
    status: response.status,
    payload: text.length === 0 ? null : (JSON.parse(text) as unknown),
  };
};

/** Raise the boundary's stable error for any non-200 reply. */
function assertOk(reply: ResultReply): void {
  if (reply.status === 200) return;
  const raw =
    typeof reply.payload === "object" && reply.payload !== null
      ? (reply.payload as Record<string, unknown>)
      : {};
  throw new ExperimentBoundaryError(
    typeof raw.error === "string" ? raw.error : "RUN_FAILED",
    typeof raw.detail === "string"
      ? raw.detail
      : `local boundary responded ${reply.status}`,
  );
}

function experimentRoot(baseUrl: string, experimentId: string): string {
  return `${baseUrl}${EXPERIMENT_API_PATH}/${encodeURIComponent(experimentId)}`;
}

/** Read one experiment's terminal result. */
export async function fetchExperimentResult(
  baseUrl: string,
  experimentId: string,
  transport: ResultTransport = defaultTransport,
): Promise<RunOutcome> {
  const reply = await transport(`${experimentRoot(baseUrl, experimentId)}/result`);
  assertOk(reply);
  return parseRunOutcome(reply.payload);
}

/** Read one experiment's current lifecycle status. */
async function fetchStatus(
  baseUrl: string,
  experimentId: string,
  transport: ResultTransport,
): Promise<string> {
  const reply = await transport(experimentRoot(baseUrl, experimentId));
  assertOk(reply);
  const raw =
    typeof reply.payload === "object" && reply.payload !== null
      ? (reply.payload as Record<string, unknown>)
      : {};
  const status = raw.status;
  if (typeof status !== "string") {
    throw new Error("local boundary returned a lifecycle without a status");
  }
  return status;
}

export interface AwaitOptions {
  transport?: ResultTransport;
  /** Maximum status reads before giving up. */
  attempts?: number;
  /** Pause between status reads. Zero in tests, which assert the contract. */
  delayMs?: number;
}

const DEFAULT_ATTEMPTS = 120;
const DEFAULT_DELAY_MS = 500;

function sleep(ms: number): Promise<void> {
  return ms <= 0 ? Promise.resolve() : new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Wait for the boundary's own terminal status, then read the result.
 *
 * The boundary exposes no push channel, so this reads its status route. The
 * loop is bounded and terminates on the boundary's reported state rather than
 * on elapsed time, so a slow machine waits longer instead of reporting a
 * falsely failed run.
 */
export async function awaitExperimentResult(
  baseUrl: string,
  experimentId: string,
  options: AwaitOptions = {},
): Promise<RunOutcome> {
  const transport = options.transport ?? defaultTransport;
  const attempts = options.attempts ?? DEFAULT_ATTEMPTS;
  const delayMs = options.delayMs ?? DEFAULT_DELAY_MS;

  for (let attempt = 0; attempt < attempts; attempt += 1) {
    const status = await fetchStatus(baseUrl, experimentId, transport);
    if (isTerminal(status)) {
      return fetchExperimentResult(baseUrl, experimentId, transport);
    }
    if (attempt + 1 < attempts) {
      await sleep(delayMs);
    }
  }
  throw new Error(
    `experiment ${experimentId} is still running after ${attempts} status reads; ` +
      "check the local runner before retrying",
  );
}
