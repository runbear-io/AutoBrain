/**
 * Experiment setup state and readiness rules.
 *
 * This is a pure module: every transition is a reducer step and readiness is a
 * derived projection, so the wizard's behavior is fully testable without a DOM
 * or a running boundary. It owns exactly one policy - what the operator must
 * choose before a Preview may be submitted - and expresses every unmet
 * requirement as guidance rather than a machine code.
 *
 * It deliberately does NOT evaluate anything. Scoring, provider access, and
 * lifecycle policy all remain in the Python engine behind the job boundary.
 */

import type {
  CandidateId,
  ExperimentIdentityContract,
  ExperimentReadinessState,
  ExperimentRequestContract,
  ReadinessCheckState,
} from "../data/experimentContracts";
import {
  SourceImportError,
  parseSourceImport,
  sourceCapability,
  type SourceCapabilityId,
  type SourceImportBatch,
  type SourceImportFormat,
} from "../data/sourceContracts";
import { sha256Hex } from "./sha256";

/** Evaluators offered by the wizard. Retrieval-only is the product default. */
export type EvaluatorId = "retrieval";

/** Subscriptions that can score a run. Mirrors the demo vocabulary. */
export type SubscriptionId = "codex" | "claude";

export interface SetupState {
  source: SourceCapabilityId | null;
  format: SourceImportFormat;
  payload: string;
  /** Records parsed from `payload`, or null when absent or invalid. */
  parsed: SourceImportBatch | null;
  /** Human-readable parse failure, or null when the payload is usable. */
  importError: string | null;
  candidates: CandidateId[];
  subscription: SubscriptionId | null;
  evaluator: EvaluatorId;
  evaluationMode: "retrieval_only";
}

export type SetupEvent =
  | { type: "source/select"; source: SourceCapabilityId }
  | { type: "source/format"; format: SourceImportFormat }
  | { type: "source/payload"; payload: string }
  | { type: "candidate/toggle"; candidate: CandidateId }
  | { type: "subscription/select"; subscription: SubscriptionId }
  | { type: "reset" };

export const SELECTABLE_CANDIDATES: readonly CandidateId[] = ["gbrain", "llm-wiki", "mem0"];

export function initialSetup(): SetupState {
  return {
    source: null,
    format: "JSON",
    payload: "",
    parsed: null,
    importError: null,
    candidates: [],
    subscription: null,
    evaluator: "retrieval",
    evaluationMode: "retrieval_only",
  };
}

/**
 * Re-parse the current payload against the current source and format.
 *
 * Parsing runs on every relevant change so the operator sees a rejection at the
 * moment they cause it, rather than at submission time. An empty payload is not
 * an error - it simply means no records have been provided yet.
 */
function withParsedImport(state: SetupState): SetupState {
  if (state.source === null || state.payload.trim().length === 0) {
    return { ...state, parsed: null, importError: null };
  }
  try {
    return {
      ...state,
      parsed: parseSourceImport(state.source, state.payload, state.format),
      importError: null,
    };
  } catch (cause) {
    const message =
      cause instanceof SourceImportError
        ? cause.message.replace(/^invalid source import: /, "")
        : String(cause);
    return { ...state, parsed: null, importError: message };
  }
}

export function reduceSetup(state: SetupState, event: SetupEvent): SetupState {
  switch (event.type) {
    case "source/select":
      return withParsedImport({ ...state, source: event.source });

    case "source/format":
      return withParsedImport({ ...state, format: event.format });

    case "source/payload":
      return withParsedImport({ ...state, payload: event.payload });

    case "candidate/toggle": {
      const selected = state.candidates.includes(event.candidate)
        ? state.candidates.filter((item) => item !== event.candidate)
        : [...state.candidates, event.candidate];
      // Keep a stable canonical order so the request is deterministic.
      return {
        ...state,
        candidates: SELECTABLE_CANDIDATES.filter((item) => selected.includes(item)),
      };
    }

    case "subscription/select":
      return { ...state, subscription: event.subscription };

    case "reset":
      return initialSetup();
  }
}

/** One unmet requirement, stated so a non-CLI user knows what to do. */
export interface SetupBlocker {
  code: string;
  title: string;
  guidance: string;
}

export interface SetupReadiness {
  state: ExperimentReadinessState;
  checks: {
    source: ReadinessCheckState;
    candidates: ReadinessCheckState;
    evaluator: ReadinessCheckState;
    subscription: ReadinessCheckState;
  };
  blockers: SetupBlocker[];
}

/**
 * Project the current configuration onto the shared readiness vocabulary.
 *
 * READY is only reachable with zero blockers, matching the Python contract, so
 * the wizard can never present a runnable state the boundary would refuse.
 */
export function setupReadiness(state: SetupState): SetupReadiness {
  const blockers: SetupBlocker[] = [];
  let source: ReadinessCheckState = "READY";

  if (state.source === null) {
    source = "NOT_CONFIGURED";
    blockers.push({
      code: "SOURCE_NOT_SELECTED",
      title: "No input source chosen",
      guidance: "Choose an official export or approved read-only source to build the frozen corpus.",
    });
  } else {
    const capability = sourceCapability(state.source);
    if (state.source === "fixture") {
      source = "BLOCKED";
      blockers.push({
        code: "SOURCE_NOT_SUPPORTED",
        title: "Local fixtures are not official input sources",
        guidance: "Choose an official export or approved read-only source for a production Preview.",
      });
    } else if (capability.readiness === "GATED") {
      source = "BLOCKED";
      blockers.push({
        code: "SOURCE_GATED",
        title: `${capability.label} is not available yet`,
        guidance:
          capability.remediation ??
          "This source is waiting on connector approval and a verified read-only transport.",
      });
    } else if (capability.readiness === "AUTH_REQUIRED") {
      source = "NOT_CONFIGURED";
      blockers.push({
        code: "SOURCE_AUTH_REQUIRED",
        title: `${capability.label} needs to be reconnected`,
        guidance:
          capability.remediation ?? "Reauthorize this source from the local runner, then retry.",
      });
    } else if (state.importError !== null) {
      source = "NOT_CONFIGURED";
      blockers.push({
        code: "SOURCE_IMPORT_INVALID",
        title: "The pasted export could not be read",
        guidance: `Fix the export and paste it again: ${state.importError}.`,
      });
    } else if (state.format === "JSONL" && state.parsed === null) {
      source = "NOT_CONFIGURED";
      blockers.push({
        code: "SOURCE_IMPORT_MISSING",
        title: "No records were pasted",
        guidance: "Paste one JSON record per line so the corpus can be frozen before the run.",
      });
    }
  }

  const candidates: ReadinessCheckState = state.candidates.length > 0 ? "READY" : "NOT_CONFIGURED";
  if (candidates !== "READY") {
    blockers.push({
      code: "NO_CANDIDATES",
      title: "No Brain candidates selected",
      guidance: "Select at least one Brain so the Preview has something to compare.",
    });
  }

  const subscription: ReadinessCheckState =
    state.subscription !== null ? "CONFIGURED" : "NOT_CONFIGURED";
  if (subscription !== "CONFIGURED") {
    blockers.push({
      code: "NO_SUBSCRIPTION",
      title: "No AI subscription selected",
      guidance: "Choose the existing plan the local runner should use to score this Preview.",
    });
  }

  return {
    state: blockers.length === 0 ? "READY" : "BLOCKED",
    checks: { source, candidates, evaluator: "READY", subscription },
    blockers,
  };
}

/**
 * Canonical fingerprint of the frozen corpus.
 *
 * Fixture runs hash the fixture identity; imports hash the normalized record
 * identity and text so a changed export produces a different experiment.
 */
function corpusIdentity(state: SetupState): { sha256: string; document_count: number } {
  if (state.parsed === null) {
    return { sha256: sha256Hex(`autobrain:fixture:${state.source ?? ""}`), document_count: 0 };
  }
  const canonical = state.parsed.records
    .map((record) => `${record.source_id}\u0000${record.title}\u0000${record.text}`)
    .join("\u0001");
  return { sha256: sha256Hex(canonical), document_count: state.parsed.records.length };
}

/** Protocol identifier recorded in the immutable experiment identity. */
export const RETRIEVAL_PROTOCOL = "retrieval-v1";

/**
 * Build the request submitted to the local job boundary.
 *
 * Only fingerprints and selections cross this boundary - never corpus text and
 * never a credential - so a submitted request is safe to persist and compare.
 */
export function buildExperimentRequest(
  state: SetupState,
  experimentId: string,
): ExperimentRequestContract {
  const readiness = setupReadiness(state);
  if (readiness.state !== "READY") {
    throw new Error(`NOT_READY: ${readiness.blockers.map((item) => item.code).join(", ")}`);
  }
  const corpus = corpusIdentity(state);
  const identity: ExperimentIdentityContract = {
    corpus,
    // The benchmark is derived from the frozen corpus and protocol until the
    // engine reports its own generated benchmark for this experiment.
    benchmark_sha256: sha256Hex(`${RETRIEVAL_PROTOCOL}:${corpus.sha256}`),
    protocol: RETRIEVAL_PROTOCOL,
    evaluator: state.evaluator,
    provider: null,
    model: null,
    configuration_hash: null,
    code_version: null,
  };
  return {
    schema_version: 1,
    experiment_id: experimentId,
    identity,
    candidates: [...state.candidates],
    evaluation_mode: state.evaluationMode,
  };
}
