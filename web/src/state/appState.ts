/**
 * Application state machine.
 *
 * All navigation and demo progression is expressed as pure reducer transitions
 * so tests can drive the entire journey by dispatching events, with no timers
 * and no wall-clock dependency. Runtime animation is layered on top by the UI,
 * which simply dispatches `diagnosis/tick` and `optimization/tick`.
 */

import { DIAGNOSIS_STAGES, SOURCES, SUBSCRIPTIONS, TOTAL_STAGE_WEIGHT } from "../data/diagnosis";
import { DEFAULT_TUNING, OPTIMIZATION_STAGES, TRIALS } from "../data/optimization";
import type {
  ConnectionState,
  KnowledgeSource,
  SourceId,
  SourceStage,
  Subscription,
  SubscriptionId,
  TrialState,
  TuningConfig,
} from "../data/types";

export type Route =
  | "welcome"
  | "subscriptions"
  | "sources"
  | "quick-start"
  | "diagnosis-run"
  | "diagnosis-report"
  | "tuning-setup"
  | "optimization-run"
  | "optimization-report"
  | "history"
  /**
   * Local runner fixture. Deliberately excluded from JOURNEY so it is
   * reachable from the sidebar without altering the synthetic demo's
   * next/back progression.
   */
  | "local-fixture"
  /**
   * Experiment setup wizard over the local job boundary. Excluded from
   * JOURNEY for the same reason as the local fixture: it is a real local
   * surface, not a step in the synthetic demo.
   */
  | "experiment-setup";

/** Ordered primary journey used by the stepper and by next/back navigation. */
export const JOURNEY: Route[] = [
  "welcome",
  "subscriptions",
  "sources",
  "quick-start",
  "diagnosis-run",
  "diagnosis-report",
  "tuning-setup",
  "optimization-run",
  "optimization-report",
];

export type AdoptionDecision = "undecided" | "adopted" | "confirmed" | "kept-default";

export interface AppState {
  route: Route;
  subscriptions: Subscription[];
  sources: KnowledgeSource[];
  /** Accumulated diagnosis progress in stage-weight units. */
  diagnosisProgress: number;
  diagnosisComplete: boolean;
  tuning: TuningConfig;
  /** Number of trials that have finished running. */
  completedTrials: number;
  optimizationComplete: boolean;
  trialFilter: TrialFilter;
  selectedTrial: number | null;
  selectedCaseId: string | null;
  adoption: AdoptionDecision;
  selectedHistoryId: string | null;
}

export type TrialFilter = "all" | "feasible" | "violations" | "pruned";

export type AppEvent =
  | { type: "navigate"; route: Route }
  | { type: "journey/next" }
  | { type: "journey/back" }
  | { type: "subscription/connect"; id: SubscriptionId }
  | { type: "subscription/disconnect"; id: SubscriptionId }
  | { type: "source/advance"; id: SourceId }
  | { type: "source/reconnect"; id: SourceId }
  | { type: "diagnosis/start" }
  | { type: "diagnosis/tick"; amount: number }
  | { type: "diagnosis/finish" }
  | { type: "tuning/update"; patch: Partial<TuningConfig> }
  | { type: "tuning/reset" }
  | { type: "optimization/start" }
  | { type: "optimization/tick" }
  | { type: "optimization/finish" }
  | { type: "optimization/filter"; filter: TrialFilter }
  | { type: "optimization/select-trial"; index: number | null }
  | { type: "diagnosis/select-case"; caseId: string | null }
  | { type: "adoption/decide"; decision: AdoptionDecision }
  | { type: "history/select"; id: string | null };

export const TOTAL_TRIALS = TRIALS.length;

export function initialState(): AppState {
  return {
    route: "welcome",
    subscriptions: SUBSCRIPTIONS.map((s) => ({ ...s })),
    sources: SOURCES.map((s) => ({ ...s })),
    diagnosisProgress: 0,
    diagnosisComplete: false,
    tuning: { ...DEFAULT_TUNING },
    completedTrials: 0,
    optimizationComplete: false,
    trialFilter: "all",
    selectedTrial: null,
    selectedCaseId: null,
    adoption: "undecided",
    selectedHistoryId: null,
  };
}

/** Next stage in the source lifecycle. IMPORTED is terminal. */
function nextSourceStage(stage: SourceStage): SourceStage {
  switch (stage) {
    case "DISCONNECTED":
      return "CONNECTED";
    case "CONNECTED":
      return "CONFIGURED";
    case "CONFIGURED":
      return "IMPORTED";
    case "IMPORTED":
      return "IMPORTED";
  }
}

function stageConnectionState(stage: SourceStage): ConnectionState {
  return stage === "DISCONNECTED" ? "DISCONNECTED" : "CONNECTED";
}

function stageDetail(source: KnowledgeSource, stage: SourceStage): string {
  const noun = source.id === "slack" ? "workspace export" : "read-only MCP snapshot";
  switch (stage) {
    case "DISCONNECTED":
      return `Not connected. Authorize to read the ${noun}.`;
    case "CONNECTED":
      return `Authorized. Choose which ${noun} content to include.`;
    case "CONFIGURED":
      return `Scope selected. Import to freeze the ${noun} into the corpus.`;
    case "IMPORTED":
      return `Imported and frozen. Verified by SHA-256 before every run.`;
  }
}

function step(route: Route, delta: number): Route {
  const index = JOURNEY.indexOf(route);
  if (index === -1) return route;
  const next = index + delta;
  if (next < 0 || next >= JOURNEY.length) return route;
  return JOURNEY[next] as Route;
}

export function reduce(state: AppState, event: AppEvent): AppState {
  switch (event.type) {
    case "navigate":
      return { ...state, route: event.route };

    case "journey/next":
      return { ...state, route: step(state.route, 1) };

    case "journey/back":
      return { ...state, route: step(state.route, -1) };

    case "subscription/connect":
      return {
        ...state,
        subscriptions: state.subscriptions.map((s) =>
          s.id === event.id
            ? {
                ...s,
                state: "CONNECTED" as ConnectionState,
                account: s.account ?? `demo@runbear.io`,
              }
            : s,
        ),
      };

    case "subscription/disconnect":
      return {
        ...state,
        subscriptions: state.subscriptions.map((s) =>
          s.id === event.id ? { ...s, state: "DISCONNECTED" as ConnectionState } : s,
        ),
      };

    case "source/advance":
      return {
        ...state,
        sources: state.sources.map((s) => {
          if (s.id !== event.id) return s;
          const stage = nextSourceStage(s.stage);
          return {
            ...s,
            stage,
            state: stageConnectionState(stage),
            detail: stageDetail(s, stage),
          };
        }),
      };

    case "source/reconnect":
      return {
        ...state,
        sources: state.sources.map((s) =>
          s.id === event.id
            ? {
                ...s,
                state: "CONNECTED" as ConnectionState,
                stage: "IMPORTED" as SourceStage,
                detail: stageDetail(s, "IMPORTED"),
              }
            : s,
        ),
      };

    case "diagnosis/start":
      return {
        ...state,
        route: "diagnosis-run",
        diagnosisProgress: 0,
        diagnosisComplete: false,
      };

    case "diagnosis/tick": {
      const progress = Math.min(TOTAL_STAGE_WEIGHT, state.diagnosisProgress + event.amount);
      return {
        ...state,
        diagnosisProgress: progress,
        diagnosisComplete: progress >= TOTAL_STAGE_WEIGHT,
      };
    }

    case "diagnosis/finish":
      return { ...state, diagnosisProgress: TOTAL_STAGE_WEIGHT, diagnosisComplete: true };

    case "tuning/update":
      return { ...state, tuning: { ...state.tuning, ...event.patch } };

    case "tuning/reset":
      return { ...state, tuning: { ...DEFAULT_TUNING } };

    case "optimization/start":
      return {
        ...state,
        route: "optimization-run",
        completedTrials: 0,
        optimizationComplete: false,
        selectedTrial: null,
      };

    case "optimization/tick": {
      const completed = Math.min(TOTAL_TRIALS, state.completedTrials + 1);
      return {
        ...state,
        completedTrials: completed,
        optimizationComplete: completed >= TOTAL_TRIALS,
      };
    }

    case "optimization/finish":
      return { ...state, completedTrials: TOTAL_TRIALS, optimizationComplete: true };

    case "optimization/filter":
      return { ...state, trialFilter: event.filter };

    case "optimization/select-trial":
      return { ...state, selectedTrial: event.index };

    case "diagnosis/select-case":
      return { ...state, selectedCaseId: event.caseId };

    case "adoption/decide":
      return { ...state, adoption: event.decision };

    case "history/select":
      return { ...state, selectedHistoryId: event.id };
  }
}

/* --------------------------------------------------------------- selectors */

export function diagnosisPercent(state: AppState): number {
  return Math.round((state.diagnosisProgress / TOTAL_STAGE_WEIGHT) * 100);
}

/** Per-stage completion derived from accumulated progress weight. */
export function stageStates(state: AppState): { id: string; status: "pending" | "active" | "complete" }[] {
  let consumed = 0;
  return DIAGNOSIS_STAGES.map((stage) => {
    const start = consumed;
    consumed += stage.weight;
    if (state.diagnosisProgress >= consumed) return { id: stage.id, status: "complete" as const };
    if (state.diagnosisProgress > start) return { id: stage.id, status: "active" as const };
    return { id: stage.id, status: "pending" as const };
  });
}

export function optimizationPercent(state: AppState): number {
  return Math.round((state.completedTrials / TOTAL_TRIALS) * 100);
}

/** Trial rows with a runtime state derived from how many trials have finished. */
export function visibleTrials(state: AppState) {
  return TRIALS.map((t) => {
    let runtimeState: TrialState;
    if (t.index <= state.completedTrials) {
      runtimeState = t.state === "pruned" ? "pruned" : "complete";
    } else if (t.index === state.completedTrials + 1) {
      runtimeState = "running";
    } else {
      runtimeState = "queued";
    }
    return { ...t, state: runtimeState };
  });
}

export function filteredTrials(state: AppState) {
  const rows = visibleTrials(state);
  switch (state.trialFilter) {
    case "feasible":
      return rows.filter((t) => t.state === "complete" && !t.violatesConstraint);
    case "violations":
      return rows.filter((t) => t.violatesConstraint);
    case "pruned":
      return rows.filter((t) => t.state === "pruned");
    case "all":
      return rows;
  }
}

/** Best trial among those that have actually finished running. */
export function currentBestTrial(state: AppState) {
  const done = visibleTrials(state).filter((t) => t.state === "complete" && !t.violatesConstraint);
  return done.reduce<(typeof done)[number] | null>((best, current) => {
    if (!best) return current;
    if (current.quality > best.quality) return current;
    if (current.quality === best.quality && current.latencyP95Ms < best.latencyP95Ms) return current;
    return best;
  }, null);
}

export function optimizationStageStates(state: AppState) {
  let consumed = 0;
  return OPTIMIZATION_STAGES.map((stage) => {
    const start = consumed;
    consumed += stage.trials;
    if (state.completedTrials >= consumed) return { id: stage.id, status: "complete" as const };
    if (state.completedTrials > start) return { id: stage.id, status: "active" as const };
    return { id: stage.id, status: "pending" as const };
  });
}

/** A run is launchable only with a connected evaluator and at least one imported source. */
export function canRunDiagnosis(state: AppState): boolean {
  const hasEvaluator = state.subscriptions.some((s) => s.state === "CONNECTED");
  const hasSource = state.sources.some((s) => s.stage === "IMPORTED");
  return hasEvaluator && hasSource;
}
