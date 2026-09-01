/** Deterministic history entries for the demo's run navigator. */

import type { HistoryEntry } from "./types";

export const HISTORY: HistoryEntry[] = [
  {
    id: "run-2026-08-24-a41f",
    kind: "diagnosis",
    label: "Slack + Notion baseline diagnosis",
    startedAt: "2026-08-24T09:12:00Z",
    status: "OK",
    headline: "GBrain recommended - 82.4 recall, +10.8 over LLM Wiki",
  },
  {
    id: "opt-2026-08-24-b73c",
    kind: "optimization",
    label: "GBrain retrieval optimization",
    startedAt: "2026-08-24T10:41:00Z",
    status: "OK",
    headline: "Trial 9 best - 93.6 recall within latency and cost limits",
  },
  {
    id: "run-2026-08-17-9d02",
    kind: "diagnosis",
    label: "Slack-only smoke diagnosis",
    startedAt: "2026-08-17T14:03:00Z",
    status: "INSUFFICIENT_BENCHMARK",
    headline: "12 scored cases - below the 20-case eligibility gate",
  },
  {
    id: "opt-2026-08-11-4e55",
    kind: "optimization",
    label: "Cost-first retrieval sweep",
    startedAt: "2026-08-11T08:26:00Z",
    status: "BUDGET_EXCEEDED",
    headline: "Stopped at trial 7 - $25.00 hard budget guard reached",
  },
];
