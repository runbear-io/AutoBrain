/**
 * The retrieval results route as a user actually reads it.
 *
 * These tests mount the real `App`, navigate via the sidebar, and read a result
 * from a stubbed local boundary, so they prove the completed, failed, and
 * cancelled surfaces each render what a non-CLI user needs - and that none of
 * them claims recommendation-grade evidence.
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import App from "../App";

let container: HTMLDivElement;
let root: Root;
const realFetch = globalThis.fetch;

const CORPUS = "a".repeat(64);
const BENCHMARK = "b".repeat(64);

function candidate(overrides: Record<string, unknown> = {}) {
  return {
    candidate: "gbrain",
    status: "OK",
    quality_score: 81.5,
    answer_success_rate: 0.9,
    source_support_rate: 0.8,
    contradiction_count: 0,
    scored_cases: 20,
    answered_cases: 18,
    cost_status: "COST_COMPLETE",
    total_cost_usd: 1.5,
    query_p50_ms: 100,
    query_p95_ms: 220,
    operating_burden: 2,
    ...overrides,
  };
}

function projection(overrides: Record<string, unknown> = {}) {
  return {
    schema_version: 1,
    run_id: "run-1",
    status: "OK",
    verdict: "gbrain",
    rationale: "GBrain led grounded retrieval recall on the frozen corpus.",
    corpus_hash: CORPUS,
    benchmark_hash: BENCHMARK,
    candidates: [
      candidate(),
      candidate({ candidate: "mem0", scored_cases: 20, answered_cases: 11, query_p95_ms: 410 }),
    ],
    warnings: [],
    ...overrides,
  };
}

function json(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

/** Serve one terminal lifecycle plus its result from the stubbed boundary. */
function stubBoundary(lifecycleStatus: string, result: unknown) {
  globalThis.fetch = (async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.endsWith("/result")) return json(result);
    return json({ experiment_id: "exp-1", status: lifecycleStatus, updated_at: null });
  }) as typeof fetch;
}

function mount() {
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
  act(() => {
    root.render(<App />);
  });
}

function navButton(label: string): HTMLElement {
  const match = [...container.querySelectorAll<HTMLElement>(".sidebar nav button")].find(
    (node) => node.querySelector("span")?.textContent?.trim().toLowerCase() === label.toLowerCase(),
  );
  if (match === undefined) throw new Error(`no sidebar entry labeled "${label}"`);
  return match;
}

function testId(id: string): HTMLElement {
  const match = container.querySelector<HTMLElement>(`[data-testid="${id}"]`);
  if (match === null) throw new Error(`no element with data-testid="${id}"`);
  return match;
}

function maybeTestId(id: string): HTMLElement | null {
  return container.querySelector<HTMLElement>(`[data-testid="${id}"]`);
}

function click(element: HTMLElement) {
  act(() => {
    element.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
}

async function clickAsync(element: HTMLElement) {
  await act(async () => {
    element.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
}

function openResults() {
  click(navButton("Results"));
}

/**
 * Submit a Preview from the wizard so the results route has an experiment to
 * read. This mirrors the real path: a user configures a run, then reads it.
 */
async function submitPreview() {
  // The caller has already stubbed the boundary for the *result* read; keep it
  // and restore it after the submission so the read still sees that stub.
  const resultStub = globalThis.fetch;
  globalThis.fetch = (async (input: RequestInfo | URL) => {
    const url = String(input);
    const status = url.endsWith("/start")
      ? "RUNNING"
      : url.endsWith("/validate")
        ? "READY"
        : "CREATED";
    return json({ experiment_id: "exp-1", status, updated_at: null }, url.endsWith("/start") ? 202 : url.endsWith("/validate") ? 200 : 201);
  }) as typeof fetch;

  click(navButton("New experiment"));
  click(testId("source-option-slack-export"));
  click(testId("candidate-option-gbrain"));
  click(testId("subscription-option-codex"));
  await clickAsync(testId("submit-preview"));

  globalThis.fetch = resultStub;
}

/** Open the route and read the currently stubbed result. */
async function loadResults() {
  await submitPreview();
  openResults();
  await clickAsync(testId("load-results"));
}

beforeEach(mount);

afterEach(() => {
  act(() => {
    root.unmount();
  });
  container.remove();
  globalThis.fetch = realFetch;
});

describe("retrieval results reachability", () => {
  test("the sidebar exposes a results entry", () => {
    expect(navButton("Results")).toBeDefined();
  });

  test("the route states it reads the local boundary, not a hosted service", () => {
    openResults();
    const scope = container.querySelector(".results__scope");
    expect(scope?.textContent?.toLowerCase()).toContain("local");
    expect(scope?.textContent?.toLowerCase()).toContain("not hosted");
  });

  test("nothing is claimed before a result has been read", () => {
    openResults();
    expect(maybeTestId("retrieval-table")).toBeNull();
    expect(container.textContent).not.toContain("Engine decision");
  });

  test("without a submitted Preview the read is disabled and says why", () => {
    openResults();
    expect((testId("load-results") as HTMLButtonElement).disabled).toBe(true);
    expect(testId("results-empty").textContent?.toLowerCase()).toContain("wizard");
  });

  test("the synthetic demo remains reachable and unchanged", () => {
    openResults();
    click(navButton("Diagnosis"));
    expect(container.textContent).toContain("GBrain is the strongest starting point");
    expect(container.querySelector(".results")).toBeNull();
  });
});

describe("a completed run", () => {
  test("shows a per-Brain retrieval row for every candidate", async () => {
    stubBoundary("SUCCEEDED", { status: "SUCCEEDED", run_id: "run-1", projection: projection() });
    await loadResults();

    const rows = testId("retrieval-table").querySelectorAll("tbody tr");
    expect(rows).toHaveLength(2);
    expect(testId("retrieval-row-gbrain").textContent).toContain("gbrain");
    expect(testId("retrieval-row-mem0").textContent).toContain("mem0");
  });

  test("reports recall, missing evidence, noise, latency, and cost per Brain", async () => {
    stubBoundary("SUCCEEDED", { status: "SUCCEEDED", run_id: "run-1", projection: projection() });
    await loadResults();

    const row = testId("retrieval-row-gbrain").textContent ?? "";
    // 18 of 20 grounded cases, so 2 are missing evidence.
    expect(row).toContain("90.0%");
    expect(row).toContain("2");
    expect(row).toContain("220");
    expect(row).toContain("1.50");
  });

  test("marks the leading Brain by measured recall, not by claim", async () => {
    stubBoundary("SUCCEEDED", { status: "SUCCEEDED", run_id: "run-1", projection: projection() });
    await loadResults();

    expect(testId("retrieval-row-gbrain").getAttribute("data-leader")).toBe("true");
    expect(testId("retrieval-row-mem0").getAttribute("data-leader")).toBe("false");
  });

  test("labels confidence and never claims recommendation-grade evidence", async () => {
    stubBoundary("SUCCEEDED", { status: "SUCCEEDED", run_id: "run-1", projection: projection() });
    await loadResults();

    const confidence = testId("evidence-confidence");
    expect(confidence.getAttribute("data-level")).toBe("ENGINE_DECISION");
    expect(confidence.getAttribute("data-recommendation-grade")).toBe("false");
    expect(confidence.textContent?.toLowerCase()).toContain("not a production recommendation");
  });

  test("shows the frozen corpus and the engine benchmark as provenance", async () => {
    stubBoundary("SUCCEEDED", { status: "SUCCEEDED", run_id: "run-1", projection: projection() });
    await loadResults();

    const provenance = testId("result-provenance").textContent ?? "";
    expect(provenance).toContain(CORPUS.slice(0, 12));
    expect(provenance).toContain(BENCHMARK.slice(0, 12));
    expect(provenance).toContain("run-1");
  });

  test("the engine benchmark replaces the placeholder and says so", async () => {
    stubBoundary("SUCCEEDED", { status: "SUCCEEDED", run_id: "run-1", projection: projection() });
    await loadResults();

    const benchmark = testId("benchmark-identity");
    expect(benchmark.getAttribute("data-source")).toBe("engine");
    expect(benchmark.getAttribute("data-placeholder")).toBe("false");
    expect(benchmark.textContent?.toLowerCase()).not.toContain("placeholder");
  });

  test("a Brain the engine could not score is shown as not comparable, with a reason", async () => {
    stubBoundary("SUCCEEDED", {
      status: "SUCCEEDED",
      run_id: "run-1",
      projection: projection({
        candidates: [
          candidate(),
          candidate({
            candidate: "mem0",
            status: "ENV_UNAVAILABLE",
            scored_cases: 0,
            answered_cases: 0,
          }),
        ],
      }),
    });
    await loadResults();

    const row = testId("retrieval-row-mem0");
    expect(row.getAttribute("data-comparable")).toBe("false");
    expect(row.textContent?.toLowerCase()).toContain("not scored");
  });

  test("a NO_RECOMMENDATION verdict is never rendered as a winner", async () => {
    stubBoundary("SUCCEEDED", {
      status: "SUCCEEDED",
      run_id: "run-1",
      projection: projection({ verdict: "NO_RECOMMENDATION", status: "NO_RECOMMENDATION" }),
    });
    await loadResults();

    const confidence = testId("evidence-confidence");
    expect(confidence.getAttribute("data-level")).toBe("NO_RECOMMENDATION");
    expect(confidence.textContent?.toLowerCase()).toContain("did not name");
    expect(container.textContent?.toLowerCase()).not.toContain("recommended:");
  });

  test("engine warnings are shown rather than swallowed", async () => {
    stubBoundary("SUCCEEDED", {
      status: "SUCCEEDED",
      run_id: "run-1",
      projection: projection({ warnings: ["cost telemetry was incomplete for mem0"] }),
    });
    await loadResults();

    expect(testId("evidence-confidence").textContent).toContain(
      "cost telemetry was incomplete for mem0",
    );
  });
});

describe("before and after comparison", () => {
  test("comparing a run against itself reports every Brain as unchanged", async () => {
    stubBoundary("SUCCEEDED", { status: "SUCCEEDED", run_id: "run-1", projection: projection() });
    await loadResults();
    await clickAsync(testId("compare-with-current"));

    const comparison = testId("comparison-table");
    expect(comparison.querySelectorAll("tbody tr")).toHaveLength(2);
    expect(testId("comparison-row-gbrain").getAttribute("data-direction")).toBe("unchanged");
  });

  test("an improved Brain reads as improved with a signed delta", async () => {
    stubBoundary("SUCCEEDED", {
      status: "SUCCEEDED",
      run_id: "run-1",
      projection: projection({ candidates: [candidate({ answered_cases: 12 })] }),
    });
    await loadResults();

    // Re-read with a better second run, then compare against the first.
    stubBoundary("SUCCEEDED", {
      status: "SUCCEEDED",
      run_id: "run-2",
      projection: projection({
        run_id: "run-2",
        candidates: [candidate({ answered_cases: 18 })],
      }),
    });
    await clickAsync(testId("load-results"));
    await clickAsync(testId("compare-with-previous"));

    const row = testId("comparison-row-gbrain");
    expect(row.getAttribute("data-direction")).toBe("improved");
    expect(row.textContent).toContain("+30.0");
  });

  test("runs over a different benchmark are refused instead of compared", async () => {
    stubBoundary("SUCCEEDED", { status: "SUCCEEDED", run_id: "run-1", projection: projection() });
    await loadResults();

    stubBoundary("SUCCEEDED", {
      status: "SUCCEEDED",
      run_id: "run-2",
      projection: projection({ run_id: "run-2", benchmark_hash: "c".repeat(64) }),
    });
    await clickAsync(testId("load-results"));
    await clickAsync(testId("compare-with-previous"));

    expect(maybeTestId("comparison-table")).toBeNull();
    expect(testId("comparison-blocker").textContent?.toLowerCase()).toContain("benchmark");
  });
});

describe("failed and cancelled recovery", () => {
  test("a failed run shows the engine detail and actionable recovery steps", async () => {
    stubBoundary("FAILED", {
      status: "FAILED",
      projection: null,
      error: "candidate transport was unavailable",
    });
    await loadResults();

    const recovery = testId("recovery-notice");
    expect(recovery.getAttribute("data-tone")).toBe("danger");
    expect(recovery.textContent).toContain("candidate transport was unavailable");
    expect(testId("recovery-action-rerun")).toBeDefined();
    expect(maybeTestId("retrieval-table")).toBeNull();
  });

  test("a cancelled run says nothing was scored and offers a rerun", async () => {
    stubBoundary("CANCELLED", { status: "CANCELLED", projection: null });
    await loadResults();

    const recovery = testId("recovery-notice");
    expect(recovery.getAttribute("data-tone")).toBe("neutral");
    expect(recovery.textContent?.toLowerCase()).toContain("no brain was scored");
    expect(testId("recovery-action-rerun")).toBeDefined();
  });

  test("a failed run never claims a recommendation or shows a leader", async () => {
    stubBoundary("FAILED", { status: "FAILED", projection: null, error: "boom" });
    await loadResults();

    expect(maybeTestId("evidence-confidence")).toBeNull();
    expect(container.textContent?.toLowerCase()).not.toContain("engine decision");
  });

  test("a boundary rejection is shown as guidance, not a raw code", async () => {
    globalThis.fetch = (async () =>
      json({ error: "NOT_FOUND", detail: "exp-1" }, 404)) as typeof fetch;
    await loadResults();

    const error = testId("results-error");
    expect(error.textContent?.length).toBeGreaterThan(40);
    expect(error.textContent?.trim()).not.toBe("NOT_FOUND");
  });

  test("an unreachable runner is reported without claiming a result", async () => {
    globalThis.fetch = (async () => {
      throw new Error("connection refused");
    }) as typeof fetch;
    await loadResults();

    expect(testId("results-error").textContent?.toLowerCase()).toContain("could not reach");
    expect(maybeTestId("retrieval-table")).toBeNull();
  });
});
