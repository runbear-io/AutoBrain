/**
 * Reachability of the local fixture route inside the real app.
 *
 * These tests mount the actual `App` into a real DOM and navigate the way a
 * user does - by clicking the sidebar - so they prove the route is genuinely
 * reachable rather than merely exported. They also assert the synthetic demo
 * is still reachable and unchanged, since the local fixture is an addition to
 * that journey, not a replacement for it.
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import App from "../App";

const HASH_A = "a".repeat(64);
const HASH_B = "b".repeat(64);

const PROJECTION = {
  schema_version: 1,
  run_id: "RUN-A41F",
  status: "OK",
  verdict: "gbrain",
  rationale: "GBrain leads grounded recall.",
  corpus_hash: HASH_A,
  benchmark_hash: HASH_B,
  candidates: [
    {
      candidate: "gbrain",
      status: "OK",
      quality_score: 93.6,
      answer_success_rate: 0.93,
      source_support_rate: 0.79,
      contradiction_count: 1,
      scored_cases: 30,
      answered_cases: 28,
      cost_status: "COST_COMPLETE",
      total_cost_usd: 1.25,
      query_p50_ms: 820,
      query_p95_ms: 2460,
      operating_burden: 2,
    },
  ],
  warnings: [],
};

let container: HTMLDivElement;
let root: Root;
const realFetch = globalThis.fetch;

function mount() {
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
  act(() => {
    root.render(<App />);
  });
}

function unmount() {
  act(() => {
    root.unmount();
  });
  container.remove();
}

/** Find a button by its trimmed visible text. */
function byText(selector: string, text: string): HTMLElement {
  const match = [...container.querySelectorAll<HTMLElement>(selector)].find(
    (node) => node.textContent?.trim().toLowerCase() === text.toLowerCase(),
  );
  if (match === undefined) {
    throw new Error(`no ${selector} with text "${text}"`);
  }
  return match;
}

/**
 * Find a sidebar entry by its label.
 *
 * Nav buttons render an icon alongside the label, so the label is matched on
 * the inner span rather than the button's full text content.
 */
function navButton(label: string): HTMLElement {
  const match = [...container.querySelectorAll<HTMLElement>(".sidebar nav button")].find(
    (node) => node.querySelector("span")?.textContent?.trim().toLowerCase() === label.toLowerCase(),
  );
  if (match === undefined) {
    throw new Error(`no sidebar entry labeled "${label}"`);
  }
  return match;
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

/** Navigate to the local fixture route the way a user would. */
function openLocalFixture() {
  click(navButton("Local runner"));
}

beforeEach(mount);

afterEach(() => {
  unmount();
  globalThis.fetch = realFetch;
});

describe("local fixture route reachability", () => {
  test("the sidebar exposes a local runner entry", () => {
    expect(navButton("Local runner")).toBeDefined();
  });

  test("clicking it renders the local runner panel", () => {
    openLocalFixture();
    expect(container.querySelector(".live-panel")).not.toBeNull();
    expect(container.textContent).toContain("Local runner");
  });

  test("the route is labeled as a local fixture and not hosted", () => {
    openLocalFixture();
    const scope = container.querySelector(".live-panel__scope");
    expect(scope).not.toBeNull();
    expect(scope?.textContent?.toLowerCase()).toContain("local fixture");
    expect(scope?.textContent?.toLowerCase()).toContain("not hosted");
  });

  test("the panel shows the loopback origin it would read from", () => {
    openLocalFixture();
    expect(container.textContent).toContain("127.0.0.1");
  });

  test("no run has been read before the operator asks for one", () => {
    openLocalFixture();
    expect(container.textContent).toContain("No local run has been read yet.");
  });

  test("reading a run invokes fetchRunOutcome and renders the projection", async () => {
    const requested: string[] = [];
    globalThis.fetch = (async (input: RequestInfo | URL) => {
      requested.push(String(input));
      return new Response(JSON.stringify({ status: "SUCCEEDED", projection: PROJECTION, error: null }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }) as typeof fetch;

    openLocalFixture();
    await clickAsync(byText("button", "Read local run"));

    expect(requested).toEqual(["http://127.0.0.1:8765/api/v1/run"]);
    expect(container.textContent).toContain("RUN-A41F");
    expect(container.textContent).toContain("run SUCCEEDED");
    expect(container.textContent).toContain("Recommended: gbrain");
  });

  test("a cancelled run is reported as cancelled, not failed", async () => {
    globalThis.fetch = (async () =>
      new Response(JSON.stringify({ status: "CANCELLED", projection: null, error: null }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      })) as typeof fetch;

    openLocalFixture();
    await clickAsync(byText("button", "Read local run"));

    expect(container.textContent).toContain("Run cancelled");
    expect(container.textContent).not.toContain("Run failed");
  });

  test("a malformed payload surfaces an error instead of a bogus summary", async () => {
    globalThis.fetch = (async () =>
      new Response(
        JSON.stringify({
          status: "SUCCEEDED",
          projection: { ...PROJECTION, corpus_hash: "not-a-hash" },
          error: null,
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      )) as typeof fetch;

    openLocalFixture();
    await clickAsync(byText("button", "Read local run"));

    expect(container.textContent).toContain("corpus_hash");
    expect(container.textContent).not.toContain("Recommended:");
  });
});

describe("synthetic demo is preserved", () => {
  test("the app still opens on the synthetic demo journey", () => {
    expect(container.textContent).toContain("Find the best brain for your company");
    expect(container.textContent).toContain("Synthetic demo");
  });

  test("the synthetic demo routes remain reachable alongside the fixture", () => {
    click(navButton("Diagnosis"));
    expect(container.textContent).toContain("GBrain is the strongest starting point");

    openLocalFixture();
    expect(container.querySelector(".live-panel")).not.toBeNull();

    click(navButton("Diagnosis"));
    expect(container.textContent).toContain("GBrain is the strongest starting point");
    expect(container.querySelector(".live-panel")).toBeNull();
  });
});
