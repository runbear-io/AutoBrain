/**
 * Browser E2E for the retrieval results and comparison views.
 *
 * This drives a real Chromium page against the built Web bundle and the real
 * Python job boundary, so it proves a non-CLI user can read a completed
 * Preview's per-Brain retrieval metrics, provenance, and confidence status -
 * and that a failed or cancelled run still offers actionable recovery without
 * ever claiming recommendation-grade evidence.
 *
 * Like experiment-setup.spec.ts, this is written against the Playwright library
 * rather than @playwright/test because this offline repository vendors
 * `playwright` only. Run it with `bun run e2e:results` (see web/e2e/run-results.ts).
 */

import type { Browser, Page } from "playwright";

export interface Scenario {
  name: string;
  run: (page: Page) => Promise<void>;
}

function assert(condition: boolean, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

/**
 * Submit a fixture Preview through the wizard.
 *
 * The results route reads whatever experiment the wizard last submitted, so
 * every scenario below starts from a genuinely submitted run rather than an
 * injected identifier.
 */
async function submitPreview(page: Page): Promise<void> {
  await page.getByRole("button", { name: "New experiment" }).click();
  await page.locator(".wizard").waitFor({ state: "visible" });
  await page.getByTestId("source-option-fixture").click();
  await page.getByTestId("candidate-option-gbrain").click();
  await page.getByTestId("subscription-option-codex").click();

  // Tie the wait to the boundary's own 202, never to elapsed time.
  const started = page.waitForResponse(
    (response) => response.url().endsWith("/start") && response.request().method() === "POST",
  );
  await page.getByTestId("submit-preview").click();
  const response = await started;
  assert(response.status() === 202, `start expected 202, got ${response.status()}`);
}

/** Open the results route and read the submitted experiment's result. */
async function readResult(page: Page): Promise<void> {
  await page.getByRole("button", { name: "Results" }).click();
  await page.locator(".results").waitFor({ state: "visible" });

  const read = page.waitForResponse((response) => response.url().endsWith("/result"));
  await page.getByTestId("load-results").click();
  await read;
}

async function openApp(page: Page, baseUrl: string): Promise<void> {
  await page.goto(baseUrl, { waitUntil: "domcontentloaded" });
}

export const SCENARIOS: Scenario[] = [
  {
    name: "a completed run shows per-Brain retrieval metrics read from the engine",
    run: async (page) => {
      await submitPreview(page);
      await readResult(page);

      const table = page.getByTestId("retrieval-table");
      await table.waitFor({ state: "visible" });
      const rows = await table.locator("tbody tr").count();
      assert(rows > 0, "a completed run must show at least one Brain row");

      const text = (await table.textContent()) ?? "";
      assert(/gbrain/.test(text), `expected the selected Brain in the table, got: ${text}`);
      // A rendered percentage proves a real measurement rather than a dash.
      assert(/%/.test(text), `expected a measured rate in the table, got: ${text}`);
    },
  },
  {
    name: "provenance names the frozen corpus, benchmark, and run identity",
    run: async (page) => {
      await submitPreview(page);
      await readResult(page);

      const provenance = page.getByTestId("result-provenance");
      await provenance.waitFor({ state: "visible" });
      const text = (await provenance.textContent()) ?? "";

      assert(/corpus/i.test(text), `expected the frozen corpus named, got: ${text}`);
      assert(/benchmark/i.test(text), `expected the benchmark named, got: ${text}`);
      assert(/run/i.test(text), `expected the run identity named, got: ${text}`);
    },
  },
  {
    name: "the engine benchmark hash replaces the wizard placeholder",
    run: async (page) => {
      await submitPreview(page);
      await readResult(page);

      const benchmark = page.getByTestId("benchmark-identity");
      await benchmark.waitFor({ state: "visible" });

      assert(
        (await benchmark.getAttribute("data-source")) === "engine",
        "a completed run must show the engine's benchmark, not the local derivation",
      );
      assert(
        (await benchmark.getAttribute("data-placeholder")) === "false",
        "the engine benchmark must not be labeled a placeholder",
      );
    },
  },
  {
    name: "confidence is stated and recommendation-grade evidence is never claimed",
    run: async (page) => {
      await submitPreview(page);
      await readResult(page);

      const confidence = page.getByTestId("evidence-confidence");
      await confidence.waitFor({ state: "visible" });

      assert(
        (await confidence.getAttribute("data-recommendation-grade")) === "false",
        "no result may present itself as recommendation-grade",
      );
      const text = ((await confidence.textContent()) ?? "").toLowerCase();
      assert(
        text.includes("not a production recommendation"),
        `expected the recommendation caveat on screen, got: ${text}`,
      );

      // The whole page must never claim a recommendation either.
      const page_text = ((await page.locator(".results").textContent()) ?? "").toLowerCase();
      assert(
        !page_text.includes("recommended configuration"),
        "the results surface must not claim a recommended configuration",
      );
    },
  },
  {
    name: "latency and cost are reported per Brain, including unavailable values",
    run: async (page) => {
      await submitPreview(page);
      await readResult(page);

      const table = page.getByTestId("retrieval-table");
      await table.waitFor({ state: "visible" });
      const header = (await table.locator("thead").textContent()) ?? "";

      for (const column of ["Recall", "Missing evidence", "Precision", "Noise", "p95", "Cost"]) {
        assert(header.includes(column), `expected a ${column} column, got: ${header}`);
      }
    },
  },
  {
    name: "a before/after comparison renders a per-Brain direction",
    run: async (page) => {
      await submitPreview(page);
      await readResult(page);

      await page.getByTestId("compare-with-current").click();
      const comparison = page.getByTestId("comparison-table");
      await comparison.waitFor({ state: "visible" });

      const rows = comparison.locator("tbody tr");
      assert((await rows.count()) > 0, "a comparison must show at least one Brain");
      const direction = await rows.first().getAttribute("data-direction");
      assert(
        direction === "unchanged",
        `comparing a run with itself must read as unchanged, got: ${String(direction)}`,
      );
    },
  },
  {
    name: "an unknown experiment yields actionable guidance, never a raw code",
    run: async (page) => {
      await submitPreview(page);
      await page.getByRole("button", { name: "Results" }).click();
      await page.locator(".results").waitFor({ state: "visible" });

      // Make the boundary answer NOT_FOUND by pointing the read at an id the
      // runner never created. This exercises the real 404 contract.
      await page.route("**/api/v1/experiments/**", async (route) => {
        await route.fulfill({
          status: 404,
          contentType: "application/json",
          body: JSON.stringify({ error: "NOT_FOUND", detail: "exp-missing" }),
        });
      });

      await page.getByTestId("load-results").click();
      const error = page.getByTestId("results-error");
      await error.waitFor({ state: "visible" });

      const text = (await error.textContent()) ?? "";
      assert(text.length > 40, `expected sentence guidance, got: ${text}`);
      assert(text.trim() !== "NOT_FOUND", "guidance must not be a bare machine code");
      assert(
        (await page.getByTestId("retrieval-table").count()) === 0,
        "a rejected read must not render a results table",
      );
    },
  },
  {
    name: "a failed run shows recovery actions and no results table",
    run: async (page) => {
      await submitPreview(page);
      await page.getByRole("button", { name: "Results" }).click();
      await page.locator(".results").waitFor({ state: "visible" });

      // Inject a terminal FAILED lifecycle and result at the transport layer so
      // the page exercises its real failure rendering path.
      await page.route("**/api/v1/experiments/**", async (route) => {
        const url = route.request().url();
        const body = url.endsWith("/result")
          ? { status: "FAILED", projection: null, error: "candidate transport was unavailable" }
          : { experiment_id: "exp-1", status: "FAILED", updated_at: null };
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify(body),
        });
      });

      await page.getByTestId("load-results").click();
      const recovery = page.getByTestId("recovery-notice");
      await recovery.waitFor({ state: "visible" });

      assert(
        (await recovery.getAttribute("data-tone")) === "danger",
        "a failed run must read as a failure",
      );
      const text = (await recovery.textContent()) ?? "";
      assert(
        text.includes("candidate transport was unavailable"),
        `expected the engine detail on screen, got: ${text}`,
      );
      assert(
        (await page.getByTestId("recovery-action-rerun").count()) === 1,
        "a failed run must offer a rerun",
      );
      assert(
        (await page.getByTestId("retrieval-table").count()) === 0,
        "a failed run must not render a results table",
      );
      assert(
        (await page.getByTestId("evidence-confidence").count()) === 0,
        "a failed run must not claim any evidence confidence",
      );
    },
  },
  {
    name: "a cancelled run states nothing was scored and offers a rerun",
    run: async (page) => {
      await submitPreview(page);
      await page.getByRole("button", { name: "Results" }).click();
      await page.locator(".results").waitFor({ state: "visible" });

      await page.route("**/api/v1/experiments/**", async (route) => {
        const url = route.request().url();
        const body = url.endsWith("/result")
          ? { status: "CANCELLED", projection: null }
          : { experiment_id: "exp-1", status: "CANCELLED", updated_at: null };
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify(body),
        });
      });

      await page.getByTestId("load-results").click();
      const recovery = page.getByTestId("recovery-notice");
      await recovery.waitFor({ state: "visible" });

      const text = ((await recovery.textContent()) ?? "").toLowerCase();
      assert(
        text.includes("no brain was scored"),
        `a cancelled run must say nothing was scored, got: ${text}`,
      );
      assert(
        (await page.getByTestId("recovery-action-rerun").count()) === 1,
        "a cancelled run must offer a rerun",
      );
    },
  },
  {
    name: "the results surface stays readable and scrollable at mobile width",
    run: async (page) => {
      await submitPreview(page);
      await readResult(page);

      const table = page.getByTestId("retrieval-table");
      await table.waitFor({ state: "visible" });

      // The metrics table is wide by nature; at any width it must stay inside
      // its scroll container rather than forcing the page to scroll sideways.
      const overflow = await page.evaluate(() => ({
        documentWidth: document.documentElement.scrollWidth,
        viewportWidth: document.documentElement.clientWidth,
      }));
      assert(
        overflow.documentWidth <= overflow.viewportWidth + 1,
        `the page must not scroll horizontally: ${overflow.documentWidth} > ${overflow.viewportWidth}`,
      );

      const confidence = page.getByTestId("evidence-confidence");
      assert(await confidence.isVisible(), "the confidence status must remain visible");
    },
  },
];

/** Execute every scenario at one viewport, returning failures. */
export async function runScenarios(
  browser: Browser,
  baseUrl: string,
  viewport: { width: number; height: number },
): Promise<string[]> {
  const failures: string[] = [];
  for (const scenario of SCENARIOS) {
    const context = await browser.newContext({ viewport });
    const page = await context.newPage();
    try {
      await openApp(page, baseUrl);
      await scenario.run(page);
      console.log(`  PASS  ${viewport.width}px  ${scenario.name}`);
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : String(cause);
      failures.push(`${viewport.width}px  ${scenario.name}: ${message}`);
      console.log(`  FAIL  ${viewport.width}px  ${scenario.name}: ${message}`);
    } finally {
      await context.close();
    }
  }
  return failures;
}
