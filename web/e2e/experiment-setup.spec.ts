/**
 * Browser E2E for the experiment setup wizard.
 *
 * This drives a real Chromium page against the built Web bundle and the real
 * Python job boundary, so it proves a non-CLI user can configure and submit a
 * Preview and that blocked or invalid selections produce actionable guidance.
 *
 * The spec is written against the Playwright library rather than
 * @playwright/test because this offline repository vendors `playwright` only.
 * Run it with `bun run e2e` (see web/e2e/run.ts), which starts the boundary and
 * a static server, then executes every scenario below.
 */

import type { Browser, Page } from "playwright";

export interface Scenario {
  name: string;
  run: (page: Page) => Promise<void>;
}

function assert(condition: boolean, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

/** Drive the wizard to a ready official-source configuration. */
async function configureOfficialPreview(page: Page): Promise<void> {
  await page.getByTestId("source-option-slack-export").click();
  await page.getByTestId("candidate-option-gbrain").click();
  await page.getByTestId("subscription-option-codex").click();
}

async function openWizard(page: Page, baseUrl: string): Promise<void> {
  await page.goto(baseUrl, { waitUntil: "domcontentloaded" });
  await page.getByRole("button", { name: "New experiment" }).click();
  await page.locator(".wizard").waitFor({ state: "visible" });
}

export const SCENARIOS: Scenario[] = [
  {
    name: "an official-source Preview is configured and submitted without a CLI command",
    run: async (page) => {
      const submit = page.getByTestId("submit-preview");
      assert(await submit.isDisabled(), "submit should start disabled");

      await configureOfficialPreview(page);
      assert(await submit.isEnabled(), "submit should enable once readiness is satisfied");

      // Wait for the start call the boundary answers with 202, so the
      // assertion is tied to the real request rather than to elapsed time.
      const started = page.waitForResponse(
        (response) => response.url().endsWith("/start") && response.request().method() === "POST",
      );
      await submit.click();
      const response = await started;
      assert(response.status() === 202, `start expected 202, got ${response.status()}`);

      const status = page.getByTestId("submission-status");
      await status.waitFor({ state: "visible" });
      const text = (await status.textContent()) ?? "";
      assert(text.includes("RUNNING"), `expected a RUNNING preview, got: ${text}`);
      assert(
        (await status.getAttribute("data-tone")) === "success",
        "an accepted preview should read as success",
      );
    },
  },
  {
    name: "a JSONL import reports its document count and unblocks submission",
    run: async (page) => {
      await configureOfficialPreview(page);
      await page.getByTestId("format-option-JSONL").click();
      await page.getByTestId("source-payload").fill(
        [
          JSON.stringify({ source_id: "doc-1", title: "Refunds", text: "Refunds within 30 days." }),
          JSON.stringify({ source_id: "doc-2", title: "Escalation", text: "Escalate promptly." }),
        ].join("\n"),
      );

      const summary = (await page.getByTestId("corpus-summary").textContent()) ?? "";
      assert(summary.includes("2 documents"), `expected a 2 document corpus, got: ${summary}`);
      assert(
        await page.getByTestId("submit-preview").isEnabled(),
        "a valid JSONL import should allow submission",
      );
    },
  },
  {
    name: "malformed JSONL is refused with the offending line named on screen",
    run: async (page) => {
      await configureOfficialPreview(page);
      await page.getByTestId("format-option-JSONL").click();
      await page.getByTestId("source-payload").fill('{"source_id":"doc-1"}\nnot json');

      const blockers = (await page.getByTestId("readiness-blockers").textContent()) ?? "";
      assert(blockers.includes("line 2"), `expected the failing line named, got: ${blockers}`);
      assert(
        await page.getByTestId("submit-preview").isDisabled(),
        "an invalid import must block submission",
      );
    },
  },
  {
    name: "a gated source is blocked with its remediation, never runnable",
    run: async (page) => {
      await configureOfficialPreview(page);
      await page.getByTestId("source-option-approved-read-only-connector").click();

      const blockers = (await page.getByTestId("readiness-blockers").textContent()) ?? "";
      assert(
        blockers.toLowerCase().includes("approval"),
        `expected gated remediation guidance, got: ${blockers}`,
      );
      assert(
        await page.getByTestId("submit-preview").isDisabled(),
        "a gated source must never be submittable",
      );
    },
  },
  {
    name: "readiness guidance is human readable, not a bare machine code",
    run: async (page) => {
      const blockers = (await page.getByTestId("readiness-blockers").textContent()) ?? "";
      assert(blockers.length > 40, `expected sentence guidance, got: ${blockers}`);
      assert(
        /choose|select|paste/i.test(blockers),
        `expected an actionable instruction, got: ${blockers}`,
      );
      assert(!/^[A-Z_]+$/.test(blockers.trim()), "guidance must not be a raw code");
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
      await openWizard(page, baseUrl);
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
