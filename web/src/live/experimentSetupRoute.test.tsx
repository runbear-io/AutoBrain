/**
 * The setup wizard as a user actually drives it.
 *
 * These tests mount the real `App`, navigate via the sidebar, and click through
 * the wizard, so they prove a non-CLI user can configure and submit a Preview.
 * No test types a CLI command or reaches past the local job boundary.
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import App from "../App";

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

/** Type into a controlled textarea the way React's onChange expects. */
function typeInto(element: HTMLTextAreaElement, value: string) {
  act(() => {
    const setter = Object.getOwnPropertyDescriptor(
      HTMLTextAreaElement.prototype,
      "value",
    )?.set;
    setter?.call(element, value);
    element.dispatchEvent(new Event("input", { bubbles: true }));
  });
}

function openWizard() {
  click(navButton("New experiment"));
}

/** Drive the wizard to a ready official-source configuration. */
function configureOfficialPreview() {
  openWizard();
  click(testId("source-option-notion-snapshot"));
  click(testId("candidate-option-gbrain"));
  click(testId("subscription-option-codex"));
}

const JSONL = [
  JSON.stringify({ source_id: "doc-1", title: "Refund policy", text: "Refunds within 30 days." }),
  JSON.stringify({ source_id: "doc-2", title: "Escalation", text: "Escalate after two replies." }),
].join("\n");

beforeEach(mount);

afterEach(() => {
  unmount();
  globalThis.fetch = realFetch;
});

describe("experiment setup wizard reachability", () => {
  test("the sidebar exposes a new experiment entry", () => {
    expect(navButton("New experiment")).toBeDefined();
  });

  test("opening it renders the wizard with the retrieval-only evaluator", () => {
    openWizard();
    expect(container.querySelector(".wizard")).not.toBeNull();
    expect(container.textContent).toContain("Retrieval only");
  });

  test("production setup exposes local files but not fixture sources", () => {
    openWizard();
    expect(container.querySelector('[data-testid="source-option-fixture"]')).toBeNull();
    expect(container.querySelector('[data-testid="source-option-local-file"]')).not.toBeNull();
    expect(container.querySelector('[data-testid="source-option-notion-snapshot"]')).not.toBeNull();
    expect(container.querySelector('[data-testid="source-option-slack-export"]')).toBeNull();
  });

  test("the wizard states that it drives the local boundary, not a hosted service", () => {
    openWizard();
    const scope = container.querySelector(".wizard__scope");
    expect(scope?.textContent?.toLowerCase()).toContain("local");
    expect(scope?.textContent?.toLowerCase()).toContain("not hosted");
  });

  test("the synthetic demo remains reachable and unchanged", () => {
    openWizard();
    click(navButton("Diagnosis"));
    expect(container.textContent).toContain("GBrain is the strongest starting point");
    expect(container.querySelector(".wizard")).toBeNull();
  });
});

describe("configuring a preview without the CLI", () => {
  test("submission is disabled until every requirement is satisfied", () => {
    openWizard();
    expect((testId("submit-preview") as HTMLButtonElement).disabled).toBe(true);
    configureOfficialPreview();
    expect((testId("submit-preview") as HTMLButtonElement).disabled).toBe(false);
  });

  test("readiness lists human-readable blockers, never bare codes", () => {
    openWizard();
    const blockers = testId("readiness-blockers");
    expect(blockers.textContent).toContain("Choose");
    expect(blockers.textContent?.trim().length).toBeGreaterThan(20);
    expect(blockers.textContent).not.toMatch(/^[A-Z_]+$/);
  });

  test("supported local HTML is accepted through the setup path", async () => {
    openWizard();
    click(testId("source-option-local-file"));
    const input = testId("local-file-input") as HTMLInputElement;
    const file = new File(["<h1>Policy</h1><p>Visible</p>"], "policy.html", { type: "text/html" });
    await act(async () => {
      Object.defineProperty(input, "files", { value: [file] });
      input.dispatchEvent(new Event("change", { bubbles: true }));
    });
    click(testId("candidate-option-gbrain"));
    click(testId("subscription-option-codex"));
    expect(testId("corpus-summary").textContent).toContain("1 document");
    expect((testId("submit-preview") as HTMLButtonElement).disabled).toBe(false);
  });

  test("PDF and DOCX local files are blocked as unavailable", async () => {
    for (const name of ["report.pdf", "brief.docx"]) {
      openWizard();
      click(testId("source-option-local-file"));
      const input = testId("local-file-input") as HTMLInputElement;
      const file = new File(["binary"], name);
      const transfer = new DataTransfer();
      transfer.items.add(file);
      await act(async () => {
        Object.defineProperty(input, "files", {
          configurable: true,
          value: transfer.files,
        });
        input.dispatchEvent(new Event("change", { bubbles: true }));
      });
      expect(testId("readiness-blockers").textContent?.toLowerCase()).toContain("unavailable");
      expect((testId("submit-preview") as HTMLButtonElement).disabled).toBe(true);
      click(navButton("New experiment"));
    }
  });

  test("gated sources are omitted from official setup", () => {
    openWizard();
    expect(container.querySelector('[data-testid="source-option-slack-export"]')).toBeNull();
    expect(
      container.querySelector('[data-testid="source-option-approved-read-only-connector"]'),
    ).toBeNull();
  });

  test("malformed JSONL shows the offending line and blocks submission", () => {
    configureOfficialPreview();
    click(testId("format-option-JSONL"));
    typeInto(testId("source-payload") as HTMLTextAreaElement, '{"source_id":"a"}\nnope');
    expect(testId("readiness-blockers").textContent).toContain("line 2");
    expect((testId("submit-preview") as HTMLButtonElement).disabled).toBe(true);
  });

  test("a valid JSONL import reports its document count and unblocks submission", () => {
    configureOfficialPreview();
    click(testId("format-option-JSONL"));
    typeInto(testId("source-payload") as HTMLTextAreaElement, JSONL);
    expect(testId("corpus-summary").textContent).toContain("2");
    expect((testId("submit-preview") as HTMLButtonElement).disabled).toBe(false);
  });

  test("submitting an official-source preview drives create, validate, and start", async () => {
    const calls: string[] = [];
    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      calls.push(`${init?.method ?? "GET"} ${new URL(url).pathname}`);
      const status = url.endsWith("/start")
        ? "RUNNING"
        : url.endsWith("/validate")
          ? "READY"
          : "CREATED";
      return new Response(JSON.stringify({ experiment_id: "exp-1", status, updated_at: null }), {
        status: url.endsWith("/start") ? 202 : url.endsWith("/validate") ? 200 : 201,
        headers: { "Content-Type": "application/json" },
      });
    }) as typeof fetch;

    configureOfficialPreview();
    await clickAsync(testId("submit-preview"));

    expect(calls).toEqual([
      "POST /api/v1/experiments",
      "POST /api/v1/experiments/exp-1/validate",
      "POST /api/v1/experiments/exp-1/start",
    ]);
    expect(testId("submission-status").textContent).toContain("RUNNING");
  });

  test("a rejected submission shows recovery guidance instead of a raw code", async () => {
    globalThis.fetch = (async () =>
      new Response(
        JSON.stringify({ error: "NOT_READY", detail: "source transport is unavailable" }),
        { status: 409, headers: { "Content-Type": "application/json" } },
      )) as typeof fetch;

    configureOfficialPreview();
    await clickAsync(testId("submit-preview"));

    const status = testId("submission-status");
    expect(status.textContent).toContain("source transport is unavailable");
    expect(status.textContent?.length).toBeGreaterThan("NOT_READY".length + 20);
    expect(status.getAttribute("data-tone")).toBe("danger");
  });

  test("a transport failure is reported without claiming the run started", async () => {
    globalThis.fetch = (async () => {
      throw new Error("connection refused");
    }) as typeof fetch;

    configureOfficialPreview();
    await clickAsync(testId("submit-preview"));

    const status = testId("submission-status");
    expect(status.textContent?.toLowerCase()).toContain("could not reach");
    expect(status.textContent).not.toContain("RUNNING");
  });
});
