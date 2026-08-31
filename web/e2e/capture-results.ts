/**
 * Capture browser evidence for the retrieval results views.
 *
 * Reuses the same real boundary + real bundle path as run.ts, then screenshots
 * the completed, comparison, failed, and cancelled surfaces at desktop and
 * mobile widths. Evidence only; it asserts nothing.
 */

import { spawn, type ChildProcess } from "node:child_process";
import { createReadStream, existsSync, mkdirSync } from "node:fs";
import { createServer, type Server } from "node:http";
import { extname, join, normalize } from "node:path";
import { chromium, type Page } from "playwright";

const WEB_ROOT = new URL("..", import.meta.url).pathname;
const DIST = join(WEB_ROOT, "dist");
const EVIDENCE = join(
  WEB_ROOT,
  "..",
  ".omo",
  "evidence",
  "autobrain-web-first-experiment",
  "todo-6",
);

const MIME: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json",
};

function startBoundary(): Promise<{ process: ChildProcess; baseUrl: string }> {
  const child = spawn("uv", ["run", "python", join(WEB_ROOT, "e2e", "serve_boundary.py")], {
    cwd: join(WEB_ROOT, ".."),
    stdio: ["pipe", "pipe", "inherit"],
  });
  return new Promise((resolve, reject) => {
    let buffered = "";
    child.stdout?.on("data", (chunk: Buffer) => {
      buffered += chunk.toString();
      const newline = buffered.indexOf("\n");
      if (newline === -1) return;
      const { base_url } = JSON.parse(buffered.slice(0, newline)) as { base_url: string };
      resolve({ process: child, baseUrl: base_url });
    });
    child.on("error", reject);
  });
}

function buildBundle(baseUrl: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const build = spawn("bunx", ["vite", "build", "--base", "/"], {
      cwd: WEB_ROOT,
      stdio: "inherit",
      env: { ...process.env, VITE_LOCAL_RUNNER_URL: baseUrl },
    });
    build.on("exit", (code) =>
      code === 0 ? resolve() : reject(new Error(`vite build failed with code ${code}`)),
    );
  });
}

function serveDist(): Promise<{ server: Server; origin: string }> {
  const server = createServer((request, response) => {
    const requested = normalize(
      decodeURIComponent(new URL(request.url ?? "/", "http://x").pathname),
    );
    const candidate = join(DIST, requested);
    const file =
      candidate.startsWith(DIST) && existsSync(candidate) && extname(candidate) !== ""
        ? candidate
        : join(DIST, "index.html");
    response.writeHead(200, { "Content-Type": MIME[extname(file)] ?? "application/octet-stream" });
    createReadStream(file).pipe(response);
  });
  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      if (address === null || typeof address === "string") throw new Error("no address");
      resolve({ server, origin: `http://127.0.0.1:${address.port}` });
    });
  });
}

async function submitPreview(page: Page): Promise<void> {
  await page.getByRole("button", { name: "New experiment" }).click();
  await page.locator(".wizard").waitFor({ state: "visible" });
  await page.getByTestId("source-option-slack-export").click();
  await page.getByTestId("candidate-option-gbrain").click();
  await page.getByTestId("subscription-option-codex").click();
  const started = page.waitForResponse((r) => r.url().endsWith("/start"));
  await page.getByTestId("submit-preview").click();
  await started;
}

async function openResults(page: Page): Promise<void> {
  await page.getByRole("button", { name: "Results" }).click();
  await page.locator(".results").waitFor({ state: "visible" });
}

async function main(): Promise<void> {
  mkdirSync(EVIDENCE, { recursive: true });
  const boundary = await startBoundary();
  await buildBundle(boundary.baseUrl);
  const { server, origin } = await serveDist();
  const browser = await chromium.launch({ headless: true });

  try {
    for (const [label, viewport] of [
      ["desktop", { width: 1280, height: 900 }],
      ["mobile", { width: 375, height: 812 }],
    ] as const) {
      // Completed run and comparison.
      let context = await browser.newContext({ viewport });
      let page = await context.newPage();
      await page.goto(origin, { waitUntil: "domcontentloaded" });
      await submitPreview(page);
      await openResults(page);
      const read = page.waitForResponse((r) => r.url().endsWith("/result"));
      await page.getByTestId("load-results").click();
      await read;
      await page.getByTestId("retrieval-table").waitFor({ state: "visible" });
      await page.screenshot({ path: join(EVIDENCE, `results-${label}-completed.png`), fullPage: true });

      await page.getByTestId("compare-with-current").click();
      await page.getByTestId("comparison-table").waitFor({ state: "visible" });
      await page.screenshot({ path: join(EVIDENCE, `results-${label}-comparison.png`), fullPage: true });
      await context.close();

      // Failed and cancelled recovery.
      for (const state of ["FAILED", "CANCELLED"] as const) {
        context = await browser.newContext({ viewport });
        page = await context.newPage();
        await page.goto(origin, { waitUntil: "domcontentloaded" });
        await submitPreview(page);
        await openResults(page);
        await page.route("**/api/v1/experiments/**", async (route) => {
          const url = route.request().url();
          const body = url.endsWith("/result")
            ? {
                status: state,
                projection: null,
                ...(state === "FAILED" ? { error: "candidate transport was unavailable" } : {}),
              }
            : { experiment_id: "exp-1", status: state, updated_at: null };
          await route.fulfill({
            status: 200,
            contentType: "application/json",
            body: JSON.stringify(body),
          });
        });
        await page.getByTestId("load-results").click();
        await page.getByTestId("recovery-notice").waitFor({ state: "visible" });
        await page.screenshot({
          path: join(EVIDENCE, `results-${label}-${state.toLowerCase()}.png`),
          fullPage: true,
        });
        await context.close();
      }
      console.log(`captured ${label}`);
    }
  } finally {
    await browser.close();
    server.close();
    boundary.process.kill();
  }
  console.log(`evidence written to ${EVIDENCE}`);
}

await main();
