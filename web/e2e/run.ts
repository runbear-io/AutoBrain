/**
 * Browser E2E runner for the local experiment surfaces.
 *
 * Boots the real Python job boundary, builds the Web bundle pointed at it,
 * serves that bundle over loopback, and drives every scenario in
 * experiment-setup.spec.ts and retrieval-results.spec.ts through Chromium at
 * desktop and mobile widths.
 *
 * Pass a suite name to run one of them (`bun run e2e results`); with no
 * argument both run. Everything is loopback and credential-free; nothing is
 * deployed.
 */

import { spawn, type ChildProcess } from "node:child_process";
import { createReadStream, existsSync } from "node:fs";
import { createServer, type Server } from "node:http";
import { extname, join, normalize } from "node:path";
import { chromium } from "playwright";
import { runScenarios as runSetupScenarios } from "./experiment-setup.spec";
import { runScenarios as runResultsScenarios } from "./retrieval-results.spec";

type SuiteRunner = (
  browser: import("playwright").Browser,
  baseUrl: string,
  viewport: { width: number; height: number },
) => Promise<string[]>;

const SUITES: Record<string, SuiteRunner> = {
  setup: runSetupScenarios,
  results: runResultsScenarios,
};

const WEB_ROOT = new URL("..", import.meta.url).pathname;
const DIST = join(WEB_ROOT, "dist");

const MIME: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".svg": "image/svg+xml",
  ".json": "application/json",
};

/** Read the boundary's base_url from its first line of stdout. */
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
      try {
        const { base_url } = JSON.parse(buffered.slice(0, newline)) as { base_url: string };
        resolve({ process: child, baseUrl: base_url });
      } catch (cause) {
        reject(cause);
      }
    });
    child.on("error", reject);
    child.on("exit", (code) => reject(new Error(`boundary exited early with code ${code}`)));
  });
}

/** Build the bundle with the wizard pointed at the live boundary. */
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
    build.on("error", reject);
  });
}

/** Serve the built bundle from loopback with SPA fallback. */
function serveDist(): Promise<{ server: Server; origin: string }> {
  const server = createServer((request, response) => {
    const requested = normalize(decodeURIComponent(new URL(request.url ?? "/", "http://x").pathname));
    const candidate = join(DIST, requested);
    const file = candidate.startsWith(DIST) && existsSync(candidate) && extname(candidate) !== ""
      ? candidate
      : join(DIST, "index.html");
    response.writeHead(200, { "Content-Type": MIME[extname(file)] ?? "application/octet-stream" });
    createReadStream(file).pipe(response);
  });
  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      if (address === null || typeof address === "string") throw new Error("no server address");
      resolve({ server, origin: `http://127.0.0.1:${address.port}` });
    });
  });
}

async function main(): Promise<void> {
  const boundary = await startBoundary();
  console.log(`boundary: ${boundary.baseUrl}`);
  await buildBundle(boundary.baseUrl);
  const { server, origin } = await serveDist();
  console.log(`web:      ${origin}`);

  const requested = process.argv[2];
  const selected =
    requested === undefined
      ? Object.entries(SUITES)
      : Object.entries(SUITES).filter(([name]) => name === requested);
  if (selected.length === 0) {
    throw new Error(`unknown suite "${requested}"; expected one of ${Object.keys(SUITES).join(", ")}`);
  }

  const browser = await chromium.launch({ headless: true });
  const failures: string[] = [];
  try {
    for (const [name, runScenarios] of selected) {
      for (const viewport of [
        { width: 1280, height: 900 },
        { width: 375, height: 812 },
      ]) {
        console.log(`\n${name} @ ${viewport.width}px`);
        failures.push(...(await runScenarios(browser, origin, viewport)));
      }
    }
  } finally {
    await browser.close();
    server.close();
    boundary.process.kill();
  }

  if (failures.length > 0) {
    console.error(`\n${failures.length} browser scenario(s) failed`);
    process.exit(1);
  }
  console.log("\nall browser scenarios passed");
}

await main();
