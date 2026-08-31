/**
 * Real HTTP integration against an actual local fixture server.
 *
 * Nothing here is stubbed: each test spawns the real Python server as a
 * subprocess, reads the port it announces on stdout, and talks to it with the
 * browser's own `fetch`. That is what makes it a genuine reachability proof -
 * a CORS or port regression fails these tests, where a stubbed transport would
 * happily keep passing.
 *
 * The CORS expectations are asserted as explicit response headers because
 * happy-dom does not enforce the response-side CORS check itself. The headers
 * asserted here are exactly the ones a real browser requires before it will
 * expose the body to the page.
 */

import { afterEach, describe, expect, test } from "bun:test";
import { resolve } from "node:path";
import { fetchRunOutcome, isLoopbackUrl, localRunnerUrl, summarize } from "./runClient";

const REPO_ROOT = resolve(import.meta.dir, "../../..");
const PYTHON = resolve(REPO_ROOT, ".venv/bin/python");
const INSTALLED_BINARY = process.env.AUTOBRAIN_BINARY ?? resolve(REPO_ROOT, ".venv/bin/autobrain");
const DRIVER = resolve(REPO_ROOT, "tests/support/serve_local_fixture.py");

/** Origin a browser would send when the app is served by `vite dev`. */
const DEV_ORIGIN = "http://localhost:5173";

type Mode = "succeeded" | "failed" | "cancelled" | "malformed";

interface Fixture {
  baseUrl: string;
  stop: () => void;
}

const running: Fixture[] = [];

/**
 * Issue a GET over a raw socket with an arbitrary Origin.
 *
 * Needed because `Origin` is a forbidden header name in `fetch`, so a remote
 * origin cannot be simulated through the browser API.
 */
async function rawGet(url: URL, origin: string): Promise<string> {
  const chunks: string[] = [];
  let settle!: () => void;
  // Resolved by the socket close event rather than a timer: the request sends
  // `Connection: close`, so the server closing is the completion signal.
  const complete = new Promise<void>((resolve) => {
    settle = resolve;
  });
  const socket = await Bun.connect({
    hostname: url.hostname,
    port: Number(url.port),
    socket: {
      data: (_socket, data) => {
        chunks.push(new TextDecoder().decode(data));
      },
      close: () => settle(),
      error: () => settle(),
    },
  });
  socket.write(
    `GET ${url.pathname} HTTP/1.1\r\nHost: ${url.host}\r\n` +
      `Origin: ${origin}\r\nConnection: close\r\n\r\n`,
  );
  await complete;
  return chunks.join("");
}

/** Start the real server and wait until it announces its bound URL. */
async function startFixture(mode: Mode): Promise<Fixture> {
  const child = Bun.spawn([PYTHON, DRIVER, mode], {
    cwd: REPO_ROOT,
    stdout: "pipe",
    stderr: "pipe",
    env: {
      ...process.env,
      AUTOBRAIN_BINARY: INSTALLED_BINARY,
      PYTHONPATH: [REPO_ROOT, process.env.PYTHONPATH].filter(Boolean).join(":"),
    },
  });

  const reader = child.stdout.getReader();
  const decoder = new TextDecoder();
  let buffered = "";
  // Read until the announcement line arrives; no fixed sleep is involved.
  while (!buffered.includes("\n")) {
    const { value, done } = await reader.read();
    if (done) {
      const error = decoder.decode(await new Response(child.stderr).arrayBuffer());
      throw new Error(`fixture ${mode} exited before listening: ${error}`);
    }
    buffered += decoder.decode(value, { stream: true });
  }
  reader.releaseLock();

  const match = /listening (\S+)/.exec(buffered);
  if (match?.[1] === undefined) {
    throw new Error(`fixture ${mode} announced no url: ${buffered}`);
  }

  const fixture: Fixture = { baseUrl: match[1], stop: () => child.kill() };
  running.push(fixture);
  return fixture;
}

afterEach(() => {
  while (running.length > 0) {
    running.pop()?.stop();
  }
});

describe("real HTTP against a live local fixture", () => {
  test("a succeeded run is read over the wire and summarized", async () => {
    const { baseUrl } = await startFixture("succeeded");

    const outcome = await fetchRunOutcome(baseUrl);
    const summary = summarize(outcome);

    expect(outcome.status).toBe("SUCCEEDED");
    expect(outcome.projection?.run_id).toBe("RUN-A41F");
    expect(outcome.projection?.schema_version).toBe(1);
    expect(summary.headline).toBe("Recommended: gbrain");
  });

  test("a failed run is read over the wire without a projection", async () => {
    const { baseUrl } = await startFixture("failed");

    const outcome = await fetchRunOutcome(baseUrl);

    expect(outcome.status).toBe("FAILED");
    expect(outcome.projection).toBeNull();
    expect(outcome.error).toContain("no comparison.json");
    expect(summarize(outcome).tone).toBe("danger");
  });

  test("a cancelled run is distinct from a failure over the wire", async () => {
    const { baseUrl } = await startFixture("cancelled");

    const outcome = await fetchRunOutcome(baseUrl);
    const summary = summarize(outcome);

    expect(outcome.status).toBe("CANCELLED");
    expect(outcome.projection).toBeNull();
    expect(summary.headline).toBe("Run cancelled");
    expect(summary.tone).toBe("neutral");
  });

  test("a malformed projection is rejected rather than rendered", async () => {
    const { baseUrl } = await startFixture("malformed");

    const attempt = fetchRunOutcome(baseUrl);

    await expect(attempt).rejects.toThrow(/corpus_hash/);
  });

  test("unknown paths 404 over real HTTP", async () => {
    const { baseUrl } = await startFixture("succeeded");

    const response = await fetch(`${baseUrl}/index.html`);

    expect(response.status).toBe(404);
  });
});

describe("CORS grants only loopback dev origins", () => {
  test("a vite dev origin is granted access", async () => {
    const { baseUrl } = await startFixture("succeeded");

    const response = await fetch(`${baseUrl}/api/v1/run`, {
      headers: { Origin: DEV_ORIGIN },
    });

    expect(response.status).toBe(200);
    expect(response.headers.get("access-control-allow-origin")).toBe(DEV_ORIGIN);
    expect(response.headers.get("vary")).toContain("Origin");
  });

  test("the grant is never a wildcard and never credentialed", async () => {
    const { baseUrl } = await startFixture("succeeded");

    const response = await fetch(`${baseUrl}/api/v1/run`, {
      headers: { Origin: DEV_ORIGIN },
    });

    expect(response.headers.get("access-control-allow-origin")).not.toBe("*");
    expect(response.headers.get("access-control-allow-credentials")).toBeNull();
  });

  test("a remote origin receives no CORS grant", async () => {
    const { baseUrl } = await startFixture("succeeded");

    // `Origin` is a forbidden header name, so a browser always sends its own
    // document origin and this must be asserted below the browser. The raw
    // socket request here is what a non-loopback page would actually produce.
    const url = new URL(`${baseUrl}/api/v1/run`);
    const raw = await rawGet(url, "https://evil.example");

    expect(raw.toLowerCase()).not.toContain("access-control-allow-origin");
  });

  test("a browser preflight is answered for a dev origin", async () => {
    const { baseUrl } = await startFixture("succeeded");

    const response = await fetch(`${baseUrl}/api/v1/run`, {
      method: "OPTIONS",
      headers: {
        Origin: DEV_ORIGIN,
        "Access-Control-Request-Method": "GET",
      },
    });

    expect(response.status).toBe(204);
    expect(response.headers.get("access-control-allow-origin")).toBe(DEV_ORIGIN);
    expect(response.headers.get("access-control-allow-methods")).toContain("GET");
  });
});

describe("runner url resolution", () => {
  test("falls back to the documented loopback default", () => {
    expect(localRunnerUrl(undefined)).toBe("http://127.0.0.1:8765");
  });

  test("accepts a loopback override and trims a trailing slash", () => {
    expect(localRunnerUrl("http://127.0.0.1:9000/")).toBe("http://127.0.0.1:9000");
    expect(localRunnerUrl("http://localhost:4173")).toBe("http://localhost:4173");
  });

  test("ignores a non-loopback override", () => {
    expect(localRunnerUrl("https://evil.example")).toBe("http://127.0.0.1:8765");
    expect(localRunnerUrl("http://192.168.1.10:8765")).toBe("http://127.0.0.1:8765");
  });

  test.each([
    "http://localhost:5173",
    "http://127.0.0.1:8765",
    "https://localhost:5173",
  ])("recognizes %s as loopback", (url) => {
    expect(isLoopbackUrl(url)).toBe(true);
  });

  test.each([
    "https://evil.example",
    "http://localhost.evil.example",
    "not-a-url",
    "file:///tmp",
    "http://user:password@127.0.0.1:8765",
    "http://127.0.0.1:8765/path",
    "http://127.0.0.1:8765?token=secret",
  ])(
    "rejects %s as non-loopback",
    (url) => {
      expect(isLoopbackUrl(url)).toBe(false);
    },
  );
});
