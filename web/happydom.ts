/**
 * Registers a real DOM for `bun test`.
 *
 * Loaded via the `preload` entry in bunfig.toml so component tests can mount
 * React into an actual document instead of asserting against rendered strings.
 * Pure-logic tests are unaffected by the presence of these globals.
 */

import { GlobalRegistrator } from "@happy-dom/global-registrator";

/**
 * Pin the document origin to the Vite dev origin.
 *
 * Without this the document has an opaque `about:blank` origin and happy-dom
 * refuses every outbound request, so the HTTP integration test could not reach
 * a local fixture at all.
 *
 * Note that happy-dom does not enforce the response-side CORS check the way a
 * real browser does: it will surface a cross-origin body even when no
 * Access-Control-Allow-Origin header is present. The integration test
 * therefore asserts the required CORS headers explicitly rather than relying
 * on the runtime to reject a missing grant.
 */
GlobalRegistrator.register({ url: "http://localhost:5173/" });

// Opts React into act() semantics so state updates are flushed synchronously
// inside act(...) instead of warning and leaving assertions racing the render.
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
