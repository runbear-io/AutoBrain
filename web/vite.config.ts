import { defineConfig } from "vite";

// This project is built and tested in an offline environment, so it avoids
// optional toolchain plugins. Vite's built-in esbuild pipeline compiles JSX
// via the automatic runtime, which is all this app needs.
export default defineConfig({
  base: "/autobrain-demo/",
  esbuild: {
    jsx: "automatic",
  },
  server: {
    port: 5173,
  },
});
