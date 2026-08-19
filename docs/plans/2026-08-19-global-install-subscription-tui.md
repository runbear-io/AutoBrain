# Global Install and Subscription-Only TUI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Install AutoBrain once as a global tool, launch it with bare `autobrain`, and keep the public TUI exclusively on the ChatGPT subscription path.

**Architecture:** Keep AutoBrain as a Python package and use `uv tool install` as the global package-manager boundary already supported by `pyproject.toml`. The installed console script continues to call the existing Typer bootstrap and curses cockpit. Remove only the TUI's API-key fallback; retain explicit headless API mode for compatibility, but remove it from the public README.

**Tech Stack:** Python 3.12–3.13, uv tools, Typer, curses, pytest, basedpyright, Ruff.

---

### Task 1: Lock the subscription-only planning contract

**Files:**
- Modify: `tests/test_experiment_setup.py`
- Modify: `src/autobrain/experiment.py`
- Modify: `src/autobrain/tui_runtime.py`

1. Add a regression test proving an `OPENAI_API_KEY` cannot make an unauthenticated TUI plan runnable.
2. Run the focused test and confirm it fails because API mode is selected.
3. Remove `api_key_available` from automatic planning.
4. Return `SUBSCRIPTION_AUTH_UNAVAILABLE` unless the local Codex subscription is ready.
5. Run focused TUI and experiment tests.

### Task 2: Make the installed command the primary UX

**Files:**
- Modify: `README.md`

1. Replace user-facing `uv run autobrain ...` commands with `autobrain ...`.
2. Document one-time installation with `uv tool install git+https://github.com/runbear-io/AutoBrain.git`.
3. Remove `OPENAI_API_KEY` and public API-provider instructions.
4. Keep `uv` commands only in the development section.

### Task 3: Verify the real installed surface

**Files:**
- No source changes expected.

1. Install the local package into isolated `UV_TOOL_DIR` and `UV_TOOL_BIN_DIR` locations.
2. Run the installed `autobrain --help`.
3. Launch bare `autobrain` in a PTY and confirm the cockpit renders.
4. Run the full tests, basedpyright, Ruff, offline build, and `git diff --check`.

No commits or pushes are part of this plan, per the requested direct-delivery mode.
