# Contributing to AutoBrain

Thanks for helping improve AutoBrain. The project is experimental, so changes
that improve reproducibility, safety, and truthful evaluation evidence are
especially welcome.

## Before opening a pull request

1. Explain the user-facing problem and the smallest correct change.
2. Add or update a regression test for behavioral changes.
3. Keep provider credentials and private source content out of commits,
   fixtures, logs, screenshots, and issue reports.
4. Run the checks that cover your change:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run basedpyright
```

For Web changes, also run `cd web && bun test && bun run typecheck && bun run build`.

## Pull requests

Describe the observable behavior, verification commands, known limitations,
and any source/provider access that was not available. Do not claim a live
provider or recommendation-grade result from fixture-only evidence.

## Development boundaries

AutoBrain must not silently collect credentials, upload source content, or
turn a local evaluation into a hosted service. New source and provider
integrations need explicit readiness, provenance, failure, and redaction
contracts.
