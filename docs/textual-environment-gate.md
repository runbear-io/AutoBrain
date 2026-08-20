# Textual environment gate

Todo 7 declares Textual as a direct MIT runtime dependency in `pyproject.toml`.
The lockfile is intentionally unchanged: this workstation has no DNS, no local
Textual wheel, and `uv` cannot resolve because its macOS SystemConfiguration
resolver panics. No hash or lock metadata was invented. After network/package
access is restored, run `uv lock` and commit the resulting `uv.lock` update.

Pure reducer/effect/view-model tests and source AST/import checks run without
Textual. Textual Pilot tests are retained as real tests for an environment with
the declared dependency; no tests are skipped to hide this gate. Visual/manual
Textual QA is not claimed here.
