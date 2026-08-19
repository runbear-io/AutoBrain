# Reading an AutoBrain report

Start with the verdict, then check whether the evidence supports it.

## Status and coverage

`OK` means the stage completed; it does not mean the company corpus was
complete. Read each connector row:

- `SEARCH_DISCOVERED` means the connector reported a bounded discovery scope;
- `PARTIAL_RATE_LIMIT` means the crawl was interrupted by provider limits;
- `UNKNOWN` means completeness was not established, especially for Notion.

`MCP_AUTH_UNAVAILABLE`, `MISSING_PROVIDER`, and
`CAPABILITY_UNAVAILABLE` describe setup blockers. `INSUFFICIENT_BENCHMARK`
means fewer than 20 source-verifiable cases reached the benchmark gate.

## Candidate table

Compare status, score, scored cases, latency, and cost together. `unknown`
cost is incomplete telemetry, not free operation. A failed candidate does not
silently disappear; its typed detail and per-candidate JSON artifact remain in
the run directory. A budget-exceeded candidate was not started.

## Bias and limitations

The evaluator uses the same `gpt-5-mini` family across candidates. Treat that
as a disclosed judge bias, not proof of general superiority. Generated cases
are labeled in the manifest. Holdout source IDs and oracle-owned material do
not enter candidate context. The report does not claim statistical
significance, universal quality, or complete workspace coverage.

## Reopening and evidence

`autobrain report <run-id> --no-open` reopens an existing local report without
rerunning or modifying it. `manifest.json` is the source of truth for stage
status, command ledger, hashes, pins, model/pricing versions, timings,
coverage, benchmark provenance, candidate artifacts, metrics, verdict, and
report hashes.
