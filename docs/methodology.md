# AutoBrain methodology

## One frozen run

Every invocation creates a new run directory and executes these stages in
order:

1. preflight and candidate pin validation;
2. missing-auth gate;
3. read-only capability probe;
4. Slack and Notion crawl;
5. coverage snapshot;
6. benchmark and holdout construction;
7. final corpus freeze and hashes;
8. post-crawl budget estimate;
9. sequential LLM Wiki, Mem0, and GBrain ingest/query;
10. deterministic evaluation and verdict;
11. self-contained report generation and optional local browser open;
12. guaranteed candidate cleanup.

The stage ledger and command ledger are persisted in `manifest.json`. A
candidate exception becomes typed `FAILED` evidence, its cleanup still runs,
and later candidates are allowed to start. A hard budget stop becomes typed
`BUDGET_EXCEEDED` and later candidates do not start.

## Benchmark and holdout

Slack thread questions are preferred when connector data supplies them.
Document-derived questions are marked `generated` and are never presented as
human-authored cases. The last bounded slice is holdout-owned and is excluded
from candidate-facing documents. Candidate context contains only frozen
documents, questions, and corpus/benchmark hashes; it contains no oracle text,
reference answer, or heldout source.

At least 20 benchmark cases are required. The evaluator records answered cases,
scored cases, score, latency, cost, status, and artifact references. A missing
cost remains unknown; it is never estimated as zero. The current thin Task 10
orchestration boundary delegates native candidate behavior to injected pinned
adapters and retains their run-local artifacts.

## Verdict

The recommendation is quality-first: only successful candidates with at least
20 scored cases are eligible, and the highest score wins. If no candidate is
eligible, the report says `NO_RECOMMENDATION`. A run can still be successful
when one candidate fails, because the failure and remaining candidates are
visible in the report. This is a comparison aid, not a statistically
significant universal ranking.

## Cost and model caveats

The default hard cap is `$25` before and during candidate calls. The manifest
records the configured cap, observed candidate costs, pricing version, and
unknown telemetry. The judge model is `gpt-5-mini`; using the same family for
evaluation is a known source of bias and is always disclosed.
