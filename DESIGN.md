# AutoBrain Task 9 report design

## Information hierarchy

The report answers one decision in the first viewport: **which candidate is
recommended, or why no recommendation is safe**. The hierarchy is:

1. Decision banner: canonical verdict, confidence/status, and one-sentence
   rationale.
2. Evidence summary: scored cases, eligibility gates, quality, cost
   completeness, latency, and coverage caveats.
3. Candidate comparison: one card per pinned candidate with status, quality,
   cost, latency, operational footprint, and explicit unknowns.
4. Methodology: benchmark composition, fixed rubric weights, same-model judge
   bias, decision thresholds, and limitations.
5. Case evidence: filterable/keyboard-accessible case rows with answer status,
   score components, cited source links, and failure details.
6. Provenance and artifacts: corpus/price/pricing evidence paths, native
   artifact paths, hash values, and privacy/redaction statements.

No oracle answer, reference reply, secret, or holdout marker is rendered.
Evidence is presented as source IDs plus safe source URLs and artifact paths,
never as evaluator-only text.

## Visual system

The static page uses a dark graphite base, near-white text, muted slate
secondary text, and a single semantic accent family: cyan for measured
evidence, green for eligible/recommended, amber for caveats, and red for
failures. The design uses system fonts only, no remote assets, gradients, or
decorative animation. Cards use a 1px border, 12px radius, and a consistent
8px spacing scale. Numeric facts use tabular numerals and labels remain close
to their values.

Charts are intentionally compact and semantic: CSS bars compare quality and
cost only when those values are complete; latency uses a labeled p50/p95
table. Every visual comparison has an adjacent textual value, and charts are
not the sole carrier of meaning.

## Responsive breakpoints

- Base/mobile: 0–639px; one column, full-width cards, horizontally scrollable
  comparison tables with visible labels, and no page-level horizontal
  overflow.
- Tablet: 640–1023px; two-column candidate cards and stacked evidence sections.
- Desktop: 1024px and wider; constrained 1180px content column, three candidate
  cards, and two-column methodology/provenance sections.

The case evidence table is allowed to scroll within its own labeled region;
the document itself must never overflow the viewport. Controls wrap rather
than compress below readable tap/keyboard targets.

## Accessibility and contrast

The document starts with a descriptive title and landmark structure:
`header`, `main`, `section`, and `footer`. Heading levels are sequential.
All status colors have text labels and symbols are never color-only. Body text
and controls target WCAG AA contrast (at least 4.5:1 for normal text and
3:1 for large text/UI boundaries). Focus rings are visible, links have
descriptive text, tables include captions and header scopes, and status/live
content uses `aria-live` only for the decision banner. The report is usable
with keyboard navigation and remains meaningful with styles disabled.

## Status vocabulary

The canonical machine vocabulary is shared by `comparison.json` and the HTML:
`OK`, `ENV_UNAVAILABLE`, `MISSING_PROVIDER`, `MCP_AUTH_UNAVAILABLE`,
`CAPABILITY_UNAVAILABLE`, `BUDGET_EXCEEDED`, `INSUFFICIENT_BENCHMARK`,
`LEAKAGE_DETECTED`, `FAILED`, `NO_DECISION`, and `NO_RECOMMENDATION`.
Candidate verdicts are exactly `llm-wiki`, `mem0`, `gbrain`,
`NO_DECISION`, or `NO_RECOMMENDATION`. Cost telemetry uses
`COST_COMPLETE`, `COST_INCOMPLETE`, and `COST_UNAVAILABLE`; incomplete or
unavailable cost is never rendered as `$0.00`.

## Evidence presentation

Every aggregate metric identifies its numerator/denominator or completeness
status. Quality shows all four weighted components out of 45/25/20/10.
Candidate-native evidence is linked by confined relative path. Cost records
retain raw proxy/native usage, the price-sheet version, and reconciliation
warnings. Coverage retains `UNKNOWN` rather than inferring completeness.
Warnings are written as visible caveats, not hidden tooltips. Deterministic
JSON and HTML are produced from canonicalized typed data, escaped before
interpolation, and written atomically.
