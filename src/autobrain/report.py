"""Deterministic, self-contained static comparison report generation."""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from pydantic import ValidationError

from autobrain.models import (
    BenchmarkProvenance,
    CandidateCaseEvidence,
    CandidateEvaluation,
    ComparisonArtifact,
    CoverageRecord,
    DecisionResult,
    RunManifest,
    Sha256,
    Status,
)

_SECRET: Final = re.compile(
    r"(?i)(?:sk-[a-z0-9_-]{8,}|xox[a-z]-[a-z0-9-]{8,}|"
    r"bearer\s+[a-z0-9._~+/=-]{8,}|(?:api[_-]?key|token|password)=?[a-z0-9._~+/=-]{8,}|"
    r"//[^/\s:@]+:[^@\s]+@)"
)
_ORACLE: Final = re.compile(
    r"(?i)\b(?:oracle|holdout|reference[\s_-]*answer|evaluator[\s_-]*only|"
    r"expected[\s_-]*claim|raw[\s_-]*reply)\b"
)
_SENSITIVE_KEY: Final = re.compile(
    r"(?i)(?:secret|token|password|api[_-]?key|authorization|credential|raw[_-]?metering)"
)


@dataclass(frozen=True)
class ReportArtifacts:
    comparison_json: Path
    report_html: Path
    comparison_sha256: Sha256
    report_sha256: Sha256


def build_comparison(
    *,
    run_id: str,
    status: Status = Status.OK,
    corpus_hash: str,
    benchmark_hash: str,
    coverage: list[CoverageRecord],
    candidates: list[CandidateEvaluation],
    decision: DecisionResult,
    evidence: list[CandidateCaseEvidence],
    provenance: BenchmarkProvenance | None = None,
    artifact_paths: dict[str, str] | None = None,
    warnings: list[str] | None = None,
    price_sheet_version: str | None = None,
) -> ComparisonArtifact:
    """Build the single typed source of truth consumed by JSON and HTML."""
    artifact = ComparisonArtifact(
        schema_version=2,
        run_id=run_id,
        status=status,
        corpus_hash=corpus_hash,
        benchmark_hash=benchmark_hash,
        verdict=decision.verdict,
        decision=decision,
        coverage=coverage,
        candidates=candidates,
        evidence=evidence,
        provenance=provenance or BenchmarkProvenance(),
        methodology={
            "quality_weights": "retrieval recall over gold source IDs, scaled to 0-100",
            "eligibility": "20 scored cases; 90% valid answers; quality >=60; "
            "source support >=50%; valid pin and corpus hash; zero direct leakage",
            "judge_model": "gpt-5-mini at temperature 0; same-model family bias applies",
            "decision_rule": (
                "quality floor first; within five points cost, then latency and operations"
            ),
        },
        artifact_paths=artifact_paths or {},
        warnings=warnings or [],
        price_sheet_version=price_sheet_version,
    )
    return _redact_artifact(artifact)


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _canonical_json(artifact: ComparisonArtifact) -> bytes:
    artifact = _redact_artifact(artifact)
    return (
        json.dumps(
            artifact.model_dump(mode="json"),
            ensure_ascii=True,
            sort_keys=True,
            indent=2,
            separators=(",", ": "),
        )
        + "\n"
    ).encode("utf-8")


def _schema_payload(payload: object) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise ValueError("artifact payload must be an object")
    typed_payload = cast(Mapping[str, object], payload)
    schema_version = typed_payload.get("schema_version")
    if type(schema_version) is not int:
        raise ValueError("schema_version must be a JSON integer")
    return typed_payload


def _migrate_comparison(payload: object) -> object:
    typed_payload = _schema_payload(payload)
    if typed_payload["schema_version"] == 1:
        migrated = dict(typed_payload)
        migrated["schema_version"] = 2
        migrated.setdefault("provenance", BenchmarkProvenance().model_dump(mode="json"))
        return migrated
    return dict(typed_payload)


def load_comparison(path: Path) -> ComparisonArtifact:
    """Load a comparison artifact and migrate supported legacy schemas."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return ComparisonArtifact.model_validate(_migrate_comparison(payload), strict=False)
    except (OSError, ValueError, ValidationError) as error:
        raise ValueError(f"corrupt comparison artifact: {path}") from error


def _migrate_manifest(payload: object) -> object:
    typed_payload = _schema_payload(payload)
    if typed_payload["schema_version"] == 1:
        migrated = dict(typed_payload)
        migrated.setdefault("provenance", BenchmarkProvenance().model_dump(mode="json"))
        return migrated
    return dict(typed_payload)


def load_manifest(path: Path) -> RunManifest:
    """Load manifest provenance and migrate only explicit schema version 1."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return RunManifest.model_validate(_migrate_manifest(payload), strict=False)
    except (OSError, ValueError, ValidationError) as error:
        raise ValueError(f"corrupt run manifest: {path}") from error


def _redact_string(value: str, *, key: str | None = None) -> str:
    replacement = "redacted:source" if key == "source_ids" else "[REDACTED]"
    if key is not None and (
        _SENSITIVE_KEY.search(key) or _ORACLE.search(key) or "BROWSER_UNAVAILABLE" in value
    ):
        return replacement
    redacted = _SECRET.sub("[REDACTED]", value)
    if _ORACLE.search(redacted) or "BROWSER_UNAVAILABLE" in redacted:
        return replacement
    return redacted


def _redact_value(value: object, *, key: str | None = None) -> object:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {
            str(item_key): _redact_value(item, key=str(item_key))
            for item_key, item in mapping.items()
        }
    if isinstance(value, list):
        return [_redact_value(item, key=key) for item in cast(list[object], value)]
    if isinstance(value, tuple):
        return tuple(_redact_value(item) for item in cast(tuple[object, ...], value))
    if isinstance(value, str):
        return _redact_string(value, key=key)
    return value


def _redact_artifact(artifact: ComparisonArtifact) -> ComparisonArtifact:
    payload = cast(dict[str, object], _redact_value(artifact.model_dump(mode="python")))
    return ComparisonArtifact.model_validate(payload, strict=False)


def _escape(value: object) -> str:
    return html.escape(_redact_string(str(value)), quote=True)


def _latency_value(duration_ms: float | None) -> str:
    return f"{duration_ms:g} ms" if duration_ms is not None else "unavailable"


def _candidate_card(candidate: CandidateEvaluation) -> str:
    cost = (
        f"${candidate.total_cost_usd:.4f}"
        if candidate.cost_status.value == "COST_COMPLETE" and candidate.total_cost_usd is not None
        else f"unknown ({candidate.cost_status.value})"
    )
    latency = (
        f"p50 {candidate.query_p50_ms:.0f} ms / p95 {candidate.query_p95_ms:.0f} ms"
        if candidate.query_p50_ms is not None and candidate.query_p95_ms is not None
        else "unknown"
    )
    workspace = candidate.workspace_bytes if candidate.workspace_bytes is not None else "unknown"
    burden = candidate.operating_burden if candidate.operating_burden is not None else "unknown"
    return f"""
      <article class="candidate-card">
        <h3>{_escape(candidate.candidate.value)}</h3>
        <p class="status status-{_escape(candidate.status.value)}">
          {_escape(candidate.status.value)}
        </p>
        <dl class="metrics">
          <div><dt>Quality</dt><dd>{candidate.quality_score:.2f}/100</dd></div>
          <div><dt>Answered</dt><dd>{candidate.answered_cases}/{candidate.scored_cases}</dd></div>
          <div><dt>Partial failures</dt><dd>{candidate.partial_failures}</dd></div>
          <div><dt>Generated cases</dt><dd>{candidate.generated_cases}</dd></div>
          <div><dt>Source support</dt><dd>{candidate.source_support_rate * 100:.1f}%</dd></div>
          <div><dt>Measured cost</dt><dd>{_escape(cost)}</dd></div>
          <div><dt>Query latency</dt><dd>{_escape(latency)}</dd></div>
          <div><dt>Workspace</dt><dd>
            {workspace} bytes
          </dd></div>
        </dl>
        <p class="muted">Cost telemetry: {_escape(candidate.cost_status.value)}
          · Operating burden: {_escape(burden)}
        </p>
        <p class="muted">{_escape("; ".join(candidate.eligibility_reasons))}</p>
      </article>
    """


def _evidence_row(item: CandidateCaseEvidence) -> str:
    source_links = (
        " ".join(f'<a href="{_escape(url)}">source</a>' for url in item.source_urls) or "none"
    )
    return (
        f"<tr><th scope='row'>{_escape(item.case_id)}</th>"
        f"<td>{_escape(item.candidate.value)}</td><td>{_escape(item.status.value)}</td>"
        f"<td>{item.score:.2f}</td><td>{item.cited_claims}/{item.required_claims}</td>"
        f"<td>{source_links}</td></tr>"
    )


def render_report(artifact: ComparisonArtifact) -> str:
    """Render escaped semantic HTML with no external assets or executable code."""
    artifact = _redact_artifact(artifact)
    candidate_cards = "".join(_candidate_card(candidate) for candidate in artifact.candidates)
    coverage_rows = "".join(
        f"<tr><th scope='row'>{_escape(record.source.value)}</th>"
        f"<td>{_escape(record.completeness.value)}</td><td>{record.discovered}</td>"
        f"<td>{record.fetched}</td><td>{record.denied}</td></tr>"
        for record in artifact.coverage
    )
    evidence_rows = "".join(_evidence_row(item) for item in artifact.evidence)
    warnings = "".join(f"<li>{_escape(warning)}</li>" for warning in artifact.warnings)
    paths = "".join(
        f"<li><code>{_escape(name)}</code>: <code>{_escape(path)}</code></li>"
        for name, path in sorted(artifact.artifact_paths.items())
    )
    source_provenance = "".join(
        f"<li><code>{_escape(source.source)}</code>: "
        f"<code>{_escape(source.mutability.value)}</code></li>"
        for source in artifact.provenance.sources
    )
    latency_spans = "".join(
        f"<li><code>{_escape(span.name.value)}</code>"
        f"{f' ({_escape(span.candidate.value)})' if span.candidate is not None else ''}: "
        f"<code>{_escape(_latency_value(span.duration_ms))}</code></li>"
        for span in artifact.provenance.latency_spans
    )
    chat_provider = artifact.provenance.chat.provider or "unavailable"
    chat_model = artifact.provenance.chat.model or "unavailable"
    embedding_backend = artifact.provenance.embedding.backend or "unavailable"
    embedding_quality = (
        artifact.provenance.embedding.quality.value
        if artifact.provenance.embedding.quality is not None
        else "unavailable"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AutoBrain comparison — {_escape(artifact.verdict.value)}</title>
  <style>
    :root {{ color-scheme: dark; --bg:#111827; --panel:#1f2937; --panel-2:#263448;
      --text:#f8fafc; --muted:#b6c2d1; --line:#64748b; --cyan:#67e8f9;
      --green:#86efac; --amber:#fcd34d; --red:#fca5a5; }}
    * {{ box-sizing:border-box; }}
    html, body {{ width:100%; max-width:100%; }}
    body {{ margin:0; background:var(--bg); color:var(--text);
      font:16px/1.55 system-ui,sans-serif; }}
    a {{ color:var(--cyan); text-underline-offset:3px; overflow-wrap:anywhere; }}
    a:focus-visible, :focus-visible {{ outline:3px solid var(--amber); outline-offset:3px; }}
    .page {{ width:100%; min-width:0; max-width:1180px; margin:0 auto; padding:32px 20px 64px; }}
    header, main, footer, section, .cards, .two-col, .candidate-card, .decision,
    .warning, .warning ul, .warning li, p, h1, h2, h3, dl, dt, dd, code {{
      min-width:0; max-width:100%;
      overflow-wrap:anywhere; word-break:break-word;
    }}
    header {{ border-bottom:1px solid var(--line); padding-bottom:24px; margin-bottom:24px; }}
    h1,h2,h3 {{ line-height:1.2; margin-top:0; }}
    h1 {{ font-size:clamp(2rem,5vw,4rem); max-width:850px; }}
    h2 {{ margin-top:40px; font-size:1.6rem; }}
    .eyebrow {{ color:var(--cyan); font-weight:700; letter-spacing:.08em;
      text-transform:uppercase; }}
    .decision {{ background:var(--panel-2); border:1px solid var(--cyan);
      border-radius:12px; padding:20px; }}
    .decision strong {{ color:var(--green); font-size:1.3rem; }}
    .muted {{ color:var(--muted); }}
    .warning {{ border-left:4px solid var(--amber); padding:12px 16px; background:#332b18; }}
    .cards {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:16px; }}
    .cards > *, .two-col > *, .metrics > * {{ min-width:0; }}
    .candidate-card {{ background:var(--panel); border:1px solid var(--line);
      border-radius:12px; padding:18px; }}
    .status {{ font-weight:700; }}
    .status-OK {{ color:var(--green); }} .status-FAILED {{ color:var(--red); }}
    .metrics {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; }}
    dt {{ color:var(--muted); font-size:.85rem; }}
    dd {{ margin:0; font-variant-numeric:tabular-nums; }}
    .table-wrap {{ width:100%; min-width:0; max-width:100%; overflow-x: auto; overflow-y:hidden;
      border:1px solid var(--line); border-radius:8px; }}
    table {{ width:100%; min-width:620px; border-collapse:collapse; }}
    caption {{ text-align:left; padding:12px; color:var(--muted); }}
    th,td {{ border-top:1px solid var(--line); padding:10px 12px;
      text-align:left; vertical-align:top; overflow-wrap:anywhere; word-break:break-word; }}
    th {{ color:var(--muted); }}
    code {{ overflow-wrap:anywhere; }}
    .two-col {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
    footer {{ border-top:1px solid var(--line); margin-top:40px; padding-top:20px; }}
    @media (max-width:1023px) {{ .cards {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} }}
    @media (max-width:639px) {{ .page {{ padding:24px 14px 48px; }}
      .cards,.two-col {{ grid-template-columns:1fr; }}
      .metrics {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} h1 {{ font-size:2.2rem; }} }}
  </style>
</head>
<body>
  <div class="page">
    <header>
      <p class="eyebrow">AutoBrain local evaluation</p>
      <h1>Candidate comparison: {_escape(artifact.verdict.value)}</h1>
      <p class="muted">Run <code>{_escape(artifact.run_id)}</code> · corpus
        <code>{_escape(artifact.corpus_hash)}</code></p>
      <div class="decision" aria-live="polite" aria-label="Recommendation">
        <strong>{_escape(artifact.verdict.value)}</strong>
        <p>{_escape(artifact.decision.rationale)}</p>
        <p class="muted">Run status: {_escape(artifact.status.value)}</p>
        <p class="muted">Decision status: {_escape(artifact.decision.status.value)}</p>
      </div>
    </header>
    <main aria-label="Comparison report">
      <section aria-labelledby="provenance">
        <h2 id="provenance">Benchmark provenance</h2><div class="two-col">
        <div><dl class="metrics">
          <div><dt>Chat provider</dt><dd>{_escape(chat_provider)}</dd></div>
          <div><dt>Chat model</dt><dd>{_escape(chat_model)}</dd></div>
          <div><dt>Embedding backend</dt><dd>{_escape(embedding_backend)}</dd></div>
          <div><dt>Embedding quality</dt><dd>{_escape(embedding_quality)}</dd></div>
          <div><dt>Usage source</dt><dd>
            {_escape(artifact.provenance.usage_source.value)}
          </dd></div>
        </dl></div>
        <div><p class="muted">Source mutability</p>
          <ul>{source_provenance or "<li>unavailable</li>"}</ul>
        <p class="muted">Named latency spans</p>
          <ul>{latency_spans or "<li>unavailable</li>"}</ul></div>
      </div></section>
      <section aria-labelledby="candidates">
        <h2 id="candidates">Candidate evidence</h2>
        <div class="cards">{candidate_cards}</div>
      </section>
      <section aria-labelledby="method">
        <h2 id="method">Methodology and caveats</h2><div class="two-col">
        <div><p>Quality is retrieval Recall over gold source IDs, scaled to 0-100.
          Extra retrieved documents do not raise the score. Generated answer text
          is not scored.</p>
        <p>The evaluator uses <strong>gpt-5-mini</strong> at temperature 0.
          Using the same model family for evaluation can bias comparisons toward
          that model family; this report makes no statistical-significance or
          universal-best claim.</p></div>
        <div><p>Cost is measured only when proxy/native usage reconciles against
          price sheet <code>{_escape(artifact.price_sheet_version or "unknown")}</code>.
          Incomplete cost is shown as unknown, never as zero.</p>
        <p>Operating burden and workspace bytes are local evidence only and do
          not represent production capacity or hosted-service economics.</p></div>
      </div></section>
      <section aria-labelledby="coverage">
        <h2 id="coverage">Corpus coverage</h2><div class="table-wrap"><table>
          <caption>Source crawl coverage; UNKNOWN remains unknown.</caption>
        <thead><tr><th scope="col">Source</th><th scope="col">Completeness</th>
          <th scope="col">Discovered</th><th scope="col">Fetched</th>
          <th scope="col">Denied</th></tr></thead><tbody>{coverage_rows}</tbody>
      </table></div></section>
      <section aria-labelledby="cases">
        <h2 id="cases">Per-case evidence</h2><div class="table-wrap"><table>
          <caption>Answers expose safe citations and scores only.</caption>
        <thead><tr><th scope="col">Case</th><th scope="col">Candidate</th>
          <th scope="col">Status</th><th scope="col">Score</th>
          <th scope="col">Citations</th><th scope="col">Sources</th></tr></thead>
          <tbody>{evidence_rows}</tbody>
      </table></div></section>
      <section aria-labelledby="warnings">
        <h2 id="warnings">Warnings and limitations</h2><div class="warning">
          <ul>{warnings or "<li>No additional warnings recorded.</li>"}</ul>
        </div>
      </section>
      <section aria-labelledby="artifacts">
        <h2 id="artifacts">Evidence paths</h2>
        <ul>{paths or "<li>No additional artifact paths recorded.</li>"}</ul>
      </section>
    </main>
    <footer><p class="muted">Local-only report. No analytics, external scripts,
      fonts, or candidate-facing oracle content.</p></footer>
  </div>
</body>
</html>
"""


def write_artifacts(artifact: ComparisonArtifact, output_dir: Path) -> ReportArtifacts:
    """Write comparison JSON and HTML atomically and return their content hashes."""
    comparison_bytes = _canonical_json(artifact)
    html_bytes = render_report(artifact).encode("utf-8")
    comparison_path = output_dir / "comparison.json"
    report_path = output_dir / "report.html"
    _atomic_write(comparison_path, comparison_bytes)
    _atomic_write(report_path, html_bytes)
    return ReportArtifacts(
        comparison_json=comparison_path,
        report_html=report_path,
        comparison_sha256=hashlib.sha256(comparison_bytes).hexdigest(),
        report_sha256=hashlib.sha256(html_bytes).hexdigest(),
    )
