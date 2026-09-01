/**
 * Local runner panel.
 *
 * Renders a run outcome fetched from the loopback fixture. This is a separate
 * surface from the synthetic demo in `src/data`, which is left untouched, and
 * it is labeled as a local fixture rather than a hosted deployment.
 */

import type { RunOutcome, RunSummary } from "./runClient";
import { summarize } from "./runClient";
import "./live.css";

function formatMs(value: number | null): string {
  return value === null ? "unavailable" : `${Math.round(value)} ms`;
}

function formatCost(value: number | null): string {
  return value === null ? "unavailable" : `$${value.toFixed(4)}`;
}

function formatRate(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function shortHash(value: string): string {
  return value.slice(0, 12);
}

export function LocalRunPanel({
  outcome,
  baseUrl,
}: {
  outcome: RunOutcome | null;
  baseUrl: string;
}) {
  if (outcome === null) {
    return (
      <section className="live-panel">
        <div className="live-panel__head">
          <div>
            <h2 className="live-panel__title">Local runner</h2>
            <p className="live-panel__origin">{baseUrl}</p>
          </div>
          <span className="live-panel__scope">Local fixture · not hosted</span>
        </div>
        <p className="live-panel__empty">
          No local run has been read yet. Start one with <code>autobrain serve</code>.
        </p>
      </section>
    );
  }

  const summary: RunSummary = summarize(outcome);
  const projection = outcome.projection;

  return (
    <section className="live-panel">
      <div className="live-panel__head">
        <div>
          <h2 className="live-panel__title">Local runner</h2>
          <p className="live-panel__origin">{baseUrl}</p>
        </div>
        <span className="live-panel__scope">Local fixture · not hosted</span>
      </div>

      <div className="live-status" data-tone={summary.tone}>
        <div className="live-status__badges">
          <span className="live-chip" data-tone={summary.tone}>
            run {summary.runStatus}
          </span>
          {summary.engineStatus !== null && (
            <span className="live-chip">engine {summary.engineStatus}</span>
          )}
        </div>
        <p className="live-status__headline">{summary.headline}</p>
        <p className="live-status__detail">{summary.detail}</p>
      </div>

      {projection !== null && (
        <>
          <dl className="live-meta">
            <div>
              <dt>Run</dt>
              <dd>{projection.run_id}</dd>
            </div>
            <div>
              <dt>Schema</dt>
              <dd>v{projection.schema_version}</dd>
            </div>
            <div>
              <dt>Corpus</dt>
              <dd>{shortHash(projection.corpus_hash)}</dd>
            </div>
            <div>
              <dt>Benchmark</dt>
              <dd>{shortHash(projection.benchmark_hash)}</dd>
            </div>
          </dl>

          <table className="live-candidates">
            <thead>
              <tr>
                <th>Candidate</th>
                <th>Status</th>
                <th className="num">Quality</th>
                <th className="num">Success</th>
                <th className="num">p95</th>
                <th className="num">Cost</th>
              </tr>
            </thead>
            <tbody>
              {projection.candidates.map((candidate) => (
                <tr key={candidate.candidate}>
                  <td>{candidate.candidate}</td>
                  <td>{candidate.status}</td>
                  <td className="num">{candidate.quality_score.toFixed(1)}</td>
                  <td className="num">{formatRate(candidate.answer_success_rate)}</td>
                  <td className="num">{formatMs(candidate.query_p95_ms)}</td>
                  <td className="num">{formatCost(candidate.total_cost_usd)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </section>
  );
}
