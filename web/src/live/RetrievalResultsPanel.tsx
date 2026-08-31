/**
 * Retrieval results and comparison panel.
 *
 * Renders one experiment's measured retrieval behavior per Brain, the identity
 * of the evidence it was measured against, and - when a previous run shares
 * that identity - a before/after comparison.
 *
 * Every claim on this surface is traceable to the engine projection. The panel
 * derives nothing on its own: ranking, confidence, comparability, and recovery
 * all come from `resultsModel`, which keeps a missing measurement null rather
 * than rendering a flattering zero. The confidence notice is always present on
 * a completed run, because a number without its status is a claim this surface
 * is not allowed to make.
 */

import { useCallback, useState } from "react";
import type { CandidateId } from "../data/types";
import { ExperimentBoundaryError, describeExperimentError } from "./experimentClient";
import type { RunOutcome, RunProjection } from "./runClient";
import { awaitExperimentResult } from "./resultsClient";
import {
  compareRetrieval,
  describeRecovery,
  evidenceConfidence,
  resolveBenchmarkHash,
  retrievalRows,
  type ComparisonRow,
  type RetrievalComparison,
  type RetrievalRow,
} from "./resultsModel";
import "./retrievalResults.css";

/** Stable candidate identity colors, shared with the rest of the product. */
const CANDIDATE_COLORS: Record<CandidateId, string> = {
  gbrain: "var(--candidate-gbrain)",
  "llm-wiki": "var(--candidate-llm-wiki)",
  mem0: "var(--candidate-mem0)",
};

function formatRate(value: number | null): string {
  return value === null ? "—" : `${(value * 100).toFixed(1)}%`;
}

function formatSignedRate(value: number | null): string {
  if (value === null) return "—";
  const points = value * 100;
  return `${points > 0 ? "+" : ""}${points.toFixed(1)}`;
}

function formatCount(value: number | null): string {
  return value === null ? "—" : String(value);
}

function formatMs(value: number | null): string {
  return value === null ? "—" : `${Math.round(value)} ms`;
}

function formatCost(row: RetrievalRow): string {
  if (row.costUsd !== null) return `$${row.costUsd.toFixed(2)}`;
  return row.costStatus === "COST_UNAVAILABLE" ? "unavailable" : "incomplete";
}

function shortHash(value: string): string {
  return value.slice(0, 12);
}

function CandidateName({ candidate }: { candidate: CandidateId }) {
  return (
    <span className="results-table__candidate">
      <i
        className="results-table__swatch"
        style={{ background: CANDIDATE_COLORS[candidate] }}
        aria-hidden="true"
      />
      {candidate}
    </span>
  );
}

/* ------------------------------------------------------------------ tables */

function RetrievalTable({ rows }: { rows: RetrievalRow[] }) {
  return (
    <div className="results-scroll">
      <table className="results-table" data-testid="retrieval-table">
        <thead>
          <tr>
            <th>Brain</th>
            <th className="num">Recall</th>
            <th className="num">Missing evidence</th>
            <th className="num">Precision</th>
            <th className="num">Noise</th>
            <th className="num">p50</th>
            <th className="num">p95</th>
            <th className="num">Cost</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr
              key={row.candidate}
              data-testid={`retrieval-row-${row.candidate}`}
              data-leader={String(row.leader)}
              data-comparable={String(row.comparable)}
            >
              <td>
                <CandidateName candidate={row.candidate} />
                {!row.comparable && (
                  <small className="results-table__note">Not scored — {row.statusNote}</small>
                )}
              </td>
              <td className="num">{formatRate(row.recall)}</td>
              <td className="num">{formatCount(row.missingEvidence)}</td>
              <td className="num">{formatRate(row.precision)}</td>
              <td className="num">{formatRate(row.noise)}</td>
              <td className="num">{formatMs(row.latencyP50Ms)}</td>
              <td className="num">{formatMs(row.latencyP95Ms)}</td>
              <td className="num">{formatCost(row)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ComparisonTable({ rows }: { rows: ComparisonRow[] }) {
  return (
    <div className="results-scroll">
      <table className="results-table" data-testid="comparison-table">
        <thead>
          <tr>
            <th>Brain</th>
            <th className="num">Before</th>
            <th className="num">After</th>
            <th className="num">Change</th>
            <th>Direction</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr
              key={row.candidate}
              data-testid={`comparison-row-${row.candidate}`}
              data-direction={row.direction}
            >
              <td>
                <CandidateName candidate={row.candidate} />
              </td>
              <td className="num">{formatRate(row.beforeRecall)}</td>
              <td className="num">{formatRate(row.afterRecall)}</td>
              <td className="num">
                <span className="results-table__delta" data-direction={row.direction}>
                  {formatSignedRate(row.recallDelta)}
                </span>
              </td>
              <td>{row.direction}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ------------------------------------------------------------------ panel */

interface ResultsError {
  headline: string;
  detail: string;
}

export function RetrievalResultsPanel({
  baseUrl,
  experimentId,
  /** The wizard's locally derived benchmark fingerprint, if one exists. */
  derivedBenchmarkHash,
}: {
  baseUrl: string;
  /** Null until a Preview has been submitted from the wizard. */
  experimentId: string | null;
  derivedBenchmarkHash: string | null;
}) {
  const [outcome, setOutcome] = useState<RunOutcome | null>(null);
  const [previous, setPrevious] = useState<RunProjection | null>(null);
  const [comparison, setComparison] = useState<RetrievalComparison | null>(null);
  const [error, setError] = useState<ResultsError | null>(null);
  const [pending, setPending] = useState(false);

  const load = useCallback(async () => {
    if (experimentId === null) return;
    setPending(true);
    try {
      const next = await awaitExperimentResult(baseUrl, experimentId);
      // Keep the run we were showing as the "before" side, so a second read is
      // immediately comparable against the first without another request.
      setPrevious(outcome?.status === "SUCCEEDED" ? outcome.projection : null);
      setOutcome(next);
      setComparison(null);
      setError(null);
    } catch (cause) {
      if (cause instanceof ExperimentBoundaryError) {
        const described = describeExperimentError(cause.code, cause.detail);
        setError({ headline: described.title, detail: described.guidance });
      } else {
        setError({
          headline: "Could not reach the local runner",
          detail: `Start it on your machine, then read the result again. ${
            cause instanceof Error ? cause.message : String(cause)
          }`,
        });
      }
      setOutcome(null);
      setComparison(null);
    } finally {
      setPending(false);
    }
  }, [baseUrl, experimentId, outcome]);

  const projection = outcome?.status === "SUCCEEDED" ? outcome.projection : null;

  const compareWith = useCallback(
    (before: RunProjection | null) => {
      if (before === null || projection === null) return;
      setComparison(compareRetrieval(before, projection));
    },
    [projection],
  );

  const recovery = outcome === null ? null : describeRecovery(outcome);
  const benchmark = resolveBenchmarkHash(derivedBenchmarkHash, outcome);

  return (
    <section className="results">
      <div className="results__head">
        <div>
          <h2 className="results__title">Retrieval results</h2>
          <p className="results__origin">{baseUrl}</p>
        </div>
        <span className="results__scope">Local boundary · not hosted</span>
      </div>

      <div className="results__actions">
        <button
          type="button"
          className="btn btn--primary"
          data-testid="load-results"
          disabled={pending || experimentId === null}
          onClick={() => void load()}
        >
          {pending ? "Reading…" : "Read result"}
        </button>
        {projection !== null && (
          <>
            <button
              type="button"
              className="btn btn--secondary"
              data-testid="compare-with-current"
              onClick={() => compareWith(projection)}
            >
              Compare with itself
            </button>
            <button
              type="button"
              className="btn btn--secondary"
              data-testid="compare-with-previous"
              disabled={previous === null}
              onClick={() => compareWith(previous)}
            >
              Compare with previous run
            </button>
          </>
        )}
      </div>

      {error !== null && (
        <div className="results-error" data-testid="results-error">
          <p className="results-error__headline">{error.headline}</p>
          <p className="results-error__detail">{error.detail}</p>
        </div>
      )}

      {recovery !== null && (
        <div className="results-recovery" data-testid="recovery-notice" data-tone={recovery.tone}>
          <p className="results-recovery__headline">{recovery.headline}</p>
          <p className="results-recovery__detail">{recovery.detail}</p>
          <div className="results-recovery__actions">
            {recovery.actions.map((action) =>
              // Rerun is the only action this panel can perform, so it is the
              // only button. The rest are guidance pointing at surfaces this
              // panel does not own; rendering them as disabled buttons would
              // read as broken controls rather than as next steps.
              action.id === "rerun" ? (
                <button
                  key={action.id}
                  type="button"
                  className="results-recovery__action"
                  data-testid={`recovery-action-${action.id}`}
                  disabled={pending || experimentId === null}
                  onClick={() => void load()}
                >
                  <strong>{action.label}</strong>
                  <span>{action.guidance}</span>
                </button>
              ) : (
                <div
                  key={action.id}
                  className="results-recovery__action results-recovery__action--note"
                  data-testid={`recovery-action-${action.id}`}
                >
                  <strong>{action.label}</strong>
                  <span>{action.guidance}</span>
                </div>
              ),
            )}
          </div>
        </div>
      )}

      {projection !== null && (
        <>
          {(() => {
            const confidence = evidenceConfidence(projection);
            return (
              <div
                className="results-confidence"
                data-testid="evidence-confidence"
                data-level={confidence.level}
                data-recommendation-grade={String(confidence.recommendationGrade)}
              >
                <p className="results-confidence__label">{confidence.label}</p>
                {projection.rationale.length > 0 && (
                  <p className="results-confidence__rationale">{projection.rationale}</p>
                )}
                <p className="results-confidence__caveat">{confidence.caveat}</p>
                {confidence.warnings.length > 0 && (
                  <ul className="results-confidence__warnings">
                    {confidence.warnings.map((warning) => (
                      <li key={warning}>{warning}</li>
                    ))}
                  </ul>
                )}
              </div>
            );
          })()}

          <dl className="results-provenance" data-testid="result-provenance">
            <div>
              <dt>Run</dt>
              <dd>{projection.run_id}</dd>
            </div>
            <div>
              <dt>Frozen corpus</dt>
              <dd>{shortHash(projection.corpus_hash)}</dd>
            </div>
            <div data-testid="benchmark-identity" data-source={benchmark.source} data-placeholder={String(benchmark.placeholder)}>
              <dt>Benchmark</dt>
              <dd>
                {benchmark.sha256 === null ? "—" : shortHash(benchmark.sha256)}
                {benchmark.placeholder && (
                  <span className="results-provenance__flag"> placeholder</span>
                )}
              </dd>
            </div>
            <div>
              <dt>Projection</dt>
              <dd>v{projection.schema_version}</dd>
            </div>
          </dl>

          <div className="results-section">
            <h3 className="results-section__title">Per-Brain retrieval</h3>
            <p className="results-section__hint">
              Every Brain was measured on the same frozen corpus using its own native retrieval.
              Recall is the share of scored cases that produced grounded evidence; missing evidence
              is the remainder. A dash means the engine reported no measurement.
            </p>
            <RetrievalTable rows={retrievalRows(projection)} />
          </div>

          {comparison !== null && (
            <div className="results-section">
              <h3 className="results-section__title">Before and after</h3>
              <p className="results-section__hint">
                {comparison.beforeRunId} → {comparison.afterRunId}
              </p>
              {comparison.comparable ? (
                <ComparisonTable rows={comparison.rows} />
              ) : (
                <p className="results-blocker" data-testid="comparison-blocker">
                  {comparison.blocker}
                </p>
              )}
            </div>
          )}
        </>
      )}

      {outcome === null && error === null && (
        <p className="results__empty" data-testid="results-empty">
          {experimentId === null
            ? "No Preview has been submitted yet. Configure one in the experiment wizard, then read its result here."
            : "No result has been read yet. Read the result of the Preview you submitted."}
        </p>
      )}
    </section>
  );
}
