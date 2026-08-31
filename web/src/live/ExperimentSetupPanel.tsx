/**
 * Experiment setup wizard.
 *
 * Lets an operator configure and submit a retrieval Preview against the local
 * job boundary without typing a CLI command. Every requirement that is not yet
 * satisfied is shown as guidance, and the submit control stays disabled until
 * readiness is genuinely READY - the same rule the Python contract enforces.
 */

import { useCallback, useMemo, useReducer, useState } from "react";
import { SOURCE_CAPABILITIES, type SourceImportFormat } from "../data/sourceContracts";
import type { CandidateId } from "../data/experimentContracts";
import {
  ExperimentBoundaryError,
  describeExperimentError,
  submitExperiment,
  type ExperimentLifecycleView,
} from "./experimentClient";
import {
  SELECTABLE_CANDIDATES,
  buildExperimentRequest,
  initialSetup,
  reduceSetup,
  setupReadiness,
  type SubscriptionId,
} from "./setupModel";
import "./experimentSetup.css";

const CANDIDATE_LABELS: Record<CandidateId, { name: string; detail: string }> = {
  gbrain: { name: "GBrain", detail: "Native graph retrieval over the frozen corpus." },
  "llm-wiki": { name: "LLM Wiki", detail: "Native wiki retrieval; no re-indexing is performed." },
  mem0: { name: "Mem0", detail: "Open-source memory retrieval lifecycle." },
};

const SUBSCRIPTION_LABELS: Record<SubscriptionId, { name: string; detail: string }> = {
  codex: { name: "Codex", detail: "Uses your existing plan through the local runner." },
  claude: { name: "Claude", detail: "Uses your existing plan through the local runner." },
};

const FORMATS: readonly SourceImportFormat[] = ["JSON", "JSONL"];

/** Tone used to pick a status token; mirrors the local runner panel. */
type Tone = "success" | "warning" | "danger" | "neutral";

interface Submission {
  tone: Tone;
  headline: string;
  detail: string;
}

function describeLifecycle(lifecycle: ExperimentLifecycleView): Submission {
  if (lifecycle.status === "RUNNING") {
    return {
      tone: "success",
      headline: `Preview ${lifecycle.status}`,
      detail: `The local runner accepted experiment ${lifecycle.experiment_id} and is scoring it now.`,
    };
  }
  if (lifecycle.status === "FAILED") {
    return {
      tone: "danger",
      headline: `Preview ${lifecycle.status}`,
      detail:
        "The local runner rejected this configuration during validation. Adjust the setup and submit again.",
    };
  }
  return {
    tone: "warning",
    headline: `Preview ${lifecycle.status}`,
    detail: `The local runner stopped at ${lifecycle.status.toLowerCase()} and did not start scoring.`,
  };
}

export function ExperimentSetupPanel({
  baseUrl,
  onSubmitted,
}: {
  baseUrl: string;
  /**
   * Reports the identity of an accepted Preview so the results route can read
   * it back without the operator copying an id by hand. Called only after the
   * boundary accepted the request, never on a rejection.
   */
  onSubmitted?: (experiment: { experimentId: string; derivedBenchmarkHash: string }) => void;
}) {
  const [setup, dispatch] = useReducer(reduceSetup, undefined, initialSetup);
  const [submission, setSubmission] = useState<Submission | null>(null);
  const [pending, setPending] = useState(false);
  const readiness = useMemo(() => setupReadiness(setup), [setup]);

  const submit = useCallback(async () => {
    setPending(true);
    try {
      const request = buildExperimentRequest(setup, crypto.randomUUID());
      setSubmission(describeLifecycle(await submitExperiment(baseUrl, request)));
      onSubmitted?.({
        experimentId: request.experiment_id,
        derivedBenchmarkHash: request.identity.benchmark_sha256,
      });
    } catch (cause) {
      if (cause instanceof ExperimentBoundaryError) {
        const described = describeExperimentError(cause.code, cause.detail);
        setSubmission({ tone: "danger", headline: described.title, detail: described.guidance });
      } else {
        setSubmission({
          tone: "danger",
          headline: "Could not reach the local runner",
          detail: `Start it on your machine, then submit again. ${
            cause instanceof Error ? cause.message : String(cause)
          }`,
        });
      }
    } finally {
      setPending(false);
    }
  }, [baseUrl, setup, onSubmitted]);

  return (
    <section className="wizard">
      <div className="wizard__head">
        <div>
          <h2 className="wizard__title">New experiment</h2>
          <p className="wizard__origin">{baseUrl}</p>
        </div>
        <span className="wizard__scope">Local boundary · not hosted</span>
      </div>

      <ol className="wizard__steps">
        <li className="wizard-step">
          <p className="wizard-step__index">01</p>
          <div className="wizard-step__body">
            <h3 className="wizard-step__title">Input source</h3>
            <p className="wizard-step__hint">
              Pick where the frozen corpus comes from. Nothing is uploaded; the local runner reads
              it on your machine.
            </p>
            <div className="wizard-options">
              {SOURCE_CAPABILITIES.map((capability) => (
                <button
                  key={capability.id}
                  type="button"
                  className="wizard-option"
                  data-testid={`source-option-${capability.id}`}
                  aria-pressed={setup.source === capability.id}
                  data-readiness={capability.readiness}
                  onClick={() => dispatch({ type: "source/select", source: capability.id })}
                >
                  <span className="wizard-option__title">{capability.label}</span>
                  <span className="wizard-option__detail">{capability.detail}</span>
                  {capability.readiness !== "READY" && (
                    <span className="wizard-option__flag">{capability.readiness}</span>
                  )}
                </button>
              ))}
            </div>

            <div className="wizard-field">
              <span className="wizard-field__label">Import format</span>
              <div className="wizard-options wizard-options--inline">
                {FORMATS.map((format) => (
                  <button
                    key={format}
                    type="button"
                    className="wizard-option"
                    data-testid={`format-option-${format}`}
                    aria-pressed={setup.format === format}
                    onClick={() => dispatch({ type: "source/format", format })}
                  >
                    <span className="wizard-option__title">{format}</span>
                  </button>
                ))}
              </div>
            </div>

            <label className="wizard-field">
              <span className="wizard-field__label">Normalized records</span>
              <textarea
                className="wizard-textarea"
                data-testid="source-payload"
                rows={5}
                spellCheck={false}
                placeholder={
                  setup.format === "JSONL"
                    ? '{"source_id": "doc-1", "title": "Refund policy", "text": "…"}'
                    : '[{"source_id": "doc-1", "title": "Refund policy", "text": "…"}]'
                }
                value={setup.payload}
                onChange={(event) =>
                  dispatch({ type: "source/payload", payload: event.target.value })
                }
              />
              <span className="wizard-field__hint" data-testid="corpus-summary">
                {setup.parsed === null
                  ? "Leave empty to use the source as the local runner already has it."
                  : `${setup.parsed.records.length} documents will be frozen for this run.`}
              </span>
            </label>
          </div>
        </li>

        <li className="wizard-step">
          <p className="wizard-step__index">02</p>
          <div className="wizard-step__body">
            <h3 className="wizard-step__title">Brain candidates</h3>
            <p className="wizard-step__hint">
              Every selected Brain is measured on the same frozen corpus using its own native
              retrieval.
            </p>
            <div className="wizard-options">
              {SELECTABLE_CANDIDATES.map((candidate) => (
                <button
                  key={candidate}
                  type="button"
                  className="wizard-option"
                  data-testid={`candidate-option-${candidate}`}
                  aria-pressed={setup.candidates.includes(candidate)}
                  onClick={() => dispatch({ type: "candidate/toggle", candidate })}
                >
                  <span className="wizard-option__title">{CANDIDATE_LABELS[candidate].name}</span>
                  <span className="wizard-option__detail">
                    {CANDIDATE_LABELS[candidate].detail}
                  </span>
                </button>
              ))}
            </div>
          </div>
        </li>

        <li className="wizard-step">
          <p className="wizard-step__index">03</p>
          <div className="wizard-step__body">
            <h3 className="wizard-step__title">AI subscription</h3>
            <p className="wizard-step__hint">
              The local runner uses your existing plan. No key is entered in or stored by the
              browser.
            </p>
            <div className="wizard-options">
              {(Object.keys(SUBSCRIPTION_LABELS) as SubscriptionId[]).map((subscription) => (
                <button
                  key={subscription}
                  type="button"
                  className="wizard-option"
                  data-testid={`subscription-option-${subscription}`}
                  aria-pressed={setup.subscription === subscription}
                  onClick={() => dispatch({ type: "subscription/select", subscription })}
                >
                  <span className="wizard-option__title">
                    {SUBSCRIPTION_LABELS[subscription].name}
                  </span>
                  <span className="wizard-option__detail">
                    {SUBSCRIPTION_LABELS[subscription].detail}
                  </span>
                </button>
              ))}
            </div>
          </div>
        </li>

        <li className="wizard-step">
          <p className="wizard-step__index">04</p>
          <div className="wizard-step__body">
            <h3 className="wizard-step__title">Evaluator</h3>
            <p className="wizard-step__hint">
              Preview compares retrieval quality only. Final answer generation is a separate mode
              and is not part of this run.
            </p>
            <div className="wizard-options">
              <button
                type="button"
                className="wizard-option"
                data-testid="evaluator-option-retrieval"
                aria-pressed={true}
                onClick={() => undefined}
              >
                <span className="wizard-option__title">Retrieval only</span>
                <span className="wizard-option__detail">
                  Recall, precision, missing evidence, latency, and cost status.
                </span>
              </button>
            </div>
          </div>
        </li>
      </ol>

      <div className="wizard__footer">
        <div
          className="wizard-readiness"
          data-testid="readiness-blockers"
          data-state={readiness.state}
        >
          {readiness.blockers.length === 0 ? (
            <p className="wizard-readiness__ready">
              Ready to submit. The local runner will validate this setup before it scores anything.
            </p>
          ) : (
            <ul className="wizard-readiness__list">
              {readiness.blockers.map((blocker) => (
                <li key={blocker.code}>
                  <strong>{blocker.title}</strong>
                  <span>{blocker.guidance}</span>
                </li>
              ))}
            </ul>
          )}
        </div>

        <button
          type="button"
          className="wizard__submit"
          data-testid="submit-preview"
          disabled={readiness.state !== "READY" || pending}
          onClick={() => void submit()}
        >
          {pending ? "Submitting…" : "Submit Preview"}
        </button>
      </div>

      {submission !== null && (
        <div className="wizard-status" data-testid="submission-status" data-tone={submission.tone}>
          <p className="wizard-status__headline">{submission.headline}</p>
          <p className="wizard-status__detail">{submission.detail}</p>
        </div>
      )}
    </section>
  );
}
