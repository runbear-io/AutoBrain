"""Task 10 local orchestration boundary.

This module intentionally owns the magic-box workflow rather than any
connector or candidate implementation.  Real integrations are injected at
this boundary, which keeps the fake-MCP E2E deterministic and prevents the
CLI from silently falling back to a direct REST crawler or a hosted service.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import uuid
import webbrowser
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any, Protocol, cast

from autobrain.auth.models import Provider
from autobrain.auth.service import ConnectionManager
from autobrain.benchmark import LeakageScanResult, scan_benchmark_leakage
from autobrain.cancellation import RunCancellation, RunCancelled
from autobrain.candidates.gbrain_config import GBrainExecutionConfig
from autobrain.corpus import normalize_raw_items
from autobrain.decision import eligibility_reasons, select_winner
from autobrain.embedding import (
    EmbeddingBackend,
    EmbeddingBackendConfig,
    EmbeddingBackendRegistry,
    build_gemini_embedding_upstream,
    build_openai_embedding_upstream,
    production_embedding_registry,
)
from autobrain.evaluate import evaluate_candidate, evaluate_case
from autobrain.experiment import automatic_experiment_copy
from autobrain.integration_provenance import integration_catalog
from autobrain.lifecycle import CleanupReceipt, cleanup_receipt_complete
from autobrain.model_access import ModelAccess, ModelAccessRegistry, ModelCapability
from autobrain.models import (
    BenchmarkCase,
    BenchmarkProvenance,
    CandidateCaseEvidence,
    CandidateEvaluation,
    CandidateId,
    CandidateObservation,
    ChatProvenance,
    CostStatus,
    CoverageCompleteness,
    CoverageRecord,
    DecisionResult,
    LatencySpan,
    LatencySpanKind,
    NativeCandidateResult,
    SourceKind,
    SourceMutability,
    SourceProvenance,
    Status,
    UsageSource,
    Verdict,
    normalize_safe_source_url,
)
from autobrain.paths import AutoBrainPaths, OccupiedRunError
from autobrain.preflight_support import candidate_pin_matches, load_candidate_pins
from autobrain.report import build_comparison, load_comparison, write_artifacts
from autobrain.secrets import redact
from autobrain.subscription_domain import ProviderId

MAX_CANDIDATES = 3
MIN_BENCHMARK_CASES = 20
MAX_BENCHMARK_CASES = 30
DEFAULT_BUDGET_USD = 25.0
_QUESTION_WORDS = re.compile(
    r"^(?:how|what|when|where|which|who|why|can|could|should|is|are|do|does|"
    r"has|have|will|would)\b",
    re.IGNORECASE,
)
_ACKNOWLEDGEMENT = re.compile(
    r"^(?:thanks|thank you|thx|ty|great|awesome|nice|got it|lgtm|agreed|same|"
    r"welcome|congrats|congratulations|lol|haha|cool|\+1)[!. ]*$",
    re.IGNORECASE,
)
_SOCIAL_CHATTER = re.compile(
    r"\b(?:happy birthday|happy anniversary|welcome to the team|lunch|coffee|"
    r"happy friday|weekend plans|poll|vote|voting|survey|react with)\b",
    re.IGNORECASE,
)
_SPECULATION = re.compile(
    r"^(?:i think|maybe|perhaps|not sure|probably|might be|could be|guessing)\b",
    re.IGNORECASE,
)
_PROMPT_LIKE_SOURCE = re.compile(
    r"\b(?:ignore|disregard|override)\b.*\b(?:instruction|prompt|tool)\b|"
    r"\b(?:system prompt|follow these instructions|call a (?:write|tool))\b",
    re.IGNORECASE,
)
_MARKDOWN_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
_MARKDOWN_BULLET = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)(.+)$")


def _empty_artifact() -> dict[str, Any]:
    return {}


class Connector(Protocol):
    """A read-only source crawler.

    Source-neutral by construction: provider-specific options such as Slack DM
    inclusion belong to the implementer's constructor, never to this seam.
    Cancellation is optional so a connector may be driven outside a run.
    """

    provider: str

    def probe(self, cancellation: RunCancellation | None = None) -> Mapping[str, Any]: ...

    def crawl(self, *, cancellation: RunCancellation | None = None) -> ConnectorSnapshot: ...


@dataclass(frozen=True)
class StageEvent:
    """One immutable stage entry reconstructed from the persisted manifest value."""

    sequence: int
    run_id: str
    name: str
    status: Status
    detail: str
    started_at: str

    def as_manifest_entry(self) -> dict[str, int | str]:
        return {
            "sequence": self.sequence,
            "run_id": self.run_id,
            "name": self.name,
            "status": self.status.value,
            "detail": self.detail,
            "started_at": self.started_at,
        }


StageEventSink = Callable[[StageEvent], None]


class Candidate(Protocol):
    candidate_id: str

    def run(self, context: CandidateContext) -> CandidateOutcome: ...

    def cleanup(self) -> CleanupReceipt | None: ...


def retain_selected_candidates(
    candidates: Sequence[Candidate],
    *,
    selected_ids: set[str],
) -> tuple[Candidate, ...]:
    """Keep selected native candidates and settle every discarded lifecycle."""
    selected: list[Candidate] = []
    cleanup_errors: list[str] = []
    for candidate in candidates:
        if candidate.candidate_id in selected_ids:
            selected.append(candidate)
            continue
        try:
            candidate.cleanup()
        except Exception as exc:
            cleanup_errors.append(f"{candidate.candidate_id}: {exc}")
    if cleanup_errors:
        for candidate in selected:
            try:
                candidate.cleanup()
            except Exception as exc:
                cleanup_errors.append(f"{candidate.candidate_id}: {exc}")
        raise RuntimeError("discarded candidate cleanup failed: " + "; ".join(cleanup_errors))
    return tuple(selected)


@dataclass(frozen=True)
class ConnectorSnapshot:
    provider: str
    documents: Sequence[Mapping[str, Any]]
    coverage: Mapping[str, Any]


@dataclass(frozen=True)
class CandidateContext:
    documents: tuple[Mapping[str, Any], ...]
    questions: tuple[str, ...]
    case_ids: tuple[str, ...]
    cancellation: RunCancellation = field(default_factory=RunCancellation)

    @property
    def normalized_documents(self) -> tuple[Any, ...]:
        from autobrain.models import NormalizedDocument

        return tuple(
            normalize_raw_items(
                [
                    document if isinstance(document, NormalizedDocument) else dict(document)
                    for document in self.documents
                ]
            )
        )


@dataclass(frozen=True)
class EvaluatorCaseRecord:
    case: BenchmarkCase
    reference_text: str = ""
    reply_texts: tuple[str, ...] = ()
    reference_confidence: float = 1.0


@dataclass(frozen=True)
class CorpusBoundary:
    """Keep candidate-visible documents separate from evaluator evidence."""

    candidate_documents: tuple[Mapping[str, Any], ...]
    evaluator_cases: tuple[EvaluatorCaseRecord, ...]


@dataclass(frozen=True)
class CandidateOutcome:
    candidate: str
    status: Status
    score: float = 0.0
    answered_cases: int = 0
    scored_cases: int = 0
    cost_usd: float | None = None
    latency_ms: int = 0
    latency_spans: tuple[LatencySpan, ...] = ()
    detail: str = ""
    artifact: Mapping[str, Any] = field(default_factory=_empty_artifact)
    observations: tuple[CandidateObservation, ...] = ()
    cost_status: CostStatus = CostStatus.COMPLETE
    usage_source: UsageSource = UsageSource.MEASURED
    evaluation: CandidateEvaluation | None = None
    native_result: NativeCandidateResult | None = None


@dataclass(frozen=True)
class RunConfig:
    budget_usd: float = DEFAULT_BUDGET_USD
    max_questions: int = 30
    include_dms: bool = False
    open_report: bool = True
    output: Path | None = None
    run_id: str | None = None
    provider_mode: str = "api"
    embedding_backend: EmbeddingBackend | str | None = None
    embedding_registry: EmbeddingBackendRegistry = field(
        default_factory=production_embedding_registry,
        repr=False,
        compare=False,
    )
    selected_sources: tuple[Provider, ...] = (Provider.SLACK, Provider.NOTION)
    selected_candidates: tuple[CandidateId, ...] = tuple(CandidateId)
    slack_export_path: Path | None = None
    slack_export_sha256: str | None = None
    notion_snapshot_path: Path | None = None
    experiment_title: str = ""
    experiment_description: str = ""
    gbrain_config: GBrainExecutionConfig = field(
        default_factory=GBrainExecutionConfig.quick_start, repr=False
    )

    def __post_init__(self) -> None:
        if self.budget_usd <= 0:
            raise ValueError("budget_usd must be greater than 0")
        if not MIN_BENCHMARK_CASES <= self.max_questions <= MAX_BENCHMARK_CASES:
            raise ValueError(
                f"max_questions must be between {MIN_BENCHMARK_CASES} and {MAX_BENCHMARK_CASES}"
            )
        if self.output is not None:
            AutoBrainPaths.validate_output_root(self.output)
        subscription_modes = {f"{provider.value}-subscription" for provider in ProviderId}
        if self.provider_mode not in {"api", *subscription_modes}:
            raise ValueError("provider_mode must be api or <codex|claude|kimi|grok>-subscription")
        requested_embedding = self.embedding_backend or (
            EmbeddingBackend.OPENAI if self.provider_mode == "api" else EmbeddingBackend.LOCAL_HASH
        )
        embedding = EmbeddingBackendConfig.from_environ(
            {},
            requested=requested_embedding,
            registry=self.embedding_registry,
        )
        object.__setattr__(self, "embedding_backend", embedding.backend)
        if self.provider_mode == "api" and embedding.backend != EmbeddingBackend.OPENAI.value:
            raise ValueError("api provider mode requires embedding_backend=openai")
        if not self.selected_sources:
            raise ValueError("selected_sources must include Slack or Notion")
        if len(set(self.selected_sources)) != len(self.selected_sources):
            raise ValueError("selected_sources must be unique")
        if len(self.selected_candidates) < 2:
            raise ValueError("selected_candidates must include at least two candidates")
        if len(set(self.selected_candidates)) != len(self.selected_candidates):
            raise ValueError("selected_candidates must be unique")
        if (self.slack_export_path is None) != (self.slack_export_sha256 is None):
            raise ValueError("slack export path and sha256 must be provided together")
        if self.slack_export_path is not None and Provider.SLACK not in self.selected_sources:
            raise ValueError("slack export requires Slack to be selected")
        if self.notion_snapshot_path is not None and Provider.NOTION not in self.selected_sources:
            raise ValueError("Notion snapshot requires Notion to be selected")
        automatic_title, automatic_description = automatic_experiment_copy(
            sources=self.selected_sources,
            candidates=self.selected_candidates,
        )
        if not self.experiment_title:
            object.__setattr__(self, "experiment_title", automatic_title)
        if not self.experiment_description:
            object.__setattr__(self, "experiment_description", automatic_description)

    def embedding_config(
        self,
        *,
        environ: Mapping[str, str],
        api_key: str | None = None,
    ) -> EmbeddingBackendConfig:
        return EmbeddingBackendConfig.from_environ(
            environ,
            requested=self.embedding_backend,
            api_key=api_key,
            registry=self.embedding_registry,
        )


@dataclass(frozen=True)
class RunResult:
    run_id: str
    run_dir: Path
    status: Status
    report_path: Path | None
    candidate_results: tuple[CandidateOutcome, ...]
    verdict: str
    event_sink_errors: tuple[str, ...] = ()


def _default_connector_builder(
    manager: ConnectionManager,
    include_dms: bool,
    selected_sources: tuple[Provider, ...] = (Provider.SLACK, Provider.NOTION),
    *,
    slack_export_path: Path | None = None,
    slack_export_sha256: str | None = None,
    notion_snapshot_path: Path | None = None,
) -> Sequence[Connector]:
    from autobrain.production import build_production_connectors

    return build_production_connectors(
        manager,
        include_dms=include_dms,
        providers=selected_sources,
        slack_export_path=slack_export_path,
        slack_export_sha256=slack_export_sha256,
        notion_snapshot_path=notion_snapshot_path,
    )


def _default_candidate_builder(
    run_dir: Path,
    api_key: str,
    budget_usd: float,
    provider_upstream: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    candidate_ids: tuple[CandidateId, ...] = tuple(CandidateId),
    gbrain_config: GBrainExecutionConfig | None = None,
) -> Sequence[Candidate]:
    from autobrain.production import build_production_candidates

    return build_production_candidates(
        run_dir,
        api_key=api_key,
        budget_usd=budget_usd,
        provider_upstream=provider_upstream,
        candidate_ids=candidate_ids,
        gbrain_config=gbrain_config,
    )


class RunOrchestrator:
    """Execute one immutable, inspectable comparison run."""

    def __init__(
        self,
        *,
        config: RunConfig,
        connectors: Sequence[Connector],
        candidates: Sequence[Candidate],
        provider_available: bool,
        provider_detail: str = "",
        provider_status: Status = Status.MCP_AUTH_UNAVAILABLE,
        test_mode: Mapping[str, Any] | None = None,
        candidate_builder: Callable[..., Sequence[Candidate]] | None = None,
        provider_key: str | None = None,
        chat_provenance_provider: Callable[[], ChatProvenance] | None = None,
        browser_open: Callable[[str], bool] = webbrowser.open,
        now: Callable[[], datetime] | None = None,
        monotonic_clock: Callable[[], float] = monotonic,
        stage_event_sink: StageEventSink | None = None,
        cancellation: RunCancellation | None = None,
    ) -> None:
        self.config = config
        self.connectors = tuple(connectors)
        self.candidates = tuple(candidates)
        self.provider_available = provider_available
        self.provider_detail = provider_detail
        self.provider_status = (
            Status.MISSING_PROVIDER
            if provider_status is Status.MCP_AUTH_UNAVAILABLE
            and provider_detail.startswith("MISSING_PROVIDER:")
            else provider_status
        )
        self.test_mode = dict(test_mode or {"enabled": False})
        self._candidate_builder = candidate_builder
        self._provider_key = provider_key
        self._chat_provenance_provider = chat_provenance_provider
        self._embedding_descriptor = config.embedding_config(
            environ={},
            api_key=provider_key,
        ).descriptor
        self.browser_open = browser_open
        self.now = now or (lambda: datetime.now(UTC))
        self._monotonic = monotonic_clock
        self._stage_event_sink = stage_event_sink
        self._event_sink_errors: list[str] = []
        self._run_started = False
        self.cancellation = cancellation or RunCancellation()
        self._stages: list[dict[str, Any]] = []
        self._ledger: list[dict[str, Any]] = []
        self._manifest: dict[str, Any] = {}
        self._cleanup_receipts: dict[str, CleanupReceipt] = {}
        self._cleanup_attempted: set[str] = set()
        self._cleanup_errors: list[str] = []
        self._started = 0.0
        self._known_secrets = tuple(
            secret
            for secret in (self._provider_key, *os.environ.values())
            if secret and len(secret) >= 8
        )

    @classmethod
    def local(
        cls,
        config: RunConfig,
        *,
        connection_manager: ConnectionManager | None = None,
        connector_builder: Callable[[ConnectionManager, bool], Sequence[Connector]] | None = None,
        candidate_builder: Callable[..., Sequence[Candidate]] | None = None,
        api_key: str | None = None,
        stage_event_sink: StageEventSink | None = None,
        cancellation: RunCancellation | None = None,
    ) -> RunOrchestrator:
        """Build the production workflow without opening provider connections."""
        state_root = AutoBrainPaths.from_home().root
        manager = connection_manager or ConnectionManager(state_root)
        connections = manager.status().connections
        selected_sources = set(config.selected_sources)
        oauth_sources = {
            provider
            for provider in selected_sources
            if not (
                (provider is Provider.SLACK and config.slack_export_path is not None)
                or (provider is Provider.NOTION and config.notion_snapshot_path is not None)
            )
        }
        disconnected = [
            item.provider.value
            for item in connections
            if item.provider in oauth_sources and item.state.value != "CONNECTED"
        ]
        if disconnected:
            return cls(
                config=config,
                connectors=(),
                candidates=(),
                provider_available=False,
                provider_detail="MCP_AUTH_UNAVAILABLE: " + ", ".join(disconnected),
                provider_status=Status.MCP_AUTH_UNAVAILABLE,
                stage_event_sink=stage_event_sink,
                cancellation=cancellation,
            )

        resolved_api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        embedding_config = config.embedding_config(
            environ=os.environ,
            api_key=(
                resolved_api_key
                if config.embedding_backend == EmbeddingBackend.OPENAI.value
                else api_key
            ),
        )
        embedding_readiness = embedding_config.readiness()
        if (
            embedding_config.descriptor.recommendation_eligible
            and not embedding_readiness.recommendation_ready
        ):
            return cls(
                config=config,
                connectors=(),
                candidates=(),
                provider_available=False,
                provider_detail=embedding_readiness.detail,
                provider_status=embedding_readiness.status,
                stage_event_sink=stage_event_sink,
                cancellation=cancellation,
            )

        try:
            if connector_builder is None:
                connectors = tuple(
                    _default_connector_builder(
                        manager,
                        config.include_dms,
                        config.selected_sources,
                        slack_export_path=config.slack_export_path,
                        slack_export_sha256=config.slack_export_sha256,
                        notion_snapshot_path=config.notion_snapshot_path,
                    )
                )
            else:
                selected_source_values = {provider.value for provider in config.selected_sources}
                connectors = tuple(
                    connector
                    for connector in connector_builder(manager, config.include_dms)
                    if connector.provider in selected_source_values
                )
        except Exception as exc:
            message = str(exc)
            status = (
                Status.MCP_AUTH_UNAVAILABLE
                if message.startswith("MCP_AUTH_UNAVAILABLE:")
                else Status.CAPABILITY_UNAVAILABLE
            )
            return cls(
                config=config,
                connectors=(),
                candidates=(),
                provider_available=False,
                provider_detail=message
                if message.startswith(("MCP_AUTH_UNAVAILABLE:", "CAPABILITY_UNAVAILABLE:"))
                else f"{status.value}: {exc}",
                provider_status=status,
                stage_event_sink=stage_event_sink,
                cancellation=cancellation,
            )

        subscription_upstream: Callable[[dict[str, Any]], dict[str, Any]] | None = None
        chat_provenance_provider: Callable[[], ChatProvenance] | None = None
        if config.provider_mode.endswith("-subscription"):
            from autobrain.subscription import (
                ProviderId,
                SubscriptionStatus,
                build_subscription_upstream,
                provider_registry,
            )

            provider_id = ProviderId(config.provider_mode.removesuffix("-subscription"))
            registry = provider_registry(cancellation=cancellation)
            subscription_client = registry.get(provider_id)
            subscription_report = registry.probe(provider_id, refresh=True)
            if subscription_report.status is not SubscriptionStatus.READY:
                return cls(
                    config=config,
                    connectors=connectors,
                    candidates=(),
                    provider_available=False,
                    provider_detail=(
                        f"{subscription_report.status.value}: "
                        f"{subscription_report.detail or provider_id.value}"
                    ),
                    provider_status=Status.CAPABILITY_UNAVAILABLE,
                    stage_event_sink=stage_event_sink,
                    cancellation=cancellation,
                )
            subscription_identity = subscription_client.probe_identity()
            embedding_upstream = (
                build_openai_embedding_upstream(embedding_config)
                if embedding_config.descriptor.openai_transport
                else build_gemini_embedding_upstream(embedding_config)
                if embedding_config.descriptor.gemini_transport
                else None
            )
            model_access = (
                ModelAccessRegistry(
                    {
                        ModelCapability.SEMANTIC_EMBEDDING: ModelAccess(
                            capability=ModelCapability.SEMANTIC_EMBEDDING,
                            provider=embedding_config.descriptor.provenance_backend,
                            handler=embedding_upstream,
                        )
                    }
                )
                if embedding_upstream is not None
                else None
            )
            subscription_upstream = build_subscription_upstream(
                subscription_client,
                embedding_upstream=embedding_upstream,
                model_access=model_access,
            )

            def selected_chat_provenance() -> ChatProvenance:
                answer = subscription_client.last_answer
                identity = answer.identity if answer is not None else subscription_identity
                return ChatProvenance(
                    provider=identity.provider.value,
                    model=identity.model,
                    cli_version=identity.cli_version,
                    auth_kind=identity.auth_kind.value,
                )

            chat_provenance_provider = selected_chat_provenance

        candidate_api_key = embedding_config.candidate_api_key

        if candidate_builder is None:

            def selected_candidate_builder(
                run_dir: Path,
                api_key: str,
                budget_usd: float,
            ) -> Sequence[Candidate]:
                return _default_candidate_builder(
                    run_dir,
                    api_key,
                    budget_usd,
                    provider_upstream=subscription_upstream,
                    candidate_ids=config.selected_candidates,
                    gbrain_config=config.gbrain_config,
                )

        else:

            def selected_candidate_builder(
                run_dir: Path,
                api_key: str,
                budget_usd: float,
            ) -> Sequence[Candidate]:
                selected_candidate_values = {
                    candidate.value for candidate in config.selected_candidates
                }
                parameters = tuple(inspect.signature(candidate_builder).parameters.values())
                accepts_budget = (
                    any(
                        parameter.kind is parameter.VAR_POSITIONAL
                        or parameter.kind is parameter.VAR_KEYWORD
                        for parameter in parameters
                    )
                    or sum(
                        parameter.kind
                        in {parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD}
                        for parameter in parameters
                    )
                    >= 3
                )
                built_candidates = (
                    candidate_builder(run_dir, api_key, budget_usd)
                    if accepts_budget
                    else candidate_builder(run_dir, api_key)
                )
                return retain_selected_candidates(
                    built_candidates,
                    selected_ids=selected_candidate_values,
                )

        return cls(
            config=config,
            connectors=connectors,
            candidates=(),
            provider_available=True,
            candidate_builder=selected_candidate_builder,
            provider_key=candidate_api_key,
            chat_provenance_provider=chat_provenance_provider,
            stage_event_sink=stage_event_sink,
            cancellation=cancellation,
        )

    @classmethod
    def fixture(
        cls,
        config: RunConfig,
        *,
        fixture_id: str,
        fixture_sha256: str,
        connectors: Sequence[Connector],
        candidates: Sequence[Candidate],
        stage_event_sink: StageEventSink | None = None,
        cancellation: RunCancellation | None = None,
    ) -> RunOrchestrator:
        return cls(
            config=config,
            connectors=connectors,
            candidates=candidates,
            provider_available=True,
            test_mode={
                "enabled": True,
                "fixture_id": fixture_id,
                "fixture_sha256": fixture_sha256,
            },
            stage_event_sink=stage_event_sink,
            cancellation=cancellation,
        )

    def cancel(self) -> None:
        self.cancellation.cancel()

    def run(self) -> RunResult:
        if self._run_started:
            raise RuntimeError("RunOrchestrator is single-use; construct a new instance")
        self._run_started = True
        result = self._run_workflow()
        cleanup_errors: list[str] = [*self._cleanup_errors]
        cleanup_interruptions: list[str] = []
        for candidate in self.candidates:
            if candidate.candidate_id in self._cleanup_attempted:
                continue
            self._cleanup_attempted.add(candidate.candidate_id)
            try:
                receipt = candidate.cleanup()
                if receipt is not None:
                    self._cleanup_receipts[candidate.candidate_id] = receipt
                    cleanup_manifest = self._manifest.setdefault("cleanup", {})
                    cleanup_manifest[candidate.candidate_id] = receipt.model_dump(mode="json")
                if receipt is not None and not cleanup_receipt_complete(receipt):
                    cleanup_errors.append(f"{candidate.candidate_id}: incomplete cleanup receipt")
                    self._ledger.append(
                        {
                            "kind": "cleanup",
                            "candidate": candidate.candidate_id,
                            "status": Status.FAILED.value,
                            "detail": "cleanup receipt did not prove complete removal",
                            "receipt": receipt.model_dump(mode="json"),
                        }
                    )
                    continue
            except KeyboardInterrupt:
                detail = "cleanup interrupted"
                cleanup_interruptions.append(f"{candidate.candidate_id}: {detail}")
                self._ledger.append(
                    {
                        "kind": "cleanup",
                        "candidate": candidate.candidate_id,
                        "status": Status.CANCELLED.value,
                        "detail": detail,
                    }
                )
            except Exception as exc:
                detail = str(exc)
                cleanup_errors.append(f"{candidate.candidate_id}: {detail}")
                self._ledger.append(
                    {
                        "kind": "cleanup",
                        "candidate": candidate.candidate_id,
                        "status": Status.FAILED.value,
                        "detail": detail,
                    }
                )
            else:
                self._ledger.append(
                    {
                        "kind": "cleanup",
                        "candidate": candidate.candidate_id,
                        "status": Status.OK.value,
                        "receipt": receipt.model_dump(mode="json") if receipt is not None else None,
                    }
                )

        final_status = (
            Status.FAILED
            if cleanup_errors
            else Status.CANCELLED
            if cleanup_interruptions
            else result.status
        )
        warning = (
            "candidate cleanup interrupted: " + "; ".join(cleanup_interruptions)
            if cleanup_interruptions
            else ""
        )
        if result.report_path is None and final_status is Status.CANCELLED:
            report_path = self._write_terminal_artifacts(
                result.run_dir,
                status=final_status,
                warning=warning or "run interrupted; no resume is attempted",
            )
            result = replace(result, report_path=report_path)
        self._manifest["status"] = final_status.value
        if cleanup_errors and result.report_path is not None:
            artifact = load_comparison(result.run_dir / "comparison.json")
            artifact = artifact.model_copy(
                update={
                    "status": Status.FAILED,
                    "warnings": [
                        *artifact.warnings,
                        "candidate cleanup failed: " + "; ".join(cleanup_errors),
                    ],
                }
            )
            report_artifacts = write_artifacts(artifact, result.run_dir)
            self._manifest["report"] = {
                "path": str(report_artifacts.report_html),
                "sha256": report_artifacts.report_sha256,
            }
        cleanup_status = (
            Status.FAILED
            if cleanup_errors
            else Status.CANCELLED
            if cleanup_interruptions
            else Status.OK
        )
        cleanup_detail = (
            "; ".join(cleanup_errors) if cleanup_errors else "candidate process trees settled"
        )
        if cleanup_interruptions:
            cleanup_detail = "; ".join(cleanup_interruptions)
        self._stage(result.run_dir, "cleanup", cleanup_status, cleanup_detail)
        total_ms = round((self._monotonic() - self._started) * 1000)
        self._manifest["timings"]["total_ms"] = total_ms if total_ms > 0 else None
        self._persist(result.run_dir)
        return replace(
            result,
            status=final_status,
            event_sink_errors=tuple(self._event_sink_errors),
        )

    def coverage_eligibility_reasons(self) -> list[str]:
        """Return source-scope reasons that make a recommendation partial/non-final."""
        reasons: list[str] = []
        if Provider.SLACK not in self.config.selected_sources:
            reasons.append("Slack source absent; source coverage is partial and non-final")
        if self.config.notion_snapshot_path is not None:
            reasons.append("Notion snapshot coverage is partial and non-final")
        return reasons

    def _run_workflow(self) -> RunResult:
        run_id = self.config.run_id or self._new_run_id()
        run_dir = self._create_run_dir(run_id)
        self._manifest = {
            "schema_version": 2,
            "run_id": run_id,
            "created_at": self.now().isoformat(),
            "config": {
                "budget_usd": self.config.budget_usd,
                "max_questions": self.config.max_questions,
                "include_dms": self.config.include_dms,
                "open_report": self.config.open_report,
                "provider_mode": self.config.provider_mode,
                "embedding_backend": self._embedding_descriptor.selector,
                "selected_sources": [provider.value for provider in self.config.selected_sources],
                "selected_candidates": [
                    candidate.value for candidate in self.config.selected_candidates
                ],
                "experiment_title": self.config.experiment_title,
                "experiment_description": self.config.experiment_description,
                "gbrain": self.config.gbrain_config.safe_metadata(),
            },
            "stages": self._stages,
            "commands": self._ledger,
            "coverage": {},
            "benchmark": {},
            "candidates": [],
            "hashes": {},
            "pins": {},
            "models": {"judge": "gpt-5-mini", "provider": "openai"},
            "provenance": self.benchmark_provenance().model_dump(mode="json"),
            "pricing": {"version": "local-metering-v1"},
            "timings": {},
            "status": Status.NO_DECISION.value,
            "verdict": Status.NO_DECISION.value,
            "report": {},
            "test_mode": dict(self.test_mode),
        }
        self._persist(run_dir)
        started = self._monotonic()
        candidate_results: list[CandidateOutcome] = []
        report_path: Path | None = None
        final_status = Status.OK
        verdict = Status.NO_DECISION.value
        self._started = started
        try:
            self._stage(run_dir, "preflight", Status.OK, "local paths and candidate pins checked")
            self._record_pins(run_dir)
            if not self.provider_available:
                final_status = self.provider_status
                detail = self.provider_detail or "MCP_AUTH_UNAVAILABLE"
                self._stage(run_dir, "missing-auth-gate", final_status, detail)
                return RunResult(run_id, run_dir, final_status, None, (), verdict)
            if len(self.connectors) != len(self.config.selected_sources):
                final_status = Status.CAPABILITY_UNAVAILABLE
                self._stage(
                    run_dir,
                    "capability-probe",
                    final_status,
                    "selected knowledge source connectors are unavailable",
                )
                return RunResult(run_id, run_dir, final_status, None, (), verdict)

            try:
                probes = self._probe(run_dir)
            except RunCancelled:
                raise
            except Exception as exc:
                final_status = Status.CAPABILITY_UNAVAILABLE
                self._stage(run_dir, "capability-probe", final_status, type(exc).__name__)
                return RunResult(run_id, run_dir, final_status, None, (), verdict)
            if any(
                probe.get("capability_available") is False or not probe.get("allowed")
                for probe in probes.values()
            ):
                final_status = Status.CAPABILITY_UNAVAILABLE
                self._stage(run_dir, "capability-probe", final_status, "no read-only capability")
                return RunResult(run_id, run_dir, final_status, None, (), verdict)

            snapshots = self._crawl(run_dir)
            documents = [
                dict(document) for snapshot in snapshots for document in snapshot.documents
            ]
            self._manifest["coverage"] = {
                snapshot.provider: dict(snapshot.coverage) for snapshot in snapshots
            }
            if Provider.SLACK not in self.config.selected_sources:
                self._manifest["coverage"][Provider.SLACK.value] = {
                    "completeness": CoverageCompleteness.UNKNOWN.value,
                    "discovered": 0,
                    "fetched": 0,
                    "unsupported": 1,
                    "crawl_provenance": {
                        "source_state": "absent",
                        "partial": "true",
                        "final": "false",
                    },
                }
            self._stage(run_dir, "coverage", Status.OK, f"{len(documents)} documents")
            cases, holdout_ids = self._build_benchmark(documents, self.config.max_questions)
            normalized_candidate_documents = normalize_raw_items(
                self._candidate_documents(documents, cases, holdout_ids)
            )
            corpus = CorpusBoundary(
                candidate_documents=tuple(
                    document.model_dump(mode="json") for document in normalized_candidate_documents
                ),
                evaluator_cases=self._evaluator_cases(documents, cases),
            )
            candidate_documents = corpus.candidate_documents
            evaluator_cases = corpus.evaluator_cases
            self._manifest["benchmark"] = {
                "case_count": len(cases),
                "holdout_count": len(holdout_ids),
                "provenance": "local Slack thread questions with document fallback",
                "same_model_judge": "gpt-5-mini",
                "generated_case_count": sum(1 for case in cases if bool(case.get("generated"))),
            }
            benchmark_sha = self._hash_json(cases)
            self._manifest["hashes"]["benchmark_sha256"] = benchmark_sha
            self._stage(run_dir, "benchmark-holdout", Status.OK, f"{len(cases)} cases")
            if len(cases) < MIN_BENCHMARK_CASES:
                final_status = Status.INSUFFICIENT_BENCHMARK
                self._stage(
                    run_dir,
                    "benchmark-gate",
                    final_status,
                    f"need {MIN_BENCHMARK_CASES}, found {len(cases)}",
                )
                return RunResult(run_id, run_dir, final_status, None, (), verdict)

            holdout_markers = self._holdout_markers(documents, holdout_ids)
            leakage = self._scan_candidate_leakage(
                candidate_documents,
                cases,
                holdout_markers,
            )
            if not leakage.clean:
                final_status = Status.LEAKAGE_DETECTED
                self._stage(
                    run_dir,
                    "leakage-gate",
                    final_status,
                    ", ".join(leakage.matched_tokens)
                    or "holdout identifiers or evaluator markers reached candidate input",
                )
                return RunResult(run_id, run_dir, final_status, None, (), verdict)
            self._stage(run_dir, "leakage-gate", Status.OK, "candidate input is holdout-clean")
            corpus_sha = self._hash_json(candidate_documents)
            self._manifest["hashes"]["corpus_sha256"] = corpus_sha
            corpus_path = run_dir / "corpus-freeze.json"
            self._write_json(
                corpus_path,
                {"documents": candidate_documents, "sha256": corpus_sha},
            )
            self._stage(run_dir, "final-corpus-freeze", Status.OK, str(corpus_path))

            self._stage(
                run_dir,
                "post-crawl-budget-estimate",
                Status.OK,
                f"hard cap ${self.config.budget_usd:.2f}",
            )
            if self._candidate_builder is not None and not self.candidates:
                self.candidates = tuple(self._build_candidates(run_dir))
                self._stage(
                    run_dir,
                    "candidate-construction",
                    Status.OK,
                    f"{len(self.candidates)} pinned candidate adapters constructed",
                )
            if not self.candidates:
                final_status = Status.CAPABILITY_UNAVAILABLE
                self._stage(
                    run_dir,
                    "candidate-construction",
                    final_status,
                    "at least one candidate is required",
                )
                return RunResult(run_id, run_dir, final_status, None, (), verdict)
            context = CandidateContext(
                documents=tuple(candidate_documents),
                questions=tuple(case["question"] for case in cases),
                case_ids=tuple(case["case_id"] for case in cases),
                cancellation=self.cancellation,
            )
            self._stage(run_dir, "evaluation-gate", Status.OK, "benchmark and corpus gates passed")
            candidate_results = self._run_candidates(run_dir, context)
            self._candidate_outcomes = tuple(candidate_results)
            if any(outcome.status is Status.CANCELLED for outcome in candidate_results):
                final_status = Status.CANCELLED
                warning = (
                    "run interrupted during candidate evaluation; no later candidate was started"
                )
                self._manifest.setdefault("warnings", []).append(warning)
                self._stage(run_dir, "cancelled", final_status, warning)
                return RunResult(
                    run_id,
                    run_dir,
                    final_status,
                    None,
                    tuple(candidate_results),
                    verdict,
                )
            if any(outcome.status is Status.BUDGET_EXCEEDED for outcome in candidate_results):
                final_status = Status.BUDGET_EXCEEDED
                warning = "hard budget cap exhausted; no later candidate was started"
                self._manifest.setdefault("warnings", []).append(warning)
            output_leakage = self._scan_candidate_outputs(
                run_dir,
                candidate_results,
                holdout_markers,
            )
            if not output_leakage.clean:
                final_status = Status.LEAKAGE_DETECTED
                self._stage(
                    run_dir,
                    "candidate-output-leakage-gate",
                    final_status,
                    ", ".join(output_leakage.matched_tokens)
                    or "candidate output retained evaluator-only data",
                )
                return RunResult(
                    run_id,
                    run_dir,
                    final_status,
                    None,
                    tuple(candidate_results),
                    verdict,
                )
            self._cleanup_candidates(run_dir)
            evaluations = self._canonical_evaluations(
                candidate_results,
                evaluator_cases,
                corpus_sha,
            )
            provenance = self.benchmark_provenance(evaluations)
            coverage_reasons = self.coverage_eligibility_reasons()
            evaluations = [
                evaluation.model_copy(
                    update={
                        "eligible_override": (
                            False if coverage_reasons else evaluation.eligible_override
                        ),
                        "eligibility_reasons": [
                            *eligibility_reasons(
                                evaluation.model_copy(
                                    update={"eligible_override": False} if coverage_reasons else {}
                                ),
                                embedding=self._embedding_descriptor,
                                embedding_registry=self.config.embedding_registry,
                            ),
                            *coverage_reasons,
                        ],
                    }
                )
                for evaluation in evaluations
            ]
            candidate_results = [
                replace(
                    outcome,
                    score=evaluation.quality_score,
                    answered_cases=evaluation.answered_cases,
                    scored_cases=evaluation.scored_cases,
                    cost_usd=evaluation.total_cost_usd,
                    evaluation=evaluation,
                )
                for outcome, evaluation in zip(candidate_results, evaluations, strict=True)
            ]
            self._manifest["candidates"] = [self._outcome_json(item) for item in candidate_results]
            decision = select_winner(
                evaluations,
                embedding=self._embedding_descriptor,
                embedding_registry=self.config.embedding_registry,
            )
            self._manifest["evaluations"] = [
                evaluation.model_dump(mode="json") for evaluation in evaluations
            ]
            self._manifest["provenance"] = provenance.model_dump(mode="json")
            self._manifest["decision"] = decision.model_dump(mode="json")
            self._stage(
                run_dir,
                "evaluation",
                final_status,
                "canonical 45/25/20/10 candidate evaluations reconciled",
            )
            verdict = decision.verdict.value
            self._manifest["verdict"] = verdict
            self._stage(run_dir, "verdict", Status.OK, verdict)
            self._write_evaluator_holdout(run_dir, documents, holdout_ids)
            report_path = self._write_report(
                run_dir,
                candidate_results,
                verdict,
                evaluator_cases=evaluator_cases,
                candidate_documents=candidate_documents,
                corpus_sha=corpus_sha,
                benchmark_sha=benchmark_sha,
                decision=decision,
                evaluations=evaluations,
                status=final_status,
            )
            self._manifest["report"] = {
                "path": str(report_path),
                "sha256": self._sha256(report_path),
            }
            self._stage(run_dir, "report", Status.OK, str(report_path))
            if self.config.open_report:
                self.browser_open(report_path.as_uri())
                self._stage(run_dir, "browser-open", Status.OK, "local report requested")
            return RunResult(
                run_id, run_dir, final_status, report_path, tuple(candidate_results), verdict
            )
        except (KeyboardInterrupt, RunCancelled):
            final_status = Status.CANCELLED
            warning = (
                "operator cancelled run; no resume is attempted"
                if self.cancellation.cancelled
                else "run interrupted; no resume is attempted"
            )
            self._manifest.setdefault("warnings", []).append(warning)
            self._stage(
                run_dir,
                "cancelled" if self.cancellation.cancelled else "interrupted",
                final_status,
                warning,
            )
            return RunResult(
                run_id, run_dir, final_status, report_path, tuple(candidate_results), verdict
            )
        except Exception as exc:
            final_status = Status.FAILED
            self._stage(run_dir, "failed", final_status, str(exc))
            return RunResult(
                run_id, run_dir, final_status, report_path, tuple(candidate_results), verdict
            )

    def _run_candidates(self, run_dir: Path, context: CandidateContext) -> list[CandidateOutcome]:
        results: list[CandidateOutcome] = []
        spent = 0.0
        estimate = max(0.5, len(context.questions) * 0.05)
        budget_exhausted = False
        for candidate in self.candidates:
            if self.cancellation.cancelled:
                results.append(
                    CandidateOutcome(
                        candidate=candidate.candidate_id,
                        status=Status.CANCELLED,
                        detail="operator cancelled run",
                    )
                )
                break
            if budget_exhausted or spent + estimate > self.config.budget_usd:
                outcome = CandidateOutcome(
                    candidate=candidate.candidate_id,
                    status=Status.BUDGET_EXCEEDED,
                    detail=f"hard cap ${self.config.budget_usd:.2f} reached before candidate start",
                    cost_status=CostStatus.INCOMPLETE,
                )
                results.append(outcome)
                self._ledger.append(self._outcome_json(outcome))
                self._write_json(
                    run_dir / "candidates" / f"{candidate.candidate_id}.json",
                    self._outcome_json(outcome),
                )
                if "run_id" in self._manifest:
                    self._stage(
                        run_dir,
                        f"candidate:{candidate.candidate_id}",
                        outcome.status,
                        outcome.detail,
                    )
                budget_exhausted = True
                continue
            started = self._monotonic()
            try:
                outcome = candidate.run(context)
            except (KeyboardInterrupt, RunCancelled):
                outcome = CandidateOutcome(
                    candidate=candidate.candidate_id,
                    status=Status.CANCELLED,
                    detail=(
                        "operator cancelled run"
                        if self.cancellation.cancelled
                        else "candidate interrupted"
                    ),
                )
            except Exception as exc:
                outcome = CandidateOutcome(
                    candidate=candidate.candidate_id,
                    status=Status.FAILED,
                    detail=str(exc),
                )
            elapsed = round((self._monotonic() - started) * 1000)
            if elapsed > 0 and not any(
                span.name is LatencySpanKind.END_TO_END for span in outcome.latency_spans
            ):
                outcome = replace(
                    outcome,
                    latency_ms=outcome.latency_ms or elapsed,
                    latency_spans=(
                        *outcome.latency_spans,
                        LatencySpan(
                            name=LatencySpanKind.END_TO_END,
                            duration_ms=elapsed,
                            candidate=CandidateId(outcome.candidate),
                        ),
                    ),
                )
            if outcome.cost_usd is not None:
                spent += outcome.cost_usd
                if spent > self.config.budget_usd:
                    outcome = replace(
                        outcome,
                        status=Status.BUDGET_EXCEEDED,
                        detail=(
                            f"hard cap ${self.config.budget_usd:.2f} exceeded by measured "
                            f"candidate cost ${spent:.4f}"
                        ),
                    )
                    budget_exhausted = True
            if outcome.status is Status.BUDGET_EXCEEDED:
                budget_exhausted = True
            results.append(outcome)
            self._ledger.append(self._outcome_json(outcome))
            self._write_json(
                run_dir / "candidates" / f"{candidate.candidate_id}.json",
                self._outcome_json(outcome),
            )
            if "run_id" in self._manifest:
                self._stage(
                    run_dir,
                    f"candidate:{candidate.candidate_id}",
                    outcome.status,
                    outcome.detail or f"{outcome.answered_cases} cases answered",
                )
            if spent >= self.config.budget_usd:
                estimate = self.config.budget_usd
                budget_exhausted = True
            if outcome.status is Status.CANCELLED:
                break
        return results

    def _build_candidates(self, run_dir: Path) -> Sequence[Candidate]:
        assert self._candidate_builder is not None
        builder = self._candidate_builder
        parameters = tuple(inspect.signature(builder).parameters.values())
        accepts_budget = (
            any(
                parameter.kind is parameter.VAR_POSITIONAL
                or parameter.kind is parameter.VAR_KEYWORD
                for parameter in parameters
            )
            or sum(
                parameter.kind in {parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD}
                for parameter in parameters
            )
            >= 3
        )
        if accepts_budget:
            return builder(
                run_dir,
                self._provider_key or "",
                self.config.budget_usd,
            )
        return builder(run_dir, self._provider_key or "")

    def _probe(self, run_dir: Path) -> dict[str, Mapping[str, Any]]:
        results: dict[str, Mapping[str, Any]] = {}
        for connector in self.connectors:
            self.cancellation.raise_if_cancelled()
            result = connector.probe(cancellation=self.cancellation)
            results[connector.provider] = dict(result)
            self._ledger.append(
                {"kind": "probe", "provider": connector.provider, "read_only": True}
            )
        self._stage(run_dir, "capability-probe", Status.OK, "read-only allowlists only")
        return results

    def _crawl(self, run_dir: Path) -> list[ConnectorSnapshot]:
        snapshots: list[ConnectorSnapshot] = []
        for connector in self.connectors:
            self.cancellation.raise_if_cancelled()
            snapshot = connector.crawl(cancellation=self.cancellation)
            snapshots.append(snapshot)
            self._ledger.append(
                {
                    "kind": "crawl",
                    "provider": snapshot.provider,
                    "documents": len(snapshot.documents),
                    "read_only": True,
                }
            )
        self._stage(run_dir, "crawl", Status.OK, "Slack and Notion read-only MCP crawl")
        return snapshots

    @staticmethod
    def _build_benchmark(
        documents: Sequence[Mapping[str, Any]],
        max_questions: int,
    ) -> tuple[list[dict[str, Any]], set[str]]:
        cases: list[dict[str, Any]] = []
        replies_by_parent: dict[str, list[str]] = {}
        ordered_documents = sorted(
            documents,
            key=lambda document: (
                RunOrchestrator._natural_source_key(str(document.get("source_id", ""))),
                json.dumps(document, sort_keys=True, separators=(",", ":"), default=str),
            ),
        )
        for document in ordered_documents:
            parent = RunOrchestrator._parent_source_id(document)
            text = str(document.get("text", "")).strip()
            if isinstance(parent, str) and parent and text:
                replies_by_parent.setdefault(parent, []).append(text)
        seen_questions: set[str] = set()
        for document in ordered_documents:
            source_id = str(document.get("source_id", ""))
            raw_question = str(document.get("question", "")).strip()
            text = str(document.get("text", "")).strip()
            if not raw_question and RunOrchestrator._is_notion_document(document):
                for generated in RunOrchestrator._notion_document_cases(document):
                    normalized_question = " ".join(
                        re.findall(r"\w+", str(generated["question"]).casefold())
                    )
                    if normalized_question in seen_questions:
                        continue
                    seen_questions.add(normalized_question)
                    cases.append(generated)
                continue
            question = raw_question
            question_is_generated = False
            if not question and text.endswith("?"):
                question = text
            if not question:
                title = str(document.get("title", "document")).strip() or "document"
                question = f"What does {title} say?"
                question_is_generated = True
            evidence = RunOrchestrator._evidence_text(
                document,
                replies_by_parent.get(source_id, ()),
            )
            if not RunOrchestrator._qualifies_question(question, evidence):
                continue
            normalized_question = " ".join(re.findall(r"\w+", question.casefold()))
            if normalized_question in seen_questions:
                continue
            seen_questions.add(normalized_question)
            cases.append(
                {
                    "case_id": f"case-{hashlib.sha256(source_id.encode()).hexdigest()[:16]}",
                    "question": question,
                    "source_ids": [source_id],
                    "generated": question_is_generated,
                }
            )
        holdout_target = (
            min(
                max(1, len(cases) // 10),
                max(1, len(cases) - MIN_BENCHMARK_CASES),
            )
            if cases
            else 0
        )
        holdout_ids: set[str] = set()
        holdout_case_count = 0
        for case in reversed(cases):
            source_ids = {str(source_id) for source_id in case["source_ids"] if source_id}
            if source_ids & holdout_ids:
                continue
            source_case_count = sum(
                1
                for candidate in cases
                if source_ids.intersection(str(item) for item in candidate["source_ids"])
            )
            if len(cases) - holdout_case_count - source_case_count < MIN_BENCHMARK_CASES:
                if not holdout_ids:
                    holdout_ids.update(source_ids)
                break
            holdout_ids.update(source_ids)
            holdout_case_count += source_case_count
            if holdout_case_count >= holdout_target:
                break
        benchmark_cases = [
            case
            for case in cases
            if not holdout_ids.intersection(str(item) for item in case["source_ids"])
        ]
        cap = min(max_questions, MAX_BENCHMARK_CASES)
        benchmark_cases = benchmark_cases[:cap]
        if len(benchmark_cases) < MIN_BENCHMARK_CASES:
            return [], holdout_ids
        return benchmark_cases, holdout_ids

    @staticmethod
    def _evaluator_cases(
        documents: Sequence[Mapping[str, Any]],
        cases: Sequence[Mapping[str, Any]],
    ) -> tuple[EvaluatorCaseRecord, ...]:
        replies_by_parent: dict[str, list[str]] = {}
        for document in documents:
            parent = RunOrchestrator._parent_source_id(document)
            text = str(document.get("text", "")).strip()
            if isinstance(parent, str) and parent and text:
                replies_by_parent.setdefault(parent, []).append(text)
        documents_by_source = {
            str(document.get("source_id")): document
            for document in documents
            if document.get("source_id")
        }
        records: list[EvaluatorCaseRecord] = []
        for safe_case in cases:
            source_ids = [
                str(source_id)
                for source_id in cast(Sequence[Any], safe_case.get("source_ids", []))
                if isinstance(source_id, str)
            ]
            source_document = documents_by_source.get(source_ids[0], {})
            reference = RunOrchestrator._case_reference_text(
                source_document,
                safe_case,
                replies_by_parent.get(source_ids[0], ()) if source_ids else (),
            )
            raw_claims = source_document.get("expected_claims")
            expected_claims = (
                [
                    claim.strip()
                    for claim in cast(Sequence[Any], raw_claims)
                    if isinstance(claim, str) and claim.strip()
                ]
                if isinstance(raw_claims, Sequence) and not isinstance(raw_claims, str | bytes)
                else []
            )
            if not expected_claims and reference:
                expected_claims = [reference]
            raw_forbidden = source_document.get("forbidden_contradictions", [])
            forbidden = (
                [
                    contradiction.strip()
                    for contradiction in cast(Sequence[Any], raw_forbidden)
                    if isinstance(contradiction, str) and contradiction.strip()
                ]
                if isinstance(raw_forbidden, Sequence)
                and not isinstance(raw_forbidden, str | bytes)
                else []
            )
            raw_confidence = source_document.get("reference_confidence", 1.0)
            try:
                confidence = float(raw_confidence)
            except (TypeError, ValueError):
                confidence = 1.0
            confidence = max(0.0, min(1.0, confidence))
            records.append(
                EvaluatorCaseRecord(
                    case=BenchmarkCase(
                        case_id=str(safe_case["case_id"]),
                        question=str(safe_case["question"]),
                        source_ids=source_ids,
                        expected_claims=expected_claims,
                        forbidden_contradictions=forbidden,
                        generated=bool(safe_case.get("generated")),
                    ),
                    reference_text=reference,
                    reply_texts=tuple(replies_by_parent.get(source_ids[0], ()))
                    if source_ids
                    else (),
                    reference_confidence=confidence,
                )
            )
        return tuple(records)

    @staticmethod
    def _is_notion_document(document: Mapping[str, Any]) -> bool:
        source_id = str(document.get("source_id", ""))
        source_kind = str(document.get("source_kind", ""))
        return source_id.startswith("notion:") or source_kind.casefold() in {
            "notion_page",
            "sourcekind.notion_page",
        }

    @staticmethod
    def _notion_document_cases(document: Mapping[str, Any]) -> list[dict[str, Any]]:
        source_id = str(document.get("source_id", ""))
        title = str(document.get("title", "document")).strip() or "document"
        text = str(document.get("text", ""))
        heading = title
        heading_items: dict[str, int] = {}
        cases: list[dict[str, Any]] = []
        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.strip()
            heading_match = _MARKDOWN_HEADING.match(line)
            if heading_match is not None:
                heading = heading_match.group(1).strip()
                continue
            bullet_match = _MARKDOWN_BULLET.match(line)
            if bullet_match is None or _PROMPT_LIKE_SOURCE.search(line):
                continue
            evidence = bullet_match.group(1).strip()
            if len(re.findall(r"\w+", evidence, re.UNICODE)) < 6:
                continue
            heading_items[heading] = heading_items.get(heading, 0) + 1
            item_number = heading_items[heading]
            question = f"What does {title} state in {heading}, item {item_number}?"
            if not RunOrchestrator._qualifies_question(question, evidence):
                continue
            case_key = f"{source_id}\x00{line_number}\x00{question}"
            cases.append(
                {
                    "case_id": f"case-{hashlib.sha256(case_key.encode()).hexdigest()[:16]}",
                    "question": question,
                    "source_ids": [source_id],
                    "generated": True,
                    "provenance": [source_id, f"markdown:line:{line_number}"],
                }
            )
            if len(cases) == 6:
                break
        return cases

    @staticmethod
    def _case_reference_text(
        document: Mapping[str, Any],
        case: Mapping[str, Any],
        thread_replies: Sequence[str] = (),
    ) -> str:
        provenance = case.get("provenance")
        if isinstance(provenance, Sequence) and not isinstance(provenance, str | bytes):
            for item in cast(Sequence[Any], provenance):
                if not isinstance(item, str) or not item.startswith("markdown:line:"):
                    continue
                try:
                    line_number = int(item.rsplit(":", 1)[1])
                except ValueError:
                    continue
                lines = str(document.get("text", "")).splitlines()
                if 1 <= line_number <= len(lines):
                    bullet = _MARKDOWN_BULLET.match(lines[line_number - 1].strip())
                    if bullet is not None:
                        return bullet.group(1).strip()
        return RunOrchestrator._evidence_text(document, thread_replies)

    @staticmethod
    def _parent_source_id(document: Mapping[str, Any]) -> str | None:
        value = document.get("parent_source_id")
        if isinstance(value, str) and value:
            return value
        provenance = document.get("crawl_provenance")
        if isinstance(provenance, Mapping):
            provenance_mapping = cast(Mapping[str, Any], provenance)
            value = provenance_mapping.get("parent_source_id")
            if isinstance(value, str) and value:
                return value
        return None

    @staticmethod
    def _candidate_documents(
        documents: Sequence[Mapping[str, Any]],
        cases: Sequence[Mapping[str, Any]],
        holdout_ids: set[str],
    ) -> list[dict[str, Any]]:
        benchmark_root_ids = set(holdout_ids)
        benchmark_root_ids.update(
            str(source_id)
            for case in cases
            for source_id in cast(Sequence[Any], case.get("source_ids", ()))
            if isinstance(source_id, str) and source_id
        )
        return [
            RunOrchestrator._candidate_document(document)
            for document in documents
            if str(document.get("source_id", "")) not in holdout_ids
            and RunOrchestrator._parent_source_id(document) not in benchmark_root_ids
        ]

    @staticmethod
    def _natural_source_key(source_id: str) -> tuple[str, ...]:
        return tuple(
            part.zfill(20) if part.isdigit() else part for part in re.split(r"(\d+)", source_id)
        )

    @staticmethod
    def _candidate_document(document: Mapping[str, Any]) -> dict[str, Any]:
        evaluator_keys = {
            "answer",
            "evidence",
            "evidence_reply",
            "expected",
            "expected_claims",
            "forbidden_contradictions",
            "oracle",
            "question",
            "reference_answer",
            "reference_text",
            "replies",
            "reply",
        }
        return {
            str(key): value
            for key, value in document.items()
            if str(key).casefold() not in evaluator_keys
        }

    @staticmethod
    def _holdout_markers(
        documents: Sequence[Mapping[str, Any]],
        holdout_ids: set[str],
    ) -> set[str]:
        markers = set(holdout_ids)
        for document in documents:
            source_id = str(document.get("source_id", ""))
            parent_source_id = RunOrchestrator._parent_source_id(document)
            if source_id not in holdout_ids and parent_source_id not in holdout_ids:
                continue
            text = document.get("text")
            if isinstance(text, str) and text.strip():
                markers.add(text.strip())
            for key in (
                "question",
                "evidence_reply",
                "reply",
                "answer",
                "evidence",
                "expected",
                "reference_answer",
                "reference_text",
            ):
                value = document.get(key)
                if isinstance(value, str) and value.strip():
                    markers.add(value.strip())
        return markers

    @staticmethod
    def _evidence_text(
        document: Mapping[str, Any],
        thread_replies: Sequence[str] = (),
    ) -> str:
        for key in ("evidence_reply", "reply", "answer", "evidence", "expected"):
            value = document.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        replies = document.get("replies")
        if isinstance(replies, Sequence) and not isinstance(replies, str | bytes):
            values: list[str] = []
            for item in cast(Sequence[Any], replies):
                if isinstance(item, str):
                    values.append(item)
                elif isinstance(item, Mapping):
                    mapping = cast(Mapping[str, Any], item)
                    values.append(str(mapping.get("text", "")))
            joined = " ".join(value for value in values if value.strip())
            if joined:
                return joined
        return " ".join(value for value in thread_replies if value.strip())

    @staticmethod
    def _qualifies_question(question: str, evidence: str) -> bool:
        words = re.findall(r"[A-Za-z0-9]+", question)
        evidence_words = re.findall(r"[A-Za-z0-9]+", evidence)
        return not (
            not question
            or _ACKNOWLEDGEMENT.fullmatch(question)
            or _SOCIAL_CHATTER.search(question)
            or _SPECULATION.match(question)
            or len(words) < 5
            or not (_QUESTION_WORDS.search(question) or question.endswith("?"))
            or len(evidence_words) < 5
            or _SPECULATION.match(evidence)
        )

    @staticmethod
    def _scan_candidate_leakage(
        documents: Sequence[Mapping[str, Any]],
        cases: Sequence[Mapping[str, Any]],
        forbidden_tokens: set[str],
    ) -> LeakageScanResult:
        payload = [
            json.dumps(value, sort_keys=True, separators=(",", ":"))
            for value in (*documents, *cases)
        ]
        return scan_benchmark_leakage(
            texts=payload,
            serialized_artifacts=payload,
            forbidden_tokens=tuple(sorted(forbidden_tokens)),
        )

    @staticmethod
    def _scan_candidate_outputs(
        run_dir: Path,
        outcomes: Sequence[CandidateOutcome],
        forbidden_tokens: set[str],
    ) -> LeakageScanResult:
        candidate_workspaces = tuple(
            path for path in (run_dir / "native", run_dir / "candidates") if path.exists()
        )
        return scan_benchmark_leakage(
            outputs=tuple(
                json.dumps(
                    RunOrchestrator._outcome_json(outcome),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                for outcome in outcomes
            ),
            candidate_workspaces=candidate_workspaces,
            forbidden_tokens=tuple(sorted(forbidden_tokens)),
        )

    def _canonical_evaluations(
        self,
        outcomes: Sequence[CandidateOutcome],
        evaluator_cases: Sequence[EvaluatorCaseRecord],
        corpus_sha: str,
    ) -> list[CandidateEvaluation]:
        evaluations: list[CandidateEvaluation] = []
        for outcome in outcomes:
            candidate = CandidateId(outcome.candidate)
            observations_by_case = {
                observation.case_id: observation
                for observation in outcome.observations
                if observation.case_id and observation.candidate == candidate
            }
            evaluated_cases: list[tuple[Any, ...]] = []
            missing_identity = False
            for evaluator_case in evaluator_cases:
                observation = observations_by_case.get(evaluator_case.case.case_id)
                if observation is None:
                    missing_identity = True
                    evaluated_cases.append(
                        (
                            evaluator_case.case,
                            "",
                            [],
                            evaluator_case.reference_confidence,
                            0,
                            Status.FAILED,
                            "candidate produced no observation",
                        )
                    )
                    continue
                evaluated_cases.append(
                    (
                        evaluator_case.case,
                        observation.answer,
                        observation.source_ids,
                        evaluator_case.reference_confidence,
                        observation.latency_ms,
                        observation.status,
                        "; ".join(observation.warnings),
                    )
                )
            cost_status = outcome.cost_status
            if outcome.evaluation is not None:
                cost_status = outcome.evaluation.cost_status
            complete_cost = outcome.cost_usd is not None and cost_status is CostStatus.COMPLETE
            effective_cost_status = (
                CostStatus.COMPLETE
                if complete_cost
                else CostStatus.INCOMPLETE
                if cost_status is CostStatus.COMPLETE
                else cost_status
            )
            evaluation = evaluate_candidate(
                candidate,
                evaluated_cases,
                total_cost_usd=outcome.cost_usd if complete_cost else None,
                cost_status=effective_cost_status,
                usage_source=outcome.usage_source,
                valid_pin=(
                    True
                    if outcome.native_result is None
                    else self._native_pin_is_valid(outcome.native_result)
                ),
                corpus_hash=corpus_sha,
                query_wall_time_ms=sum(
                    observation.latency_ms for observation in outcome.observations
                ),
            )
            execution_config = outcome.artifact.get("execution_config")
            keyword_only = (
                isinstance(execution_config, Mapping)
                and cast(Mapping[str, object], execution_config).get("keyword_only") is True
            )
            if outcome.native_result is not None:
                evaluation = evaluation.model_copy(update={"native_result": outcome.native_result})
            if outcome.status is not Status.OK:
                evaluation = evaluation.model_copy(
                    update={"status": outcome.status, "eligible_override": False}
                )
            elif keyword_only:
                evaluation = evaluation.model_copy(
                    update={
                        "eligible_override": False,
                        "eligibility_reasons": [
                            "GBrain keyword-only retrieval has no measured semantic quality"
                        ],
                    }
                )
            elif missing_identity:
                evaluation = evaluation.model_copy(
                    update={"status": Status.FAILED, "eligible_override": False}
                )
            receipt = self._cleanup_receipts.get(outcome.candidate)
            if outcome.native_result is not None and not cleanup_receipt_complete(receipt):
                evaluation = evaluation.model_copy(
                    update={
                        "eligible_override": False,
                        "eligibility_reasons": [
                            *evaluation.eligibility_reasons,
                            "cleanup receipt does not prove complete candidate cleanup",
                        ],
                    }
                )
            evaluations.append(evaluation.model_copy(update={"corpus_hash": corpus_sha}))
        return evaluations

    def _native_pin_is_valid(self, native: NativeCandidateResult | None) -> bool:
        if native is None:
            return False
        try:
            pin = next(
                item for item in load_candidate_pins().candidates if item.id is native.candidate
            )
        except (OSError, ValueError, StopIteration):
            return False
        return candidate_pin_matches(
            pin,
            distribution=native.backend.name,
            version=native.backend.version or "",
            commit=native.backend.commit,
        )

    def _cleanup_candidates(self, run_dir: Path) -> None:
        for candidate in self.candidates:
            if candidate.candidate_id in self._cleanup_attempted:
                continue
            self._cleanup_attempted.add(candidate.candidate_id)
            try:
                receipt = candidate.cleanup()
            except BaseException as exc:
                detail = f"{type(exc).__name__}: {exc}"
                self._cleanup_errors.append(f"{candidate.candidate_id}: {detail}")
                self._manifest.setdefault("cleanup", {})[candidate.candidate_id] = {
                    "candidate": candidate.candidate_id,
                    "complete": False,
                    "error": detail,
                }
                self._ledger.append(
                    {
                        "kind": "cleanup",
                        "candidate": candidate.candidate_id,
                        "status": Status.FAILED.value,
                        "detail": detail,
                    }
                )
                continue
            if receipt is None:
                continue
            self._cleanup_receipts[candidate.candidate_id] = receipt
            self._manifest.setdefault("cleanup", {})[candidate.candidate_id] = receipt.model_dump(
                mode="json"
            )
            complete = cleanup_receipt_complete(receipt)
            if not complete:
                self._cleanup_errors.append(f"{candidate.candidate_id}: incomplete cleanup receipt")
            self._ledger.append(
                {
                    "kind": "cleanup",
                    "candidate": candidate.candidate_id,
                    "status": Status.OK.value if complete else Status.FAILED.value,
                    "receipt": receipt.model_dump(mode="json"),
                }
            )
        self._persist(run_dir)

    def _write_evaluator_holdout(
        self,
        run_dir: Path,
        documents: Sequence[Mapping[str, Any]],
        holdout_ids: set[str],
    ) -> None:
        holdout_documents = [
            dict(document)
            for document in documents
            if str(document.get("source_id", "")) in holdout_ids
        ]
        self._write_json(
            run_dir / "evaluator" / "holdout.json",
            {
                "source_ids": sorted(holdout_ids),
                "documents": holdout_documents,
            },
        )
        self._manifest["benchmark"]["holdout_source_ids"] = sorted(holdout_ids)

    def _write_report(
        self,
        run_dir: Path,
        outcomes: Sequence[CandidateOutcome],
        verdict: str,
        *,
        evaluator_cases: Sequence[EvaluatorCaseRecord],
        candidate_documents: Sequence[Mapping[str, Any]],
        corpus_sha: str,
        benchmark_sha: str,
        decision: Any,
        evaluations: Sequence[CandidateEvaluation],
        status: Status = Status.OK,
    ) -> Path:
        del verdict
        coverage, coverage_warnings = self._canonical_coverage()
        warnings = [
            "same-model judge bias",
            "coverage/cost may be incomplete",
            *coverage_warnings,
        ]
        if self.test_mode.get("enabled") is True:
            warnings.append(
                "TEST_MODE: deterministic local fixture; no provider or network access."
            )
        artifact = build_comparison(
            run_id=str(self._manifest["run_id"]),
            status=status,
            corpus_hash=corpus_sha,
            benchmark_hash=benchmark_sha,
            coverage=coverage,
            candidates=list(evaluations),
            decision=decision,
            evidence=self._canonical_evidence(outcomes, evaluator_cases, candidate_documents),
            provenance=self.benchmark_provenance(evaluations),
            artifact_paths={
                "comparison_json": "comparison.json",
                "corpus_freeze": "corpus-freeze.json",
                "manifest": "manifest.json",
                "report_html": "report.html",
            },
            warnings=warnings,
            price_sheet_version="local-metering-v1",
        )
        canonical_run_dir = run_dir.resolve()
        for artifact_path in (run_dir / "report.html", run_dir / "comparison.json"):
            if artifact_path.is_symlink() or not artifact_path.resolve(strict=False).is_relative_to(
                canonical_run_dir
            ):
                raise ValueError(f"report artifact escapes run directory: {artifact_path}")
        return write_artifacts(artifact, run_dir).report_html

    def _write_terminal_artifacts(
        self,
        run_dir: Path,
        *,
        status: Status,
        warning: str,
    ) -> Path:
        candidate_ids = [
            CandidateId(candidate.candidate_id)
            for candidate in self.candidates
            if candidate.candidate_id in {item.value for item in CandidateId}
        ]
        if not candidate_ids:
            candidate_ids = [CandidateId.LLM_WIKI]
        evaluations = [
            CandidateEvaluation(
                candidate=candidate_id,
                status=status,
                scored_cases=0,
                answered_cases=0,
                quality_score=0,
                answer_success_rate=0,
                source_support_rate=0,
                contradiction_count=0,
                total_input_tokens=0,
                total_output_tokens=0,
                total_cost_usd=None,
                cost_status=CostStatus.INCOMPLETE,
                valid_pin=False,
                eligibility_reasons=[status.value, "run did not complete"],
                eligible_override=False,
            )
            for candidate_id in candidate_ids
        ]
        decision = DecisionResult(
            status=status,
            verdict=Verdict.NO_RECOMMENDATION,
            rationale="Run did not complete; no recommendation is available.",
            considered_candidates=candidate_ids,
        )
        artifact = build_comparison(
            run_id=str(self._manifest["run_id"]),
            status=status,
            corpus_hash=str(self._manifest.get("hashes", {}).get("corpus_sha256", "0" * 64)),
            benchmark_hash=str(self._manifest.get("hashes", {}).get("benchmark_sha256", "0" * 64)),
            coverage=[],
            candidates=evaluations,
            decision=decision,
            evidence=[],
            provenance=self.benchmark_provenance(evaluations),
            artifact_paths={
                "comparison_json": "comparison.json",
                "corpus_freeze": "corpus-freeze.json",
                "manifest": "manifest.json",
                "report_html": "report.html",
            },
            warnings=[warning],
            price_sheet_version="local-metering-v1",
        )
        artifacts = write_artifacts(artifact, run_dir)
        self._manifest["report"] = {
            "path": str(artifacts.report_html),
            "sha256": artifacts.report_sha256,
        }
        return artifacts.report_html

    def benchmark_provenance(
        self,
        evaluations: Sequence[CandidateEvaluation] = (),
    ) -> BenchmarkProvenance:
        subscription = self.config.provider_mode.endswith("-subscription")
        if self._chat_provenance_provider is not None:
            chat = self._chat_provenance_provider()
        else:
            chat = ChatProvenance(
                provider=(
                    self.config.provider_mode.removesuffix("-subscription")
                    if subscription
                    else "openai"
                ),
                model=(os.environ.get("AUTOBRAIN_SUBSCRIPTION_MODEL") or None)
                if subscription
                else "gpt-5-mini",
                cli_version=None,
                auth_kind="consumer_subscription" if subscription else "api_key",
            )
        embedding = self._embedding_descriptor.provenance
        sources = [
            SourceProvenance(
                source=provider.value,
                mutability=(
                    SourceMutability.FROZEN_EXPORT
                    if provider is Provider.SLACK and self.config.slack_export_path is not None
                    else SourceMutability.LIVE_MCP_CAPTURED
                ),
            )
            for provider in self.config.selected_sources
        ]
        spans = [
            span
            for outcome in getattr(self, "_candidate_outcomes", ())
            for span in outcome.latency_spans
        ]
        if not spans:
            spans = [
                LatencySpan(
                    name=LatencySpanKind.CANDIDATE_QUERY,
                    duration_ms=evaluation.query_wall_time_ms or None,
                    candidate=evaluation.candidate,
                )
                for evaluation in evaluations
            ]
        usage_sources = {evaluation.usage_source for evaluation in evaluations}
        usage_source = (
            next(iter(usage_sources)) if len(usage_sources) == 1 else UsageSource.UNAVAILABLE
        )
        return BenchmarkProvenance(
            chat=chat,
            embedding=embedding,
            usage_source=usage_source,
            sources=sources,
            latency_spans=spans,
            integrations=list(integration_catalog()),
        )

    def _canonical_coverage(self) -> tuple[list[CoverageRecord], list[str]]:
        source_kinds = {
            "slack": SourceKind.SLACK_MESSAGE,
            "notion": SourceKind.NOTION_PAGE,
        }
        records: list[CoverageRecord] = []
        warnings: list[str] = []
        raw_coverage = cast(Mapping[str, Mapping[str, Any]], self._manifest.get("coverage", {}))
        for provider, value in sorted(raw_coverage.items()):
            raw_completeness = str(value.get("completeness", CoverageCompleteness.UNKNOWN.value))
            try:
                completeness = CoverageCompleteness(raw_completeness)
            except ValueError:
                completeness = CoverageCompleteness.UNKNOWN
                warnings.append(f"{provider} coverage: {raw_completeness}")
            records.append(
                CoverageRecord(
                    source=source_kinds.get(provider, SourceKind.SLACK_MESSAGE),
                    completeness=completeness,
                    discovered=self._coverage_count(value, "discovered"),
                    fetched=self._coverage_count(value, "fetched"),
                    skipped=self._coverage_count(value, "skipped"),
                    truncated=self._coverage_count(value, "truncated"),
                    denied=self._coverage_count(value, "denied"),
                    rate_limited=self._coverage_count(value, "rate_limited"),
                    unsupported=self._coverage_count(value, "unsupported"),
                )
            )
        return records, warnings

    @staticmethod
    def _coverage_count(value: Mapping[str, Any], key: str) -> int:
        count = value.get(key, 0)
        return count if isinstance(count, int) and count >= 0 else 0

    @staticmethod
    def _canonical_evidence(
        outcomes: Sequence[CandidateOutcome],
        evaluator_cases: Sequence[EvaluatorCaseRecord],
        candidate_documents: Sequence[Mapping[str, Any]],
    ) -> list[CandidateCaseEvidence]:
        urls_by_source = {
            str(document.get("source_id")): normalized
            for document in candidate_documents
            if isinstance(document.get("source_id"), str)
            and isinstance(document.get("canonical_url"), str)
            and (normalized := normalize_safe_source_url(str(document["canonical_url"])))
            is not None
        }
        evidence: list[CandidateCaseEvidence] = []
        for outcome in outcomes:
            candidate = CandidateId(outcome.candidate)
            observations_by_case = {
                observation.case_id: observation
                for observation in outcome.observations
                if observation.case_id and observation.candidate == candidate
            }
            for evaluator_case in evaluator_cases:
                observation = observations_by_case.get(evaluator_case.case.case_id)
                if observation is None:
                    case_evaluation = evaluate_case(
                        evaluator_case.case,
                        answer="",
                        cited_source_ids=[],
                        reference_confidence=evaluator_case.reference_confidence,
                        status=Status.FAILED,
                        failure_detail="candidate produced no observation",
                        candidate=candidate,
                    )
                    cited_source_ids: list[str] = []
                else:
                    cited_source_ids = [
                        source_id
                        for source_id in observation.source_ids
                        if re.fullmatch(r"[a-z][a-z0-9_-]*:.+", source_id)
                    ]
                    case_evaluation = evaluate_case(
                        evaluator_case.case,
                        answer=observation.answer,
                        cited_source_ids=cited_source_ids,
                        reference_confidence=evaluator_case.reference_confidence,
                        status=observation.status,
                        failure_detail="; ".join(observation.warnings),
                        latency_ms=observation.latency_ms,
                        candidate=candidate,
                    )
                evidence.append(
                    CandidateCaseEvidence(
                        candidate=candidate,
                        case_id=evaluator_case.case.case_id,
                        status=case_evaluation.status,
                        score=case_evaluation.score,
                        source_ids=cited_source_ids,
                        source_urls=[
                            urls_by_source[source_id]
                            for source_id in cited_source_ids
                            if source_id in urls_by_source
                        ],
                        cited_claims=case_evaluation.cited_claims,
                        required_claims=case_evaluation.required_claims,
                        failure_detail=(
                            case_evaluation.failure_detail
                            if case_evaluation.status is not Status.OK
                            else ""
                        ),
                    )
                )
        return evidence

    @staticmethod
    def _cost_label(outcome: CandidateOutcome) -> str:
        evaluation = outcome.evaluation
        if (
            evaluation is None
            or evaluation.cost_status is not CostStatus.COMPLETE
            or evaluation.total_cost_usd is None
        ):
            status = (
                evaluation.cost_status.value
                if evaluation is not None
                else CostStatus.INCOMPLETE.value
            )
            return f"unknown ({status})"
        return f"${evaluation.total_cost_usd:.4f}"

    def _stage(self, run_dir: Path, name: str, status: Status, detail: str) -> None:
        raw_entry = {
            "sequence": len(self._stages) + 1,
            "run_id": str(self._manifest["run_id"]),
            "name": name,
            "status": status.value,
            "detail": detail,
            "started_at": self.now().isoformat(),
        }
        persisted_entry = cast(
            dict[str, Any],
            redact(raw_entry, known_secrets=self._known_secrets),
        )
        self._stages.append(persisted_entry)
        self._persist(run_dir)
        if self._stage_event_sink is None:
            return
        event = StageEvent(
            sequence=int(persisted_entry["sequence"]),
            run_id=str(persisted_entry["run_id"]),
            name=str(persisted_entry["name"]),
            status=Status(str(persisted_entry["status"])),
            detail=str(persisted_entry["detail"]),
            started_at=str(persisted_entry["started_at"]),
        )
        try:
            self._stage_event_sink(event)
        except BaseException as exc:
            diagnostic = self._sink_diagnostic(name, exc)
            self._event_sink_errors.append(diagnostic)
            warnings = self._manifest.setdefault("warnings", [])
            if diagnostic not in warnings:
                warnings.append(diagnostic)

    def _sink_diagnostic(self, stage_name: str, exc: BaseException) -> str:
        raw = f"stage event sink failed after {stage_name}: {type(exc).__name__}"
        sanitized = str(redact(raw, known_secrets=self._known_secrets))
        return sanitized if len(sanitized) <= 500 else sanitized[:497] + "..."

    def _record_pins(self, run_dir: Path) -> None:
        try:
            pins = load_candidate_pins()
            self._manifest["pins"] = {
                "candidates": [item.id.value for item in pins.candidates],
                "versions": {item.id.value: item.version for item in pins.candidates},
            }
        except Exception as exc:
            self._manifest["pins"] = {"status": Status.FAILED.value, "detail": str(exc)}
        self._persist(run_dir)

    def _create_run_dir(self, run_id: str) -> Path:
        root = self.config.output or AutoBrainPaths.from_home().runs
        AutoBrainPaths.validate_output_root(root)
        if root.is_symlink():
            raise ValueError(f"output root cannot be a symlink: {root}")
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if root.resolve() != root.parent.resolve() / root.name:
            raise ValueError(f"output root escapes its canonical parent: {root}")
        if (
            not run_id
            or run_id in {".", ".."}
            or not all(
                character.isascii() and (character.isalnum() or character in "._-")
                for character in run_id
            )
        ):
            raise ValueError(f"invalid run id: {run_id!r}")
        run_dir = root / run_id
        if run_dir.is_symlink() or not run_dir.resolve(strict=False).is_relative_to(root.resolve()):
            raise ValueError(f"run path escapes output root: {run_dir}")
        try:
            run_dir.mkdir(mode=0o700)
        except FileExistsError as exc:
            raise OccupiedRunError(f"run already exists: {run_dir}") from exc
        (run_dir / "candidates").mkdir(mode=0o700)
        return run_dir

    @staticmethod
    def _new_run_id() -> str:
        return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]

    @staticmethod
    def _hash_json(value: object) -> str:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _outcome_json(outcome: CandidateOutcome) -> dict[str, Any]:
        return {
            "candidate": outcome.candidate,
            "status": outcome.status.value,
            "score": outcome.score,
            "answered_cases": outcome.answered_cases,
            "scored_cases": outcome.scored_cases,
            "cost_usd": outcome.cost_usd,
            "latency_ms": outcome.latency_ms or None,
            "latency_spans": [span.model_dump(mode="json") for span in outcome.latency_spans],
            "detail": outcome.detail,
            "cost_status": outcome.cost_status.value,
            "usage_source": outcome.usage_source.value,
            "artifact": dict(outcome.artifact),
            "observations": [
                observation.model_dump(mode="json") for observation in outcome.observations
            ],
            "evaluation": (
                outcome.evaluation.model_dump(mode="json")
                if outcome.evaluation is not None
                else None
            ),
            "native_result": (
                outcome.native_result.model_dump(mode="json")
                if outcome.native_result is not None
                else None
            ),
        }

    def _persist(self, run_dir: Path) -> None:
        self._manifest["stages"] = list(self._stages)
        self._manifest["commands"] = list(self._ledger)
        self._write_json(run_dir / "manifest.json", self._manifest)

    def _write_json(self, path: Path, value: object) -> None:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        payload = (
            json.dumps(
                redact(value, known_secrets=self._known_secrets),
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(path)


def locate_run(run_id: str, *, roots: Sequence[Path]) -> Path | None:
    """Find a run without following arbitrary symlinks or touching the network."""
    if (
        not run_id
        or run_id in {".", ".."}
        or any(
            not (character.isascii() and (character.isalnum() or character in "._-"))
            for character in run_id
        )
    ):
        return None
    for root in roots:
        if root.is_symlink() or not root.is_dir():
            continue
        canonical_root = root.resolve()
        candidate = root / run_id
        if (
            candidate.is_dir()
            and not candidate.is_symlink()
            and candidate.resolve(strict=False).is_relative_to(canonical_root)
            and (candidate / "manifest.json").is_file()
            and not (candidate / "manifest.json").is_symlink()
            and (candidate / "report.html")
            .resolve(strict=False)
            .is_relative_to(candidate.resolve())
        ):
            return candidate
        for manifest in root.rglob("manifest.json"):
            if (
                manifest.parent.name == run_id
                and not manifest.is_symlink()
                and manifest.parent.resolve(strict=False).is_relative_to(canonical_root)
                and (manifest.parent / "report.html")
                .resolve(strict=False)
                .is_relative_to(manifest.parent.resolve())
            ):
                return manifest.parent
    return None
