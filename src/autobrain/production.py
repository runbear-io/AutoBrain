"""Production composition for the approved MCP and candidate adapters.

This module contains no product-service clients.  It only composes the
provider-specific read-only MCP transport and the already pinned candidate
adapters into the orchestration protocol.
"""

from __future__ import annotations

import asyncio
import json
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from time import monotonic
from typing import Any, cast

import anyio

from autobrain.auth.models import Provider
from autobrain.auth.oauth import OAuthManager
from autobrain.auth.providers import config_for
from autobrain.auth.service import ConnectionManager
from autobrain.cancellation import RunCancellation, RunCancelled
from autobrain.candidates.gbrain import (
    GBrainAdapter,
    GBrainMissingProviderError,
    GBrainProcessError,
)
from autobrain.candidates.llm_wiki import LLMWikiAdapter, LLMWikiConfig, LLMWikiRunResult
from autobrain.candidates.mem0 import (
    Mem0Adapter,
    Mem0AdapterConfig,
    Mem0MissingProviderError,
)
from autobrain.connectors.slack_export import SlackExportConnector, SlackExportCrawlResult
from autobrain.mcp.transport import StreamableHttpConnection
from autobrain.metering import (
    BudgetExceededError,
    LoopbackMeteringProxy,
    MeteringBudget,
    MeteringRole,
    MeteringSummary,
    load_price_sheet,
    reconcile_usage,
)
from autobrain.models import (
    CandidateId,
    CandidateObservation,
    CandidateQuery,
    CostStatus,
    LatencySpan,
    LatencySpanKind,
    Status,
)
from autobrain.orchestration import Candidate, CandidateContext, CandidateOutcome, ConnectorSnapshot
from autobrain.paths import AutoBrainPaths


def _provider_upstream(
    api_key: str,
    base_url: str | None,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    root = (base_url or "https://api.openai.com/v1").rstrip("/")

    def request(payload: dict[str, Any]) -> dict[str, Any]:
        model = str(payload.get("model", ""))
        suffix = "/embeddings" if model.startswith("text-embedding") else "/chat/completions"
        endpoint = root + suffix if root.endswith("/v1") else root + "/v1" + suffix
        outgoing = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(outgoing, timeout=1.0) as response:
                value = json.loads(response.read())
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"provider request failed ({type(error).__name__})") from None
        if not isinstance(value, dict):
            raise RuntimeError("provider returned a non-object response")
        return cast(dict[str, Any], value)

    return request


def _proxy_summary(
    proxy: LoopbackMeteringProxy,
    candidate: str,
    *,
    native_usage: Mapping[str, int] | None = None,
) -> MeteringSummary:
    return reconcile_usage(
        proxy.events,
        proxy.prices,
        native_usage=native_usage,
        role=MeteringRole.CANDIDATE,
        candidate=candidate,
    )


def _provider_spans(
    proxy: LoopbackMeteringProxy,
    candidate: CandidateId,
) -> tuple[LatencySpan, ...]:
    measured = [
        event.provider_execution_ms
        for event in proxy.events
        if event.candidate == candidate.value and event.provider_execution_ms is not None
    ]
    return (
        LatencySpan(
            name=LatencySpanKind.PROVIDER_EXECUTION,
            duration_ms=sum(measured) or None,
            candidate=candidate,
        ),
    )


def _budget_outcome(
    candidate: str,
    proxy: LoopbackMeteringProxy,
    detail: str,
) -> CandidateOutcome:
    return CandidateOutcome(
        candidate=candidate,
        status=Status.BUDGET_EXCEEDED,
        detail=detail,
        cost_status=CostStatus.INCOMPLETE,
        usage_source=_proxy_summary(proxy, candidate).usage_source,
        artifact={"metering": _proxy_summary(proxy, candidate).model_dump(mode="json")},
    )


def _run_async(coro: Any, cancellation: RunCancellation | None = None) -> Any:
    async def execute() -> Any:
        task = asyncio.create_task(coro)
        loop = asyncio.get_running_loop()

        def cancel_task() -> None:
            loop.call_soon_threadsafe(task.cancel)

        remove_callback = (
            cancellation.add_callback(cancel_task) if cancellation is not None else lambda: None
        )
        try:
            if cancellation is not None:
                cancellation.raise_if_cancelled()
            return await task
        except asyncio.CancelledError:
            if cancellation is not None and cancellation.cancelled:
                raise RunCancelled("operator cancelled run") from None
            raise
        finally:
            remove_callback()

    return anyio.run(execute)


class _McpConnector:
    def __init__(self, provider: Provider, connection: StreamableHttpConnection) -> None:
        self.provider = provider.value
        self.connection = connection

    def probe(self, cancellation: RunCancellation | None = None) -> Mapping[str, Any]:
        async def inspect() -> Mapping[str, Any]:
            async with self.connection as connection:
                snapshot = connection.snapshot
                if self.provider_enum is Provider.SLACK:
                    required_groups = (
                        {"slack-channel-list", "slack_channels_list"},
                        {"slack-channel-history", "slack_channel_history"},
                    )
                else:
                    required_groups = (
                        {"notion-search", "notion_search"},
                        {"notion-fetch", "notion_fetch"},
                    )
                allowed = set(snapshot.allowed)
                return {
                    "advertised": list(snapshot.advertised),
                    "allowed": list(snapshot.allowed),
                    "required": [sorted(group) for group in required_groups],
                    "capability_available": all(group & allowed for group in required_groups),
                }

        return cast(Mapping[str, Any], _run_async(inspect(), cancellation))

    @property
    def provider_enum(self) -> Provider:
        return Provider(self.provider)


class SlackMcpConnector(_McpConnector):
    def __init__(
        self,
        connection: StreamableHttpConnection,
        *,
        include_dms: bool = False,
    ) -> None:
        super().__init__(Provider.SLACK, connection)
        self.include_dms = include_dms

    def crawl(
        self,
        *,
        include_dms: bool,
        cancellation: RunCancellation | None = None,
    ) -> ConnectorSnapshot:
        from autobrain.connectors.slack import SlackCrawler

        async def collect() -> ConnectorSnapshot:
            async with self.connection as connection:
                result = await SlackCrawler(
                    connection,
                    include_dms=include_dms or self.include_dms,
                ).crawl(scopes=config_for(Provider.SLACK).scopes)
                return ConnectorSnapshot(
                    provider=self.provider,
                    documents=tuple(
                        document.model_dump(mode="json") for document in result.documents
                    ),
                    coverage=result.coverage.model_dump(mode="json"),
                )

        return cast(ConnectorSnapshot, _run_async(collect(), cancellation))


class SlackExportSourceConnector:
    provider = Provider.SLACK.value

    def __init__(
        self,
        archive_path: Path,
        *,
        expected_sha256: str | None = None,
    ) -> None:
        self.archive_path = archive_path
        self._connector = SlackExportConnector(
            archive_path,
            expected_sha256=expected_sha256,
        )

    def probe(self, cancellation: RunCancellation | None = None) -> Mapping[str, Any]:
        return cast(Mapping[str, Any], _run_async(self._connector.probe(), cancellation))

    def crawl(
        self,
        *,
        include_dms: bool,
        cancellation: RunCancellation | None = None,
    ) -> ConnectorSnapshot:
        del include_dms
        result = cast(
            SlackExportCrawlResult,
            _run_async(self._connector.crawl(), cancellation),
        )
        return ConnectorSnapshot(
            provider=self.provider,
            documents=tuple(document.model_dump(mode="json") for document in result.documents),
            coverage=result.coverage,
        )


class NotionMcpConnector(_McpConnector):
    def __init__(self, connection: StreamableHttpConnection) -> None:
        super().__init__(Provider.NOTION, connection)

    def crawl(
        self,
        *,
        include_dms: bool,
        cancellation: RunCancellation | None = None,
    ) -> ConnectorSnapshot:
        del include_dms
        from autobrain.connectors.notion import NotionCrawler

        async def collect() -> ConnectorSnapshot:
            async with self.connection as connection:
                result = await NotionCrawler(connection).crawl()
                return ConnectorSnapshot(
                    provider=self.provider,
                    documents=tuple(
                        document.model_dump(mode="json") for document in result.documents
                    ),
                    coverage=result.coverage.model_dump(mode="json"),
                )

        return cast(ConnectorSnapshot, _run_async(collect(), cancellation))


def build_production_connectors(
    manager: ConnectionManager,
    *,
    include_dms: bool = False,
    providers: Sequence[Provider] = (Provider.SLACK, Provider.NOTION),
    slack_export_path: Path | None = None,
    slack_export_sha256: str | None = None,
) -> tuple[SlackMcpConnector | SlackExportSourceConnector | NotionMcpConnector, ...]:
    oauth = OAuthManager(manager.store)
    connectors: dict[
        Provider,
        SlackMcpConnector | SlackExportSourceConnector | NotionMcpConnector,
    ] = {}
    if Provider.SLACK in providers:
        if slack_export_path is not None:
            connectors[Provider.SLACK] = SlackExportSourceConnector(
                slack_export_path,
                expected_sha256=slack_export_sha256,
            )
        else:
            slack_token = manager.token_for(Provider.SLACK)
            if slack_token is None:
                raise ValueError("MCP_AUTH_UNAVAILABLE: authenticated Slack token required")
            connectors[Provider.SLACK] = SlackMcpConnector(
                StreamableHttpConnection.with_oauth(
                    Provider.SLACK,
                    config_for(Provider.SLACK).resource,
                    slack_token,
                    manager=oauth,
                ),
                include_dms=include_dms,
            )
    if Provider.NOTION in providers:
        notion_token = manager.token_for(Provider.NOTION)
        if notion_token is None:
            raise ValueError("MCP_AUTH_UNAVAILABLE: authenticated Notion token required")
        connectors[Provider.NOTION] = NotionMcpConnector(
            StreamableHttpConnection.with_oauth(
                Provider.NOTION,
                config_for(Provider.NOTION).resource,
                notion_token,
                manager=oauth,
            )
        )
    return tuple(connectors[provider] for provider in providers)


def _queries(context: CandidateContext) -> tuple[CandidateQuery, ...]:
    return tuple(
        CandidateQuery(
            case_id=case_id,
            question=question,
        )
        for case_id, question in zip(context.case_ids, context.questions, strict=True)
    )


class LLMWikiCandidate:
    candidate_id = CandidateId.LLM_WIKI.value

    def __init__(
        self,
        adapter: LLMWikiAdapter,
        *,
        api_key: str,
        metering_proxy: LoopbackMeteringProxy,
    ) -> None:
        self.adapter = adapter
        self.api_key = api_key
        self.metering_proxy = metering_proxy

    def run(self, context: CandidateContext) -> CandidateOutcome:
        context.cancellation.raise_if_cancelled()
        queries = _queries(context)
        try:
            result: LLMWikiRunResult = self.adapter.run(
                context.normalized_documents,
                queries,
                api_key=self.api_key,
                cancellation=context.cancellation,
            )
            context.cancellation.raise_if_cancelled()
        except BudgetExceededError as exc:
            return _budget_outcome(self.candidate_id, self.metering_proxy, str(exc))
        incomplete_codes = {
            "COST_INCOMPLETE",
            "METERING_MALFORMED",
            "METERING_UNAVAILABLE",
        }
        summary = _proxy_summary(self.metering_proxy, self.candidate_id)
        if self.metering_proxy.budget_exceeded:
            return _budget_outcome(
                self.candidate_id,
                self.metering_proxy,
                "BUDGET_EXCEEDED: provider usage crossed the hard cap",
            )
        complete_cost = (
            summary.cost_status is CostStatus.COMPLETE
            and summary.usd is not None
            and result.measured_cost_usd is not None
            and not any(warning.code in incomplete_codes for warning in result.warnings)
        )
        candidate_observations = tuple(
            CandidateObservation(
                candidate=CandidateId.LLM_WIKI,
                case_id=observation.case_id,
                status=Status.OK,
                answer=observation.answer,
                source_ids=list(observation.source_ids),
                latency_ms=observation.latency_ms,
                warnings=list(observation.warnings),
            )
            for observation in result.observations
        )
        return CandidateOutcome(
            candidate=self.candidate_id,
            status=result.status,
            answered_cases=len(result.observations),
            scored_cases=len(queries),
            cost_usd=summary.usd if complete_cost else None,
            latency_ms=result.elapsed_ms,
            detail="; ".join(warning.message for warning in result.warnings),
            artifact={
                **asdict(result),
                "metering": summary.model_dump(mode="json"),
            },
            observations=candidate_observations,
            cost_status=CostStatus.COMPLETE if complete_cost else CostStatus.INCOMPLETE,
            usage_source=summary.usage_source,
            latency_spans=(
                LatencySpan(
                    name=LatencySpanKind.CANDIDATE_QUERY,
                    duration_ms=sum(item.latency_ms for item in candidate_observations) or None,
                    candidate=CandidateId.LLM_WIKI,
                ),
                *_provider_spans(self.metering_proxy, CandidateId.LLM_WIKI),
            ),
        )

    def cleanup(self) -> None:
        self.metering_proxy.close()


class Mem0Candidate:
    candidate_id = CandidateId.MEM0.value

    def __init__(
        self,
        adapter: Mem0Adapter,
        *,
        metering_proxy: LoopbackMeteringProxy,
    ) -> None:
        self.adapter = adapter
        self.metering_proxy = metering_proxy

    def run(self, context: CandidateContext) -> CandidateOutcome:
        try:
            context.cancellation.raise_if_cancelled()
            queries = _queries(context)
            ingest_started = monotonic()
            self.adapter.ingest(
                context.normalized_documents,
                cancellation=context.cancellation,
            )
            context.cancellation.raise_if_cancelled()
            ingest_ms = round((monotonic() - ingest_started) * 1000)
            candidate_observations: list[CandidateObservation] = []
            query_ms = 0
            for query in queries:
                context.cancellation.raise_if_cancelled()
                query_started = monotonic()
                question = query.question
                native = self.adapter.search_native(
                    question,
                    cancellation=context.cancellation,
                )
                answer = self.adapter.answer(
                    question,
                    native["results"],
                    cancellation=context.cancellation,
                )
                latency_ms = round((monotonic() - query_started) * 1000)
                query_ms += latency_ms
                candidate_observations.append(
                    CandidateObservation(
                        candidate=CandidateId.MEM0,
                        case_id=query.case_id,
                        status=Status.OK,
                        answer=answer.answer,
                        source_ids=list(answer.source_ids),
                        latency_ms=latency_ms,
                    )
                )
            summary = _proxy_summary(self.metering_proxy, self.candidate_id)
            if self.metering_proxy.budget_exceeded:
                return _budget_outcome(
                    self.candidate_id,
                    self.metering_proxy,
                    "BUDGET_EXCEEDED: provider usage crossed the hard cap",
                )
            return CandidateOutcome(
                candidate=self.candidate_id,
                status=Status.OK,
                answered_cases=len(candidate_observations),
                scored_cases=len(queries),
                cost_usd=summary.usd if summary.cost_status is CostStatus.COMPLETE else None,
                latency_ms=ingest_ms + query_ms,
                artifact={
                    "pin": "mem0ai==2.0.18",
                    "usage_events": [
                        event.model_dump(mode="json") for event in self.metering_proxy.events
                    ],
                    "metering": summary.model_dump(mode="json"),
                },
                observations=tuple(candidate_observations),
                cost_status=(
                    CostStatus.COMPLETE
                    if summary.cost_status is CostStatus.COMPLETE
                    else CostStatus.INCOMPLETE
                ),
                usage_source=summary.usage_source,
                latency_spans=(
                    LatencySpan(
                        name=LatencySpanKind.CANDIDATE_INGEST,
                        duration_ms=ingest_ms or None,
                        candidate=CandidateId.MEM0,
                    ),
                    LatencySpan(
                        name=LatencySpanKind.CANDIDATE_QUERY,
                        duration_ms=query_ms or None,
                        candidate=CandidateId.MEM0,
                    ),
                    *_provider_spans(self.metering_proxy, CandidateId.MEM0),
                ),
            )
        except Mem0MissingProviderError as exc:
            return CandidateOutcome(
                candidate=self.candidate_id,
                status=Status.MISSING_PROVIDER,
                detail=str(exc),
                cost_status=CostStatus.INCOMPLETE,
            )
        except Exception:
            if self.metering_proxy.budget_exceeded:
                return _budget_outcome(
                    self.candidate_id,
                    self.metering_proxy,
                    "BUDGET_EXCEEDED: provider usage crossed the hard cap",
                )
            raise

    def cleanup(self) -> None:
        try:
            self.adapter.cleanup()
        finally:
            self.metering_proxy.close()


class GBrainCandidate:
    candidate_id = CandidateId.GBRAIN.value

    def __init__(
        self,
        adapter: GBrainAdapter,
        *,
        base_url: str | None = None,
        metering_proxy: LoopbackMeteringProxy,
    ):
        self.adapter = adapter
        self.base_url = base_url
        self.metering_proxy = metering_proxy

    def run(self, context: CandidateContext) -> CandidateOutcome:
        context.cancellation.raise_if_cancelled()
        queries = _queries(context)
        try:
            results = self.adapter.run(
                context.normalized_documents,
                context.questions,
                base_url=self.base_url,
                strict_base_url=True,
                cancellation=context.cancellation,
            )
            context.cancellation.raise_if_cancelled()
        except GBrainMissingProviderError as exc:
            return CandidateOutcome(
                candidate=self.candidate_id,
                status=Status.MISSING_PROVIDER,
                detail=str(exc),
                cost_status=CostStatus.INCOMPLETE,
            )
        except GBrainProcessError as exc:
            return CandidateOutcome(
                candidate=self.candidate_id,
                status=Status.FAILED,
                detail=str(exc),
                cost_status=CostStatus.INCOMPLETE,
            )
        known_sources = {document.source_id for document in context.normalized_documents}
        candidate_observations: list[CandidateObservation] = []
        for query, result in zip(queries, results, strict=False):
            context.cancellation.raise_if_cancelled()
            cited_sources = [
                citation
                for citation in result.citations
                if isinstance(citation, str) and citation in known_sources
            ]
            candidate_observations.append(
                CandidateObservation(
                    candidate=CandidateId.GBRAIN,
                    case_id=query.case_id,
                    status=Status.OK if result.status == "OK" else Status.FAILED,
                    answer=result.answer,
                    source_ids=cited_sources,
                    latency_ms=result.timings_ms.get("total_query", 0),
                    warnings=list(result.warnings),
                )
            )
        for query in queries[len(results) :]:
            candidate_observations.append(
                CandidateObservation(
                    candidate=CandidateId.GBRAIN,
                    case_id=query.case_id,
                    status=Status.FAILED,
                    latency_ms=0,
                    warnings=["candidate produced no observation"],
                )
            )
        successful = [item for item in results if item.status == "OK"]
        native_usage_values = [item.usage for item in results]
        native_usage = (
            {
                "input_tokens": sum(item["input_tokens"] for item in native_usage_values if item),
                "output_tokens": sum(item["output_tokens"] for item in native_usage_values if item),
            }
            if results and all(item is not None for item in native_usage_values)
            else None
        )
        summary = _proxy_summary(
            self.metering_proxy,
            self.candidate_id,
            native_usage=native_usage,
        )
        if self.metering_proxy.budget_exceeded:
            return _budget_outcome(
                self.candidate_id,
                self.metering_proxy,
                "BUDGET_EXCEEDED: provider usage crossed the hard cap",
            )
        complete_cost = summary.cost_status is CostStatus.COMPLETE and summary.usd is not None
        total_cost = summary.usd if complete_cost else None
        return CandidateOutcome(
            candidate=self.candidate_id,
            status=Status.OK if len(successful) == len(results) else Status.FAILED,
            answered_cases=len(successful),
            scored_cases=len(queries),
            cost_usd=total_cost,
            artifact={
                "pin": {
                    "version": "0.46.19.0",
                    "commit": "f49ca569232dbc0d8e0783d84606115e3bfe5ab1",
                },
                "results": [asdict(item) for item in results],
                "metering": summary.model_dump(mode="json"),
            },
            observations=tuple(candidate_observations),
            cost_status=CostStatus.COMPLETE if complete_cost else CostStatus.INCOMPLETE,
            usage_source=summary.usage_source,
            latency_spans=(
                LatencySpan(
                    name=LatencySpanKind.CANDIDATE_QUERY,
                    duration_ms=(
                        sum(observation.latency_ms for observation in candidate_observations)
                        or None
                    ),
                    candidate=CandidateId.GBRAIN,
                ),
                *_provider_spans(self.metering_proxy, CandidateId.GBRAIN),
            ),
        )

    def cleanup(self) -> None:
        try:
            self.adapter.cleanup()
        finally:
            self.metering_proxy.close()


def build_production_candidates(
    run_dir: Path,
    *,
    api_key: str,
    paths: AutoBrainPaths | None = None,
    base_url: str | None = None,
    budget_usd: float = 25.0,
    provider_upstream: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    candidate_ids: Sequence[CandidateId] = tuple(CandidateId),
) -> tuple[Candidate, ...]:
    if not api_key:
        raise ValueError("MISSING_PROVIDER: OPENAI_API_KEY is unavailable")
    state = paths or AutoBrainPaths.from_home()
    native_root = run_dir / "native"
    price_sheet = load_price_sheet()
    budget = MeteringBudget(budget_usd, price_sheet)
    upstream = provider_upstream or _provider_upstream(api_key, base_url)
    proxies = {
        candidate_id: LoopbackMeteringProxy(
            upstream,
            budget_state=budget,
            default_candidate=candidate_id.value,
        )
        for candidate_id in candidate_ids
    }
    for proxy in proxies.values():
        proxy.__enter__()
    adapters: dict[CandidateId, Candidate] = {}
    if CandidateId.LLM_WIKI in proxies:
        adapters[CandidateId.LLM_WIKI] = LLMWikiCandidate(
            LLMWikiAdapter(
                LLMWikiConfig(
                    workspace=native_root / "llm-wiki",
                    tool_cache=state.tool_dir(CandidateId.LLM_WIKI.value),
                    base_url=proxies[CandidateId.LLM_WIKI].base_url,
                )
            ),
            api_key=api_key,
            metering_proxy=proxies[CandidateId.LLM_WIKI],
        )
    if CandidateId.MEM0 in proxies:
        adapters[CandidateId.MEM0] = Mem0Candidate(
            Mem0Adapter(
                Mem0AdapterConfig(
                    run_id=run_dir.name,
                    run_dir=native_root / "mem0",
                    heldout_source_ids=set(),
                    api_key=api_key,
                    base_url=proxies[CandidateId.MEM0].base_url,
                )
            ),
            metering_proxy=proxies[CandidateId.MEM0],
        )
    if CandidateId.GBRAIN in proxies:
        adapters[CandidateId.GBRAIN] = GBrainCandidate(
            GBrainAdapter(
                tools_root=state.tools,
                run_root=native_root / "gbrain",
            ),
            base_url=proxies[CandidateId.GBRAIN].base_url,
            metering_proxy=proxies[CandidateId.GBRAIN],
        )
    return tuple(adapters[candidate_id] for candidate_id in candidate_ids)
