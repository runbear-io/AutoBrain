"""Run-local OpenAI-compatible metering and honest cost reconciliation."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import monotonic
from typing import Any, cast

from pydantic import Field, model_validator

from autobrain.models import CostStatus, StrictModel, UsageSource

COST_COMPLETE = CostStatus.COMPLETE
COST_INCOMPLETE = CostStatus.INCOMPLETE
COST_UNAVAILABLE = CostStatus.UNAVAILABLE


class BudgetExceededError(RuntimeError):
    """A measured metered operation would exceed the configured hard cap."""

    def __init__(self, budget_usd: float, spent_usd: float) -> None:
        self.budget_usd = budget_usd
        self.spent_usd = spent_usd
        super().__init__(
            f"BUDGET_EXCEEDED: measured spend ${spent_usd:.10f} exceeds hard cap ${budget_usd:.10f}"
        )


class MeteringRole(StrEnum):
    GENERATOR = "generator"
    ORACLE = "oracle"
    HARNESS = "harness"
    CANDIDATE = "candidate"


class PriceQuote(StrictModel):
    input_usd_per_million: float = Field(ge=0)
    output_usd_per_million: float = Field(ge=0)


class PriceSheet(StrictModel):
    version: str = Field(min_length=1)
    effective_date: str = Field(min_length=1)
    models: dict[str, PriceQuote] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def qualify_legacy_openai_keys(cls, data: object) -> object:
        if not isinstance(data, Mapping):
            return data
        values = dict(cast(Mapping[str, Any], data))
        raw_models = values.get("models")
        if isinstance(raw_models, Mapping):
            models = cast(Mapping[str, Any], raw_models)
            values["models"] = {
                key if ":" in str(key) else f"openai:{key}": value for key, value in models.items()
            }
        return values


class MeteringEvent(StrictModel):
    event_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    candidate: str = Field(min_length=1)
    phase: str = Field(min_length=1)
    role: MeteringRole = MeteringRole.CANDIDATE
    provider: str = Field(default="openai", min_length=1)
    model: str = Field(min_length=1)
    usage_source: UsageSource = UsageSource.MEASURED
    provider_execution_ms: float | None = Field(default=None, gt=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    raw_usage: dict[str, Any] = Field(default_factory=dict)
    tags: dict[str, str] = Field(default_factory=dict)
    native: bool = False

    @model_validator(mode="before")
    @classmethod
    def canonical_tags(cls, data: object) -> object:
        if not isinstance(data, Mapping):
            return data
        values = dict(cast(Mapping[str, Any], data))
        candidate = values.get("candidate")
        phase = values.get("phase")
        role = values.get("role", MeteringRole.CANDIDATE)
        role_value = role.value if isinstance(role, MeteringRole) else str(role)
        required = {"candidate": candidate, "phase": phase, "role": role_value}
        tags = dict(cast(Mapping[str, str], values.get("tags", {})))
        for key, value in required.items():
            if value is None:
                continue
            if key in tags and tags[key] != value:
                raise ValueError(f"metering tag {key!r} does not match event fields")
            tags[key] = value
        values["tags"] = tags
        return values


class MeteringSummary(StrictModel):
    cost_status: CostStatus
    usage_source: UsageSource = UsageSource.UNAVAILABLE
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    usd: float | None = Field(default=None, ge=0)
    price_sheet_version: str | None = None
    native_usage: dict[str, int] | None = None
    raw_events: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


@dataclass
class MeteringBudget:
    """Shared run-local spend ledger used by every candidate boundary."""

    budget_usd: float
    prices: PriceSheet
    spent_usd: float = 0.0
    budget_exceeded: bool = False
    events: list[MeteringEvent] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        if self.budget_usd <= 0:
            raise ValueError("budget_usd must be greater than 0")


def load_price_sheet(path: Path | None = None) -> PriceSheet:
    """Load and validate the committed, versioned price sheet."""
    price_path = path or Path(__file__).with_name("openai-prices-2026-08.json")
    try:
        payload = json.loads(price_path.read_text(encoding="utf-8"))
        return PriceSheet.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"corrupt price sheet: {price_path}") from error


def _usage_count(raw_usage: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = raw_usage.get(key)
        if type(value) is int and value >= 0:
            return value
    return None


def _event_cost(event: MeteringEvent, quote: PriceQuote) -> float | None:
    if event.input_tokens is None or event.output_tokens is None:
        return None
    return (
        event.input_tokens * quote.input_usd_per_million / 1_000_000
        + event.output_tokens * quote.output_usd_per_million / 1_000_000
    )


def _canonicalize_value(value: object) -> object:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {
            str(key): _canonicalize_value(item)
            for key, item in sorted(mapping.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, list):
        return [_canonicalize_value(item) for item in cast(list[object], value)]
    if isinstance(value, tuple):
        return [_canonicalize_value(item) for item in cast(tuple[object, ...], value)]
    return value


def _canonical_event_payload(event: MeteringEvent) -> dict[str, Any]:
    payload = _canonicalize_value(event.model_dump(mode="json"))
    return cast(dict[str, Any], payload)


def _event_sort_key(event: MeteringEvent) -> tuple[str, ...]:
    return (
        event.role.value,
        event.candidate,
        event.phase,
        event.event_id,
        event.request_id,
        event.provider,
        event.model,
        json.dumps(
            _canonical_event_payload(event),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def reconcile_usage(
    events: Sequence[MeteringEvent],
    prices: PriceSheet | None,
    native_usage: Mapping[str, int] | None = None,
    *,
    role: MeteringRole | str | None = None,
    candidate: str | None = None,
    phase: str | None = None,
    required_phases: Sequence[str] = ("ingest", "query"),
) -> MeteringSummary:
    """Reconcile raw proxy/native usage without turning unknown cost into zero."""
    selected = sorted(
        filter_events(events, role=role, candidate=candidate, phase=phase),
        key=_event_sort_key,
    )
    roles = {event.role for event in events}
    mixed_roles = role is None and len(roles) > 1
    if not events or not selected:
        return MeteringSummary(
            cost_status=COST_UNAVAILABLE,
            usage_source=UsageSource.UNAVAILABLE,
            input_tokens=0,
            output_tokens=0,
            usd=None,
            price_sheet_version=prices.version if prices else None,
            native_usage=dict(native_usage) if native_usage is not None else None,
            warnings=[
                "COST_UNAVAILABLE: no attributable usage events were recorded"
                if events
                else "COST_UNAVAILABLE: no usage events were recorded"
            ],
        )
    input_tokens = sum(event.input_tokens or 0 for event in selected)
    output_tokens = sum(event.output_tokens or 0 for event in selected)
    warnings: list[str] = []
    if mixed_roles:
        warnings.append("METERING_ROLE_MIXED: pass role/candidate filters before reconciliation")
    costs: list[float] = []
    cost_complete = prices is not None and not mixed_roles
    for event in selected:
        if event.usage_source is not UsageSource.MEASURED:
            cost_complete = False
            warnings.append(
                f"COST_INCOMPLETE: {event.usage_source.value} usage for {event.request_id}"
            )
            continue
        if event.input_tokens is None or event.output_tokens is None:
            cost_complete = False
            warnings.append(f"COST_INCOMPLETE: missing usage for {event.request_id}")
            continue
        price_key = f"{event.provider}:{event.model}"
        if prices is None or price_key not in prices.models:
            cost_complete = False
            warnings.append(f"COST_INCOMPLETE: no price for {price_key}")
            continue
        cost = _event_cost(event, prices.models[price_key])
        if cost is not None:
            costs.append(cost)
    has_gbrain = any(event.candidate == "gbrain" for event in selected)
    if has_gbrain and native_usage is None:
        cost_complete = False
        warnings.append("COST_INCOMPLETE: GBrain native usage was unavailable")
    if native_usage is not None:
        native_input = native_usage.get("input_tokens")
        native_output = native_usage.get("output_tokens")
        query_events = [event for event in selected if event.phase == "query"]
        proxy_input = sum(event.input_tokens or 0 for event in query_events)
        proxy_output = sum(event.output_tokens or 0 for event in query_events)
        if (native_input, native_output) != (proxy_input, proxy_output):
            cost_complete = False
            warnings.append("METERING_USAGE_MISMATCH: native and proxy query usage differ")
    present_phases = {event.phase for event in selected}
    for required_phase in required_phases:
        if required_phase not in present_phases:
            cost_complete = False
            warnings.append(f"COST_INCOMPLETE: required paid phase {required_phase} is missing")
    usage_sources = {event.usage_source for event in selected}
    missing_usage = any(
        event.input_tokens is None or event.output_tokens is None for event in selected
    )
    summary_usage_source = (
        UsageSource.UNAVAILABLE
        if missing_usage or UsageSource.UNAVAILABLE in usage_sources
        else UsageSource.ESTIMATED
        if UsageSource.ESTIMATED in usage_sources
        else UsageSource.MEASURED
    )
    only_unavailable_usage = all(
        event.input_tokens is None or event.output_tokens is None for event in selected
    )
    return MeteringSummary(
        cost_status=(
            COST_COMPLETE
            if cost_complete
            else COST_UNAVAILABLE
            if only_unavailable_usage
            else COST_INCOMPLETE
        ),
        usage_source=summary_usage_source,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        usd=round(sum(costs), 10) if cost_complete else None,
        price_sheet_version=prices.version if prices else None,
        native_usage=(
            {key: native_usage[key] for key in sorted(native_usage)}
            if native_usage is not None
            else None
        ),
        raw_events=[_canonical_event_payload(event) for event in selected],
        warnings=sorted(warnings),
    )


def filter_events(
    events: Sequence[MeteringEvent],
    *,
    role: MeteringRole | str | None = None,
    candidate: str | None = None,
    phase: str | None = None,
) -> list[MeteringEvent]:
    """Select a single attributable metering slice without relying on arrival order."""
    requested_role = MeteringRole(role) if role is not None else None
    return [
        event
        for event in events
        if (requested_role is None or event.role is requested_role)
        and (candidate is None or event.candidate == candidate)
        and (phase is None or event.phase == phase)
    ]


class _ProxyHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = cast(dict[str, Any], json.loads(self.rfile.read(length)))
        server = cast(_ProxyServer, self.server)
        provider_started = server.owner.monotonic_clock()
        try:
            response = server.upstream(payload)
        except BudgetExceededError:
            raise
        provider_execution_ms = round(
            (server.owner.monotonic_clock() - provider_started) * 1000,
            6,
        )
        if provider_execution_ms > 0:
            response.setdefault("_autobrain_execution_ms", provider_execution_ms)
        usage = response.get("usage")
        raw_usage = cast(dict[str, Any], usage) if isinstance(usage, Mapping) else {}
        try:
            server.record(
                model=str(payload.get("model", "")),
                candidate=self.headers.get("X-AutoBrain-Candidate", ""),
                phase=self.headers.get("X-AutoBrain-Phase", ""),
                role=self.headers.get("X-AutoBrain-Role", MeteringRole.CANDIDATE.value),
                response=response,
                raw_usage=raw_usage,
                path=self.path,
            )
        except BudgetExceededError as error:
            body = json.dumps(
                {
                    "error": {
                        "type": "budget_exceeded",
                        "message": str(error),
                    }
                }
            ).encode("utf-8")
            self.send_response(402)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


class _ProxyServer(ThreadingHTTPServer):
    def __init__(
        self,
        address: tuple[str, int],
        upstream: Callable[[dict[str, Any]], dict[str, Any]],
        owner: LoopbackMeteringProxy,
    ) -> None:
        self.upstream = upstream
        self.owner = owner
        super().__init__(address, _ProxyHandler)

    def record(
        self,
        *,
        model: str,
        candidate: str,
        phase: str,
        role: MeteringRole | str,
        response: Mapping[str, Any],
        raw_usage: dict[str, Any],
        path: str,
    ) -> None:
        self.owner.record_event(
            model,
            candidate or self.owner.default_candidate,
            phase or self.owner.phase_for_path(path),
            response,
            raw_usage,
            role=role,
        )


class LoopbackMeteringProxy:
    """Small local OpenAI-compatible endpoint used by candidate adapters."""

    def __init__(
        self,
        upstream: Callable[[dict[str, Any]], dict[str, Any]],
        *,
        budget_usd: float | None = None,
        prices: PriceSheet | None = None,
        budget_state: MeteringBudget | None = None,
        default_candidate: str = "",
        monotonic_clock: Callable[[], float] = monotonic,
    ) -> None:
        self.upstream = upstream
        if budget_state is not None:
            if budget_usd is not None and budget_usd != budget_state.budget_usd:
                raise ValueError("budget_usd does not match shared metering budget")
            self._budget_state = budget_state
        elif budget_usd is not None:
            self._budget_state = MeteringBudget(
                budget_usd,
                prices or load_price_sheet(),
            )
        else:
            self._budget_state = None
        self.budget_usd = self._budget_state.budget_usd if self._budget_state is not None else None
        self.prices = self._budget_state.prices if self._budget_state is not None else prices
        self.default_candidate = default_candidate
        self.monotonic_clock = monotonic_clock
        self._local_events: list[MeteringEvent] = []
        self._server: _ProxyServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def spent_usd(self) -> float:
        return self._budget_state.spent_usd if self._budget_state is not None else 0.0

    @property
    def budget_exceeded(self) -> bool:
        return self._budget_state.budget_exceeded if self._budget_state is not None else False

    @property
    def events(self) -> list[MeteringEvent]:
        return self._budget_state.events if self._budget_state is not None else self._local_events

    def phase_for_path(self, path: str) -> str:
        return "ingest" if "/embeddings" in path else "query"

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("metering proxy is not running")
        return f"http://127.0.0.1:{self._server.server_port}/v1"

    def __enter__(self) -> LoopbackMeteringProxy:
        self._server = _ProxyServer(("127.0.0.1", 0), self.upstream, self)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._server = None
        self._thread = None

    def chat(
        self,
        payload: dict[str, Any],
        *,
        candidate: str,
        phase: str,
        role: MeteringRole | str = MeteringRole.CANDIDATE,
    ) -> dict[str, Any]:
        if self.budget_exceeded:
            assert self.budget_usd is not None
            raise BudgetExceededError(self.budget_usd, self.spent_usd)
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-AutoBrain-Candidate": candidate,
                "X-AutoBrain-Phase": phase,
                "X-AutoBrain-Role": MeteringRole(role).value,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return cast(dict[str, Any], json.loads(response.read()))
        except urllib.error.HTTPError as error:
            if error.code == 402:
                error.read()
                raise BudgetExceededError(
                    self.budget_usd or 0.0,
                    self.spent_usd,
                ) from error
            raise

    def record_event(
        self,
        model: str,
        candidate: str,
        phase: str,
        response: Mapping[str, Any],
        raw_usage: dict[str, Any],
        *,
        role: MeteringRole | str = MeteringRole.CANDIDATE,
    ) -> None:
        if self.budget_exceeded:
            assert self.budget_usd is not None
            raise BudgetExceededError(self.budget_usd, self.spent_usd)
        request_id = str(response.get("id", f"proxy-{len(self.events) + 1}"))
        provider = str(response.get("_autobrain_provider", "openai"))
        actual_model = str(response.get("model", model))
        usage_source_raw = response.get("_autobrain_usage_source", UsageSource.MEASURED.value)
        try:
            usage_source = UsageSource(str(usage_source_raw))
        except ValueError:
            usage_source = UsageSource.UNAVAILABLE
        input_tokens = _usage_count(raw_usage, "prompt_tokens", "input_tokens")
        output_tokens = _usage_count(raw_usage, "completion_tokens", "output_tokens")
        if input_tokens is None or output_tokens is None:
            usage_source = UsageSource.UNAVAILABLE
        event = MeteringEvent(
            event_id=f"{request_id}-{len(self.events) + 1}",
            request_id=request_id,
            candidate=candidate,
            phase=phase,
            role=MeteringRole(role),
            provider=provider,
            model=actual_model,
            usage_source=usage_source,
            provider_execution_ms=(
                float(response["_autobrain_execution_ms"])
                if isinstance(response.get("_autobrain_execution_ms"), int | float)
                and not isinstance(response.get("_autobrain_execution_ms"), bool)
                and float(response["_autobrain_execution_ms"]) > 0
                else None
            ),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            raw_usage=raw_usage,
            tags={"candidate": candidate, "phase": phase},
        )
        if self._budget_state is None:
            self._local_events.append(event)
            return
        with self._budget_state.lock:
            self._budget_state.events.append(event)
            price_key = f"{event.provider}:{event.model}"
            quote = self.prices.models.get(price_key) if self.prices is not None else None
            cost = _event_cost(event, quote) if quote is not None else None
            budget_usd = self.budget_usd
            if cost is not None and budget_usd is not None:
                self._budget_state.spent_usd += cost
                if self._budget_state.spent_usd > budget_usd:
                    self._budget_state.budget_exceeded = True
                    raise BudgetExceededError(budget_usd, self.spent_usd)
