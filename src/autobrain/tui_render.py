"""Pure terminal rendering for the AutoBrain setup cockpit."""

from __future__ import annotations

from autobrain.auth.models import Provider
from autobrain.experiment import ExperimentPlan
from autobrain.models import CandidateId, ConnectionState
from autobrain.orchestration import RunResult
from autobrain.subscription import SubscriptionStatus
from autobrain.terminal_text import truncate_terminal_text

MIN_TERMINAL_WIDTH = 60
MIN_TERMINAL_HEIGHT = 23


def terminal_too_small(*, width: int, height: int | None) -> bool:
    return width < MIN_TERMINAL_WIDTH or (height is not None and height < MIN_TERMINAL_HEIGHT)


def render_dashboard(
    *,
    section: str,
    selected_sources: tuple[Provider, ...],
    selected_candidates: tuple[CandidateId, ...],
    source_states: dict[Provider, ConnectionState],
    subscription_status: SubscriptionStatus,
    plan: ExperimentPlan | None,
    setup_error: str,
    result: RunResult | None,
    elapsed_seconds: int,
    width: int,
    height: int | None = None,
    source_details: dict[Provider, str] | None = None,
) -> list[str]:
    """Return a size-bounded dashboard that is easy to snapshot and test."""
    usable_width = max(1, width - 2)
    if terminal_too_small(width=width, height=height):
        if usable_width <= 8:
            lines = [
                "AUTO",
                "TOO",
                "SMALL",
                f">={MIN_TERMINAL_WIDTH}x{MIN_TERMINAL_HEIGHT}",
                "Q quit",
            ]
        else:
            lines = [
                "AUTOBRAIN",
                "TERMINAL_TOO_SMALL",
                f"Resize to at least {MIN_TERMINAL_WIDTH}x{MIN_TERMINAL_HEIGHT}.",
                "Q quit",
            ]
        return [truncate_terminal_text(line, usable_width) for line in lines]

    if section == "running":
        lines = [
            "AUTOBRAIN",
            "Running the experiment AutoBrain designed from your selected scope.",
            "",
            "RUNNING",
            f"   {plan.title if plan is not None else 'Automatic experiment'}",
            f"   Sources      {', '.join(item.value.title() for item in selected_sources)}",
            f"   Candidates   {', '.join(item.value for item in selected_candidates)}",
            f"   Elapsed      {elapsed_seconds}s",
            "",
            _footer(section),
        ]
        return [truncate_terminal_text(line, usable_width) for line in lines]
    if section == "results" and result is not None:
        lines = [
            "AUTOBRAIN",
            "Experiment complete. Inspect the evidence before accepting the verdict.",
            "",
            "RESULTS",
            f"   Status       {result.status.value}",
        ]
        for outcome in result.candidate_results:
            lines.append(
                f"   {outcome.candidate:<12} score {outcome.score:>5.1f}   {outcome.status.value}"
            )
        lines.extend(
            [
                "",
                f"   Verdict      {result.verdict}",
                f"   Report       {result.report_path or 'not generated'}",
                "",
                _footer(section, report_available=result.report_path is not None),
            ]
        )
        return [truncate_terminal_text(line, usable_width) for line in lines]

    lines = [
        "AUTOBRAIN",
        "One grounded experiment. You choose ChatGPT, knowledge sources, and candidates.",
        "",
        _heading("connections", "1  ChatGPT", section),
        _connection_line("C", "ChatGPT", subscription_status is SubscriptionStatus.READY),
        _heading("knowledge_sources", "2  Knowledge", section),
        _knowledge_line(
            "S",
            "Slack",
            selected=Provider.SLACK in selected_sources,
            connected=source_states.get(Provider.SLACK) is ConnectionState.CONNECTED,
            connected_status=(source_details or {}).get(Provider.SLACK),
        ),
        _knowledge_line(
            "N",
            "Notion",
            selected=Provider.NOTION in selected_sources,
            connected=source_states.get(Provider.NOTION) is ConnectionState.CONNECTED,
        ),
        _heading("candidates", "3  Candidates", section),
        _toggle_line("1", "LLM Wiki", CandidateId.LLM_WIKI in selected_candidates),
        _toggle_line("2", "Mem0 OSS", CandidateId.MEM0 in selected_candidates),
        _toggle_line("3", "GBrain", CandidateId.GBRAIN in selected_candidates),
        _heading("review", "4  Automatic Experiment", section),
    ]
    if plan is not None:
        lines.extend(
            [
                f"   {plan.title}",
                f"   {plan.description}",
                f"   Provider     {_provider_label(plan.provider_mode)}",
                f"   Questions    automatic, up to {plan.max_questions}",
                f"   Budget guard automatic, ${plan.budget_usd:.0f}",
            ]
        )
    else:
        lines.append(f"   {setup_error or 'Complete the setup to preview the experiment.'}")

    lines.extend(["", _footer(section, plan_available=plan is not None)])
    return [truncate_terminal_text(line, usable_width) for line in lines]


def _heading(identifier: str, label: str, section: str) -> str:
    return f"> {label}" if identifier == section else f"  {label}"


def _connection_line(
    key: str,
    label: str,
    connected: bool,
    *,
    connected_status: str | None = None,
) -> str:
    status = (
        connected_status
        if connected and connected_status
        else ("connected" if connected else "not connected")
    )
    action = "reconnect" if connected else "connect"
    return f"   [{key}] {label:<12} {status:<14} ({action})"


def _knowledge_line(
    key: str,
    label: str,
    *,
    selected: bool,
    connected: bool,
    connected_status: str | None = None,
) -> str:
    mark = "x" if selected else " "
    status = (
        connected_status
        if connected and connected_status
        else ("connected" if connected else "not connected")
    )
    action = "reconnect" if connected else "connect"
    return f"   [{key}] [{mark}] {label:<12} {status:<14} ({action})"


def _toggle_line(key: str, label: str, selected: bool) -> str:
    return f"   [{key}] [{'x' if selected else ' '}] {label}"


def _provider_label(provider_mode: str) -> str:
    return "ChatGPT subscription" if provider_mode == "codex-subscription" else "API"


def _footer(
    section: str,
    *,
    report_available: bool = False,
    plan_available: bool = False,
) -> str:
    if section == "connections":
        return "C connect  |  Enter next  |  Q quit"
    if section == "knowledge_sources":
        return "S/N connect  |  1/2 toggle  |  Enter next  |  B back  |  Q quit"
    if section == "candidates":
        return "1/2/3 toggle  |  Enter next  |  B back  |  Q quit"
    if section == "review":
        if plan_available:
            return "Enter run experiment  |  B back  |  Q quit"
        return "B back  |  Q quit"
    if section == "running":
        return "Experiment is running. Wait for the evidence-backed result."
    if report_available:
        return "O open report  |  R new experiment  |  Q quit"
    return "R new experiment  |  Q quit"
