"""Terminal rendering for the AutoBrain cockpit."""

from autobrain.auth.models import Provider
from autobrain.experiment import ExperimentPlan
from autobrain.models import CandidateId, ConnectionState
from autobrain.orchestration import RunResult
from autobrain.subscription import SubscriptionStatus
from autobrain.terminal_text import truncate_terminal_text
from autobrain.tui_home import render_home
from autobrain.tui_interview import render_interview

MIN_TERMINAL_WIDTH = 60
MIN_TERMINAL_HEIGHT = 22


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
    usable_width = max(1, width - 2)
    details = source_details or {}
    if terminal_too_small(width=width, height=height):
        lines = (
            ["AUTO", "TOO", "SMALL", f">={MIN_TERMINAL_WIDTH}x{MIN_TERMINAL_HEIGHT}", "Q quit"]
            if usable_width <= 8
            else [
                "AutoBrain",
                "TERMINAL_TOO_SMALL",
                f"Resize to at least {MIN_TERMINAL_WIDTH}x{MIN_TERMINAL_HEIGHT}.",
                "Q quit",
            ]
        )
        return [truncate_terminal_text(line, usable_width) for line in lines]
    if section == "home":
        lines = render_home(
            selected_sources=selected_sources,
            selected_candidates=selected_candidates,
            source_states=source_states,
            subscription_status=subscription_status,
            plan=plan,
            setup_error=setup_error,
            source_details=details,
        )
    elif section == "running":
        lines = _running(plan, selected_sources, selected_candidates, elapsed_seconds)
    elif section == "results" and result is not None:
        lines = _results(result)
    else:
        lines = render_interview(
            section,
            selected_sources,
            selected_candidates,
            source_states,
            subscription_status,
            plan,
            setup_error,
            details,
        )
    return [truncate_terminal_text(line, usable_width) for line in lines]


def _running(
    plan: ExperimentPlan | None,
    selected_sources: tuple[Provider, ...],
    selected_candidates: tuple[CandidateId, ...],
    elapsed_seconds: int,
) -> list[str]:
    return [
        "AutoBrain",
        "Working through the experiment.",
        "",
        plan.title if plan is not None else "Automatic experiment",
        f"Sources     {', '.join(item.value.title() for item in selected_sources)}",
        f"Brains      {', '.join(item.value for item in selected_candidates)}",
        f"Elapsed     {elapsed_seconds}s",
        "1. Freeze corpus   2. Compile LLM Wiki",
        "3. Retrieve and answer   4. Score quality / latency / cost",
        "",
        "Experiment is running. Wait for the evidence-backed result.",
    ]


def _results(result: RunResult) -> list[str]:
    lines = [
        "AutoBrain",
        "Experiment complete.",
        "",
        f"Status      {result.status.value}",
    ]
    for outcome in result.candidate_results:
        lines.append(f"{outcome.candidate:<12}  {outcome.score:>5.1f}   {outcome.status.value}")
    report = "O open report  |  R new experiment  |  Q quit"
    if result.report_path is None:
        report = "R new experiment  |  Q quit"
    lines.extend(
        [
            f"Verdict     {result.verdict}",
            f"Report      {result.report_path or 'not generated'}",
            "",
            report,
        ]
    )
    return lines
