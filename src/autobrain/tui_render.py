"""Interview-style terminal rendering for the AutoBrain cockpit."""

from autobrain.auth.models import Provider
from autobrain.experiment import ExperimentPlan
from autobrain.models import CandidateId, ConnectionState
from autobrain.orchestration import RunResult
from autobrain.subscription import SubscriptionStatus
from autobrain.terminal_text import truncate_terminal_text

MIN_TERMINAL_WIDTH = 60
MIN_TERMINAL_HEIGHT = 22

_STEPS = (
    ("connections", "ChatGPT"),
    ("slack", "Slack"),
    ("notion", "Notion"),
    ("candidates", "Brains"),
    ("review", "Run"),
)
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
    if section == "running":
        lines = _running(plan, selected_sources, selected_candidates, elapsed_seconds)
    elif section == "results" and result is not None:
        lines = _results(result)
    else:
        lines = _interview(
            section,
            selected_sources,
            selected_candidates,
            source_states,
            subscription_status,
            plan,
            setup_error,
            source_details or {},
        )
    return [truncate_terminal_text(line, usable_width) for line in lines]

def _interview(
    section: str,
    selected_sources: tuple[Provider, ...],
    selected_candidates: tuple[CandidateId, ...],
    source_states: dict[Provider, ConnectionState],
    subscription_status: SubscriptionStatus,
    plan: ExperimentPlan | None,
    setup_error: str,
    source_details: dict[Provider, str],
) -> list[str]:
    index = next((i for i, item in enumerate(_STEPS) if item[0] == section), 0)
    names = [f"[{label}]" if i == index else label for i, (_, label) in enumerate(_STEPS)]
    slack_status = _source_status(Provider.SLACK, source_states, source_details)
    notion_status = _source_status(Provider.NOTION, source_states, source_details)
    chatgpt_ready = subscription_status is SubscriptionStatus.READY
    body = _step_body(
        section,
        chatgpt_ready=chatgpt_ready,
        slack_status=slack_status,
        notion_status=notion_status,
        selected_sources=selected_sources,
        selected_candidates=selected_candidates,
        plan=plan,
        setup_error=setup_error,
    )
    return [
        "AutoBrain",
        "Which Brain should your company build on?",
        "",
        "  ".join(names),
        "",
        f"Step {index + 1} of {len(_STEPS)}",
        *body,
        "",
        _footer(section, plan_available=plan is not None, chatgpt_ready=chatgpt_ready),
    ]


def _step_body(
    section: str,
    *,
    chatgpt_ready: bool,
    slack_status: str,
    notion_status: str,
    selected_sources: tuple[Provider, ...],
    selected_candidates: tuple[CandidateId, ...],
    plan: ExperimentPlan | None,
    setup_error: str,
) -> list[str]:
    if section == "connections":
        status = "connected" if chatgpt_ready else "not connected"
        action = "Continue" if chatgpt_ready else "Open ChatGPT in your browser"
        return [
            "Sign in with ChatGPT",
            "A browser window will open for grounded questions and scoring.",
            f"Status    {status}",
            f"Enter     {action}",
        ]
    if section == "slack":
        included = Provider.SLACK in selected_sources
        return [
            "Add Slack knowledge",
            "Import an official Slack export ZIP. No Slack app is required.",
            f"Status    {slack_status if included else 'skipped'}",
            "Enter     Import Slack export ZIP     S  Skip Slack",
        ]
    if section == "notion":
        included = Provider.NOTION in selected_sources
        return [
            "Add Notion knowledge",
            "A browser window will open for read-only Notion access.",
            f"Status    {notion_status if included else 'skipped'}",
            "Enter     Open Notion authorization     S  Skip Notion",
        ]
    if section == "candidates":
        return [
            "Choose Brains to compare",
            "LLM Wiki will compile your sources, then retrieve per question.",
            "",
            _toggle_line("1", "LLM Wiki", CandidateId.LLM_WIKI in selected_candidates),
            _toggle_line("2", "Mem0 OSS", CandidateId.MEM0 in selected_candidates),
            _toggle_line("3", "GBrain", CandidateId.GBRAIN in selected_candidates),
            "Enter     Continue",
        ]
    if plan is None:
        return [
            "Ready to run?",
            setup_error or "Finish the earlier steps to unlock the experiment.",
            "",
            "B         Go back",
        ]
    return [
        "Ready to run?",
        plan.title,
        "AutoBrain will freeze the corpus, compile LLM Wiki, retrieve",
        "evidence for each question, and score quality, latency, and cost.",
        "",
        f"Sources    {', '.join(item.value.title() for item in plan.sources)}",
        f"Brains     {', '.join(item.value for item in plan.candidates)}",
        f"Questions  up to {plan.max_questions}   Budget  ${plan.budget_usd:.0f}",
        "",
        "Enter     Start experiment",
    ]


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
        lines.append(
            f"{outcome.candidate:<12}  {outcome.score:>5.1f}   {outcome.status.value}"
        )
    lines.extend(
        [
            f"Verdict     {result.verdict}",
            f"Report      {result.report_path or 'not generated'}",
            "",
            _footer("results", report_available=result.report_path is not None),
        ]
    )
    return lines


def _source_status(
    provider: Provider,
    source_states: dict[Provider, ConnectionState],
    source_details: dict[Provider, str],
) -> str:
    if source_states.get(provider) is not ConnectionState.CONNECTED:
        return "not connected"
    return source_details.get(provider) or "connected"

def _toggle_line(key: str, label: str, selected: bool) -> str:
    return f"   [{key}] [{'x' if selected else ' '}] {label}"

def _footer(
    section: str,
    *,
    report_available: bool = False,
    plan_available: bool = False,
    chatgpt_ready: bool = False,
) -> str:
    if section == "connections":
        return "Enter open ChatGPT  |  Q quit" if not chatgpt_ready else "Enter next  |  Q quit"
    if section in {"slack", "notion"}:
        return "Enter connect  |  S skip  |  B back  |  Q quit"
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
