"""Interview-step copy for first-time AutoBrain setup."""

from autobrain.auth.models import Provider
from autobrain.experiment import ExperimentPlan
from autobrain.models import CandidateId, ConnectionState
from autobrain.subscription import ProviderId, SubscriptionStatus

_STEPS = (
    ("connections", "Provider"),
    ("slack", "Slack"),
    ("notion", "Notion"),
    ("candidates", "Brains"),
    ("review", "Run"),
)


def render_interview(
    section: str,
    selected_sources: tuple[Provider, ...],
    selected_candidates: tuple[CandidateId, ...],
    source_states: dict[Provider, ConnectionState],
    subscription_status: SubscriptionStatus,
    subscription_provider: ProviderId,
    plan: ExperimentPlan | None,
    setup_error: str,
    source_details: dict[Provider, str],
) -> list[str]:
    index = next((i for i, item in enumerate(_STEPS) if item[0] == section), 0)
    names = [f"[{label}]" if i == index else label for i, (_, label) in enumerate(_STEPS)]
    slack_status = _source_status(Provider.SLACK, source_states, source_details)
    notion_status = _source_status(Provider.NOTION, source_states, source_details)
    subscription_ready = subscription_status is SubscriptionStatus.READY
    body = _step_body(
        section,
        subscription_ready=subscription_ready,
        subscription_provider=subscription_provider,
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
        _footer(
            section,
            plan_available=plan is not None,
            subscription_ready=subscription_ready,
        ),
    ]


def _step_body(
    section: str,
    *,
    subscription_ready: bool,
    subscription_provider: ProviderId,
    slack_status: str,
    notion_status: str,
    selected_sources: tuple[Provider, ...],
    selected_candidates: tuple[CandidateId, ...],
    plan: ExperimentPlan | None,
    setup_error: str,
) -> list[str]:
    if section == "connections":
        status = "connected" if subscription_ready else "not connected"
        action = "Continue" if subscription_ready else "Open vendor login"
        return [
            "Choose a consumer subscription provider",
            "AutoBrain never falls back to another provider.",
            _provider_line("1", "Codex / ChatGPT", ProviderId.CODEX, subscription_provider),
            _provider_line("2", "Claude", ProviderId.CLAUDE, subscription_provider),
            _provider_line("3", "Kimi", ProviderId.KIMI, subscription_provider),
            _provider_line("4", "Grok", ProviderId.GROK, subscription_provider),
            f"Selected  {subscription_provider.value}   Status  {status}",
            f"Enter     {action}   R  Refresh status",
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


def _provider_line(
    key: str,
    label: str,
    provider: ProviderId,
    selected: ProviderId,
) -> str:
    return _toggle_line(key, label, provider is selected)


def _footer(section: str, *, plan_available: bool, subscription_ready: bool) -> str:
    if section == "connections":
        if subscription_ready:
            return "1/2/3/4 select  |  R refresh  |  Enter next  |  Q quit"
        return "1/2/3/4 select  |  R refresh  |  Enter login  |  Q quit"
    if section in {"slack", "notion"}:
        return "Enter connect  |  S skip  |  B back  |  Q quit"
    if section == "candidates":
        return "1/2/3 toggle  |  Enter next  |  B back  |  Q quit"
    if plan_available:
        return "Enter run experiment  |  B back  |  Q quit"
    return "B back  |  Q quit"
