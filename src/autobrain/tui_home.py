"""Main cockpit shown after AutoBrain onboarding is complete."""

from autobrain.auth.models import Provider
from autobrain.experiment import ExperimentPlan
from autobrain.models import CandidateId, ConnectionState
from autobrain.subscription import SubscriptionStatus


def render_home(
    *,
    selected_sources: tuple[Provider, ...],
    selected_candidates: tuple[CandidateId, ...],
    source_states: dict[Provider, ConnectionState],
    subscription_status: SubscriptionStatus,
    plan: ExperimentPlan | None,
    setup_error: str,
    source_details: dict[Provider, str],
) -> list[str]:
    chatgpt = (
        "connected" if subscription_status is SubscriptionStatus.READY else "not connected"
    )
    slack = _status(Provider.SLACK, selected_sources, source_states, source_details)
    notion = _status(Provider.NOTION, selected_sources, source_states, source_details)
    brains = ", ".join(item.value for item in selected_candidates) or "none"
    if plan is None:
        action = setup_error or "Open setup to finish ChatGPT or a knowledge source."
        enter = "S         Open setup"
    else:
        action = plan.title
        enter = "Enter     Run experiment"
    return [
        "AutoBrain",
        "Compare LLM Wiki, Mem0 OSS, and GBrain on your knowledge.",
        "",
        f"ChatGPT    {chatgpt}",
        f"Slack      {slack}",
        f"Notion     {notion}",
        f"Brains     {brains}",
        "",
        action,
        "",
        enter,
        "S         Setup / reconnect",
        "Q         Quit",
        "",
        "Enter run  |  S setup  |  Q quit" if plan is not None else "S setup  |  Q quit",
    ]


def _status(
    provider: Provider,
    selected_sources: tuple[Provider, ...],
    source_states: dict[Provider, ConnectionState],
    source_details: dict[Provider, str],
) -> str:
    if provider not in selected_sources:
        return "skipped"
    if source_states.get(provider) is ConnectionState.CONNECTED:
        return source_details.get(provider) or "connected"
    return "not connected"
