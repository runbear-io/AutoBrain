"""AutoBrain local comparator foundation."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

__version__ = "0.1.1"

if TYPE_CHECKING:
    from autobrain.decision import select_winner
    from autobrain.evaluate import evaluate_candidate, evaluate_case
    from autobrain.metering import (
        LoopbackMeteringProxy,
        MeteringEvent,
        PriceSheet,
        load_price_sheet,
        reconcile_usage,
    )
    from autobrain.report import build_comparison, load_comparison, render_report, write_artifacts

__all__ = [
    "LoopbackMeteringProxy",
    "MeteringEvent",
    "PriceSheet",
    "__version__",
    "build_comparison",
    "evaluate_candidate",
    "evaluate_case",
    "load_comparison",
    "load_price_sheet",
    "reconcile_usage",
    "render_report",
    "select_winner",
    "write_artifacts",
]

_EXPORTS = {
    "LoopbackMeteringProxy": ("autobrain.metering", "LoopbackMeteringProxy"),
    "MeteringEvent": ("autobrain.metering", "MeteringEvent"),
    "PriceSheet": ("autobrain.metering", "PriceSheet"),
    "build_comparison": ("autobrain.report", "build_comparison"),
    "evaluate_candidate": ("autobrain.evaluate", "evaluate_candidate"),
    "evaluate_case": ("autobrain.evaluate", "evaluate_case"),
    "load_comparison": ("autobrain.report", "load_comparison"),
    "load_price_sheet": ("autobrain.metering", "load_price_sheet"),
    "reconcile_usage": ("autobrain.metering", "reconcile_usage"),
    "render_report": ("autobrain.report", "render_report"),
    "select_winner": ("autobrain.decision", "select_winner"),
    "write_artifacts": ("autobrain.report", "write_artifacts"),
}


def __getattr__(name: str) -> Any:
    """Load public Task 9 helpers lazily to keep the CLI bootstrap dependency-free."""
    export = _EXPORTS.get(name)
    if export is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(export[0])
    return getattr(module, export[1])
