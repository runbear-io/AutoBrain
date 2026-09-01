import inspect
from pathlib import Path

from autobrain.orchestration import RunOrchestrator


def test_workflow_uses_mature_benchmark_pipeline_not_orchestration_shadow() -> None:
    source = Path(inspect.getsourcefile(RunOrchestrator.run) or "").read_text()

    assert "build_benchmark(" in source
    assert "self._build_benchmark(documents" not in source
