from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest

import autobrain.production as production
from autobrain.models import CandidateId


class _TrackingProxy:
    instances: ClassVar[list[_TrackingProxy]] = []

    def __init__(self, *_args: object, default_candidate: str, **_kwargs: object) -> None:
        self.default_candidate = default_candidate
        self.enter_count = 0
        self.close_count = 0
        self.__class__.instances.append(self)

    @property
    def base_url(self) -> str:
        return f"http://meter.test/{self.default_candidate}"

    def __enter__(self) -> _TrackingProxy:
        self.enter_count += 1
        return self

    def close(self) -> None:
        self.close_count += 1


def test_candidate_constructor_failure_closes_every_entered_proxy_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _TrackingProxy.instances.clear()

    def fail_mem0(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("injected adapter constructor failure")

    monkeypatch.setattr(production, "LoopbackMeteringProxy", _TrackingProxy)
    monkeypatch.setattr(production, "Mem0Adapter", fail_mem0)

    with pytest.raises(RuntimeError, match="injected adapter constructor failure"):
        production.build_production_candidates(
            tmp_path / "run",
            api_key="test-key",
            paths=production.AutoBrainPaths(
                root=tmp_path / "home",
                runs=tmp_path / "home" / "runs",
                tools=tmp_path / "home" / "tools",
                cache=tmp_path / "home" / "cache",
            ),
            candidate_ids=(CandidateId.LLM_WIKI, CandidateId.MEM0, CandidateId.GBRAIN),
        )

    assert [proxy.default_candidate for proxy in _TrackingProxy.instances] == [
        candidate.value for candidate in CandidateId
    ]
    assert all(proxy.enter_count == 1 for proxy in _TrackingProxy.instances)
    assert all(proxy.close_count == 1 for proxy in _TrackingProxy.instances)
