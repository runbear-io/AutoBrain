from pathlib import Path

import pytest
from harness import E2EHarness


@pytest.fixture
def e2e(tmp_path: Path) -> E2EHarness:
    return E2EHarness(tmp_path)
