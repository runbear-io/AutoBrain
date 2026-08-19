from pathlib import Path

from autobrain.onboarding import is_onboarded, mark_onboarded
from autobrain.paths import AutoBrainPaths
from autobrain.tui_state import TUIState, WizardSection


def test_onboarding_flag_round_trip(tmp_path: Path) -> None:
    paths = AutoBrainPaths.from_home(tmp_path)
    assert is_onboarded(paths) is False
    mark_onboarded(paths)
    assert is_onboarded(paths) is True
    assert oct((paths.root / "onboarding.json").stat().st_mode & 0o777) == "0o600"


def test_setup_returns_home_when_opened_from_main() -> None:
    home = TUIState(section=WizardSection.HOME).start_setup()
    assert home.section is WizardSection.CONNECTIONS
    assert home.return_home is True
    assert home.back().section is WizardSection.HOME
