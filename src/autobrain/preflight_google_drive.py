"""Google Drive readiness check for AutoBrain doctor."""

from autobrain.connectors.google_drive import google_drive_external_gate
from autobrain.models import CheckResult, Status


def check_google_drive_source() -> CheckResult:
    """Expose the typed external gate; this check never claims Drive is ready."""
    gate = google_drive_external_gate()
    return CheckResult(
        name="google_drive_source",
        status=Status.UNSUPPORTED,
        detail=gate.detail,
    )
