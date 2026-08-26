import pytest

from autobrain.connectors.google_drive import (
    ExternalGateReason,
    google_drive_external_gate,
    google_drive_fixture_corpus,
    normalize_google_drive_fixture,
)
from autobrain.models import SourceKind, Status
from autobrain.preflight_google_drive import check_google_drive_source


def test_google_drive_gate_is_typed_and_fail_closed() -> None:
    gate = google_drive_external_gate()

    assert gate.provider.value == "google_drive"
    assert gate.state == "EXTERNAL_GATE"
    assert gate.reason is ExternalGateReason.PROJECT_POLICY
    assert gate.read_only is True
    assert gate.public_api_verified is False
    assert gate.credentials_present is False
    assert "no REST, MCP, OCR, or binary fallback" in gate.detail
    assert "stable source IDs based on Drive file IDs" in gate.required_contract
    assert "unsupported MIME coverage" in gate.required_contract
    assert gate.network_allowed is False
    assert gate.source_writes_allowed is False
    assert gate.remediation


def test_google_drive_fixture_normalizer_is_read_only_and_provenance_preserving() -> None:
    documents, coverage = google_drive_fixture_corpus(
        [
            {
                "id": "file-1",
                "name": "Policy",
                "mimeType": "application/vnd.google-apps.document",
                "webViewLink": "https://drive.example.test/file-1",
                "text": "Read-only policy text.",
                "exportMimeType": "text/plain",
            }
        ]
    )

    assert len(documents) == 1
    document = documents[0]
    assert document.source_id == "google_drive:file:file-1"
    assert document.source_kind is SourceKind.GOOGLE_DRIVE_FILE
    assert document.crawl_provenance["file_id"] == "file-1"
    assert document.crawl_provenance["mime_type"] == "application/vnd.google-apps.document"
    assert document.crawl_provenance["content_mime_type"] == "text/plain"
    assert coverage.fetched == 1


def test_google_drive_fixture_rejects_unsupported_binary_and_ambiguous_export() -> None:
    binary = {
        "id": "file-1",
        "name": "Binary",
        "mime_type": "application/pdf",
        "web_view_link": "https://drive.example.test/file-1",
        "text": "not an implicit PDF extraction",
    }
    with pytest.raises(ValueError, match="unsupported Drive MIME"):
        normalize_google_drive_fixture([binary])

    workspace = {
        "id": "file-2",
        "name": "Doc",
        "mime_type": "application/vnd.google-apps.document",
        "web_view_link": "https://drive.example.test/file-2",
        "text": "text",
    }
    with pytest.raises(ValueError, match="export_mime_type"):
        normalize_google_drive_fixture([workspace])


def test_doctor_check_exposes_gate_without_claiming_readiness() -> None:
    check = check_google_drive_source()

    assert check.name == "google_drive_source"
    assert check.status is Status.UNSUPPORTED
    assert "official Drive API/SDK connector" in check.detail
