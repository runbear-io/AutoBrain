from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from typer.testing import CliRunner

from autobrain.cli import app
from autobrain.fixture import (
    FixtureDocument,
    FixtureFaultCode,
    FixtureSpec,
    build_fixture,
    fixture_json_bytes,
    write_fixture,
)


def test_fixture_document_fields_have_bounded_sizes() -> None:
    base = {
        "provider": "notion",
        "source_id": "notion:page:bounded",
        "source_kind": "NOTION_PAGE",
        "canonical_url": "https://example.test/page",
        "title": "title",
        "text": "text",
        "question": "question",
        "content_hash": hashlib.sha256(b"text").hexdigest(),
    }
    for field, value in (
        ("title", "x" * 4097),
        ("question", "x" * 4097),
        ("text", "x" * 1_000_001),
    ):
        with pytest.raises(ValueError):
            FixtureDocument.model_validate({**base, field: value})


def test_fixture_builder_is_seeded_and_round_trips_schema_v1() -> None:
    first = build_fixture(seed=17)
    second = build_fixture(seed=17)

    assert first == second
    assert first.schema_version == 1
    assert len(first.documents) == 24
    assert first.faults == []
    assert FixtureSpec.model_validate_json(fixture_json_bytes(first)) == first


def test_fixture_fault_vocabulary_is_declarative_and_non_executable() -> None:
    spec = build_fixture(seed=3, faults=[FixtureFaultCode.UNSAFE_URL])

    assert spec.faults[0].code is FixtureFaultCode.UNSAFE_URL
    assert not any(key in fixture_json_bytes(spec).decode() for key in ("command", "exec", "shell"))


def test_fixture_writer_writes_canonical_json(tmp_path: Path) -> None:
    path = write_fixture(tmp_path / "fixture.json", seed=9)

    assert (
        FixtureSpec.model_validate_json(path.read_bytes()).fixture_sha256
        == build_fixture(seed=9).fixture_sha256
    )
    assert path.read_bytes().endswith(b"\n")


def test_fixture_generate_requires_explicit_test_gate(tmp_path: Path) -> None:
    output = tmp_path / "generated.json"
    runner = CliRunner()

    denied = runner.invoke(app, ["fixture", "generate", "--seed", "7", "--output", str(output)])
    assert denied.exit_code != 0
    assert "AUTOBRAIN_ALLOW_TEST_FIXTURE=1" in denied.output
    assert not output.exists()

    allowed = runner.invoke(
        app,
        ["fixture", "generate", "--seed", "7", "--output", str(output)],
        env={"AUTOBRAIN_ALLOW_TEST_FIXTURE": "1"},
    )
    assert allowed.exit_code == 0, allowed.output
    assert output.exists()
    assert "fixture-id:" in allowed.output
