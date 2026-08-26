"""Shared lifecycle receipts for run-scoped native candidates."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from autobrain.models import CandidateId


class CleanupReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    candidate: CandidateId
    closed_resources: list[str] = Field(default_factory=list)
    removed_paths: list[str] = Field(default_factory=list)
    remaining_paths: list[str] = Field(default_factory=list)
    interrupted: bool = False
    error: str | None = None

    @property
    def complete(self) -> bool:
        return not self.interrupted and self.error is None and not self.remaining_paths


def cleanup_receipt_complete(receipt: CleanupReceipt | None) -> bool:
    return receipt is not None and receipt.complete


def remaining_paths(*paths: Path) -> tuple[str, ...]:
    return tuple(str(path) for path in paths if path.exists())
