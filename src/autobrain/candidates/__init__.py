"""Candidate-specific adapters for the pinned comparison runtimes."""

import os

# Candidate adapters capture provider usage locally; Mem0 product telemetry is out of scope.
os.environ["MEM0_TELEMETRY"] = "false"
