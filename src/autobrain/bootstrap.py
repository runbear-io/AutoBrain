"""Dependency-light runtime gate for the installed console script."""

import json
import sys
from collections.abc import Sequence

_MINIMUM = "Python >=3.12 and <3.14.0a0"


def _unsupported(version_info: tuple[int, int, int]) -> bool:
    return version_info[0] > 3 or (version_info[0] == 3 and version_info[1] >= 14)


def _diagnostic(version_info: tuple[int, int, int], json_requested: bool) -> int:
    detail = (
        "Python >=3.14 is unsupported "
        f"(running {version_info[0]}.{version_info[1]}.{version_info[2]}); "
        f"requires {_MINIMUM}"
    )
    if json_requested:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "ENV_UNAVAILABLE",
                    "checks": [{"name": "python", "status": "ENV_UNAVAILABLE", "detail": detail}],
                },
                sort_keys=True,
            )
        )
    else:
        print(f"ENV_UNAVAILABLE: {detail}")
    return 1


def main(
    argv: Sequence[str] | None = None,
    *,
    version_info: tuple[int, int, int] | None = None,
) -> int:
    """Gate unsupported interpreters before importing Typer, Pydantic, or the CLI."""
    args = list(sys.argv[1:] if argv is None else argv)
    version = version_info or sys.version_info[:3]
    if _unsupported(version):
        return _diagnostic(version, "--json" in args)
    from autobrain.cli import app

    app()
    return 0
