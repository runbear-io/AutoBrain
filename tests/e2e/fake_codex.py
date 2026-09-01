#!/usr/bin/env python3
import json
import sys

if sys.argv[1:3] == ["login", "status"]:
    print("Logged in")
elif sys.argv[1:2] == ["--version"]:
    print("codex-e2e 1.0")
elif sys.argv[1:2] == ["exec"]:
    print(
        json.dumps(
            {
                "type": "agent_message",
                "text": "Fixture answer",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        )
    )
else:
    print("unsupported", file=sys.stderr)
    raise SystemExit(2)
