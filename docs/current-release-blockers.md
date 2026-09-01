# Current release blockers

## Engineering complete for the supported local surface

- Notion snapshot, JSON/JSONL normalized imports
- Markdown, TXT, and HTML local-file imports
- Slack connector retained behind an explicit advanced/future gate (not a public v1 claim)
- PDF and DOCX typed unavailable states
- Fixture excluded from public source choices
- Security, Web, package, and installed CLI checks

## Remaining release-bound blocker

The checked-in release evidence and Homebrew metadata are bound to an older
`uv.lock` digest. The current lockfile passes `uv lock --check`, but the
retained release records do not match the current source tree. The release
formula remains intentionally unapproved, so these records must be regenerated
only after the owner freezes and approves the release source.

## Remaining external verification

- Fresh Notion read-only authorization or customer-safe snapshot
- Optional advanced/future Slack evidence is out of public v1 scope
- Provider authorization for the providers claimed in the release
- Decision on whether the public claim is “find your best company Brain” with a
  recommendation backed by real evidence
- Owner approval for the final release, tag, archive hosting, and Homebrew
  publication

Until those external checks are complete, the local surface is verified but the
public recommendation claim is not evidence-complete.
