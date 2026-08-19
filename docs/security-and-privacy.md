# Security and privacy

AutoBrain is intentionally local-first.

## Credentials

- OAuth uses provider-specific discovery, PKCE/state checks, audience-bound
  tokens, and the OS keyring where available.
- ChatGPT subscription authentication remains in the local Codex CLI
  credential store. AutoBrain checks the typed login status but does not ingest
  the credential, password, or browser token.
- Denied consent, revoked grants, admin rejection, missing app credentials,
  unavailable keyring, and callback failures settle as typed auth evidence.
- Slack and Notion credentials are never interchangeable.

## Local Slack exports

The recommended Slack source is an official local Workspace Export ZIP.
AutoBrain stores only the resolved path, SHA-256, configuration timestamp, and
non-sensitive counts under `~/.autobrain/sources/` with `0600` permissions. It
does not create a second permanent copy of the archive.

Archive members are read directly without extraction. Absolute paths,
traversal, symlinks, encrypted entries, duplicate names, unsafe sizes, invalid
JSON, and unsupported archive layouts are rejected. A missing or changed ZIP
is a typed source failure rather than a successful empty crawl.

## MCP boundary

Notion and the optional advanced live Slack connector use hosted MCP. The
policy intersects advertised tools with an explicit read allowlist. Write,
mutation, credential, token, and secret-shaped tools are refused. MCP results
and Slack archive content are untrusted data: document text cannot change
orchestration policy or invoke tools. There is no direct REST crawler, hosted
server, analytics, upload, or external publication path.

## Data locality and cleanup

Run directories contain hashes, coverage, benchmark provenance, candidate
artifacts, metrics, reports, and typed errors. The corpus freeze is local and
the report is offline. Reports do not copy OAuth tokens or secret-shaped
values. Candidate processes use run-local state and cleanup is guaranteed in a
`finally` block. Interruption does not resume or overwrite a run.

The report may link only to safe evidence. Never commit or externally send a
Slack export, run directory, corpus, screenshot containing source text, or
token unless company policy explicitly permits it.
