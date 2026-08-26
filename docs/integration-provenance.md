# Integration provenance

`integration-provenance.json` is the machine-readable inventory for the
integration surfaces AutoBrain currently reuses or deliberately gates. The
same records are embedded in `manifest.json` and `comparison.json` under
`provenance.integrations`, and the HTML report renders their status and
classification.

Each record names the provider, source, backend, version, license, auth kind,
capabilities, usage provenance, and evidence classification. A `null` version
or license means that this repository does not have a verified value for that
surface; it is not permission to infer one from a package name or executable.
Secrets and credential material are never part of the record.

## Reuse classifications

- **`direct_reuse`**: the project uses an explicitly identified local backend
  without wrapping an external consumer login or inventing a remote API.
- **`protocol_reuse`**: the project speaks an existing provider protocol or
  first-party CLI contract and records the selected auth kind and capabilities.
- **`thin_adapter`**: the native integration remains responsible for its own
  lifecycle; AutoBrain only translates the shared corpus/query boundary and
  collects bounded evidence.
- **`gated`**: the surface is represented so readiness is machine-readable,
  but execution is refused until the missing official contract or project
  policy gate is satisfied.

The catalog intentionally does not add Kimi or Grok adapters, custom OAuth,
API-key substitution for consumer subscriptions, or a new LLM Wiki indexing
path. Candidate versions and licenses come from the approved candidate pin
registry. Runtime CLI versions are observed only when a provider probe returns
them and are not fabricated in this static inventory.

## Current and gated surfaces

Current source surfaces are Slack frozen-export ingestion and read-only Notion
MCP capture. Current candidate surfaces are the pinned LLM Wiki, Mem0 OSS, and
GBrain adapters. Codex and Claude are protocol-reuse subscription adapters;
their consumer-subscription identity is distinct from API-key embedding
configuration. Local hash embedding is retained for smoke execution only.

Google Drive and Confluence are explicit readiness gates. Kimi and Grok are
explicit unsupported provider gates. Gated records have no capabilities and
unavailable usage provenance, so they cannot appear ready merely because a
binary, endpoint, or API key exists.
