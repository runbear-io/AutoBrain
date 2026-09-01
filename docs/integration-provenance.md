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

The public v1 source surface is read-only Notion MCP capture, plus normalized
JSON/JSONL and local Markdown/TXT/HTML inputs. Slack frozen-export ingestion and
the live Slack connector remain explicitly gated advanced/future functionality;
they are not official public setup inputs or public v1 release claims. Current
candidate surfaces are the pinned LLM Wiki, Mem0 OSS, and GBrain adapters. Codex
and Claude are protocol-reuse subscription adapters;
their consumer-subscription identity is distinct from API-key embedding
configuration. Local hash embedding is retained for smoke execution only.

Google Drive and Confluence are explicit readiness gates, not v1 source
connectors. SharePoint is likewise a gated Graph OAuth preview with no
production constructor. Onyx is a design-partner evaluation gate only; it has
no verified license, public API, runtime, ACL, resource, network, or teardown
proof. Kimi and Grok are explicit unsupported provider gates. Gated records
have no capabilities and unavailable usage provenance, so they cannot appear
ready merely because a binary, endpoint, or API key exists.

The v1 document boundary is intentionally narrower than a general file
indexer: public source ingestion accepts Notion MCP snapshots, normalized
JSON/JSONL records, and local Markdown/TXT/HTML files. PDF and DOCX are not
standalone v1 import formats. Slack export file links do not imply binary
extraction, and the retained Slack connector is advanced/future functionality
behind an explicit gate rather than a public setup or release claim. Fixtures
remain internal test-only data and are excluded from public setup choices.
