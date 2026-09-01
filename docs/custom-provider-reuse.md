# Custom provider reuse and verification

AutoBrain does not vendor Omo, Senpi, or GJC runtime code. Those runtimes are
JavaScript/Bun packages with different execution and distribution boundaries.
Their useful local configuration convention is represented by:

```text
provider id -> endpoint/base URL -> apiKeyEnv -> model list
```

AutoBrain reuses its existing Python implementation instead:

- OpenAI-compatible chat and embedding transport
- loopback metering proxy
- local orchestration
- keyring and 0600 fallback storage
- redacted projections and run provenance

## Verified local flow

```text
provider add
→ provider status
→ provider verify
→ custom:<provider-id> run
```

This flow was exercised against a loopback OpenAI-compatible mock implementing
`/v1/models`, `/v1/embeddings`, and `/v1/chat/completions`. A routing defect
where custom embeddings bypassed the configured endpoint was fixed and
regression-tested.

## Current limits

- Registration is local CLI-only; the Web never receives credentials.
- Custom provider cost remains incomplete unless pricing is configured.
- A custom provider is not recommendation-eligible by default.
- Real customer-source evaluation still requires authorized Slack/Notion data.
- Omo/GJC credentials and local auth stores are never imported.
