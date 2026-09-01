# AutoBrain public release checklist

This checklist separates work that can be completed locally from actions that
require an owner with external access or release authority.

## Completed locally

- Web-first retrieval diagnostic flow
- Fixture removed from official product source choices
- Slack export and Notion snapshot synthetic replay coverage
- Source readiness and fail-closed unsupported-source states
- Archive symlink, member-count, size, mutation, and redaction boundaries
- OAuth callback and malformed OAuth-index handling
- Benchmark holdout and leakage boundaries
- Streaming evaluation aggregation for large case iterables
- Ruff, basedpyright, Web tests/build, wheel/sdist, and installed CLI smoke

## Owner-required evidence and decisions

- Provide a redacted real Slack Workspace Export ZIP.
- Authorize a fresh Notion read-only test workspace or snapshot.
- Choose and authorize the provider(s) supported in the first public release.
- Decide whether the first release makes diagnostic-only claims or
  recommendation-grade claims.
- Approve the final release commit and version.
- Approve publication of the source archive and Homebrew formula.

## Final release sequence

1. Freeze the release code after real Slack, Notion, and provider checks.
2. Recompute the source digest from that exact code.
3. Update release evidence and formula metadata together.
4. Build and hash the source archive and wheel.
5. Install the artifacts in fresh environments and run the installed-binary
   checks.
6. Publish only after the owner approves the release, tag, and Homebrew write.

Until the owner-required items are complete, describe the project as a local
retrieval diagnostic preview, not a recommendation-grade hosted service.
