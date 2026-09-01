# Security policy

## Supported versions

AutoBrain is experimental. Security fixes are applied to the latest version
on the default development branch; older releases may not receive fixes.

## Reporting a vulnerability

Please do not open a public issue for a vulnerability. Report it privately to
the repository maintainers through the security contact configured for the
GitHub repository. Include:

- a concise description and impact;
- the affected version or commit;
- reproducible steps or a minimal proof of concept;
- whether private data or credentials may be exposed.

Do not include real credentials, private Slack exports, private Notion pages,
or other customer data in a report. Replace them with synthetic fixtures.

Maintainers should acknowledge reports, assess severity, and coordinate a fix
and disclosure timeline with the reporter.

## Security boundaries

AutoBrain is designed for local evaluation. Treat source exports and provider
subprocesses as sensitive. Keep credentials outside source fixtures and logs,
use isolated disposable homes for testing, and verify redaction before sharing
artifacts.
