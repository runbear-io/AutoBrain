# Export Slack data for AutoBrain

AutoBrain's recommended Slack input is the official ZIP produced by Slack's
workspace export tool. You do not need to create a Slack App, copy a bot token,
or provide OAuth client credentials.

## Before you start

Slack export access depends on your role, plan, and workspace policies:

- Workspace Owners and Admins can generally export public-channel messages.
- Private channels and direct messages require additional plan capabilities
  and, in many cases, approved export access.
- Retention settings determine how much history is available.
- Standard JSON exports contain file links, not the file binaries.

If the Export tab is missing, ask a Workspace Owner or Admin. AutoBrain cannot
bypass Slack's permissions or recover data omitted from the export.

Official Slack references:

- [Export your workspace data](https://slack.com/help/articles/201658943-Export-your-workspace-data)
- [Guide to Slack import and export tools](https://slack.com/help/articles/204897248-Guide-to-Slack-import-and-export-tools)
- [How to read Slack data exports](https://slack.com/help/articles/220556107-How-to-read-Slack-data-exports)

## Download the ZIP

1. Open Slack on desktop.
2. Click **Admin** in the sidebar.
3. Select **Workspace settings**.
4. Open **Security**.
5. Select **Import & export data**.
6. Open the **Export** tab.
7. Choose the export type and date range available to your workspace.
8. Click **Start Export**.
9. Wait for Slack's email or workspace notification.
10. Return to the export page and click **Ready for download**.

Slack downloads a `.zip` file. Do not unzip, rename its internal files, or
repackage it before giving it to AutoBrain.

## Configure AutoBrain

Use the downloaded path:

```bash
autobrain source slack --export ~/Downloads/slack-export.zip
```

A successful import prints a summary:

```text
Slack export ready: 1842 messages from 12 channels
```

Verify it at any time:

```bash
autobrain source status --json
```

Or run `autobrain`, press `S`, keep the recommended export option, and paste
the ZIP path. The Knowledge section will show:

```text
[S] [x] Slack        export ready
```

## What AutoBrain reads

AutoBrain reads the archive directly without extracting it. It recognizes:

- workspace metadata from `team.json`,
- users from `users.json`,
- public channels from `channels.json`,
- supported private or direct-message catalogs when present,
- channel message JSON files,
- thread relationships,
- exported file names and links.

Messages are converted into the same `NormalizedDocument` format used for
Notion. Exact duplicate content is collapsed, evaluator holdouts are separated,
and every candidate receives the same frozen corpus.

## What AutoBrain stores

Configuration stores only:

- the resolved local path,
- the ZIP SHA-256,
- configuration time,
- channel, user, message, and file-link counts.

The configuration is stored with `0600` permissions under:

```text
~/.autobrain/sources/slack-export.json
```

The ZIP itself remains where you downloaded it. AutoBrain does not make a
second permanent copy. During a run, normalized candidate-visible content is
written to:

```text
~/.autobrain/runs/<run-id>/corpus-freeze.json
```

If the ZIP is moved, deleted, or changed, AutoBrain refuses to treat Slack as
ready. Configure the new file again.

## Privacy warning

A Slack export can contain confidential team conversations and personal data.

- Do not commit the ZIP to Git.
- Do not place it in a shared cloud folder unless company policy permits it.
- Keep the generated AutoBrain run directory private.
- Delete local exports and run artifacts according to your organization's
  retention policy.

AutoBrain does not upload the original ZIP. Normalized source content is still
provided to the candidate systems selected for the experiment, so review those
candidate boundaries before running sensitive corpora.

## Troubleshooting

### `Slack export not found`

The configured file was moved or deleted. Download it again or configure its
new path:

```bash
autobrain source slack --export /new/path/slack-export.zip
```

### `Slack export changed after it was configured`

The file's SHA-256 no longer matches. This prevents a run from silently using
different evidence. Configure the archive again.

### `Slack export must contain one channels.json catalog`

The ZIP is not a supported official workspace JSON export, or it was
repackaged. Use the original ZIP downloaded from Slack.

### Only public channels appear

That is expected for the standard export available on many plans. Private
channels and DMs require different Slack export permissions; AutoBrain can only
read what Slack included.

### Attached file content is missing

Standard exports generally contain links, not file binaries. AutoBrain
preserves the names and links as metadata but does not fetch private files
automatically.

## Advanced live Slack MCP

Operators who require a live crawl can still use the advanced MCP path:

```bash
export AUTOBRAIN_SLACK_CLIENT_ID="<slack-app-client-id>"
export AUTOBRAIN_SLACK_CLIENT_SECRET="<slack-app-client-secret>"
autobrain source slack --live
```

For most first-time evaluations, the export ZIP is simpler, more reproducible,
and easier to audit.
