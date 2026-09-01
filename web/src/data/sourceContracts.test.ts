import { describe, expect, test } from "bun:test";
import {
  PUBLIC_SOURCE_CAPABILITIES,
  SOURCE_CAPABILITIES,
  localFileFormatForName,
  parseSourceImport,
  sourceCapability,
  SourceImportError,
} from "./sourceContracts";

describe("source capability and import contracts", () => {
  test("local file capability exposes only supported text/html plus typed unavailable formats", () => {
    const local = sourceCapability("local-file");
    expect(local.acceptedFormats).toEqual(["MARKDOWN", "TXT", "HTML", "PDF", "DOCX"]);
    expect(local.credentialsRequired).toBe(false);
    expect(local.detail).toContain("without uploading");
    expect(localFileFormatForName("policy.md")).toBe("MARKDOWN");
    expect(localFileFormatForName("notes.txt")).toBe("TXT");
    expect(localFileFormatForName("page.html")).toBe("HTML");
    expect(localFileFormatForName("report.pdf")).toBe("PDF");
    expect(localFileFormatForName("brief.docx")).toBe("DOCX");
  });

  test("parses supported local HTML and blocks PDF/DOCX extraction", () => {
    const result = parseSourceImport("local-file", "<h1>Policy</h1><script>secret</script>Visible", "HTML");
    expect(result.records[0]?.text).toBe("Policy Visible");
    expect(result.records[0]?.source_kind).toBe("LOCAL_FILE");
    expect(() => parseSourceImport("local-file", "binary", "PDF")).toThrow(/unavailable/i);
    expect(() => parseSourceImport("local-file", "binary", "DOCX")).toThrow(/unavailable/i);
  });
  test("keeps fixture utilities in the internal inventory but out of public choices", () => {
    expect(SOURCE_CAPABILITIES.map((item) => item.id)).toEqual([
      "fixture",
      "local-file",
      "slack-export",
      "notion-snapshot",
      "approved-read-only-connector",
    ]);
    expect(PUBLIC_SOURCE_CAPABILITIES.map((item) => item.id)).toEqual([
      "local-file",
      "notion-snapshot",
    ]);
    expect(sourceCapability("fixture").readiness).toBe("TEST_ONLY");
    expect(sourceCapability("fixture").credentialsRequired).toBe(false);
    expect(sourceCapability("slack-export").mutability).toBe("frozen_export");
    expect(sourceCapability("slack-export").readiness).toBe("GATED");
    expect(PUBLIC_SOURCE_CAPABILITIES.some((item) => item.id === "slack-export")).toBe(false);
    expect(sourceCapability("notion-snapshot").capabilities).toContain("fetch");
    expect(sourceCapability("approved-read-only-connector").readiness).toBe("GATED");
  });

  test("normalized export sources remain JSON or JSONL only", () => {
    for (const source of PUBLIC_SOURCE_CAPABILITIES.filter((item) => item.id !== "local-file")) {
      expect(source.acceptedFormats).toEqual(["JSON", "JSONL"]);
    }
    expect(() =>
      parseSourceImport(
        "slack-export",
        JSON.stringify({ source_id: "slack:1", title: "Notes", text: "x" }),
        "TXT" as never,
      ),
    ).toThrow(/does not accept/i);
  });

  test("parses a versioned JSON envelope and preserves normalized fields", () => {
    const result = parseSourceImport(
      "notion-snapshot",
      JSON.stringify({
        schema_version: 1,
        records: [
          {
            source_id: "notion:page:runbook",
            title: "Runbook",
            text: "Restart the worker after confirming the queue is drained.",
            source_kind: "NOTION_PAGE",
          },
        ],
      }),
      "JSON",
    );
    expect(result.schema_version).toBe(1);
    expect(result.records[0]?.source_id).toBe("notion:page:runbook");
    expect(result.records[0]?.source_kind).toBe("NOTION_PAGE");
  });

  test("parses JSONL with blank lines and stable source identity", () => {
    const result = parseSourceImport(
      "slack-export",
      [
        JSON.stringify({ source_id: "slack:message:1", title: "One", text: "First" }),
        "",
        JSON.stringify({ source_id: "slack:message:2", title: "Two", text: "Second" }),
      ].join("\n"),
      "JSONL",
    );
    expect(result.format).toBe("JSONL");
    expect(result.records.map((record) => record.source_id)).toEqual([
      "slack:message:1",
      "slack:message:2",
    ]);
  });

  test("rejects malformed envelopes, duplicate IDs, and credential-shaped fields", () => {
    expect(() => parseSourceImport("fixture", "{}", "JSON")).toThrow(SourceImportError);
    expect(() =>
      parseSourceImport(
        "fixture",
        JSON.stringify([
          { source_id: "fixture:1", title: "One", text: "x" },
          { source_id: "fixture:1", title: "Again", text: "y" },
        ]),
        "JSON",
      ),
    ).toThrow(/unique/i);
    expect(() =>
      parseSourceImport(
        "slack-export",
        JSON.stringify([{ source_id: "slack:1", title: "Leak", text: "x", access_token: "nope" }]),
        "JSON",
      ),
    ).toThrow(/not allowed/i);
  });

  test("accepts future connector-shaped data without making readiness ready", () => {
    const result = parseSourceImport(
      "approved-read-only-connector",
      JSON.stringify([{ source_id: "future:1", title: "Future", text: "x" }]),
      "JSON",
    );
    expect(result.records).toHaveLength(1);
    expect(sourceCapability(result.source).readiness).toBe("GATED");
    expect(sourceCapability(result.source).credentialsRequired).toBe(true);
  });
});
