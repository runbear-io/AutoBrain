import { describe, expect, test } from "bun:test";
import {
  PUBLIC_SOURCE_CAPABILITIES,
  SOURCE_CAPABILITIES,
  parseSourceImport,
  sourceCapability,
  SourceImportError,
} from "./sourceContracts";

describe("source capability and import contracts", () => {
  test("keeps fixture utilities in the internal inventory but out of public choices", () => {
    expect(SOURCE_CAPABILITIES.map((item) => item.id)).toEqual([
      "fixture",
      "slack-export",
      "notion-snapshot",
      "approved-read-only-connector",
    ]);
    expect(PUBLIC_SOURCE_CAPABILITIES.map((item) => item.id)).toEqual([
      "slack-export",
      "notion-snapshot",
      "approved-read-only-connector",
    ]);
    expect(sourceCapability("fixture").credentialsRequired).toBe(false);
    expect(sourceCapability("slack-export").mutability).toBe("frozen_export");
    expect(sourceCapability("notion-snapshot").capabilities).toContain("fetch");
    expect(sourceCapability("approved-read-only-connector").readiness).toBe("GATED");
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
