/**
 * Source capability inventory and import boundary for the web-first experiment.
 *
 * This is deliberately a data-only contract. It describes which local source
 * inputs are accepted by the UI and what is verified before they can be used;
 * it never performs OAuth, reads credentials, or calls a connector.
 */

export type SourceCapabilityId =
  | "fixture"
  | "local-file"
  | "slack-export"
  | "notion-snapshot"
  | "approved-read-only-connector";

export type SourceReadiness = "READY" | "CONFIGURED" | "AUTH_REQUIRED" | "GATED" | "TEST_ONLY";
export type SourceImportFormat =
  | "JSON"
  | "JSONL"
  | "MARKDOWN"
  | "TXT"
  | "HTML"
  | "PDF"
  | "DOCX"
  | "UNSUPPORTED";
export type SourceMutability = "frozen_export" | "live_mcp_captured" | "future_read_only";
export type SourceCapability = "read" | "search" | "fetch" | "normalize" | "freeze";

export interface SourceCapabilityInventoryItem {
  id: SourceCapabilityId;
  label: string;
  readiness: SourceReadiness;
  acceptedFormats: readonly SourceImportFormat[];
  mutability: SourceMutability;
  capabilities: readonly SourceCapability[];
  credentialsRequired: boolean;
  detail: string;
  remediation: string | null;
}

/** The single source inventory consumed by setup and import validation. */
export const SOURCE_CAPABILITIES: readonly SourceCapabilityInventoryItem[] = [
  {
    id: "fixture",
    label: "Local fixture",
    readiness: "TEST_ONLY",
    acceptedFormats: ["JSON", "JSONL"],
    mutability: "frozen_export",
    capabilities: ["read", "normalize", "freeze"],
    credentialsRequired: false,
    detail: "Deterministic, local, credential-free input for development and QA only.",
    remediation: null,
  },
  {
    id: "local-file",
    label: "Local file",
    readiness: "READY",
    acceptedFormats: ["MARKDOWN", "TXT", "HTML", "PDF", "DOCX"],
    mutability: "frozen_export",
    capabilities: ["read", "normalize", "freeze"],
    credentialsRequired: false,
    detail: "Read one local Markdown, TXT, or HTML document without uploading it.",
    remediation: "Choose a Markdown, TXT, or HTML file. PDF and DOCX extraction is unavailable.",
  },
  {
    id: "slack-export",
    label: "Slack export",
    readiness: "READY",
    acceptedFormats: ["JSON", "JSONL"],
    mutability: "frozen_export",
    capabilities: ["read", "normalize", "freeze"],
    credentialsRequired: false,
    detail: "A sanitized normalized export prepared from an official Slack archive.",
    remediation: null,
  },
  {
    id: "notion-snapshot",
    label: "Notion snapshot",
    readiness: "CONFIGURED",
    acceptedFormats: ["JSON", "JSONL"],
    mutability: "live_mcp_captured",
    capabilities: ["read", "search", "fetch", "normalize", "freeze"],
    credentialsRequired: false,
    detail: "A bounded, read-only snapshot captured through the approved Notion MCP flow.",
    remediation: "Capture a fresh read-only snapshot before importing.",
  },
  {
    id: "approved-read-only-connector",
    label: "Future approved read-only connector",
    readiness: "GATED",
    acceptedFormats: ["JSON", "JSONL"],
    mutability: "future_read_only",
    capabilities: ["read", "normalize", "freeze"],
    credentialsRequired: true,
    detail: "Reserved for a connector with an approved read-only contract and evidence.",
    remediation: "Wait for connector approval and a verified read-only transport.",
  },
] as const;

/** Sources that production/user setup may present as official inputs. */
export const PUBLIC_SOURCE_CAPABILITIES = SOURCE_CAPABILITIES.filter(
  (item) => item.id !== "fixture",
);

export type ImportSourceId = SourceCapabilityId;

export interface SourceImportRecord {
  source_id: string;
  title: string;
  text: string;
  canonical_url?: string;
  source_kind?: string;
  [key: string]: unknown;
}

export interface SourceImportBatch {
  schema_version: 1;
  source: ImportSourceId;
  format: SourceImportFormat;
  records: SourceImportRecord[];
}

const SOURCE_IDS: readonly ImportSourceId[] = [
  "fixture",
  "local-file",
  "slack-export",
  "notion-snapshot",
];
const FORBIDDEN_KEY = /(?:secret|token|password|api[_-]?key|authorization|credential)/i;
const MAX_IMPORT_BYTES = 8 * 1024 * 1024;
const MAX_LOCAL_FILE_BYTES = 256 * 1024;
const MAX_RECORDS = 50_000;
const MAX_TEXT_CHARS = 250_000;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isSourceId(value: unknown): value is ImportSourceId {
  return (
    typeof value === "string" &&
    (SOURCE_IDS.includes(value as ImportSourceId) || value === "approved-read-only-connector")
  );
}

function rejectForbiddenKeys(value: unknown, path: string): void {
  if (Array.isArray(value)) {
    value.forEach((item, index) => rejectForbiddenKeys(item, `${path}[${index}]`));
    return;
  }
  if (!isRecord(value)) return;
  for (const [key, item] of Object.entries(value)) {
    if (FORBIDDEN_KEY.test(key)) {
      throw new SourceImportError(`${path}.${key} is not allowed in source imports`);
    }
    rejectForbiddenKeys(item, `${path}.${key}`);
  }
}

function validateRecord(value: unknown, index: number): SourceImportRecord {
  if (!isRecord(value)) throw new SourceImportError(`records[${index}] must be an object`);
  const sourceId = value.source_id;
  const title = value.title;
  const text = value.text;
  if (typeof sourceId !== "string" || sourceId.length < 3 || sourceId.length > 512) {
    throw new SourceImportError(`records[${index}].source_id must be a bounded string`);
  }
  if (typeof title !== "string" || title.length < 1 || title.length > 10_000) {
    throw new SourceImportError(`records[${index}].title must be a bounded string`);
  }
  if (typeof text !== "string" || text.length > MAX_TEXT_CHARS) {
    throw new SourceImportError(`records[${index}].text must be a bounded string`);
  }
  return { ...value, source_id: sourceId, title, text };
}

function localFileText(payload: string, format: SourceImportFormat): string {
  if (format === "PDF" || format === "DOCX" || format === "UNSUPPORTED") {
    throw new SourceImportError(`${format} local-file extraction is unavailable`);
  }
  if (format === "HTML") {
    return payload
      .replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, " ")
      .replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, " ")
      .replace(/<[^>]+>/g, " ")
      .replace(/&nbsp;/gi, " ")
      .replace(/&amp;/gi, "&")
      .replace(/&lt;/gi, "<")
      .replace(/&gt;/gi, ">")
      .replace(/\s+/g, " ")
      .trim();
  }
  return payload;
}

export function localFileFormatForName(name: string): SourceImportFormat {
  const extension = name.slice(name.lastIndexOf(".")).toLowerCase();
  const formats: Record<string, SourceImportFormat> = {
    ".md": "MARKDOWN",
    ".markdown": "MARKDOWN",
    ".txt": "TXT",
    ".text": "TXT",
    ".html": "HTML",
    ".htm": "HTML",
    ".pdf": "PDF",
    ".docx": "DOCX",
  };
  return formats[extension] ?? "UNSUPPORTED";
}

export class SourceImportError extends Error {
  constructor(message: string) {
    super(`invalid source import: ${message}`);
    this.name = "SourceImportError";
  }
}

/**
 * Parse a normalized JSON or JSONL payload without coercing unknown shapes.
 * JSON accepts either an array of records or `{schema_version, records}`;
 * JSONL accepts one record per non-empty line and no envelope.
 */
export function parseSourceImport(
  source: ImportSourceId,
  payload: string,
  format: SourceImportFormat,
): SourceImportBatch {
  if (!isSourceId(source)) throw new SourceImportError(`unsupported source ${String(source)}`);
  const capability = SOURCE_CAPABILITIES.find((item) => item.id === source);
  if (!capability?.acceptedFormats.includes(format)) {
    throw new SourceImportError(`${source} does not accept ${format}`);
  }
  if (new TextEncoder().encode(payload).byteLength > MAX_IMPORT_BYTES) {
    throw new SourceImportError("payload exceeds the 8 MiB limit");
  }

  let records: unknown[];
  if (source === "local-file") {
    if (new TextEncoder().encode(payload).byteLength > MAX_LOCAL_FILE_BYTES) {
      throw new SourceImportError("local file exceeds the 256 KiB limit");
    }
    const text = localFileText(payload, format);
    if (text.trim().length === 0) throw new SourceImportError("local file has no readable text");
    records = [
      {
        source_id: `local_file:${format.toLowerCase()}`,
        title: `${format.toLowerCase()} local file`,
        text,
        source_kind: "LOCAL_FILE",
      },
    ];
  } else if (format === "JSONL") {
    records = payload
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line, index) => {
        try {
          return JSON.parse(line) as unknown;
        } catch {
          throw new SourceImportError(`JSONL line ${index + 1} is not valid JSON`);
        }
      });
  } else {
    let parsed: unknown;
    try {
      parsed = JSON.parse(payload) as unknown;
    } catch {
      throw new SourceImportError("JSON payload is not valid JSON");
    }
    if (Array.isArray(parsed)) {
      records = parsed;
    } else if (isRecord(parsed) && parsed.schema_version === 1 && Array.isArray(parsed.records)) {
      records = parsed.records;
    } else {
      throw new SourceImportError("JSON must be a record array or version-1 envelope");
    }
  }

  if (records.length === 0) throw new SourceImportError("at least one record is required");
  if (records.length > MAX_RECORDS) throw new SourceImportError("record count exceeds the limit");
  rejectForbiddenKeys(records, "records");
  const validated = records.map(validateRecord);
  const ids = new Set(validated.map((record) => record.source_id));
  if (ids.size !== validated.length) throw new SourceImportError("source_id values must be unique");
  return { schema_version: 1, source, format, records: validated };
}

export function sourceCapability(id: SourceCapabilityId): SourceCapabilityInventoryItem {
  const item = SOURCE_CAPABILITIES.find((candidate) => candidate.id === id);
  if (!item) throw new Error(`unknown source capability: ${id}`);
  return item;
}
