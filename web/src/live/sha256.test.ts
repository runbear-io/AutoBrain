/**
 * The synchronous digest must agree with the platform implementation.
 *
 * A corpus fingerprint that only looks like a sha256 would be a fabricated
 * identity, so these vectors are checked against both the published FIPS test
 * vectors and the runtime's own Web Crypto digest.
 */

import { describe, expect, test } from "bun:test";
import { sha256Hex } from "./sha256";

async function webCryptoHex(text: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

describe("synchronous sha256", () => {
  test("matches the published vectors", () => {
    expect(sha256Hex("")).toBe(
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    );
    expect(sha256Hex("abc")).toBe(
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    );
    expect(sha256Hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")).toBe(
      "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
    );
  });

  test("agrees with Web Crypto across block boundaries and unicode", async () => {
    const samples = [
      "",
      "a",
      "x".repeat(55),
      "x".repeat(56),
      "x".repeat(64),
      "x".repeat(1000),
      "policy · refund — “quoted” 日本語",
      JSON.stringify({ source_id: "doc-1", title: "Refund", text: "Refunds within 30 days." }),
    ];
    for (const sample of samples) {
      expect(sha256Hex(sample)).toBe(await webCryptoHex(sample));
    }
  });
});
