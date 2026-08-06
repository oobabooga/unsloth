// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  createDiagnosticsLineRedactor,
  latestRedactedDiagnosticsLine,
  redactDiagnosticsLineStream,
  redactDiagnosticsText,
} from "../src/lib/diagnostics-redaction.ts";

const ESCAPE_CHARACTER = String.fromCharCode(27);

test("strips ANSI color sequences from diagnostics", () => {
  assert.equal(
    redactDiagnosticsText(`${ESCAPE_CHARACTER}[31mfailed${ESCAPE_CHARACTER}[0m`),
    "failed",
  );
});

test("strips OSC hyperlinks terminated by BEL from diagnostics", () => {
  const bell = String.fromCharCode(7);
  const text = `${ESCAPE_CHARACTER}]8;;https://example.com${bell}docs${ESCAPE_CHARACTER}]8;;${bell}`;

  assert.equal(redactDiagnosticsText(text), "docs");
});

test("strips OSC hyperlinks terminated by ST from diagnostics", () => {
  const terminator = `${ESCAPE_CHARACTER}\\`;
  const text = `${ESCAPE_CHARACTER}]8;;https://example.com${terminator}docs${ESCAPE_CHARACTER}]8;;${terminator}`;

  assert.equal(redactDiagnosticsText(text), "docs");
});

test("strips an unterminated OSC sequence from diagnostics", () => {
  const text = `before${ESCAPE_CHARACTER}]8;;https://example.com`;

  assert.equal(redactDiagnosticsText(text), "before");
});

test("an unterminated OSC sequence does not erase later log lines", () => {
  const text = `before${ESCAPE_CHARACTER}]8;;https://example.com\ntraceback`;

  assert.equal(redactDiagnosticsText(text), "before\ntraceback");
});

test("redacts private keys split across log lines", () => {
  const text = [
    "before",
    "-----BEGIN PRIVATE KEY-----",
    "secret material",
    "-----END PRIVATE KEY-----",
    "after",
  ].join("\n");

  assert.equal(redactDiagnosticsText(text), "before\n<redacted-private-key>\nafter");
});

test("redacts an unterminated private key block", () => {
  const text = "before\n-----BEGIN PRIVATE KEY-----\nsecret material";

  assert.equal(redactDiagnosticsText(text), "before\n<redacted-private-key>");
});

test("startup preview never selects a private key body line", () => {
  assert.equal(
    latestRedactedDiagnosticsLine([
      "loading",
      "-----BEGIN PRIVATE KEY-----",
      "secret material",
    ]),
    "<redacted-private-key>",
  );
  assert.equal(
    latestRedactedDiagnosticsLine([
      "-----BEGIN PRIVATE KEY-----",
      "secret material",
      "-----END PRIVATE KEY-----",
      "ready",
    ]),
    "ready",
  );
  assert.equal(
    latestRedactedDiagnosticsLine([
      `${ESCAPE_CHARACTER}[31m-----BEGIN PRIVATE KEY-----${ESCAPE_CHARACTER}[0m`,
      "secret material",
    ]),
    "<redacted-private-key>",
  );
});

test("incremental stream redaction preserves multiline private-key state", () => {
  const first = redactDiagnosticsLineStream([
    "before",
    "-----BEGIN PRIVATE KEY-----",
    "secret material",
  ]);
  assert.deepEqual(first.displayedLines, ["before", "<redacted-private-key>"]);

  const appended = redactDiagnosticsLineStream(
    [
      "before",
      "-----BEGIN PRIVATE KEY-----",
      "secret material",
      "-----END PRIVATE KEY-----",
      "after",
    ],
    first,
  );
  assert.deepEqual(appended.displayedLines, [
    "before",
    "<redacted-private-key>",
    "after",
  ]);

  const shifted = redactDiagnosticsLineStream(
    [
      "-----BEGIN PRIVATE KEY-----",
      "secret material",
      "-----END PRIVATE KEY-----",
      "after",
      "ready",
    ],
    appended,
  );
  assert.deepEqual(shifted.displayedLines, [
    "<redacted-private-key>",
    "after",
    "ready",
  ]);
});

test("retained-log producer never stores a private-key body after rollover", () => {
  const redactor = createDiagnosticsLineRedactor();
  const retained: string[] = [];
  const append = (line: string) => {
    const displayed = redactor.redactLine(line);
    if (displayed !== null) retained.push(displayed);
    if (retained.length > 500) retained.shift();
  };

  append("before");
  append("-----BEGIN PRIVATE KEY-----");
  for (let index = 0; index < 700; index += 1) {
    append(`secret-key-body-${index}`);
  }

  assert.deepEqual(retained, ["before", "<redacted-private-key>"]);
  assert.doesNotMatch(latestRedactedDiagnosticsLine(retained), /secret-key-body/);

  append("-----END PRIVATE KEY-----");
  append("ready");
  assert.equal(latestRedactedDiagnosticsLine(retained), "ready");
});

test("redacts URL query values, fragments, and native path leases", () => {
  const redacted = redactDiagnosticsText(
    "signed=https://example.com/object?X-Amz-Signature=presignedvalue987&version=1#fragmentsecret native_path_lease=abc.DEF_123",
  );

  assert.doesNotMatch(redacted, /presignedvalue987|fragmentsecret|abc\.DEF_123/);
  assert.match(
    redacted,
    /\?X-Amz-Signature=<redacted>&version=<redacted>#<redacted>/,
  );
  assert.match(
    redacted,
    /native_path_lease=<redacted-native-path-lease>/,
  );
});

test("long benign lines stay on the linear redaction path", () => {
  const benign = "a".repeat(16 * 1024);
  const started = performance.now();
  for (let index = 0; index < 50; index += 1) {
    assert.equal(redactDiagnosticsText(benign), benign);
  }
  assert.ok(
    performance.now() - started < 1_000,
    "50 bounded benign lines should redact in under one second",
  );
});

test("long URL credentials are redacted without a length bypass", () => {
  const credential = "secret".repeat(2_000);
  assert.equal(
    redactDiagnosticsText(`https://${credential}@example.com/index`),
    "https://<redacted>@example.com/index",
  );
});
