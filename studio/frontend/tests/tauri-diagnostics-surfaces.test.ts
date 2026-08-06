// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { after, test } from "node:test";

import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { createServer } from "vite";

const ESCAPE_CHARACTER = String.fromCharCode(27);
const BELL_CHARACTER = String.fromCharCode(7);
const TOKEN = "hf_12345678901234567890";
const PRESIGNED_VALUE = "presignedvalue987";
const FRAGMENT_SECRET = "fragmentsecret";
const NATIVE_PATH_LEASE = "abc.DEF_123";
const SENSITIVE_DIAGNOSTICS = [
  `${ESCAPE_CHARACTER}[31muse https://alice:password@example.invalid/simple${ESCAPE_CHARACTER}[0m`,
  `token ${TOKEN}`,
  "unix /home/alice/project windows C:\\Users\\Alice\\project",
  `${ESCAPE_CHARACTER}]8;;https://osc-target.invalid${BELL_CHARACTER}useful-link${ESCAPE_CHARACTER}]8;;${BELL_CHARACTER}`,
  `signed=https://example.com/object?X-Amz-Signature=${PRESIGNED_VALUE}&version=1#${FRAGMENT_SECRET}`,
  `native_path_lease=${NATIVE_PATH_LEASE}`,
  "-----BEGIN PRIVATE KEY-----",
  "super-secret-key-material",
  "-----END PRIVATE KEY-----",
  "still useful … [line truncated]",
].join("\n");

const vite = await createServer({
  appType: "custom",
  logLevel: "error",
  server: { middlewareMode: true },
});

after(async () => {
  await vite.close();
});

const { StartupScreen } = await vite.ssrLoadModule(
  "/src/components/tauri/startup-screen.tsx",
);
const { UpdateScreen } = await vite.ssrLoadModule(
  "/src/components/tauri/update-screen.tsx",
);
const { UpdateBanner } = await vite.ssrLoadModule(
  "/src/components/tauri/update-banner.tsx",
);

function assertSafeAndUseful(html: string) {
  for (const secret of [
    TOKEN,
    "alice:password",
    "/home/alice",
    "C:\\Users\\Alice",
    "super-secret-key-material",
    "osc-target.invalid",
    PRESIGNED_VALUE,
    FRAGMENT_SECRET,
    NATIVE_PATH_LEASE,
    ESCAPE_CHARACTER,
  ]) {
    assert.doesNotMatch(html, new RegExp(secret.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));
  }

  assert.match(html, /hf_&lt;redacted&gt;/);
  assert.match(html, /\$HOME\/project/);
  assert.match(html, /%USERPROFILE%\\project/);
  assert.match(html, /useful-link/);
  assert.match(html, /X-Amz-Signature=&lt;redacted&gt;/);
  assert.match(html, /native_path_lease=&lt;redacted-native-path-lease&gt;/);
  assert.match(html, /still useful … \[line truncated\]/);
}

const startupDefaults = {
  logs: [] as string[],
  error: null as string | null,
  currentStepIndex: 0,
  progressDetail: null as string | null,
  startupMessage: "Loading Unsloth..." as const,
  elevationPackages: [] as string[],
  onInstall() {},
  onRetry() {},
  onRetryInstall() {},
  onApproveElevation() {},
  onStartServer() {},
  async onCopyDiagnostics() {
    return { ok: true as const };
  },
};

for (const [name, status, props] of [
  ["startup progress", "starting", { logs: [SENSITIVE_DIAGNOSTICS] }],
  ["install progress", "installing", { progressDetail: SENSITIVE_DIAGNOSTICS }],
  ["install failure", "install-error", { error: SENSITIVE_DIAGNOSTICS }],
  ["repair progress", "repairing", { progressDetail: SENSITIVE_DIAGNOSTICS }],
  ["repair failure", "repair-error", { error: SENSITIVE_DIAGNOSTICS }],
  ["startup failure", "error", { error: SENSITIVE_DIAGNOSTICS }],
] as const) {
  test(`${name} redacts retained child output before rendering`, () => {
    const html = renderToStaticMarkup(
      createElement(StartupScreen, {
        ...startupDefaults,
        ...props,
        status,
      }),
    );
    assertSafeAndUseful(html);
  });
}

test("install progress remains bounded after redaction", () => {
  const html = renderToStaticMarkup(
    createElement(StartupScreen, {
      ...startupDefaults,
      status: "installing",
      progressDetail: `visible ${"x".repeat(1_100)} hidden-tail`,
    }),
  );

  assert.match(html, /visible x+/);
  assert.match(html, /…/);
  assert.doesNotMatch(html, /hidden-tail/);
});

test("elevation package output is redacted as one bounded stream", () => {
  const html = renderToStaticMarkup(
    createElement(StartupScreen, {
      ...startupDefaults,
      status: "needs-elevation",
      elevationPackages: [
        `${ESCAPE_CHARACTER}[31muse https://alice:password@example.invalid/simple${ESCAPE_CHARACTER}[0m`,
        `token ${TOKEN} unix /home/alice/project windows C:\\Users\\Alice\\project`,
        `${ESCAPE_CHARACTER}]8;;https://osc-target.invalid${BELL_CHARACTER}useful-link${ESCAPE_CHARACTER}]8;;${BELL_CHARACTER}`,
        `signed=https://example.com/object?X-Amz-Signature=${PRESIGNED_VALUE}&version=1#${FRAGMENT_SECRET}`,
        `native_path_lease=${NATIVE_PATH_LEASE}`,
        "-----BEGIN PRIVATE KEY-----",
        "super-secret-key-material",
        "-----END PRIVATE KEY-----",
        "still useful … [line truncated]",
        `visible ${"x".repeat(1_100)} hidden-tail`,
      ],
    }),
  );

  assertSafeAndUseful(html);
  assert.doesNotMatch(html, /-----END PRIVATE KEY-----/);
  assert.match(html, /visible x+/);
  assert.match(html, /…/);
  assert.doesNotMatch(html, /hidden-tail/);
});

test("update screen redacts both logs and the active error before rendering", () => {
  const html = renderToStaticMarkup(
    createElement(UpdateScreen, {
      status: "error",
      logs: SENSITIVE_DIAGNOSTICS.split("\n"),
      progress: 42,
      error: SENSITIVE_DIAGNOSTICS,
      onRetry() {},
      onSkipRestart() {},
      async onCopyDiagnostics() {
        return { ok: true as const };
      },
    }),
  );

  assertSafeAndUseful(html);
  assert.doesNotMatch(html, /-----END PRIVATE KEY-----/);
});

test("update screen redacts an unterminated private key split across log entries", () => {
  const html = renderToStaticMarkup(
    createElement(UpdateScreen, {
      status: "updating-backend",
      logs: [
        "update started",
        "-----BEGIN PRIVATE KEY-----",
        "unterminated-secret-key-material",
      ],
      progress: 42,
      error: null,
      onRetry() {},
      onSkipRestart() {},
      async onCopyDiagnostics() {
        return { ok: true as const };
      },
    }),
  );

  assert.match(html, /update started/);
  assert.doesNotMatch(html, /-----BEGIN PRIVATE KEY-----/);
  assert.doesNotMatch(html, /unterminated-secret-key-material/);
});

test("update banner redacts a retained failure before rendering", () => {
  const html = renderToStaticMarkup(
    createElement(UpdateBanner, {
      status: "idle",
      info: null,
      dismissed: false,
      lastFailure: {
        error: SENSITIVE_DIAGNOSTICS,
        phase: "backend",
        progress: 42,
        logs: [SENSITIVE_DIAGNOSTICS],
      },
      updatePolicyMode: "in_app",
      manualReleaseUrl: null,
      onInstall() {},
      onDismiss() {},
      async onCopyDiagnostics() {
        return { ok: true as const };
      },
    }),
  );

  assertSafeAndUseful(html);
});
