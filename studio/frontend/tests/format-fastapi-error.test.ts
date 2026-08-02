// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  formatApiErrorBody,
  readFastApiError,
} from "../src/lib/format-fastapi-error.ts";

test("formats an OpenAI-compatible error envelope", () => {
  assert.equal(
    formatApiErrorBody({
      error: {
        message: "Audio file is too large (max ~25 MB).",
        type: "invalid_request_error",
      },
    }),
    "Audio file is too large (max ~25 MB).",
  );
});

test("formats an Anthropic-compatible error envelope", () => {
  assert.equal(
    formatApiErrorBody({
      type: "error",
      error: { type: "rate_limit_error", message: "Try again later." },
    }),
    "Try again later.",
  );
});

test("keeps FastAPI detail and top-level message support", () => {
  assert.equal(
    formatApiErrorBody({ detail: "Invalid request" }),
    "Invalid request",
  );
  assert.equal(
    formatApiErrorBody({ message: "Provider failed" }),
    "Provider failed",
  );
  assert.equal(
    formatApiErrorBody({
      detail: [{ loc: ["body", "messages"], msg: "Field required" }],
    }),
    "messages: Field required",
  );
});

test("reads an OpenAI-compatible error response", async () => {
  const response = new Response(
    JSON.stringify({ error: { message: "Context limit exceeded" } }),
    { status: 400, headers: { "Content-Type": "application/json" } },
  );
  assert.equal(await readFastApiError(response), "Context limit exceeded");
});

test("falls back for malformed or empty error bodies", async () => {
  for (const body of [
    null,
    {},
    { error: null },
    { error: {} },
    { error: { message: 3 } },
  ]) {
    assert.equal(formatApiErrorBody(body), null);
  }

  const response = new Response("not JSON", { status: 503 });
  assert.equal(await readFastApiError(response, "HTTP"), "HTTP (503)");
});
