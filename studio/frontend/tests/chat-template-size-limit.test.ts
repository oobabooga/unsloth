// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { registerStoreStubResolver } from "./helpers/kit.ts";

registerStoreStubResolver();
const {
  DEFAULT_PER_MODEL_CONFIG,
  MAX_CHAT_TEMPLATE_BYTES,
  isChatTemplateWithinLimit,
  normalizePerModelConfig,
} = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);
const { fromApiOverride, toApiOverride } = await import(
  "../src/features/model-picker/api/model-overrides.ts"
);

function templateOfBytes(bytes: number, unit: string): string {
  return unit.repeat(Math.ceil(bytes / unit.length)).slice(0, bytes);
}

/** The template a save sends to the server, as the saved row reads it back. */
function savedTemplate(template: string): string | null {
  const config = normalizePerModelConfig({
    ...DEFAULT_PER_MODEL_CONFIG,
    chatTemplateOverride: template,
  });
  return fromApiOverride(toApiOverride(config)).chatTemplateOverride;
}

test("a ~230 KB community chat template saves and reloads (#14841)", () => {
  const template = templateOfBytes(
    230 * 1024,
    "{%- if message['role'] == 'user' -%}{{ message['content'] }}{%- endif -%}\n",
  );
  assert.equal(isChatTemplateWithinLimit(template), true);
  assert.equal(savedTemplate(template), template);
});

test("a template at the cap still saves and reloads", () => {
  // The cap counts the template's own bytes, so quotes and newlines that grow in JSON do not count.
  const template = templateOfBytes(
    MAX_CHAT_TEMPLATE_BYTES,
    '{{- "<|im_start|>" + message["role"] + "\\n" -}}\n',
  );
  assert.equal(isChatTemplateWithinLimit(template), true);
  assert.equal(savedTemplate(template), template);
});

test("a template over the cap never reaches the saved row", () => {
  const template = "a".repeat(MAX_CHAT_TEMPLATE_BYTES + 1);
  assert.equal(isChatTemplateWithinLimit(template), false);
  assert.equal(
    normalizePerModelConfig({ chatTemplateOverride: template })
      .chatTemplateOverride,
    null,
  );
  assert.equal(savedTemplate(template), null);
});
