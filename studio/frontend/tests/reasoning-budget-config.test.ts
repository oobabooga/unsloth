// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  DEFAULT_PER_MODEL_CONFIG,
  isDefaultConfig,
  isReasoningBudgetMessageValid,
  normalizePerModelConfig,
} = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);
const { fromApiOverride, toApiOverride } = await import(
  "../src/features/model-picker/api/model-overrides.ts"
);

test("reasoning budget fields normalize to the backend defaults", () => {
  const normalized = normalizePerModelConfig({});
  assert.equal(normalized.reasoningBudget, DEFAULT_PER_MODEL_CONFIG.reasoningBudget);
  assert.equal(
    normalized.reasoningBudgetMessage,
    DEFAULT_PER_MODEL_CONFIG.reasoningBudgetMessage,
  );
  assert.equal(
    normalizePerModelConfig({ reasoningBudget: -10 }).reasoningBudget,
    -1,
  );
  assert.equal(
    normalizePerModelConfig({ reasoningBudget: 12.9 }).reasoningBudget,
    12,
  );
  assert.equal(isDefaultConfig(DEFAULT_PER_MODEL_CONFIG), true);
  assert.equal(
    isDefaultConfig({ ...DEFAULT_PER_MODEL_CONFIG, reasoningBudget: 0 }),
    false,
  );
});

test("reasoning budget messages use the subprocess byte boundary", () => {
  assert.equal(isReasoningBudgetMessageValid("😀".repeat(2_048)), true);
  assert.equal(isReasoningBudgetMessageValid("😀".repeat(2_049)), false);
  assert.equal(isReasoningBudgetMessageValid("bad\0message"), false);
  assert.equal(isReasoningBudgetMessageValid("  PAD  "), true);
  assert.equal(
    normalizePerModelConfig({ reasoningBudgetMessage: "😀".repeat(2_049) })
      .reasoningBudgetMessage,
    "",
  );
  assert.equal(
    normalizePerModelConfig({ reasoningBudgetMessage: "  PAD  " })
      .reasoningBudgetMessage,
    "  PAD  ",
  );
});

test("reasoning budget and message round-trip through the saved row", () => {
  const row = toApiOverride({
    ...DEFAULT_PER_MODEL_CONFIG,
    reasoningBudget: 2048,
    reasoningBudgetMessage: "Reasoning budget exhausted",
  });
  assert.equal(row.reasoning_budget, 2048);
  assert.equal(row.reasoning_budget_message, "Reasoning budget exhausted");

  const read = fromApiOverride(row);
  assert.equal(read.reasoningBudget, 2048);
  assert.equal(read.reasoningBudgetMessage, "Reasoning budget exhausted");
});

test("an unlimited budget and an empty message stay off the saved row", () => {
  const row = toApiOverride(DEFAULT_PER_MODEL_CONFIG);
  assert.equal("reasoning_budget" in row, false);
  assert.equal("reasoning_budget_message" in row, false);

  const read = fromApiOverride(row);
  assert.equal(read.reasoningBudget, -1);
  assert.equal(read.reasoningBudgetMessage, "");
});
