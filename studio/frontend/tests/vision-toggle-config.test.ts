// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A model's saved settings live in the server's map, so the toggle has to survive the trip there and
// back, and a record or row written before the toggle existed reads as vision on.

import assert from "node:assert/strict";
import test from "node:test";

import type { PerModelConfig } from "../src/features/model-picker/model-config/per-model-config.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { DEFAULT_PER_MODEL_CONFIG, normalizePerModelConfig } = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);
const { fromApiOverride, toApiOverride } = await import(
  "../src/features/model-picker/api/model-overrides.ts"
);
const { loadedConfigSignature } = await import(
  "../src/features/model-picker/model-config/config-signature.ts"
);

function config(overrides: Partial<PerModelConfig> = {}): PerModelConfig {
  return { ...DEFAULT_PER_MODEL_CONFIG, ...overrides };
}

test("vision is on by default", () => {
  assert.equal(DEFAULT_PER_MODEL_CONFIG.disableVision, false);
});

test("a blob written before the toggle existed still loads", () => {
  // Genuinely absent, as an older build's JSON has it.
  const { disableVision: _omitted, ...legacy } = DEFAULT_PER_MODEL_CONFIG;
  assert.equal(
    normalizePerModelConfig(legacy as Record<string, unknown>).disableVision,
    false,
  );
  for (const bad of ["true", 1, null, {}]) {
    assert.equal(
      normalizePerModelConfig({
        ...DEFAULT_PER_MODEL_CONFIG,
        disableVision: bad,
      }).disableVision,
      false,
      `for ${JSON.stringify(bad)}`,
    );
  }
});

test("the toggle round-trips through the saved row", () => {
  const row = toApiOverride(config({ disableVision: true }));
  assert.equal(row.disable_vision, true);
  assert.equal(fromApiOverride(row).disableVision, true);
});

test("vision on stays off the saved row", () => {
  // Stored only when off, so a model that never touched the switch follows the default.
  assert.equal("disable_vision" in toApiOverride(config()), false);
  assert.equal(fromApiOverride({}).disableVision, false);
});

test("flipping the toggle changes the signature, so a reload is not deduped away", () => {
  assert.notEqual(
    loadedConfigSignature(config({ disableVision: false })),
    loadedConfigSignature(config({ disableVision: true })),
  );
});
