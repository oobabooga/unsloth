// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { buildCachedInventoryRow } = await import(
  "../src/features/hub/inventory/view-models.ts"
);

const companionPayload = {
  repo_id: "unsloth/FLUX.2-VAE",
  size_bytes: 300_000_000,
  model_format: "unknown" as const,
  companion: true,
  required_by: ["unsloth/FLUX.2-klein-4B-GGUF", "unsloth/Qwen-Image-GGUF"],
};

test("a required-asset cache row is complete, flagged, and carries its dependents", () => {
  const row = buildCachedInventoryRow(companionPayload, "safetensors");

  // The whole point of the fix: the row is no longer marked partial, so it gets a real On Device
  // row with a delete action instead of a Resume/Redownload affordance for files that all arrived.
  assert.equal(row.partial, false);
  // `companion`, not `partial`, is now what keeps it out of the model pickers.
  assert.equal(row.companion, true);
  assert.deepEqual(row.requiredBy, companionPayload.required_by);
});

test("an orphaned required-asset cache reports no dependents", () => {
  const row = buildCachedInventoryRow(
    { ...companionPayload, required_by: [] },
    "safetensors",
  );

  assert.equal(row.companion, true);
  assert.deepEqual(row.requiredBy, []);
});

test("a backend that does not send required_by leaves the row deletable", () => {
  // Older backends omit the field entirely. Defaulting it to undefined would make
  // `requiredBy.length` throw in the row renderer; defaulting it to a non-empty value would
  // permanently disable Delete for every cached repo.
  const withoutField: Record<string, unknown> = { ...companionPayload };
  delete withoutField.required_by;
  const row = buildCachedInventoryRow(
    withoutField as typeof companionPayload,
    "safetensors",
  );

  assert.deepEqual(row.requiredBy, []);
});

test("an ordinary cached model is neither a required asset nor held by one", () => {
  const row = buildCachedInventoryRow(
    { repo_id: "unsloth/gemma-3-4b-it", size_bytes: 8_000_000_000 },
    "safetensors",
  );

  assert.equal(row.companion, false);
  assert.deepEqual(row.requiredBy, []);
});
