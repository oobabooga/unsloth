// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { registerStoreStubResolver } from "./helpers/kit.ts";

registerStoreStubResolver();
const { DEFAULT_PER_MODEL_CONFIG, normalizePerModelConfig } = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);
const { fromApiOverride, toApiOverride } = await import(
  "../src/features/model-picker/api/model-overrides.ts"
);
const { resolveInitialConfig, setSavedModelConfigsForTests } = await import(
  "../src/features/model-picker/model-config/saved-model-configs.ts"
);

const MODEL = "unsloth/split-GGUF";
const VARIANT = "Q4_K_M";
const configured = () => ({
  ...DEFAULT_PER_MODEL_CONFIG,
  gpuMemoryMode: "manual" as const,
  gpuLayers: 66,
  selectedGpuIds: [1, 2, 0],
  selectedGpuIndexKind: "physical" as const,
  tensorSplit: [30, 20, 16],
});

test("saving and reopening keeps the split in GPU order", () => {
  setSavedModelConfigsForTests({
    [`${MODEL}:${VARIANT}`]: toApiOverride(configured()),
  });
  const restored = resolveInitialConfig(MODEL, VARIANT);
  assert.equal(restored.remembered, true);
  assert.deepEqual(restored.config.selectedGpuIds, [1, 2, 0]);
  assert.deepEqual(restored.config.tensorSplit, [30, 20, 16]);
  assert.equal(restored.config.gpuLayers, 66);
});

test("the saved row carries the ratio in GPU order", () => {
  const payload = toApiOverride(configured());
  assert.deepEqual(payload.gpu_ids, [1, 2, 0]);
  assert.deepEqual(
    (payload as Record<string, unknown>).tensor_split,
    [30, 20, 16],
  );
  assert.deepEqual(fromApiOverride(payload).tensorSplit, [30, 20, 16]);
});

test("a row's ratio is read only with the GPU order it was saved with", () => {
  const unsplit = fromApiOverride({ gpu_ids: [0, 1, 2] });
  assert.deepEqual(unsplit.selectedGpuIds, [0, 1, 2]);
  assert.equal(unsplit.tensorSplit, null);
  assert.equal(
    fromApiOverride({ gpu_ids: [1, 2, 0], tensor_split: null }).tensorSplit,
    null,
  );
  // A ratio for another number of GPUs describes some other pick.
  assert.equal(
    fromApiOverride({ gpu_ids: [1, 2], tensor_split: [30, 20, 16] })
      .tensorSplit,
    null,
  );
});

test("a ratio without its GPU IDs is not a split", () => {
  assert.equal(fromApiOverride({ tensor_split: [3, 2, 1] }).tensorSplit, null);
  assert.equal(fromApiOverride({ tensor_split: null }).tensorSplit, null);
});

test("default and malformed ratios are not persisted", () => {
  for (const tensorSplit of [
    null,
    [],
    [0, 0, 0],
    [-1, 20, 47],
    [NaN, 20, 16],
    [Infinity, 20, 16],
    [30, 36],
    [Number.MAX_VALUE, Number.MAX_VALUE, 1],
  ]) {
    assert.equal(
      normalizePerModelConfig({ ...configured(), tensorSplit }).tensorSplit ??
        null,
      null,
    );
  }
  for (const selectedGpuIds of [
    null,
    undefined,
    [1, 1, 0],
    [1.5, 2, 0],
    [-1, 2, 0],
  ]) {
    assert.equal(
      normalizePerModelConfig({ ...configured(), selectedGpuIds })
        .tensorSplit ?? null,
      null,
    );
  }
});

test("zero shares and fractional ratios survive the saved row", () => {
  const value = { ...configured(), tensorSplit: [0, 2.5, 1] };
  assert.deepEqual(
    fromApiOverride(toApiOverride(value)).tensorSplit,
    value.tensorSplit,
  );
});

test("clearing the ratio sends an explicit clear", () => {
  // An omitted split leaves the saved one in place on the server; null clears it.
  const cleared = toApiOverride({ ...configured(), tensorSplit: null });
  assert.equal("tensor_split" in cleared, true);
  assert.equal(cleared.tensor_split, null);
  assert.equal(fromApiOverride(cleared).tensorSplit ?? null, null);
});

test("a removed GPU, changed order, or backend namespace discards the saved split", async () => {
  const { reconcileTensorSplit } = await import(
    "../src/hooks/gpu-tensor-split.ts"
  );
  const { reconcileGpuSelection } = await import(
    "../src/hooks/gpu-selection.ts"
  );
  const value = configured();
  const savedIds = value.selectedGpuIds;
  assert.deepEqual(
    reconcileTensorSplit(value.tensorSplit, savedIds, savedIds),
    value.tensorSplit,
  );
  for (const current of [null, [1, 2], [0, 1, 2]]) {
    assert.equal(
      reconcileTensorSplit(value.tensorSplit, savedIds, current),
      null,
    );
  }
  const otherBackend = reconcileGpuSelection(
    savedIds,
    "physical",
    "vulkan",
    savedIds,
  );
  assert.equal(
    reconcileTensorSplit(value.tensorSplit, savedIds, otherBackend.ids),
    null,
  );
});

test("an unpinned split survives a reload that stays unpinned", async () => {
  const { reconcileTensorSplit } = await import(
    "../src/hooks/gpu-tensor-split.ts"
  );
  assert.deepEqual(reconcileTensorSplit([3, 1], null, null), [3, 1]);
  assert.deepEqual(reconcileTensorSplit([3, 1], undefined, null), [3, 1]);
  assert.equal(reconcileTensorSplit([3, 1], null, [0, 1]), null);
  assert.equal(reconcileTensorSplit([0, 0], null, null), null);
  assert.equal(reconcileTensorSplit([3], null, null), null);
});
