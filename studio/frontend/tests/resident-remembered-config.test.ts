// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The settings for the identifier /api/inference/status reports, read from the server's map in the
// order a load reads it: the load path before its public id, a quant's row before the bare one.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import type { PerModelConfig } from "../src/features/model-picker/model-config/per-model-config.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  resolveInitialConfig,
  resolveResidentInitialConfig,
  setSavedModelConfigsForTests,
} = await import(
  "../src/features/model-picker/model-config/saved-model-configs.ts"
);
const { toApiOverride } = await import(
  "../src/features/model-picker/api/model-overrides.ts"
);
const { DEFAULT_PER_MODEL_CONFIG } = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);

/** A row as a save from Run settings writes it. */
function row(config: Partial<PerModelConfig>) {
  return toApiOverride({ ...DEFAULT_PER_MODEL_CONFIG, ...config });
}

// model_identifier after an API auto-switch is the snapshot path; settings are keyed by repo id.
const SNAPSHOT_PATH =
  "/home/u/.cache/huggingface/hub/models--unsloth--Repo-GGUF/snapshots/2f1c9ab";
const REPO_ID = "unsloth/Repo-GGUF";

test("the resident model's repo-keyed slots are found through its snapshot path", () => {
  setSavedModelConfigsForTests({
    [`${REPO_ID}:Q4_K_M`]: row({ nParallel: 4 }),
  });

  assert.equal(resolveInitialConfig(SNAPSHOT_PATH, "Q4_K_M").remembered, false);

  const resolved = resolveResidentInitialConfig(SNAPSHOT_PATH, "Q4_K_M");
  assert.equal(resolved.remembered, true);
  // Without this the control blanks and the next save writes the blank over the saved record.
  assert.equal(resolved.config.nParallel, 4);
});

test("a record under the raw identifier still wins over the repo alias", () => {
  setSavedModelConfigsForTests({
    [`${SNAPSHOT_PATH}:Q4_K_M`]: row({ nParallel: 2 }),
    [`${REPO_ID}:Q4_K_M`]: row({ nParallel: 8 }),
  });

  assert.equal(
    resolveResidentInitialConfig(SNAPSHOT_PATH, "Q4_K_M").config.nParallel,
    2,
  );
});

test("a cached repo loaded without a quant reads its repo-id row through the snapshot path", () => {
  const snapshot = "/hf/models--mlx-community--Model-4bit/snapshots/abc";
  const repo = "mlx-community/Model-4bit";
  setSavedModelConfigsForTests({ [repo]: row({ nParallel: 8 }) });

  assert.equal(
    resolveResidentInitialConfig(snapshot, null).config.nParallel,
    8,
  );
  assert.equal(resolveInitialConfig(snapshot, null).remembered, false);
});

test("the quant still separates two variants of one cached repo", () => {
  setSavedModelConfigsForTests({
    [`${REPO_ID}:Q4_K_M`]: row({ nParallel: 4 }),
  });

  assert.equal(
    resolveResidentInitialConfig(SNAPSHOT_PATH, "Q8_0").remembered,
    false,
  );
});

for (const modelId of [
  "/home/u/.lmstudio/models/unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf",
  String.raw`C:\Users\u\.lmstudio\models\unsloth\Qwen3-0.6B-GGUF\Qwen3-0.6B-Q4_K_M.gguf`,
  "Qwen3-0.6B-Q4_K_M.gguf",
]) {
  test(`a resident standalone GGUF restores the settings saved by the picker: ${modelId}`, () => {
    setSavedModelConfigsForTests({
      [modelId]: row({ nParallel: 2, customContextLength: 4096 }),
    });

    const resolved = resolveResidentInitialConfig(modelId, "Q4_K_M");
    assert.equal(resolved.remembered, true);
    assert.equal(resolved.config.customContextLength, 4096);
    assert.equal(resolved.config.nParallel, 2);
  });
}

for (const modelId of [
  "/home/u/.lmstudio/models/unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf",
  "Qwen3-0.6B-Q4_K_M.gguf",
]) {
  test(`a record the picker keyed by the quant label before #7473 is still read: ${modelId}`, () => {
    setSavedModelConfigsForTests({
      [`${modelId}:Q4_K_M`]: row({ nParallel: 5 }),
    });

    const resolved = resolveResidentInitialConfig(modelId, "Q4_K_M");
    assert.equal(resolved.remembered, true);
    assert.equal(resolved.config.nParallel, 5);

    // The bare path is what the picker saves today, and a load reads it first.
    setSavedModelConfigsForTests({
      [`${modelId}:Q4_K_M`]: row({ nParallel: 5 }),
      [modelId]: row({ nParallel: 6 }),
    });
    assert.equal(
      resolveResidentInitialConfig(modelId, "Q4_K_M").config.nParallel,
      6,
    );
  });
}

for (const modelId of [
  "/home/u/.lmstudio/models/unsloth/Qwen3-0.6B-GGUF",
  "org/model.gguf",
]) {
  test(`a model containing several GGUF variants keeps their settings separate: ${modelId}`, () => {
    setSavedModelConfigsForTests({
      [modelId]: row({ nParallel: 8 }),
      [`${modelId}:Q4_K_M`]: row({ nParallel: 2 }),
      [`${modelId}:Q8_0`]: row({ nParallel: 4 }),
    });

    assert.equal(
      resolveResidentInitialConfig(modelId, "Q4_K_M").config.nParallel,
      2,
    );
    assert.equal(
      resolveResidentInitialConfig(modelId, "Q8_0").config.nParallel,
      4,
    );
    // A quant with no row of its own loads with the bare row, so it reads that row too.
    assert.equal(
      resolveResidentInitialConfig(modelId, "Q6_K").config.nParallel,
      8,
    );
  });
}

test("a stem two models can share is never read as an alias", () => {
  // "Repo-Q4_K_M" is the stem both files collapse onto, so adopting it would cross settings.
  setSavedModelConfigsForTests({ "Repo-Q4_K_M": row({ nParallel: 4 }) });

  assert.equal(
    resolveResidentInitialConfig("/srv/models/a/Repo-Q4_K_M.gguf", null)
      .remembered,
    false,
  );
  assert.equal(
    resolveResidentInitialConfig("/srv/models/b/Repo-Q4_K_M.gguf", null)
      .remembered,
    false,
  );
});

test("a repo id resolves exactly as before", () => {
  setSavedModelConfigsForTests({
    [`${REPO_ID}:Q4_K_M`]: row({ nParallel: 4 }),
  });

  assert.equal(
    resolveResidentInitialConfig(REPO_ID, "Q4_K_M").config.nParallel,
    4,
  );
  assert.equal(
    resolveResidentInitialConfig("unsloth/Other-GGUF", "Q4_K_M").remembered,
    false,
  );
});

const composer = readFileSync(
  new URL("../src/features/chat/shared-composer.tsx", import.meta.url),
  "utf8",
);
const ensureModelLoaded = composer.slice(
  composer.indexOf("async function ensureModelLoaded("),
  composer.indexOf("if (handle1 && model1?.id)"),
);

test("compare ensureModelLoaded recovers a repo-keyed pin through a snapshot path", () => {
  setSavedModelConfigsForTests({
    [`${REPO_ID}:Q4_K_M`]: row({ customContextLength: 8192 }),
  });
  assert.equal(resolveInitialConfig(SNAPSHOT_PATH, "Q4_K_M").remembered, false);
  assert.equal(
    resolveResidentInitialConfig(SNAPSHOT_PATH, "Q4_K_M").config
      .customContextLength,
    8192,
  );
  assert.match(
    ensureModelLoaded,
    /resolveResidentInitialConfig\(\s*sel\.id,\s*sel\.ggufVariant \?\? null,?\s*\)/,
  );
  assert.doesNotMatch(ensureModelLoaded, /resolveInitialConfig\(\s*sel\.id,/);
});
