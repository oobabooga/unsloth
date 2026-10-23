// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The API page lists the copy of the saved settings the picker reads, and forgets an entry by its
// stored key, so the server clears every spelling of that model and both views repaint from its answer.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerStoreStubResolver } from "./helpers/kit.ts";

registerStoreStubResolver();

const {
  forgetModelConfig,
  readSavedModelConfigs,
  setSavedModelConfigsForTests,
} = await import(
  "../src/features/model-picker/model-config/saved-model-configs.ts"
);
const { setAuthFetchHandler } = await import("./helpers/store-stubs/auth.ts");

const SOURCE = readSrc(
  "features/api-monitor/components/saved-model-settings.tsx",
);
const READS_SHARED_COPY =
  /useSyncExternalStore\(\s*subscribeSavedModelConfigs,\s*readSavedModelConfigs,/;
const FORGETS_BY_STORED_KEY = /await forgetModelConfig\(overrideKey\);/;
const REPORTS_REFUSAL = /catch \(err: unknown\) \{\s*toast\.error\(/;
const OWN_FETCH_OR_SEQUENCE = /fetchModelOverrides|putModelOverride|loadSeq/;
const REFUSAL_REASON = /Invalid model launch override\./;

function answer(
  status: number,
  body: unknown,
  sent?: Record<string, unknown>[],
): void {
  setAuthFetchHandler((_input, init) => {
    sent?.push(JSON.parse(String(init?.body)));
    return new Response(JSON.stringify(body), {
      status,
      headers: { "content-type": "application/json" },
    });
  });
}

test("the list reads the copy the picker reads, and forgets through it", () => {
  assert.match(SOURCE, READS_SHARED_COPY);
  assert.match(SOURCE, FORGETS_BY_STORED_KEY);
  // A refused forget is reported rather than swallowed.
  assert.match(SOURCE, REPORTS_REFUSAL);
  assert.doesNotMatch(SOURCE, OWN_FETCH_OR_SEQUENCE);
});

// A quant can name a directory or a filename stem, and a colon is legal inside a local path, so
// the key is never split: the server resolves the model it names.
for (const key of [
  "unsloth/Repo-GGUF:distilled/model-Q6_K",
  "unsloth/H3-GGUF:minimax_h3_ref2va_pruned-Q6_K",
  "/home/u/models/foo:bar/baz.gguf",
  "unsloth/Repo-GGUF",
]) {
  test(`a forget sends the stored key whole: ${key}`, async () => {
    setSavedModelConfigsForTests({
      [key]: { engine: "vllm" },
      "other/model": { engine: "sglang" },
    });
    const sent: Record<string, unknown>[] = [];
    answer(200, { overrides: { "other/model": { engine: "sglang" } } }, sent);
    try {
      await forgetModelConfig(key);
      assert.equal(sent.length, 1);
      assert.equal(sent[0].model_id, key);
      assert.equal(sent[0].remove, true);
      assert.deepEqual(sent[0].llama_extra_args, []);
      assert.deepEqual(readSavedModelConfigs(), {
        "other/model": { engine: "sglang" },
      });
    } finally {
      setAuthFetchHandler(null);
      setSavedModelConfigsForTests(null);
    }
  });
}

test("a refused forget leaves the list as it was", async () => {
  const saved = { "org/model": { engine: "vllm" as const } };
  setSavedModelConfigsForTests(saved);
  answer(400, { detail: "Invalid model launch override." });
  try {
    await assert.rejects(forgetModelConfig("org/model"), REFUSAL_REASON);
    assert.equal(readSavedModelConfigs(), saved);
  } finally {
    setAuthFetchHandler(null);
    setSavedModelConfigsForTests(null);
  }
});
