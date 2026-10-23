// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// An older build's record may carry no version, or a version stamp this build no longer reads. The
// one-time import must still send every batch value it holds.

import assert from "node:assert/strict";
import test from "node:test";

import {
  installLocalStorageFake,
  registerStoreStubResolver,
} from "./helpers/kit.ts";

registerStoreStubResolver();
const { store } = installLocalStorageFake();

const { importBrowserModelConfigs } = await import(
  "../src/features/model-picker/api/migrate-model-overrides.ts"
);
const { setAuthFetchHandler } = await import("./helpers/store-stubs/auth.ts");

const KEY = "unsloth_model_configs";

test("an older build's records reach the server with every batch value they carried", async () => {
  store.clear();
  store.set(
    KEY,
    JSON.stringify({
      // No version key at all, as the earliest builds wrote it.
      "unsloth/Repo-GGUF::Q4_K_M": {
        customContextLength: 8192,
        kvCacheDtype: "q8_0",
        nParallel: 4,
        tensorParallel: true,
        nBatch: 4096,
      },
      "unsloth/Other-GGUF::Q8_0": {
        version: 1,
        customContextLength: 4096,
        nParallel: 2,
        nUbatch: 512,
      },
    }),
  );
  const puts: Record<string, unknown>[] = [];
  setAuthFetchHandler((_input, init) => {
    if (init?.method === "PUT") {
      puts.push(JSON.parse(String(init.body)));
    }
    return new Response(JSON.stringify({ overrides: {} }), { status: 200 });
  });
  try {
    await importBrowserModelConfigs();
  } finally {
    setAuthFetchHandler(null);
  }

  const byId = Object.fromEntries(
    puts.map((row) => [String(row.model_id), row]),
  );
  assert.deepEqual(Object.keys(byId).sort(), [
    "unsloth/Other-GGUF:Q8_0",
    "unsloth/Repo-GGUF:Q4_K_M",
  ]);
  const unversioned = byId["unsloth/Repo-GGUF:Q4_K_M"];
  assert.equal(unversioned.custom_context_length, 8192);
  assert.equal(unversioned.kv_cache_dtype, "q8_0");
  assert.equal(unversioned.n_parallel, 4);
  assert.equal(unversioned.tensor_parallel, true);
  assert.equal(unversioned.n_batch, 4096);
  assert.equal("n_ubatch" in unversioned, false);
  const versioned = byId["unsloth/Other-GGUF:Q8_0"];
  assert.equal(versioned.custom_context_length, 4096);
  assert.equal(versioned.n_parallel, 2);
  assert.equal(versioned.n_ubatch, 512);
  assert.equal("n_batch" in versioned, false);
});

test("a status payload from a backend that omits the batch echo is a no-op", async () => {
  const { resolveBatchSizeSeed } = await import(
    "../src/features/chat/lib/resolve-batch-size-seed.ts"
  );
  // An older backend omits requested_n_batch; that is no information, not the default.
  const pinned = { value: 4096, loaded: 4096 };
  assert.deepEqual(
    resolveBatchSizeSeed({
      incoming: undefined,
      isGguf: true,
      previous: pinned,
      seedLoadParams: true,
    }),
    {},
    "an absent echo must be a no-op, never a clear",
  );
  assert.deepEqual(
    resolveBatchSizeSeed({
      incoming: undefined,
      isGguf: true,
      previous: { value: 2048, loaded: 4096 },
      seedLoadParams: true,
    }),
    {},
  );
  assert.deepEqual(
    resolveBatchSizeSeed({
      incoming: null,
      isGguf: true,
      previous: pinned,
      seedLoadParams: true,
    }),
    { loaded: null, value: null },
    "an explicit null is information and must be honoured",
  );
});
