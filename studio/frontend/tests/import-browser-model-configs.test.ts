// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Older builds kept each model's settings in this browser. The first start of this build sends
// them to the server's map, the only store now, and removes them here once the server has them.

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
const {
  loadSavedModelConfigs,
  resolveInitialConfig,
  setSavedModelConfigsForTests,
} = await import(
  "../src/features/model-picker/model-config/saved-model-configs.ts"
);
const { invalidateLlamaFlagCatalog } = await import(
  "../src/features/model-picker/api/llama-flags.ts"
);
const { modelStorageKey } = await import(
  "../src/features/model-picker/model-config/model-identity.ts"
);
const { setAuthFetchHandler } = await import("./helpers/store-stubs/auth.ts");

const CONFIGS = "unsloth_model_configs";
const LEGACY = "unsloth_load_settings";
const LEGACY_MERGED = "unsloth_model_configs_migrated";
const BACKFILLED = "unsloth_model_overrides_backfilled_v3";
// Payload keys that say how to write, not what to store.
const WRITE_MODES = /^(model_id|remove|fill_absent_fields|mirrors_.*)$/;

type Row = Record<string, unknown>;

/** Records every override PUT and answers it; `refuse` names keys the server turns away. */
function fakeServer({
  refuse = [] as string[],
  managed = [] as string[],
} = {}) {
  const puts: Row[] = [];
  const order: string[] = [];
  let map: Record<string, Row> = {};
  setAuthFetchHandler((input, init) => {
    if (input.startsWith("/api/inference/llama-flags")) {
      return Response.json({ managed, max_bytes: 32768 });
    }
    const method = init?.method ?? "GET";
    order.push(method);
    if (method === "PUT") {
      const body = JSON.parse(String(init?.body)) as Row;
      puts.push(body);
      const key = String(body.model_id);
      if (refuse.includes(key)) {
        return Response.json({ detail: "refused" }, { status: 400 });
      }
      const fields = Object.fromEntries(
        Object.entries(body).filter(([name]) => !WRITE_MODES.test(name)),
      );
      map = { ...map, [key]: { ...fields, ...map[key] } };
    }
    return Response.json({ overrides: map });
  });
  return { puts, order };
}

function keep(records: Record<string, Row>) {
  store.set(CONFIGS, JSON.stringify(records));
}

test.beforeEach(() => {
  store.clear();
  setSavedModelConfigsForTests(null);
});

test("each record fills the server row, and the browser keys go once all are in", async () => {
  keep({
    [modelStorageKey("org/m-GGUF", "Q4_K_M")]: {
      version: 6,
      customContextLength: 32768,
      kvCacheDtype: "q8_0",
    },
    [modelStorageKey("org/plain", null)]: { version: 1, maxSeqLength: 8192 },
  });
  store.set(LEGACY_MERGED, "1");
  store.set(BACKFILLED, "1");
  const server = fakeServer();
  await importBrowserModelConfigs();
  assert.equal(server.puts.length, 2);
  for (const body of server.puts) {
    // Only what the server row lacks: a value it already holds is the newer one.
    assert.equal(body.fill_absent_fields, true);
    assert.equal(body.remove, false);
  }
  const gguf = server.puts.find(
    (body) => body.model_id === "org/m-gguf:q4_k_m",
  );
  assert.equal(gguf?.custom_context_length, 32768);
  assert.equal(gguf?.kv_cache_dtype, "q8_0");
  const plain = server.puts.find((body) => body.model_id === "org/plain");
  assert.equal(plain?.max_seq_length, 8192);
  for (const key of [CONFIGS, LEGACY, LEGACY_MERGED, BACKFILLED]) {
    assert.equal(store.has(key), false, key);
  }
});

test("a record the server turns away stays in this browser, and only that one", async () => {
  keep({
    [modelStorageKey("org/ok", null)]: { nParallel: 2 },
    [modelStorageKey("org/refused", null)]: { nParallel: 4 },
  });
  const server = fakeServer({ refuse: ["org/refused"] });
  await importBrowserModelConfigs();
  assert.equal(server.puts.length, 2);
  const left = JSON.parse(store.get(CONFIGS) ?? "{}");
  assert.deepEqual(Object.keys(left), [modelStorageKey("org/refused", null)]);
  assert.equal(left[modelStorageKey("org/refused", null)].nParallel, 4);
  // The next start sends only what is left.
  const again = fakeServer();
  await importBrowserModelConfigs();
  assert.deepEqual(
    again.puts.map((body) => body.model_id),
    ["org/refused"],
  );
  assert.equal(store.has(CONFIGS), false);
});

test("the store before that one is imported too, and the newer store wins a model both hold", async () => {
  keep({
    [modelStorageKey("org/m-GGUF", "Q4_K_M")]: { customContextLength: 32768 },
  });
  store.set(
    LEGACY,
    JSON.stringify({
      "Org/Legacy-GGUF::Q8_0": {
        contextLength: 8192,
        kvCacheDtype: "q4_0",
        tensorParallel: true,
      },
      "org/m-GGUF::Q4_K_M": { contextLength: 1024 },
      // Nothing but defaults: no row to send.
      "org/defaults-GGUF::Q8_0": { tensorParallel: false },
    }),
  );
  const server = fakeServer();
  await importBrowserModelConfigs();
  const byKey = Object.fromEntries(
    server.puts.map((body) => [String(body.model_id).toLowerCase(), body]),
  );
  assert.equal(byKey["org/legacy-gguf:q8_0"]?.custom_context_length, 8192);
  assert.equal(byKey["org/legacy-gguf:q8_0"]?.kv_cache_dtype, "q4_0");
  assert.equal(byKey["org/legacy-gguf:q8_0"]?.tensor_parallel, true);
  assert.equal(byKey["org/m-gguf:q4_k_m"]?.custom_context_length, 32768);
  assert.equal(server.puts.length, 2);
  assert.equal(store.has(LEGACY), false);
});

test("an unreadable legacy store is skipped, not thrown", async () => {
  store.set(LEGACY, "{not json");
  const server = fakeServer();
  await importBrowserModelConfigs();
  assert.equal(server.puts.length, 0);
  assert.equal(store.has(LEGACY), false);
});

test("a legacy store already merged into the newer one is not sent again", async () => {
  store.set(
    LEGACY,
    JSON.stringify({ "org/old::Q8_0": { contextLength: 8192 } }),
  );
  store.set(LEGACY_MERGED, "1");
  const server = fakeServer();
  await importBrowserModelConfigs();
  assert.equal(server.puts.length, 0);
  assert.equal(store.has(LEGACY), false);
});

test("a cached repo's snapshot-path record is sent under the repo id it is saved by now", async () => {
  const snapshot =
    "/home/u/.cache/huggingface/hub/models--org--m/snapshots/abc123";
  keep({ [modelStorageKey(snapshot, null)]: { maxSeqLength: 16384 } });
  const server = fakeServer();
  await importBrowserModelConfigs();
  assert.deepEqual(
    server.puts.map((body) => body.model_id),
    ["org/m"],
  );
});

test("a standalone .gguf record is sent under its bare path, with no quant", async () => {
  const path = "/home/u/models/Foo-Q4_K_M.gguf";
  keep({ [modelStorageKey(path, null)]: { customContextLength: 4096 } });
  const server = fakeServer();
  await importBrowserModelConfigs();
  assert.deepEqual(
    server.puts.map((body) => body.model_id),
    [path],
  );
});

test("a record at its defaults sends nothing", async () => {
  keep({ [modelStorageKey("org/m", null)]: { version: 1 } });
  const server = fakeServer();
  await importBrowserModelConfigs();
  assert.equal(server.puts.length, 0);
  assert.equal(store.has(CONFIGS), false);
});

test("a launch flag this build refuses is dropped before the record goes out", async () => {
  invalidateLlamaFlagCatalog();
  keep({
    [modelStorageKey("org/m-GGUF", "Q4_K_M")]: {
      llamaExtraArgs: ["--model", "/x.gguf", "--no-warmup"],
    },
  });
  const server = fakeServer({ managed: ["--model"] });
  await importBrowserModelConfigs();
  assert.deepEqual(server.puts[0]?.llama_extra_args, ["--no-warmup"]);
});

test("storage that refuses to be read is survived", async () => {
  const original = localStorage.getItem;
  localStorage.getItem = () => {
    throw new Error("denied");
  };
  try {
    await importBrowserModelConfigs();
  } finally {
    localStorage.getItem = original;
  }
});

test("a start imports first, then reads the map every reader uses", async () => {
  keep({ [modelStorageKey("org/m", null)]: { nParallel: 3 } });
  const server = fakeServer();
  await loadSavedModelConfigs();
  assert.deepEqual(server.order, ["PUT", "GET"]);
  const saved = resolveInitialConfig("org/m");
  assert.equal(saved.remembered, true);
  assert.equal(saved.config.nParallel, 3);
});
