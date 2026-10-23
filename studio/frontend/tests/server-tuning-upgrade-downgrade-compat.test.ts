// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// No user setting may be lost on upgrade: an older build's record reaches the server whole through
// the one-time import, and a saved row reads back as the settings it holds.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import type { PerModelConfig } from "../src/features/model-picker/model-config/per-model-config.ts";
import type { StorageFake } from "./helpers/kit.ts";
import {
  installLocalStorageFake,
  registerStoreStubResolver,
} from "./helpers/kit.ts";

registerStoreStubResolver();
const { store, storage } = installLocalStorageFake();

const { DEFAULT_PER_MODEL_CONFIG, isDefaultConfig, normalizePerModelConfig } =
  await import("../src/features/model-picker/model-config/per-model-config.ts");
const { modelStorageKey, normalizeModelIdentity } = await import(
  "../src/features/model-picker/model-config/model-identity.ts"
);
const { fromApiOverride, resolveStoredOverride, toApiOverride } = await import(
  "../src/features/model-picker/api/model-overrides.ts"
);
const { importBrowserModelConfigs } = await import(
  "../src/features/model-picker/api/migrate-model-overrides.ts"
);
const { setAuthFetchHandler } = await import("./helpers/store-stubs/auth.ts");

const STORAGE_KEY = "unsloth_model_configs";
const MODEL = "unsloth/Repo-GGUF";
const VARIANT = "Q4_K_M";

// Falsy values are deliberate: 0 checkpoints and 0/-1 cache are real decisions.
const TUNING_ONLY_PATCHES: Partial<PerModelConfig>[] = [
  { loadMode: "mmap" },
  { ctxCheckpoints: 0 },
  { ctxCheckpoints: 64 },
  { cacheRam: 0 },
  { cacheRam: -1 },
  // The dtype requires a separate-drafter mode, so the mode travels with it.
  { specDraftCacheDtype: "q8_0", speculativeType: "dspark" },
];

function config(overrides: Partial<PerModelConfig> = {}): PerModelConfig {
  return { ...DEFAULT_PER_MODEL_CONFIG, ...overrides };
}

/** Runs the one-time import and returns the bodies of the writes it sent. */
async function importedRows(): Promise<Record<string, unknown>[]> {
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
  return puts;
}

test("an older build's record reaches the server with the tuning it carried", async () => {
  store.clear();
  // No version key and the `id::variant` key, as the earliest builds wrote them.
  store.set(
    STORAGE_KEY,
    JSON.stringify({
      [`${MODEL}::${VARIANT}`]: {
        loadMode: "mmap",
        ctxCheckpoints: 0,
        cacheRam: -1,
        speculativeType: "dspark",
        specDraftCacheDtype: "q8_0",
      },
    }),
  );

  const rows = await importedRows();

  assert.equal(rows.length, 1);
  const [row] = rows;
  assert.equal(row.model_id, `${MODEL}:${VARIANT}`);
  assert.equal(row.load_mode, "mmap");
  assert.equal(row.ctx_checkpoints, 0);
  assert.equal(row.cache_ram, -1);
  assert.equal(row.spec_draft_cache_type, "q8_0");
  // Fill, never replace: a value already on the server is the newer one.
  assert.equal(row.fill_absent_fields, true);
  assert.equal(row.remove, false);
  assert.equal(
    store.has(STORAGE_KEY),
    false,
    "an imported record leaves this browser",
  );
});

test("the import sends every record, including identities only the picker can name", async () => {
  // An auto-switch never loads a materialized Ollama link or a bare file label, but the picker reads
  // their settings from the same map, so they go there too.
  store.clear();
  const template = { chatTemplateOverride: "{{ messages }}" };
  const records: [string, Partial<PerModelConfig>][] = [
    ["/Users/u/.lmstudio/models/mlx-community/Qwen3.5-4B-MLX-4bit", template],
    ["/home/u/.ollama/.studio_links/ab12/model-latest.gguf", template],
    ["Model-Q4_K_M.gguf", template],
    [
      "ollama-manifest:%2Fh%2F.ollama%2Fmanifests%2Fllama3%2Flatest",
      { cacheRam: -1 },
    ],
  ];
  store.set(
    STORAGE_KEY,
    JSON.stringify(
      Object.fromEntries(
        records.map(([id, patch]) => [
          modelStorageKey(id, null),
          config(patch),
        ]),
      ),
    ),
  );

  const rows = await importedRows();

  assert.deepEqual(
    new Set(rows.map((row) => row.model_id)),
    new Set(records.map(([id]) => normalizeModelIdentity(id))),
  );
  assert.equal(store.has(STORAGE_KEY), false);
});

test("a row that carries no tuning reads it as unset, not as a pinned default", () => {
  const read = fromApiOverride({
    // biome-ignore lint/style/useNamingConvention: API schema
    custom_context_length: 4096,
    // biome-ignore lint/style/useNamingConvention: API schema
    disable_vision: true,
  });
  assert.equal(read.customContextLength, 4096);
  assert.equal(read.disableVision, true);
  assert.equal(read.loadMode, null);
  assert.equal(read.ctxCheckpoints, null);
  assert.equal(read.cacheRam, null);
});

test("a tuning-only config is a setting, and a save sends it", () => {
  for (const patch of TUNING_ONLY_PATCHES) {
    const normalized = normalizePerModelConfig(config(patch));
    assert.equal(isDefaultConfig(normalized), false, JSON.stringify(patch));
    const read = fromApiOverride(toApiOverride(normalized));
    for (const key of Object.keys(patch) as (keyof PerModelConfig)[]) {
      assert.deepEqual(
        read[key],
        normalized[key],
        `${key} for ${JSON.stringify(patch)}`,
      );
    }
  }
});

test("a row's tuning reads back, 0 included", () => {
  const read = fromApiOverride({
    // biome-ignore lint/style/useNamingConvention: API schema
    load_mode: "mlock",
    // biome-ignore lint/style/useNamingConvention: API schema
    cache_ram: 0,
    // biome-ignore lint/style/useNamingConvention: API schema
    ctx_checkpoints: 0,
  });

  assert.equal(read.loadMode, "mlock");
  // 0 is a value (cache off, no checkpoints), not an absence.
  assert.equal(read.cacheRam, 0);
  assert.equal(read.ctxCheckpoints, 0);
});

test("a server value this build refuses reads as the app default", () => {
  const ngram = fromApiOverride({
    // biome-ignore lint/style/useNamingConvention: API schema
    speculative_type: "ngram",
    // biome-ignore lint/style/useNamingConvention: API schema
    spec_draft_cache_type: "q8_0",
  });
  assert.equal(ngram.speculativeType, "ngram");
  assert.equal(
    ngram.specDraftCacheDtype,
    null,
    "the dtype belongs to a draft context this mode never creates",
  );
  // biome-ignore lint/style/useNamingConvention: API schema
  assert.equal(fromApiOverride({ load_mode: "swap" }).loadMode, null);
});

test("an out-of-range server value clamps rather than being dropped", () => {
  // biome-ignore lint/style/useNamingConvention: API schema
  const read = fromApiOverride({ ctx_checkpoints: 1_000_000 });
  assert.equal(read.ctxCheckpoints, 256);
});

test("an empty argument list reads as a cleared box, not as nothing stored", () => {
  // [] is the route's tombstone blocking the bare-repository fallback; normalize alone collapses it.
  // biome-ignore lint/style/useNamingConvention: API schema
  const cleared = fromApiOverride({ llama_extra_args: [] });
  assert.deepEqual(cleared.llamaExtraArgs, []);
  assert.equal(isDefaultConfig(cleared), true);
  assert.deepEqual(toApiOverride(cleared).llama_extra_args, []);
  // A row that never carried the field leaves it omitted, so a save keeps the server's flags.
  assert.equal(fromApiOverride({}).llamaExtraArgs, undefined);
  assert.equal("llama_extra_args" in toApiOverride(fromApiOverride({})), false);
  assert.deepEqual(
    // biome-ignore lint/style/useNamingConvention: API schema
    fromApiOverride({ llama_extra_args: ["--numa", "distribute"] })
      .llamaExtraArgs,
    ["--numa", "distribute"],
  );
});

test("a GPU pin travels with its index space", () => {
  const pinned = config({
    selectedGpuIds: [1],
    selectedGpuIndexKind: "vulkan",
  });
  // The pin travels with its namespace so a Vulkan ordinal is never read as a physical index.
  const sent = toApiOverride(pinned);
  assert.deepEqual(sent.gpu_ids, [1]);
  assert.equal(sent.gpu_index_kind, "vulkan");
  assert.equal(
    toApiOverride({ ...pinned, selectedGpuIndexKind: "physical" })
      .gpu_index_kind,
    undefined,
  );
  const read = fromApiOverride(sent);
  assert.deepEqual(read.selectedGpuIds, [1]);
  assert.equal(read.selectedGpuIndexKind, "vulkan");
  // A row without ids pins nothing.
  // biome-ignore lint/style/useNamingConvention: API schema
  const unpinned = fromApiOverride({ gpu_ids: [] });
  assert.equal(unpinned.selectedGpuIds, null);
  assert.equal(unpinned.selectedGpuIndexKind, null);
});

// Mirrors tests/test_model_override_schema_compatibility.py::OVERRIDE_KEY_FOLDS.
const OVERRIDE_KEY_FOLDS: [string, string, boolean][] = [
  ["C:\\models\\Foo.gguf", "c:/models/foo.gguf", true],
  ["C:\\models\\Foo.gguf", "C:\\models\\Foo.gguf\\", true],
  ["//share/models/Foo.gguf", "\\\\SHARE\\models\\foo.gguf", true],
  ["/mnt/c/models/Foo.gguf", "/mnt/C/models/foo.gguf", true],
  ["/models/Foo.gguf", "/models/foo.gguf", false],
  ["unsloth/Repo-GGUF", "UNSLOTH/repo-gguf", true],
  ["unsloth/Repo-GGUF:Q4_K_M", "unsloth/repo-gguf:q4_k_m", true],
  ["/models/foo.gguf", "models/foo.gguf", false],
  ["models/foo.gguf", "/models/foo.gguf", false],
];

for (const [storedKey, lookupKey, sameModel] of OVERRIDE_KEY_FOLDS) {
  const reach = sameModel ? "is reached from" : "is not reached from";
  test(`${JSON.stringify(storedKey)} ${reach} ${JSON.stringify(lookupKey)}`, () => {
    // biome-ignore lint/style/useNamingConvention: API schema
    const row = { cache_ram: -1 };
    assert.equal(
      resolveStoredOverride({ [storedKey]: row }, [lookupKey]),
      sameModel ? row : null,
    );
  });
}

test("the MLX drafter is saved to the server and read back", () => {
  const drafter = normalizePerModelConfig({
    speculativeType: "eagle3",
    specDraftModel: "o/d",
  });
  assert.equal(toApiOverride(drafter).spec_draft_model, "o/d");
  // biome-ignore lint/style/useNamingConvention: API schema
  const row = { speculative_type: "auto", spec_draft_model: "o/d" };
  assert.equal(fromApiOverride(row).specDraftModel, "o/d");
  // Without the flag the backend keeps a drafter the save cleared.
  const overrides = path.join(
    path.dirname(fileURLToPath(import.meta.url)),
    "../src/features/model-picker/api/model-overrides.ts",
  );
  assert.match(
    readFileSync(overrides, "utf8"),
    /mirrors_spec_draft_model: true/,
  );
});

// Last, because it replaces the storage fake.

test("a localStorage that throws leaves the import with nothing to send", async () => {
  const throwing: StorageFake = {
    getItem: () => {
      throw new Error("SecurityError");
    },
    setItem: () => {
      const error = new Error("QuotaExceededError");
      error.name = "QuotaExceededError";
      throw error;
    },
    removeItem: () => undefined,
  };
  const asWindow = globalThis as unknown as {
    window: { localStorage: StorageFake };
  };
  Object.assign(globalThis, { localStorage: throwing });
  asWindow.window.localStorage = throwing;
  try {
    assert.deepEqual(await importedRows(), []);
  } finally {
    Object.assign(globalThis, { localStorage: storage });
    asWindow.window.localStorage = storage;
  }
});
