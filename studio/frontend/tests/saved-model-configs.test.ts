// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The server's override map is the only store of a model's saved load settings. These drive the
// browser's copy of it against a fake server.

import assert from "node:assert/strict";
import test from "node:test";

import {
  installLocalStorageFake,
  registerStoreStubResolver,
} from "./helpers/kit.ts";

registerStoreStubResolver();
installLocalStorageFake();

const {
  ensureSavedModelConfigs,
  forgetModelConfig,
  overrideLookupCandidates,
  readSavedModelConfigs,
  refreshSavedModelConfigs,
  resolveInitialConfig,
  resolveOnlyRememberedGgufVariant,
  resolveResidentInitialConfig,
  saveModelConfig,
  savedConfigFor,
  setSavedModelConfigsForTests,
  subscribeSavedModelConfigs,
} = await import(
  "../src/features/model-picker/model-config/saved-model-configs.ts"
);
const { DEFAULT_PER_MODEL_CONFIG } = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);
const { setAuthFetchHandler } = await import("./helpers/store-stubs/auth.ts");

const OVERRIDES_URL = "/api/settings/openai-auto-switch/overrides";
const MANAGED_REFUSAL = /--model is managed by Unsloth/;
type Row = Record<string, unknown>;
type Call = { method: string; body: Row | null };

/** A server holding `map`: a GET answers it, a PUT stores or removes the row it names. */
function fakeServer(initial: Record<string, Row> = {}) {
  let map: Record<string, Row> = { ...initial };
  const calls: Call[] = [];
  setAuthFetchHandler((input, init) => {
    if (input !== OVERRIDES_URL) {
      throw new Error(`unexpected request to ${input}`);
    }
    const method = init?.method ?? "GET";
    const body = init?.body ? (JSON.parse(String(init.body)) as Row) : null;
    calls.push({ method, body });
    if (method === "PUT" && body) {
      const { model_id: key, remove, ...rest } = body;
      const fields = Object.fromEntries(
        Object.entries(rest).filter(([name]) => !name.startsWith("mirrors_")),
      );
      map = { ...map };
      if (remove) {
        delete map[String(key)];
      } else {
        map[String(key)] = fields;
      }
    }
    return Response.json({ overrides: map });
  });
  return {
    calls,
    get map() {
      return map;
    },
  };
}

test.beforeEach(() => {
  setSavedModelConfigsForTests(null);
});

test("a save goes to the server, and its answer is what every reader sees", async () => {
  const server = fakeServer();
  let notified = 0;
  const unsubscribe = subscribeSavedModelConfigs(() => {
    notified += 1;
  });
  await saveModelConfig("org/m-GGUF", "Q4_K_M", {
    ...DEFAULT_PER_MODEL_CONFIG,
    customContextLength: 32768,
    kvCacheDtype: "q8_0",
  });
  unsubscribe();
  assert.equal(server.calls.length, 1);
  const { body } = server.calls[0];
  assert.equal(body?.model_id, "org/m-GGUF:Q4_K_M");
  assert.equal(body?.remove, false);
  assert.equal(body?.custom_context_length, 32768);
  assert.equal(body?.kv_cache_dtype, "q8_0");
  assert.equal(notified, 1);
  assert.deepEqual(readSavedModelConfigs(), server.map);
  const saved = resolveInitialConfig("org/m-GGUF", "Q4_K_M");
  assert.equal(saved.remembered, true);
  assert.equal(saved.config.customContextLength, 32768);
  assert.equal(saved.config.kvCacheDtype, "q8_0");
});

test("a refused save rejects with the server's reason and changes nothing here", async () => {
  setSavedModelConfigsForTests({ "org/m": { n_parallel: 2 } });
  const before = readSavedModelConfigs();
  setAuthFetchHandler(() =>
    Response.json(
      { detail: "llama_extra_args: --model is managed by Unsloth" },
      { status: 400 },
    ),
  );
  await assert.rejects(
    saveModelConfig("org/m", null, {
      ...DEFAULT_PER_MODEL_CONFIG,
      llamaExtraArgs: ["--model", "x"],
    }),
    MANAGED_REFUSAL,
  );
  assert.equal(readSavedModelConfigs(), before);
  assert.equal(resolveInitialConfig("org/m").config.nParallel, 2);
});

test("a forget sends an explicit remove and the row is gone for every reader", async () => {
  const server = fakeServer({ "org/m-GGUF:Q4_K_M": { n_parallel: 4 } });
  await refreshSavedModelConfigs();
  assert.equal(resolveInitialConfig("org/m-GGUF", "Q4_K_M").remembered, true);
  await forgetModelConfig("org/m-GGUF", "Q4_K_M");
  const { body } = server.calls[1];
  assert.equal(body?.remove, true);
  assert.deepEqual(body?.llama_extra_args, []);
  assert.equal(resolveInitialConfig("org/m-GGUF", "Q4_K_M").remembered, false);
});

test("a read asked for while a save is out goes after it, so it cannot bring back the old row", async () => {
  let map: Record<string, Row> = { "org/m": { n_parallel: 2 } };
  const order: string[] = [];
  let releasePut: () => void = () => undefined;
  setAuthFetchHandler(async (_input, init) => {
    const method = init?.method ?? "GET";
    order.push(`${method} sent`);
    if (method === "PUT") {
      await new Promise<void>((resolve) => {
        releasePut = resolve;
      });
      map = { "org/m": { n_parallel: 8 } };
    }
    return Response.json({ overrides: map });
  });
  const save = saveModelConfig("org/m", null, {
    ...DEFAULT_PER_MODEL_CONFIG,
    nParallel: 8,
  });
  const read = refreshSavedModelConfigs();
  while (order.length === 0) {
    await new Promise((resolve) => setTimeout(resolve, 1));
  }
  await new Promise((resolve) => setTimeout(resolve, 5));
  assert.deepEqual(order, ["PUT sent"]);
  releasePut();
  await Promise.all([save, read]);
  assert.deepEqual(order, ["PUT sent", "GET sent"]);
  assert.equal(resolveInitialConfig("org/m").config.nParallel, 8);
});

test("two saves go out one after the other, so the older answer cannot land last", async () => {
  const sent: number[] = [];
  const answered: number[] = [];
  setAuthFetchHandler(async (_input, init) => {
    const body = JSON.parse(String(init?.body)) as Row;
    const slots = Number(body.n_parallel);
    sent.push(slots);
    // The first write is the slow one.
    await new Promise((resolve) => setTimeout(resolve, slots === 2 ? 20 : 0));
    answered.push(slots);
    return Response.json({ overrides: { "org/m": { n_parallel: slots } } });
  });
  await Promise.all([
    saveModelConfig("org/m", null, {
      ...DEFAULT_PER_MODEL_CONFIG,
      nParallel: 2,
    }),
    saveModelConfig("org/m", null, {
      ...DEFAULT_PER_MODEL_CONFIG,
      nParallel: 8,
    }),
  ]);
  assert.deepEqual(sent, [2, 8]);
  assert.deepEqual(answered, [2, 8]);
  assert.equal(resolveInitialConfig("org/m").config.nParallel, 8);
});

test("a failed first read leaves the defaults and the next caller reads again", async () => {
  let attempts = 0;
  setAuthFetchHandler(() => {
    attempts += 1;
    return attempts === 1
      ? Response.json({ detail: "busy" }, { status: 503 })
      : Response.json({ overrides: { "org/m": { n_parallel: 3 } } });
  });
  await ensureSavedModelConfigs();
  assert.equal(readSavedModelConfigs(), null);
  assert.equal(resolveInitialConfig("org/m").remembered, false);
  await ensureSavedModelConfigs();
  assert.equal(attempts, 2);
  assert.equal(resolveInitialConfig("org/m").config.nParallel, 3);
  await ensureSavedModelConfigs();
  assert.equal(attempts, 2, "a read map is not read again");
});

test("the lookup order is the server's override_lookup_candidates", () => {
  assert.deepEqual(overrideLookupCandidates("org/m-GGUF", null, "Q4_K_M"), [
    "org/m-GGUF:Q4_K_M",
    "org/m-GGUF",
  ]);
  assert.deepEqual(
    overrideLookupCandidates("/cache/snap/abc", "org/m-GGUF", "Q8_0"),
    [
      "/cache/snap/abc:Q8_0",
      "org/m-GGUF:Q8_0",
      "/cache/snap/abc",
      "org/m-GGUF",
    ],
  );
  // A loose .gguf names no variant, so its filename label is tried after the bare path.
  assert.deepEqual(
    overrideLookupCandidates("/models/foo-Q4_K_M.gguf", null, null),
    ["/models/foo-Q4_K_M.gguf", "/models/foo-Q4_K_M.gguf:Q4_K_M"],
  );
});

test("a quant reads its own row, folded by case, else the model's bare row", () => {
  setSavedModelConfigsForTests({
    "org/m-gguf:q8_0": { custom_context_length: 8192 },
    "org/m-GGUF": { custom_context_length: 4096 },
  });
  assert.equal(
    resolveInitialConfig("org/m-GGUF", "Q8_0").config.customContextLength,
    8192,
  );
  const fallback = resolveInitialConfig("org/m-GGUF", "Q4_K_M");
  assert.equal(fallback.remembered, true);
  assert.equal(fallback.config.customContextLength, 4096);
  assert.equal(
    resolveInitialConfig("org/other-GGUF", "Q8_0").remembered,
    false,
  );
});

test("a load path reads before the id it is advertised under", () => {
  setSavedModelConfigsForTests({
    "org/m-GGUF:Q8_0": { n_parallel: 2 },
    "/cache/snap/abc:Q8_0": { n_parallel: 6 },
  });
  assert.equal(
    savedConfigFor("/cache/snap/abc", "org/m-GGUF", "Q8_0").config.nParallel,
    6,
  );
  assert.equal(
    savedConfigFor("/cache/other", "org/m-GGUF", "Q8_0").config.nParallel,
    2,
  );
});

test("a resident file reads the bare path its settings are saved under, then the old label key", () => {
  const path = "/home/u/models/foo-Q4_K_M.gguf";
  setSavedModelConfigsForTests({ [`${path}:Q4_K_M`]: { n_parallel: 3 } });
  assert.equal(
    resolveResidentInitialConfig(path, "Q4_K_M").config.nParallel,
    3,
  );
  setSavedModelConfigsForTests({
    [path]: { n_parallel: 5 },
    [`${path}:Q4_K_M`]: { n_parallel: 3 },
  });
  assert.equal(
    resolveResidentInitialConfig(path, "Q4_K_M").config.nParallel,
    5,
  );
});

test("a resident snapshot path finds the settings saved under its repo id", () => {
  const snapshot =
    "/home/u/.cache/huggingface/hub/models--org--m/snapshots/abc123";
  setSavedModelConfigsForTests({ "org/m": { max_seq_length: 16384 } });
  const resolved = resolveResidentInitialConfig(snapshot, null);
  assert.equal(resolved.remembered, true);
  assert.equal(resolved.config.maxSeqLength, 16384);
});

test("the only saved quant of a model is found, and two are not guessed between", () => {
  setSavedModelConfigsForTests({ "org/m-GGUF:Q4_K_M": { n_parallel: 2 } });
  assert.equal(
    resolveOnlyRememberedGgufVariant("org/m-GGUF")?.ggufVariant,
    "Q4_K_M",
  );
  setSavedModelConfigsForTests({
    "org/m-GGUF:Q4_K_M": { n_parallel: 2 },
    "org/m-GGUF:Q8_0": { n_parallel: 4 },
  });
  assert.equal(resolveOnlyRememberedGgufVariant("org/m-GGUF"), null);
});
