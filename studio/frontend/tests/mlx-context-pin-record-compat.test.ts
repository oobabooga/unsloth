// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A context pin lives in one of two fields: MLX keeps it in customContextLength and transformers in
// maxSeqLength, and a row saved before the move still holds an MLX pin in maxSeqLength. Every reader
// has to honour both, on either backend.

import assert from "node:assert/strict";
import test from "node:test";

import type { ApiModelOverride } from "../src/features/model-picker/api/model-overrides.ts";

import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
installLocalStorageFake();

const {
  CONTEXT_LENGTH_MIN,
  DEFAULT_MAX_SEQ_LENGTH,
  DEFAULT_PER_MODEL_CONFIG,
  MAX_SEQ_LENGTH_MIN,
  contextPinPatch,
  isDefaultConfig,
  normalizeMaxSeqLength,
  normalizePerModelConfig,
  savedContextPin,
} = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);
const { toApiOverride } = await import(
  "../src/features/model-picker/api/model-overrides.ts"
);
const { resolveInitialConfig, setSavedModelConfigsForTests } = await import(
  "../src/features/model-picker/model-config/saved-model-configs.ts"
);
const { resolveLoadMaxSeqLength } = await import(
  "../src/features/chat/presets/preset-policy.ts"
);

const MODEL = "mlx-community/Qwen3-8B-4bit";

/** The model's saved row as the server returns it, read the way every load path reads it. */
function stage(row: ApiModelOverride) {
  setSavedModelConfigsForTests({ [MODEL]: row });
  return resolveInitialConfig(MODEL, null);
}

type Pin = { customContextLength?: number; maxSeqLength?: number };

function rowOf(pin: Pin): ApiModelOverride {
  return {
    ...(pin.customContextLength === undefined
      ? {}
      : // biome-ignore lint/style/useNamingConvention: API schema
        { custom_context_length: pin.customContextLength }),
    ...(pin.maxSeqLength === undefined
      ? {}
      : // biome-ignore lint/style/useNamingConvention: API schema
        { max_seq_length: pin.maxSeqLength }),
  };
}

type Row = {
  name: string;
  pin: Pin;
  normalizedPin: number | null;
  rawPin: number | null;
  isDefault: boolean;
  remembered: boolean;
  mlxRequest: number;
  transformersRequest: number;
  note?: string;
};

const ROWS: Row[] = [
  {
    name: "old row: pin only in max_seq_length",
    pin: { maxSeqLength: 8192 },
    normalizedPin: 8192,
    rawPin: 8192,
    isDefault: false,
    remembered: true,
    mlxRequest: 8192,
    transformersRequest: 8192,
    note: "backwards compatibility: the pre-move field still pins, on either backend.",
  },
  {
    name: "new row: pin only in custom_context_length",
    pin: { customContextLength: 32768 },
    normalizedPin: 32768,
    rawPin: 32768,
    isDefault: false,
    remembered: true,
    mlxRequest: 32768,
    transformersRequest: 32768,
  },
  {
    name: "both fields set",
    pin: { customContextLength: 32768, maxSeqLength: 8192 },
    normalizedPin: 32768,
    rawPin: 32768,
    isDefault: false,
    remembered: true,
    mlxRequest: 32768,
    transformersRequest: 32768,
    note:
      "customContextLength wins in both the pin reader and the load resolver, so the " +
      "two agree. No edit can author this state, since contextPinPatch clears the other " +
      "field, but a legacy row can hold it.",
  },
  {
    name: "neither field set",
    pin: {},
    normalizedPin: null,
    rawPin: null,
    isDefault: true,
    // An empty row is no row: nothing is saved for the model.
    remembered: false,
    // MLX sizes its own window (0 asks for nothing); transformers gets the app default.
    mlxRequest: 0,
    transformersRequest: DEFAULT_MAX_SEQ_LENGTH,
  },
  {
    name: "zero in both fields",
    pin: { customContextLength: 0, maxSeqLength: 0 },
    normalizedPin: null,
    // `??` keeps 0, so an unnormalized value yields a falsy 0 pin.
    rawPin: 0,
    isDefault: true,
    remembered: true,
    mlxRequest: 0,
    transformersRequest: DEFAULT_MAX_SEQ_LENGTH,
  },
  {
    name: "negative in both fields",
    pin: { customContextLength: -1, maxSeqLength: -4096 },
    normalizedPin: null,
    rawPin: -1,
    isDefault: true,
    remembered: true,
    mlxRequest: 0,
    transformersRequest: DEFAULT_MAX_SEQ_LENGTH,
  },
  {
    name: "non-integer in both fields",
    pin: { customContextLength: 8192.7, maxSeqLength: 8192.7 },
    normalizedPin: 8192,
    rawPin: 8192.7,
    isDefault: false,
    remembered: true,
    mlxRequest: 8192,
    transformersRequest: 8192,
  },
];

function loadRequest(
  config: { customContextLength: number | null; maxSeqLength: number | null },
  isMlx: boolean,
): number {
  return resolveLoadMaxSeqLength({
    modelId: MODEL,
    ggufVariant: null,
    isGguf: false,
    customContextLength: config.customContextLength,
    loadedContextLength: null,
    currentCheckpoint: "",
    activeGgufVariant: null,
    isMlx,
    pinnedMaxSeqLength: normalizeMaxSeqLength(config.maxSeqLength),
    defaultMaxSeqLength: DEFAULT_MAX_SEQ_LENGTH,
    presetSource: "builtin-default",
  });
}

for (const row of ROWS) {
  test(`saved row matrix: ${row.name}`, () => {
    const { config, remembered } = stage(rowOf(row.pin));
    assert.equal(remembered, row.remembered, "remembered");
    assert.equal(
      savedContextPin(config),
      row.normalizedPin,
      "savedContextPin(read)",
    );
    assert.equal(
      savedContextPin(row.pin),
      row.rawPin,
      "savedContextPin(unnormalized)",
    );
    assert.equal(isDefaultConfig(config), row.isDefault, "isDefaultConfig");
    assert.equal(loadRequest(config, true), row.mlxRequest, "MLX /load n_ctx");
    assert.equal(
      loadRequest(config, false),
      row.transformersRequest,
      "transformers /load n_ctx",
    );
  });
}

test("contextPinPatch writes the pin in exactly one field and clears the other", () => {
  assert.deepEqual(contextPinPatch(32768, true), {
    customContextLength: 32768,
    maxSeqLength: null,
  });
  assert.deepEqual(contextPinPatch(32768, false), {
    customContextLength: null,
    maxSeqLength: 32768,
  });
});

test("contextPinPatch bounds without snapping, and never writes a blank", () => {
  assert.equal(contextPinPatch(8193, true).customContextLength, 8193);
  assert.equal(contextPinPatch(8192.7, true).customContextLength, 8192);
  assert.equal(
    contextPinPatch(1, true).customContextLength,
    MAX_SEQ_LENGTH_MIN,
  );
  // 0 and negatives become the floor, not a cleared pin: "unpin" must not route here.
  assert.equal(
    contextPinPatch(0, true).customContextLength,
    MAX_SEQ_LENGTH_MIN,
  );
  assert.equal(contextPinPatch(-1, false).maxSeqLength, MAX_SEQ_LENGTH_MIN);
  assert.equal(
    contextPinPatch(Number.NaN, true).customContextLength,
    MAX_SEQ_LENGTH_MIN,
  );
  assert.equal(
    contextPinPatch(Number.POSITIVE_INFINITY, true).customContextLength,
    MAX_SEQ_LENGTH_MIN,
  );
  assert.equal(MAX_SEQ_LENGTH_MIN, CONTEXT_LENGTH_MIN);
});

test("a patched pin round-trips through the saved row on both backends", () => {
  for (const isMlx of [true, false]) {
    const patched = {
      ...DEFAULT_PER_MODEL_CONFIG,
      ...contextPinPatch(32768, isMlx),
    };
    assert.equal(isDefaultConfig(normalizePerModelConfig(patched)), false);
    const { config, remembered } = stage(toApiOverride(patched));
    assert.ok(remembered);
    assert.equal(savedContextPin(config), 32768, `isMlx=${isMlx}`);
    assert.equal(loadRequest(config, isMlx), 32768, `isMlx=${isMlx}`);
    assert.equal(
      loadRequest(config, !isMlx),
      32768,
      `cross-read isMlx=${!isMlx}`,
    );
  }
});

test("int8 prefill defaults off, and only an enabled one is a setting", () => {
  assert.equal(normalizePerModelConfig({}).mlxInt8Prefill, false);
  assert.equal(
    normalizePerModelConfig({ mlxInt8Prefill: "true" }).mlxInt8Prefill,
    false,
  );
  assert.equal(
    isDefaultConfig({ ...DEFAULT_PER_MODEL_CONFIG, mlxInt8Prefill: true }),
    false,
  );
});

test("MLX offload keeps off or a fraction in (0, 1], and a set one is a setting", () => {
  const offload = (value: unknown) =>
    normalizePerModelConfig({ mlxExpertOffload: value }).mlxExpertOffload;
  assert.deepEqual([0.5, 1, "off"].map(offload), [0.5, 1, "off"]);
  assert.deepEqual(
    [0, 1.5, -0.2, Number.NaN, "auto", "0.5", true].map(offload),
    Array(7).fill(null),
  );
  assert.equal(
    isDefaultConfig({ ...DEFAULT_PER_MODEL_CONFIG, mlxPleOffload: "off" }),
    false,
  );
});

test("the fractional TurboQuant choice survives the saved row", () => {
  const saved = toApiOverride({
    ...DEFAULT_PER_MODEL_CONFIG,
    mlxKvQuant: "tq-3.5",
  });
  assert.equal(saved.mlx_kv_quant, "tq-3.5");
  assert.equal(stage(saved).config.mlxKvQuant, "tq-3.5");
});

test("a record holding the superseded width normalizes to the single choice", () => {
  // An older build's record, as the one-time import reads it.
  assert.equal(normalizePerModelConfig({ mlxKvBits: 8 }).mlxKvQuant, "8");
  assert.equal(
    normalizePerModelConfig({ mlxKvQuant: " TQ-4 ", mlxKvBits: 8 }).mlxKvQuant,
    "tq-4",
  );
  assert.equal(
    normalizePerModelConfig({ mlxKvQuant: null, mlxKvBits: 8 }).mlxKvQuant,
    null,
  );
});
