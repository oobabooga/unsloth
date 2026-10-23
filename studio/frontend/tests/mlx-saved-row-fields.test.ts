// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The MLX fields of a model's saved row: what a save sends, and how a row reads back. A field the
// row does not carry reads as its default, since a save sends every field the panel shows.

import assert from "node:assert/strict";
import test from "node:test";

import { registerStoreStubResolver } from "./helpers/kit.ts";

registerStoreStubResolver();

const { fromApiOverride, toApiOverride } = await import(
  "../src/features/model-picker/api/model-overrides.ts"
);
const { DEFAULT_PER_MODEL_CONFIG } = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);

test("int8 prefill reaches the server row only when on", () => {
  const on = { ...DEFAULT_PER_MODEL_CONFIG, mlxInt8Prefill: true };
  assert.equal(toApiOverride(on).mlx_int8_prefill, true);
  assert.equal(
    "mlx_int8_prefill" in toApiOverride(DEFAULT_PER_MODEL_CONFIG),
    false,
  );
  assert.equal(
    fromApiOverride({ mlx_int8_prefill: true }).mlxInt8Prefill,
    true,
  );
  assert.equal(fromApiOverride({}).mlxInt8Prefill, false);
});

test("MLX offload is always sent, null clearing to Auto, and an absent field reads as Auto", () => {
  const set = {
    ...DEFAULT_PER_MODEL_CONFIG,
    mlxPleOffload: 0.5,
    mlxExpertOffload: "off" as const,
  };
  assert.deepEqual(
    [toApiOverride(set).mlx_ple_offload, toApiOverride(set).mlx_expert_offload],
    [0.5, "off"],
  );
  const auto = toApiOverride(DEFAULT_PER_MODEL_CONFIG);
  assert.deepEqual(
    [auto.mlx_ple_offload, auto.mlx_expert_offload],
    [null, null],
  );
  assert.equal(
    fromApiOverride({ mlx_expert_offload: 0.25 }).mlxExpertOffload,
    0.25,
  );
  assert.equal(fromApiOverride({}).mlxPleOffload, null);
});

test("a server row holding the superseded cache width reads back as the single choice", () => {
  assert.equal(fromApiOverride({ mlx_kv_bits: 8 }).mlxKvQuant, "8");
  assert.equal(
    fromApiOverride({ mlx_kv_quant: "tq-4", mlx_kv_bits: 8 }).mlxKvQuant,
    "tq-4",
  );
  assert.equal(
    fromApiOverride({ mlx_kv_quant: "auto", mlx_kv_bits: 4 }).mlxKvQuant,
    null,
  );
  assert.equal(
    fromApiOverride({ mlx_kv_quant: null } as never).mlxKvQuant,
    null,
  );
  assert.equal(fromApiOverride({}).mlxKvQuant, null);
});
