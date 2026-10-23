// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Run settings saves a config as one server row (toApiOverride) and reads it back
// (fromApiOverride). A field one direction misses is saved and silently never comes back, so
// every field is driven through both. The backend's test_model_override_field_parity.py holds
// ApiModelOverride to the fields the route accepts and the load applies.

import assert from "node:assert/strict";
import test from "node:test";

import type { PerModelConfig } from "../src/features/model-picker/model-config/per-model-config.ts";
import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { fromApiOverride, toApiOverride } = await import(
  "../src/features/model-picker/api/model-overrides.ts"
);
const { DEFAULT_PER_MODEL_CONFIG, normalizePerModelConfig } = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);

// Required, so a field added to PerModelConfig fails the typecheck here until it is covered.
// Every value is away from its default and valid beside the others.
const EVERY_FIELD: Required<PerModelConfig> = {
  engineParallelism: "pipeline",
  enginePrecision: "fp8",
  engineKvCacheDtype: "fp8",
  engineSeed: 7,
  engineGpuMemoryUtilization: 0.5,
  engine: "vllm",
  loadIn4bit: false,
  customContextLength: 4096,
  maxSeqLength: 8192,
  kvCacheDtype: "q8_0",
  mlxKvQuant: "4",
  mlxInt8Prefill: true,
  mlxPleOffload: 0.5,
  mlxExpertOffload: "off",
  speculativeType: "dspark",
  specDraftNMax: 3,
  specDraftCacheDtype: "q4_0",
  specDraftModel: "org/drafter",
  nParallel: 4,
  reasoningBudget: 256,
  reasoningBudgetMessage: "Done thinking.",
  nBatch: 1024,
  nUbatch: 256,
  loadMode: "mmap",
  ctxCheckpoints: 4,
  cacheRam: 0,
  tensorParallel: true,
  disableVision: true,
  chatTemplateOverride: "{{ messages }}",
  llamaExtraArgs: ["--no-warmup"],
  gpuMemoryMode: "manual",
  gpuLayers: 10,
  nCpuMoe: 2,
  selectedGpuIds: [0, 1],
  selectedGpuIndexKind: "vulkan",
  tensorSplit: [1, 2],
};

const INTERFACE_BODY = /export interface ApiModelOverride \{([\s\S]*?)\n\}/;
const OPTIONAL_FIELD = /^\s*(\w+)\?:/gm;

function interfaceFields(): string[] {
  const source = readSrc("features/model-picker/api/model-overrides.ts");
  const body = INTERFACE_BODY.exec(source);
  if (!body) {
    throw new Error("ApiModelOverride not found");
  }
  return [...body[1].matchAll(OPTIONAL_FIELD)].map((match) => match[1]);
}

test("every value used here is one storage keeps, so no field passes by being dropped", () => {
  assert.deepEqual(normalizePerModelConfig(EVERY_FIELD), EVERY_FIELD);
});

test("every config field survives the trip through a stored row", () => {
  assert.deepEqual(fromApiOverride(toApiOverride(EVERY_FIELD)), EVERY_FIELD);
});

test("a save writes every field the row declares", () => {
  // The superseded MLX cache width is only ever read, from a row an older build wrote.
  const written = interfaceFields().filter((field) => field !== "mlx_kv_bits");
  assert.deepEqual(
    Object.keys(toApiOverride(EVERY_FIELD)).sort(),
    written.sort(),
  );
});

test("a read takes every field a row carries", () => {
  const row = toApiOverride(EVERY_FIELD);
  const read = fromApiOverride(row);
  for (const field of Object.keys(row)) {
    const without = { ...row };
    delete without[field as keyof typeof without];
    assert.notDeepEqual(
      fromApiOverride(without),
      read,
      `${field} is never read`,
    );
  }
});

test("a row written by an older build reads its superseded cache width", () => {
  // biome-ignore lint/style/useNamingConvention: API schema
  assert.equal(fromApiOverride({ mlx_kv_bits: 8 }).mlxKvQuant, "8");
});

test("defaults travel as nothing, and nothing reads back as the defaults", () => {
  const row = toApiOverride(DEFAULT_PER_MODEL_CONFIG);
  // Sent even at their defaults, because the route keeps a stored value the payload omits.
  assert.deepEqual(Object.keys(row).sort(), [
    "engine",
    "engine_gpu_memory_utilization",
    "engine_kv_cache_dtype",
    "engine_parallelism",
    "engine_precision",
    "engine_seed",
    "mlx_expert_offload",
    "mlx_ple_offload",
  ]);
  // A row without a GPU pick says Automatic. Absent in a staged load config would mean "keep
  // whatever pick is live", which is not what a saved row without one means.
  assert.deepEqual(fromApiOverride({}), {
    ...normalizePerModelConfig(DEFAULT_PER_MODEL_CONFIG),
    selectedGpuIds: null,
    selectedGpuIndexKind: null,
  });
});
