// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The server's map of saved per-model load settings, and the translation between one of its rows
// and the picker's config shape. Every load path reads the map (routes/inference.py,
// llama_compat.py, the CLI); saved-model-configs.ts is the browser's copy of it.

import { normalizeTensorSplit } from "@/hooks/gpu-tensor-split";
import type { GpuIndexKind } from "@/hooks/gpu-selection";
import { readFastApiError } from "@/lib/format-fastapi-error";
import { splitQuantSuffix } from "../model-config/model-identity";
import {
  normalizeMlxKvQuant,
  type PerModelConfig,
  normalizePerModelConfig,
} from "../model-config/per-model-config";

const OVERRIDES_URL = "/api/settings/openai-auto-switch/overrides";

async function request(init?: RequestInit): Promise<Response> {
  // Imported on first use, so the readers of saved settings stay importable without the auth module.
  const { authFetch } = await import("@/features/auth");
  return authFetch(OVERRIDES_URL, init);
}

/** One model's stored launch config, as the backend persists it. */
export interface ApiModelOverride {
  engine_parallelism?: "tensor" | "pipeline" | "data";
  engine_precision?: "auto" | "bf16" | "fp16" | "int4" | "int8" | "fp8";
  engine?: "auto" | "vllm" | "sglang";
  engine_kv_cache_dtype?: "auto" | "fp8";
  engine_seed?: number | null;
  engine_gpu_memory_utilization?: number | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  llama_extra_args?: string[];
  // biome-ignore lint/style/useNamingConvention: API schema
  max_seq_length?: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  custom_context_length?: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  kv_cache_dtype?: string;
  // biome-ignore lint/style/useNamingConvention: API schema
  mlx_kv_quant?: string;
  mlx_kv_bits?: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  mlx_int8_prefill?: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  mlx_ple_offload?: "off" | number | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  mlx_expert_offload?: "off" | number | null;
  // Stored only when false: absent is the 4-bit default.
  // biome-ignore lint/style/useNamingConvention: API schema
  load_in_4bit?: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  speculative_type?: string;
  // biome-ignore lint/style/useNamingConvention: API schema
  spec_draft_n_max?: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  spec_draft_model?: string;
  // biome-ignore lint/style/useNamingConvention: API schema
  n_parallel?: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  reasoning_budget?: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  reasoning_budget_message?: string;
  // biome-ignore lint/style/useNamingConvention: API schema
  n_batch?: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  n_ubatch?: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  load_mode?: string;
  // biome-ignore lint/style/useNamingConvention: API schema
  spec_draft_cache_type?: string;
  // biome-ignore lint/style/useNamingConvention: API schema
  ctx_checkpoints?: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  cache_ram?: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  tensor_parallel?: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  disable_vision?: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  chat_template_override?: string;
  // biome-ignore lint/style/useNamingConvention: API schema
  gpu_memory_mode?: "auto" | "manual";
  // biome-ignore lint/style/useNamingConvention: API schema
  gpu_layers?: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  n_cpu_moe?: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  tensor_split?: number[] | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  gpu_ids?: number[];
  // Which index space gpu_ids is in. Absent means "physical", all an older row could mean.
  // biome-ignore lint/style/useNamingConvention: API schema
  gpu_index_kind?: GpuIndexKind;
}

export type ApiModelOverrides = Record<string, ApiModelOverride>;

/** The key one model's config is stored under: the `repo:VARIANT` form an OpenAI request
 *  names a quant by, so two quants of one repo keep separate configs. Bare id if no variant. */
export function modelOverrideKey(
  modelId: string,
  ggufVariant?: string | null,
): string {
  return ggufVariant ? `${modelId}:${ggufVariant}` : modelId;
}

/** A path that names one file whatever the casing, folded for comparison, or null when the
 *  path is case-sensitive. Rules are _fold_case_insensitive_path's and must be, since the
 *  server applies an override this resolver has to find. A POSIX path is not folded, or
 *  "/models/Foo.gguf" would collect the arguments stored for "/models/foo.gguf". */
const WINDOWS_DRIVE_PATH = /^[a-zA-Z]:[\\/]/;
const WSL_DRIVE_PATH = /^\/mnt\/[a-zA-Z](\/|$)/;

function foldCaseInsensitivePath(key: string): string | null {
  const slashed = key.replace(/\\/g, "/");
  let minimum: number;
  if (WINDOWS_DRIVE_PATH.test(key)) {
    minimum = 3;
  } else if (slashed.startsWith("//")) {
    minimum = 2;
  } else if (WSL_DRIVE_PATH.test(slashed)) {
    minimum = 6;
  } else {
    return null;
  }
  let trimmed = slashed;
  while (trimmed.length > minimum && trimmed.endsWith("/")) {
    trimmed = trimmed.slice(0, -1);
  }
  return trimmed.toLowerCase();
}

export function foldOverrideKey(key: string): string {
  // A path that folds does so whole, separators and casing together.
  const path = foldCaseInsensitivePath(key);
  if (path !== null) {
    return path;
  }
  // splitQuantSuffix, not the last colon: a colon is legal in a POSIX filename, so
  // "/models/foo:Bar.gguf" is a whole path and reading "Bar.gguf" as a quant would fold it onto the
  // real, different file "/models/foo:bar.gguf". Mirrors the backend's split_quant_suffix.
  const split = splitQuantSuffix(key);
  const id = split ? split[0] : key;
  const quant = split ? `:${split[1].toLowerCase()}` : "";
  // A POSIX path keeps its case; only the quant folds, because the browser lowercases that
  // before storing. A repo id folds whole.
  return id.startsWith("/") ? `${id}${quant}` : `${id.toLowerCase()}${quant}`;
}

function presentOverride(
  value: ApiModelOverride | undefined | null,
): ApiModelOverride | null {
  return value && Object.keys(value).length > 0 ? value : null;
}

export function resolveStoredOverride(
  overrides: ApiModelOverrides,
  keys: readonly string[],
): ApiModelOverride | null {
  // The overrides route folds identities before reading a row and falls back from `repo:QUANT`
  // to the bare repo. Keys are tried most specific first, then folded, as whole ENTRIES,
  // stopping at the first that exists. The auto-switch loader breaks on the first non-empty
  // override, so falling through to a bare repo because the variant row carries no arguments
  // would launch with flags an API load would not use. An entry with no fields is skipped,
  // because that is what `if override: break` does on the server.
  const folded = new Map<string, ApiModelOverride | null>();
  for (const [key, value] of Object.entries(overrides)) {
    if (!presentOverride(value)) {
      continue;
    }
    const foldedKey = foldOverrideKey(key);
    // Ambiguous folds resolve to nothing, as resolve_model_override_key does: duplicate case
    // variants left by an upgrade must not be picked at enumeration order, which is another
    // model's settings half the time.
    folded.set(foldedKey, folded.has(foldedKey) ? null : value);
  }
  for (const key of keys) {
    const exact = presentOverride(overrides[key]);
    if (exact) {
      return exact;
    }
    // Folding by the backend's rule: a POSIX path is case-sensitive, so lowercasing one whole
    // would hand /models/foo.gguf the arguments stored for /models/Foo.gguf. Windows, UNC and
    // WSL paths do fold, as does the quant suffix the browser lowercases before storing.
    const match = folded.get(foldOverrideKey(key));
    if (match) {
      return match;
    }
  }
  return null;
}

export async function fetchModelOverrides(): Promise<ApiModelOverrides> {
  const res = await request();
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to load saved model settings"),
    );
  }
  const body = (await res.json()) as { overrides?: ApiModelOverrides };
  return body.overrides ?? {};
}

/** One stored row in the picker's config shape. A field the row does not carry is at its default:
 *  a save sends every field this UI shows, so the row is the whole saved config. */
export function fromApiOverride(override: ApiModelOverride): PerModelConfig {
  const gpuIds = override.gpu_ids?.length ? override.gpu_ids : null;
  const normalized = normalizePerModelConfig({
    engine: override.engine,
    engineParallelism: override.engine_parallelism,
    enginePrecision: override.engine_precision,
    engineKvCacheDtype: override.engine_kv_cache_dtype,
    engineSeed: override.engine_seed,
    engineGpuMemoryUtilization: override.engine_gpu_memory_utilization,
    loadIn4bit: override.load_in_4bit,
    customContextLength: override.custom_context_length,
    maxSeqLength: override.max_seq_length,
    kvCacheDtype: override.kv_cache_dtype,
    mlxKvQuant: normalizeMlxKvQuant(override.mlx_kv_quant, override.mlx_kv_bits),
    mlxInt8Prefill: override.mlx_int8_prefill,
    mlxPleOffload: override.mlx_ple_offload,
    mlxExpertOffload: override.mlx_expert_offload,
    speculativeType: override.speculative_type,
    specDraftNMax: override.spec_draft_n_max,
    specDraftCacheDtype: override.spec_draft_cache_type,
    specDraftModel: override.spec_draft_model,
    nParallel: override.n_parallel,
    reasoningBudget: override.reasoning_budget,
    reasoningBudgetMessage: override.reasoning_budget_message,
    nBatch: override.n_batch,
    nUbatch: override.n_ubatch,
    loadMode: override.load_mode,
    ctxCheckpoints: override.ctx_checkpoints,
    cacheRam: override.cache_ram,
    tensorParallel: override.tensor_parallel,
    disableVision: override.disable_vision,
    chatTemplateOverride: override.chat_template_override,
    llamaExtraArgs: override.llama_extra_args,
    gpuMemoryMode: override.gpu_memory_mode,
    gpuLayers: override.gpu_layers,
    nCpuMoe: override.n_cpu_moe,
    selectedGpuIds: gpuIds,
    // The pin keeps its index space; reconcileGpuSelection drops it on a host that numbers
    // devices the other way.
    selectedGpuIndexKind: gpuIds ? (override.gpu_index_kind ?? "physical") : null,
    tensorSplit: normalizeTensorSplit(override.tensor_split, gpuIds),
  });
  // normalizePerModelConfig collapses an empty list to null. The row's [] is a cleared box that
  // stops a bare-repository row from answering, so it is sent as one.
  if (Array.isArray(override.llama_extra_args)) {
    normalized.llamaExtraArgs = [...override.llama_extra_args];
  }
  return normalized;
}

/** Translate the UI's per-model config into the backend's schema. Only fields the user set are
 *  sent: an absent field reads as "app default", so nulls would pin defaults and stop the
 *  model following later global changes. A `null` config clears the entry. */
export function toApiOverride(config: PerModelConfig | null): ApiModelOverride {
  if (!config) {
    return {};
  }
  // Engine fields are always sent, defaults included: the server keeps a stored engine choice
  // when the field is absent, so omitting "auto" could never clear an earlier "vllm". It stores
  // only non-default values, so an all-default save still leaves no row.
  const payload: ApiModelOverride = {
    engine: config.engine ?? "auto",
    engine_precision: config.enginePrecision ?? "auto",
    engine_parallelism: config.engineParallelism ?? "tensor",
    engine_kv_cache_dtype: config.engineKvCacheDtype ?? "auto",
    engine_seed: config.engineSeed ?? null,
    engine_gpu_memory_utilization: config.engineGpuMemoryUtilization ?? null,
    // Sent even at Auto: the server keeps a stored knob the payload omits.
    mlx_ple_offload: config.mlxPleOffload ?? null,
    mlx_expert_offload: config.mlxExpertOffload ?? null,
  };
  if (config.maxSeqLength && config.maxSeqLength > 0) {
    payload.max_seq_length = config.maxSeqLength;
  }
  if (config.customContextLength && config.customContextLength > 0) {
    payload.custom_context_length = config.customContextLength;
  }
  if (config.kvCacheDtype) {
    payload.kv_cache_dtype = config.kvCacheDtype;
  }
  // Travels beside kv_cache_dtype, or an API auto-switch loads a remembered MLX model at full precision.
  if (config.mlxKvQuant) {
    payload.mlx_kv_quant = config.mlxKvQuant;
  }
  if (config.speculativeType) {
    payload.speculative_type = config.speculativeType;
  }
  if (config.specDraftNMax && config.specDraftNMax > 0) {
    payload.spec_draft_n_max = config.specDraftNMax;
  }
  if (config.specDraftModel) {
    payload.spec_draft_model = config.specDraftModel;
  }
  // Blank follows the server-wide --parallel default, which is the app default here.
  if (config.nParallel && config.nParallel > 0) {
    payload.n_parallel = config.nParallel;
  }
  if (config.reasoningBudget !== -1) {
    payload.reasoning_budget = config.reasoningBudget;
  }
  if (config.reasoningBudgetMessage) {
    payload.reasoning_budget_message = config.reasoningBudgetMessage;
  }
  // blank follows the llama.cpp defaults (2048 / 512)
  if (config.nBatch && config.nBatch > 0) {
    payload.n_batch = config.nBatch;
  }
  if (config.nUbatch && config.nUbatch > 0) {
    payload.n_ubatch = config.nUbatch;
  }
  // Blank follows llama.cpp's own default for each: auto, f16, 32 and 8192. cacheRam is
  // compared against null rather than truth, since 0 and -1 both mean something.
  if (config.loadMode) {
    payload.load_mode = config.loadMode;
  }
  if (config.specDraftCacheDtype) {
    payload.spec_draft_cache_type = config.specDraftCacheDtype;
  }
  if (config.ctxCheckpoints != null && config.ctxCheckpoints >= 0) {
    payload.ctx_checkpoints = config.ctxCheckpoints;
  }
  if (config.cacheRam != null) {
    payload.cache_ram = config.cacheRam;
  }
  if (config.tensorParallel) {
    payload.tensor_parallel = true;
  }
  if (config.disableVision) {
    payload.disable_vision = true;
  }
  if (config.mlxInt8Prefill) {
    payload.mlx_int8_prefill = true;
  }
  if (config.loadIn4bit === false) {
    payload.load_in_4bit = false;
  }
  if (config.chatTemplateOverride?.trim()) {
    payload.chat_template_override = config.chatTemplateOverride;
  }
  // The one field where absent does NOT mean "app default": the route preserves
  // llama_extra_args it is not sent, which kept CLI-set flags alive while this panel had no
  // control. So `undefined` stays omitted and a cleared box sends an explicit empty list.
  if (config.llamaExtraArgs !== undefined) {
    payload.llama_extra_args = config.llamaExtraArgs ?? [];
  }
  // Only "manual" is a real override; "auto" is the follow-the-global default.
  if (config.gpuMemoryMode === "manual") {
    payload.gpu_memory_mode = "manual";
  }
  // gpuLayers < 0 is Auto, which is also the default.
  if (typeof config.gpuLayers === "number" && config.gpuLayers >= 0) {
    payload.gpu_layers = config.gpuLayers;
  }
  if (typeof config.nCpuMoe === "number" && config.nCpuMoe > 0) {
    payload.n_cpu_moe = config.nCpuMoe;
  }
  // The pin travels with its namespace, or after a backend change the server pins a
  // different device; reconcileGpuSelection drops it on a mismatch instead.
  const gpuIndexKind = config.selectedGpuIndexKind ?? "physical";
  if (config.selectedGpuIds && config.selectedGpuIds.length > 0) {
    payload.gpu_ids = config.selectedGpuIds;
    const tensorSplit = normalizeTensorSplit(config.tensorSplit, config.selectedGpuIds);
    // Explicit null clears a saved split; an older client omitting it keeps its value.
    payload.tensor_split = tensorSplit;
    // Sent only when it is not the legacy default, so a physical pin's payload is
    // unchanged from before this field.
    if (gpuIndexKind !== "physical") {
      payload.gpu_index_kind = gpuIndexKind;
    }
  }
  return payload;
}

export interface PutModelOverrideOptions {
  /** Fill in only what is missing: every value already on the server stays. For the one-time
   *  import of what older builds kept in this browser, whose copy may be the older one. */
  fillAbsentFields?: boolean;
  /** Remove a legacy passthrough value only after this control was explicitly reset. */
  resetReasoningBudget?: boolean;
  resetReasoningBudgetMessage?: boolean;
}

/** Saves one model's settings, or forgets them for a null config, and returns the map after the
 *  write. Throws with the server's reason when the write is refused. `modelId` may be a stored key. */
export async function putModelOverride(
  modelId: string,
  ggufVariant: string | null | undefined,
  config: PerModelConfig | null,
  options?: PutModelOverrideOptions,
): Promise<ApiModelOverrides> {
  const res = await request({
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      // biome-ignore lint/style/useNamingConvention: API schema
      model_id: modelOverrideKey(modelId, ggufVariant),
      // This build mirrors the llama-server tuning group, so an omission here is the user clearing
      // it rather than a client that predates the fields. Without this the backend preserves the
      // stored values, which is what stops a cached older bundle from deleting settings it never
      // knew to send. An older backend ignores the key.
      // biome-ignore lint/style/useNamingConvention: API schema
      mirrors_server_tuning: true,
      // Same contract for the reasoning pair, which a build mirroring the tuning group
      // can still predate.
      // biome-ignore lint/style/useNamingConvention: API schema
      mirrors_reasoning_budget: true,
      // biome-ignore lint/style/useNamingConvention: API schema
      mirrors_spec_draft_model: true,
      // Only sent when set, so an older backend is not handed an unknown key every save.
      ...(options?.fillAbsentFields
        ? // biome-ignore lint/style/useNamingConvention: API schema
          { fill_absent_fields: true }
        : {}),
      // Say which operation this is: an all-default save carries no fields, shape-identical to a
      // forget, and guessing wrong wipes flags the UI cannot show or restore.
      remove: config === null,
      // The backend preserves the flags when omitted, which a save that never opened the box relies
      // on; a forget means all of it, so that path sends an explicit [].
      ...(config === null
        ? // biome-ignore lint/style/useNamingConvention: API schema
          { llama_extra_args: [] }
        : {}),
      ...toApiOverride(config),
      // Write-only reset markers let the backend remove legacy passthrough flags
      // shadowing these controls. Fill-only migration must never delete stored flags.
      ...(options?.resetReasoningBudget && config?.reasoningBudget === -1
        ? {
            // biome-ignore lint/style/useNamingConvention: API schema
            reasoning_budget: -1,
          }
        : {}),
      ...(options?.resetReasoningBudgetMessage && config?.reasoningBudgetMessage === ""
        ? {
            // biome-ignore lint/style/useNamingConvention: API schema
            reasoning_budget_message: "",
          }
        : {}),
    }),
  });
  if (!res.ok) {
    throw new Error(await readFastApiError(res, "Failed to save model settings"));
  }
  const body = (await res.json()) as { overrides?: ApiModelOverrides };
  return body.overrides ?? {};
}
