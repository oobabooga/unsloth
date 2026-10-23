// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { normalizeTensorSplit } from "@/hooks/gpu-tensor-split";
import type { GpuIndexKind } from "@/hooks/use-gpu-info";
import { isExternalModelId } from "@/features/chat/external-providers";
import {
  DRAFT_N_MAX_SPEC_TYPES,
  DRAFTER_MODEL_SPEC_TYPES,
  MLX_ONLY_SPEC_TYPES,
  SEPARATE_DRAFT_MODEL_SPEC_TYPES,
} from "@/lib/speculative-modes";

export interface PerModelConfig {
  engineParallelism?: "tensor" | "pipeline" | "data";
  enginePrecision?: "auto" | "bf16" | "fp16" | "int4" | "int8" | "fp8";
  engineKvCacheDtype?: "auto" | "fp8";
  engineSeed?: number | null;
  engineGpuMemoryUtilization?: number | null;
  engine?: "auto" | "vllm" | "sglang";
  /** Default engine only; optional so older blobs parse, and anything but false is the 4-bit default. */
  loadIn4bit?: boolean;
  customContextLength: number | null;
  maxSeqLength: number | null;
  kvCacheDtype: string | null;
  mlxKvQuant?: MlxKvQuant | null;
  /** MLX only: run quantized projections with int8 activations, where the model supports it. */
  mlxInt8Prefill?: boolean;
  /** MLX only: share of the PLE tables / routed experts read from disk, "off", or null for Auto. */
  mlxPleOffload?: MlxOffload;
  mlxExpertOffload?: MlxOffload;
  speculativeType: string | null;
  specDraftNMax: number | null;
  /** KV cache dtype for the DRAFT context, sized and quantized independently of kvCacheDtype.
   *  Optional so older blobs parse. */
  specDraftCacheDtype?: string | null;
  /** MLX companion drafter, a repo id or local path. Optional so older blobs parse. */
  specDraftModel?: string | null;
  nParallel: number | null;
  reasoningBudget: number;
  reasoningBudgetMessage: string;
  nBatch: number | null;
  nUbatch: number | null;
  /** --load-mode; null lets the fit decide (`none` when the load fits, else no flag).
   *  Any value set here wins. */
  loadMode?: string | null;
  /** --ctx-checkpoints; null follows the llama.cpp default (32). */
  ctxCheckpoints?: number | null;
  /** --cache-ram in MiB; null follows the llama.cpp default (8192). */
  cacheRam?: number | null;
  tensorParallel: boolean;
  /** Load a vision GGUF without its mmproj, freeing the projector's VRAM. */
  disableVision: boolean;
  chatTemplateOverride: string | null;
  /** Pass-through llama-server argv tokens, appended after Unsloth's own flags. Three states: `undefined` means
     *  this copy never read the stored value, so a save leaves the server's alone (this is what kept CLI-set flags
     *  alive); `null` means the user cleared the box and must be sent as an explicit `[]`; a non-empty list is what
     *  to launch with. */
  llamaExtraArgs?: string[] | null;
  // GPU Memory controls (per-model, GGUF-only), optional so older blobs parse. Absent or null
  // selectedGpuIds means automatic.
  gpuMemoryMode?: "auto" | "manual";
  gpuLayers?: number;
  nCpuMoe?: number;
  selectedGpuIds?: number[] | null;
  selectedGpuIndexKind?: GpuIndexKind | null;
  /** --tensor-split bound to selectedGpuIds in picker order. `undefined` defers to the store, `null` = default. */
  tensorSplit?: number[] | null;
}

export const DEFAULT_PER_MODEL_CONFIG: PerModelConfig = {
  engine: "auto",
  enginePrecision: "auto",
  engineParallelism: "tensor",
  engineKvCacheDtype: "auto",
  engineSeed: null,
  engineGpuMemoryUtilization: null,
  loadIn4bit: true,
  customContextLength: null,
  maxSeqLength: null,
  kvCacheDtype: null,
  mlxKvQuant: null,
  mlxInt8Prefill: false,
  mlxPleOffload: null,
  mlxExpertOffload: null,
  speculativeType: null,
  specDraftNMax: null,
  specDraftCacheDtype: null,
  specDraftModel: null,
  nParallel: null,
  reasoningBudget: -1,
  reasoningBudgetMessage: "",
  nBatch: null,
  nUbatch: null,
  loadMode: null,
  ctxCheckpoints: null,
  cacheRam: null,
  tensorParallel: false,
  disableVision: false,
  chatTemplateOverride: null,
};

/** vLLM --seed / SGLang --random-seed; the backend bounds it to a non-negative int32. */
export function normalizeEngineSeed(value: unknown): number | null {
  return typeof value === "number" &&
    Number.isInteger(value) &&
    value >= 0 &&
    value <= 2 ** 31 - 1
    ? value
    : null;
}

/** Ceiling on the share of each GPU a managed engine may claim, as the backend bounds it. */
export function normalizeEngineGpuMemoryUtilization(
  value: unknown,
): number | null {
  return typeof value === "number" &&
    Number.isFinite(value) &&
    value >= 0.05 &&
    value <= 0.99
    ? Math.round(value * 1000) / 1000
    : null;
}

// Mirrors llama_server_args.py PARALLEL_MIN/MAX; null = follow the server-wide default.
export const N_PARALLEL_MIN = 1;
export const N_PARALLEL_MAX = 64;

// Mirrors vram_budget_settings.py VRAM_FRACTION_MIN/MAX/DEFAULT as whole percent (the slider is integer).
// Server-wide, so these only bound the control; the value lives in /api/settings/vram-budget.
export const VRAM_BUDGET_PERCENT_MIN = 10;
export const VRAM_BUDGET_PERCENT_MAX = 100;
export const VRAM_BUDGET_PERCENT_DEFAULT = 97;
// Tenths: a whole percent is ~245 MiB of context on a 24 GB card. Mirrors
// VRAM_FRACTION_DECIMALS = 3.
export const VRAM_BUDGET_PERCENT_STEP = 0.1;

/** Percent for the slider; the fraction is rebuilt at the API boundary. */
export function vramFractionToPercent(fraction: number): number {
  return Math.round(fraction * 1000) / 10;
}

export function vramPercentToFraction(percent: number): number {
  // Three decimals, so 0.975 cannot return as 0.9750000000000001 and read as "changed" after
  // a drag that ended where it began.
  return Math.round(percent * 10) / 1000;
}

// mirrors llama_server_args.py BATCH_MIN/MAX; null = follow the llama.cpp defaults (2048 / 512)
export const N_BATCH_MIN = 1;
export const N_BATCH_MAX = 65536;
// llama.cpp's own --batch-size default, which a blank control runs at: it still caps the
// micro-batch, so advisories have to reckon with it.
export const N_BATCH_LLAMA_DEFAULT = 2048;

export const MAX_SEQ_LENGTH_MIN = 128;
export const MAX_SEQ_LENGTH_MAX = 1048576;
export const MAX_SEQ_LENGTH_STEP = 128;
// App default for a model no local backend sizes itself. Every path falls back here, not
// to an active model's runtime value, so an unconfigured pane never inherits and OOMs.
export const DEFAULT_MAX_SEQ_LENGTH = 4096;
export const CONTEXT_LENGTH_MIN = 128;

// Reasons a Mac still cannot serve with MLX.
const NO_MLX_REASONS = new Set([
  "mlx_unavailable",
  "no_torch",
  "intel_mac",
  "rosetta",
  "detection_failed",
]);

/** Whether MLX will serve this model, and so whether MLX-only settings apply. Every non-GGUF model loads through
 *  MLX on a working Mac stack, so `!isGguf` alone would show these controls to CUDA users. */
export function isServedByMlx(
  isGguf: boolean,
  deviceType: string | null | undefined,
  chatOnlyReason?: string | null,
): boolean {
  return (
    !isGguf &&
    deviceType === "mac" &&
    !NO_MLX_REASONS.has(chatOnlyReason ?? "")
  );
}

/** Whether MLX serves the RESIDENT model. The platform cannot answer it alone: the worker picks
 *  NativeAudioBackend for a native-audio checkpoint before the MLX fast-path, so those load on Apple Silicon
 *  without MLX serving them. `loadedIsMlx` is the backend's own answer, and null there means nothing is loaded
 *  yet, where the platform is still the best available one. */
export function residentIsServedByMlx(
  isGguf: boolean,
  deviceType: string | null | undefined,
  chatOnlyReason: string | null | undefined,
  loadedIsMlx: boolean | null | undefined,
): boolean {
  return (
    isServedByMlx(isGguf, deviceType, chatOnlyReason) && loadedIsMlx !== false
  );
}

export function presetLoadSettingNames(
  isGguf: boolean,
  deviceType: string | null | undefined,
  chatOnlyReason?: string | null,
): string {
  if (isGguf) {
    return "context length, KV cache dtype, speculative decoding, GPU layers";
  }
  return isServedByMlx(isGguf, deviceType, chatOnlyReason)
    ? "max seq length, KV cache dtype"
    : "max seq length";
}

/** Whether llama.cpp serves the active model. `loadedIsGguf` is the backend's own answer; the rest identify a
 *  GGUF that has not reported one yet. A context length is not among them, since MLX reports one too, so it says
 *  a model is loaded, not which backend loaded it. An external provider is excluded because its id keeps a
 *  `.gguf` suffix while the flag describes a local load. */
export function isServedByLlamaCpp(x: {
  loadedIsGguf?: boolean | null;
  activeGgufVariant?: string | null;
  activeNativePathToken?: string | null;
  checkpoint?: string | null;
}): boolean {
  if (isExternalModelId(x.checkpoint)) return false;
  // A reported non-GGUF backend settles it: the variant and path token outlive the pick
  // that set them, so neither is evidence past a reload.
  if (x.loadedIsGguf === false) return false;
  return (
    x.loadedIsGguf === true ||
    x.activeGgufVariant != null ||
    x.activeNativePathToken != null ||
    String(x.checkpoint ?? "").toLowerCase().endsWith(".gguf")
  );
}

/** Whether the backend can resume a reply stopped mid-thought: llama-server or reported MLX. */
export function resumesThought(x: {
  loadedIsGguf?: boolean | null;
  loadedIsMlx?: boolean | null;
  activeGgufVariant?: string | null;
  activeNativePathToken?: string | null;
  checkpoint?: string | null;
}): boolean {
  return (
    isServedByLlamaCpp(x) ||
    (!isExternalModelId(x.checkpoint) && x.loadedIsMlx === true)
  );
}

/** The store's record of the context window a load left behind.
 *
 *  A window counts when the backend that reported it sized one. MLX always does, so its `context_length` stands
 *  alone even with no trained window in the config. Transformers sizes nothing and echoes the requested
 *  max_seq_length, so without a native length it contributes no window. One constructor because the four move
 *  together: a window without the backend that produced it is what made a context length read as proof of a GGUF.
 */
export function loadedContextFields(resp: {
  is_gguf?: boolean;
  is_mlx?: boolean;
  context_length?: number | null;
  native_context_length?: number | null;
  max_context_length?: number | null;
  context_length_enforced?: boolean | null;
  context_unbounded_when_batched?: boolean;
  parallel_slots?: number | null;
  mlx_context_budget?: number | null;
} | null): {
  loadedContextLength: number | null;
  maxContextLength: number | null;
  nativeContextLength: number | null;
  loadedIsGguf: boolean | null;
  loadedIsMlx: boolean | null;
  loadedContextEnforced: boolean | null;
  loadedContextUnboundedWhenBatched: boolean;
  loadedParallelSlots: number | null;
  loadedContextBudget: number | null;
} {
  if (!resp) {
    return {
      loadedContextLength: null,
      maxContextLength: null,
      nativeContextLength: null,
      loadedIsGguf: null,
      loadedIsMlx: null,
      loadedContextEnforced: null,
      loadedContextUnboundedWhenBatched: false,
      loadedParallelSlots: null,
      loadedContextBudget: null,
    };
  }
  const isGguf = resp.is_gguf ?? false;
  // Unknown, not a default: a response omits it when reading the model's window failed.
  const loaded = resp.context_length ?? null;
  if (!isGguf && !resp.is_mlx && resp.native_context_length == null) {
    return {
      loadedContextLength: null,
      maxContextLength: null,
      nativeContextLength: null,
      loadedIsGguf: false,
      loadedIsMlx: resp.is_mlx ?? null,
      loadedContextEnforced: null,
      loadedContextUnboundedWhenBatched: false,
      loadedParallelSlots: null,
      loadedContextBudget: null,
    };
  }
  return {
    loadedContextLength: loaded,
    maxContextLength: resp.max_context_length ?? loaded,
    nativeContextLength: resp.native_context_length ?? null,
    loadedIsGguf: isGguf,
    // The backend's own answer, so a checkpoint the worker serves off the MLX path
    // (native audio) is not taken for MLX by the platform alone.
    loadedIsMlx: resp.is_mlx ?? null,
    // llama.cpp allocates what it reports, so GGUF is enforced by construction.
    // Everything else answers for itself, or says nothing.
    loadedContextEnforced: isGguf ? true : (resp.context_length_enforced ?? null),
    // Read from the same response as the other two so the three never mix across loads.
    loadedContextUnboundedWhenBatched: isGguf
      ? false
      : (resp.context_unbounded_when_batched ?? false),
    loadedParallelSlots: resp.parallel_slots ?? null,
    loadedContextBudget: isGguf ? null : (resp.mlx_context_budget ?? null),
  };
}

// Matches studio/backend/core/inference/llama_cpp.py _valid_cache_types (f16 is the UI default).
export const KV_CACHE_DTYPES = [
  "bf16",
  "q8_0",
  "q4_0",
  "q4_1",
  "q5_0",
  "q5_1",
  "iq4_nl",
  "f32",
] as const;

/** Menu entries for a llama.cpp backend. CUDA, ROCm and Metal have no iq4_nl FlashAttention kernel, so attention
 *  runs on the CPU there; Vulkan and CPU builds run it. A selected value stays listed so the trigger can show it. */
export function kvCacheDtypeOptions(
  backend: string | null,
  selected: string | null | undefined,
): readonly string[] {
  if (backend === "vulkan" || backend === "cpu") return KV_CACHE_DTYPES;
  return KV_CACHE_DTYPES.filter((dtype) => dtype !== "iq4_nl" || dtype === selected);
}

export const MLX_KV_QUANTS = [
  "8",
  "6",
  "5",
  "4",
  "3",
  "2",
  "tq-4",
  "tq-3.5",
  "tq-3",
  "tq-2",
] as const;
export type MlxKvQuant = (typeof MLX_KV_QUANTS)[number];
const VALID_MLX_KV_QUANTS = new Set<string>(MLX_KV_QUANTS);

export function mlxKvQuantLabel(quant: string): string {
  return quant.startsWith("tq-") ? `TurboQuant ${quant.slice(3)}-bit` : `${quant}-bit`;
}

/** A bare width only ever meant mx.quantize; null is how a saved Auto spells itself. */
export function normalizeMlxKvQuant(
  value: unknown,
  supersededBits?: unknown,
): MlxKvQuant | null {
  if (typeof value === "string") {
    // Trimmed and lower-cased to match the backend's reader, or one row means two settings.
    const named = value.trim().toLowerCase();
    return VALID_MLX_KV_QUANTS.has(named) ? (named as MlxKvQuant) : null;
  }
  if (value !== undefined) return null;
  if (typeof supersededBits !== "number" || !Number.isFinite(supersededBits)) return null;
  const name = String(supersededBits);
  return VALID_MLX_KV_QUANTS.has(name) ? (name as MlxKvQuant) : null;
}
const VALID_KV_CACHE_DTYPES = new Set<string>(KV_CACHE_DTYPES);

export type MlxOffload = "off" | number | null;

export function normalizeMlxOffload(value: unknown): MlxOffload {
  if (value === "off") return "off";
  return typeof value === "number" && Number.isFinite(value) && value > 0 && value <= 1
    ? value
    : null;
}

// llama-server's --load-mode enum in --help order. "auto" is the default: the UI shows it, storage keeps null and
// the backend emits no flag. Never sent verbatim; builds like b10360 reject "auto" as a value.
export const LOAD_MODES = [
  "auto",
  "none",
  "mmap",
  "mlock",
  "mmap+mlock",
  "dio",
] as const;
export const LOAD_MODE_DEFAULT = "auto";
const VALID_LOAD_MODES = new Set<string>(LOAD_MODES);

// --ctx-checkpoints: per-slot sliding-window snapshots, so the count is small. 0 disables
// them; the ceiling is a sanity bound, not an upstream one.
export const CTX_CHECKPOINTS_MIN = 0;
export const CTX_CHECKPOINTS_MAX = 256;
export const CTX_CHECKPOINTS_LLAMA_DEFAULT = 32;

// --cache-ram in MiB: -1 is "no limit" and 0 disables the host prompt cache, so the floor
// is -1. The 1 TiB ceiling fails a stray keystroke before the child does.
export const CACHE_RAM_MIN = -1;
export const CACHE_RAM_MAX = 1024 * 1024;
export const CACHE_RAM_LLAMA_DEFAULT = 8192;

export {
  DRAFT_N_MAX_SPEC_TYPES,
  SEPARATE_DRAFT_MODEL_SPEC_TYPES,
  SPECULATIVE_TYPES,
} from "@/lib/speculative-modes";

// Mirrors MAX_CHAT_TEMPLATE_BYTES in backend/picker/schemas.py.
export const MAX_CHAT_TEMPLATE_BYTES = 512 * 1024;
export const MAX_REASONING_BUDGET_MESSAGE_BYTES = 8_192;

export function isReasoningBudgetMessageValid(value: string): boolean {
  if (value.includes("\0")) return false;
  for (let i = 0; i < value.length; i += 1) {
    const code = value.charCodeAt(i);
    if (code >= 0xd800 && code <= 0xdbff) {
      const next = value.charCodeAt(i + 1);
      if (next < 0xdc00 || next > 0xdfff) return false;
      i += 1;
    } else if (code >= 0xdc00 && code <= 0xdfff) {
      return false;
    }
  }
  return (
    new TextEncoder().encode(value).byteLength <=
    MAX_REASONING_BUDGET_MESSAGE_BYTES
  );
}

type RawConfig = Partial<PerModelConfig> & { mlxKvBits?: unknown };

/** Keep only a list of strings, preserving the three states above. Anything that is not an array is "not loaded"
 *  (`undefined`), never "cleared": the flags a wipe would throw away are invisible in this panel until the row
 *  reads them. */
function normalizeLlamaExtraArgs(value: unknown): string[] | null | undefined {
  if (value === null) {
    return null;
  }
  if (!Array.isArray(value)) {
    return undefined;
  }
  const tokens = value.filter((entry): entry is string => typeof entry === "string");
  return tokens.length > 0 ? tokens : null;
}

function normalizeGpuFields(partial: RawConfig): {
  gpuMemoryMode?: "auto" | "manual";
  gpuLayers?: number;
  nCpuMoe?: number;
  selectedGpuIds?: number[] | null;
  selectedGpuIndexKind?: GpuIndexKind | null;
} {
  const out: {
    gpuMemoryMode?: "auto" | "manual";
    gpuLayers?: number;
    nCpuMoe?: number;
    selectedGpuIds?: number[] | null;
    selectedGpuIndexKind?: GpuIndexKind | null;
  } = {};
  // Only "manual" is a real override; persisting "auto" would stop the model following the global.
  if (partial.gpuMemoryMode === "manual") {
    out.gpuMemoryMode = "manual";
  }
  if (
    typeof partial.gpuLayers === "number" &&
    Number.isFinite(partial.gpuLayers)
  ) {
    out.gpuLayers = Math.trunc(partial.gpuLayers);
  }
  if (
    typeof partial.nCpuMoe === "number" &&
    Number.isFinite(partial.nCpuMoe) &&
    partial.nCpuMoe >= 0
  ) {
    out.nCpuMoe = Math.trunc(partial.nCpuMoe);
  }
  if (partial.selectedGpuIds === null) {
    out.selectedGpuIds = null;
  } else if (
    Array.isArray(partial.selectedGpuIds) &&
    partial.selectedGpuIds.every(
      (n) => typeof n === "number" && Number.isFinite(n),
    )
  ) {
    out.selectedGpuIds = partial.selectedGpuIds.map((n) => Math.trunc(n));
  }
  if (
    partial.selectedGpuIndexKind === "physical" ||
    partial.selectedGpuIndexKind === "vulkan" ||
    partial.selectedGpuIndexKind === null
  ) {
    out.selectedGpuIndexKind = partial.selectedGpuIndexKind;
  }
  return out;
}

function canonicalizeSpeculativeType(value: string): string | null {
  const s = value.trim().toLowerCase();
  if (!s) {
    return null;
  }
  // Kept: an MLX model's explicit Auto beats a standing "off". GGUF saves fold it via storedSpeculativeAuto.
  if (s === "auto" || s === "default") {
    return "auto";
  }
  // _LEGACY_SPEC_MODE_MAP's four. A stored override arrives raw, and the null below
  // is "follow the global preference", which would enable a drafter under Auto.
  if (s === "off" || s === "none" || s === "disable" || s === "disabled") {
    return "off";
  }
  if (s === "mtp" || s === "draft-mtp") {
    return "mtp";
  }
  if (s === "dspark" || s === "draft-dspark") {
    return "dspark";
  }
  if (s === "dflash" || s === "draft-dflash") {
    return "dflash";
  }
  if (s === "ngram" || s === "ngram-mod" || s === "ngram-simple") {
    return "ngram";
  }
  if (s === "mtp+ngram" || (MLX_ONLY_SPEC_TYPES as readonly string[]).includes(s)) {
    return s;
  }
  return null;
}

/** The config as storage keeps it: GGUF reads Auto as "follow the standing preference", spelled null,
 *  while an MLX model stores its explicit Auto. */
export function storedSpeculativeAuto<T extends Pick<PerModelConfig, "speculativeType">>(
  config: T,
  isMlx: boolean,
): T {
  return !isMlx && config.speculativeType === "auto"
    ? { ...config, speculativeType: null }
    : config;
}

/** An edit to draft tokens or the drafter. Made while the mode is unset, it pins the shown mode:
 *  storage keeps either value only beside a mode that reads it. */
export function pinSpeculativeMode(
  config: Pick<PerModelConfig, "speculativeType">,
  shownMode: string,
  patch: Partial<PerModelConfig>,
): Partial<PerModelConfig> {
  return config.speculativeType == null
    ? { ...patch, speculativeType: shownMode }
    : patch;
}

/** Canonicalize a stored --load-mode, or null to follow the llama.cpp default. "auto" folds
 *  to null: it IS the default, so storing it would pin a value the build may redefine. */
export function canonicalizeLoadMode(value: unknown): string | null {
  if (typeof value !== "string") {
    return null;
  }
  // Whitespace and case only: "mmap + mlock" is not a spelling llama-server accepts, so it is
  // refused rather than repaired.
  const mode = value.trim().toLowerCase();
  if (!mode || mode === LOAD_MODE_DEFAULT) {
    return null;
  }
  return VALID_LOAD_MODES.has(mode) ? mode : null;
}

function normalizeIntegerInRange(
  value: unknown,
  min: number,
  max: number,
): number | null {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return null;
  }
  return Math.max(min, Math.min(max, Math.round(value)));
}

export function normalizeCtxCheckpoints(value: unknown): number | null {
  return normalizeIntegerInRange(value, CTX_CHECKPOINTS_MIN, CTX_CHECKPOINTS_MAX);
}

export function normalizeCacheRam(value: unknown): number | null {
  return normalizeIntegerInRange(value, CACHE_RAM_MIN, CACHE_RAM_MAX);
}

export function normalizeMaxSeqLength(value: unknown): number | null {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    return null;
  }
  const snapped = Math.round(value / MAX_SEQ_LENGTH_STEP) * MAX_SEQ_LENGTH_STEP;
  return Math.max(MAX_SEQ_LENGTH_MIN, Math.min(MAX_SEQ_LENGTH_MAX, snapped));
}

/** The context a saved record pins, whichever field it was written in.
 *
 *  MLX's pin moved into `customContextLength` beside llama.cpp's, while transformers keeps its own in
 *  `maxSeqLength` and a record written before the move still carries an MLX pin there. The same record is read on
 *  a host that serves it with a different backend, so every read has to honour both. A *saved* record only:
 *  `currentRuntimePerModelConfig` builds the same shape from the live store, where `maxSeqLength` is the length
 *  the model resolved to rather than one anybody chose; read that through `customContextLength` alone.
 */
export function savedContextPin(config: {
  customContextLength?: number | null;
  maxSeqLength?: number | null;
}): number | null {
  return (
    config.customContextLength ??
    normalizeMaxSeqLength(config.maxSeqLength ?? null)
  );
}

/** The patch that pins a context for a non-GGUF target, on the backend serving it. An edit leaves a pin in
 *  exactly one field, clearing whichever the record arrived with: a record holding both loads at a different
 *  length depending on who asked, since the picker resolves `customContextLength` first and the API's
 *  auto-switch load `maxSeqLength`. */
export function contextPinPatch(value: number, isMlx: boolean): Partial<PerModelConfig> {
  // Held to what a load may ask for, but not rounded to the control's step: a pin taken
  // from a resolved window need not sit on that grid.
  const pin = boundContextPin(value) ?? MAX_SEQ_LENGTH_MIN;
  return isMlx
    ? { customContextLength: pin, maxSeqLength: null }
    : { customContextLength: null, maxSeqLength: pin };
}

function boundContextPin(value: unknown): number | null {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    return null;
  }
  return Math.max(
    MAX_SEQ_LENGTH_MIN,
    Math.min(MAX_SEQ_LENGTH_MAX, Math.floor(value)),
  );
}

export function floorMaxSeqLength(value: unknown): number | null {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    return null;
  }
  const snapped = Math.floor(value / MAX_SEQ_LENGTH_STEP) * MAX_SEQ_LENGTH_STEP;
  return Math.max(MAX_SEQ_LENGTH_MIN, Math.min(MAX_SEQ_LENGTH_MAX, snapped));
}

function canUseStorage(): boolean {
  return typeof window !== "undefined";
}

// Whether Run Settings shows its advanced section is a standing preference, not per-model.
// Closed until asked for.
export const ADVANCED_SETTINGS_OPEN_KEY = "unsloth_model_advanced_settings";

function loadAdvancedSettingsOpen(): boolean | null {
  if (!canUseStorage()) {
    return null;
  }
  try {
    const raw = localStorage.getItem(ADVANCED_SETTINGS_OPEN_KEY);
    return raw === "true" ? true : raw === "false" ? false : null;
  } catch {
    return null;
  }
}

// Set only when a write is refused, so the switch keeps working with storage disabled or full. Cleared by the
// next write that sticks. `stored` is what storage held then, the one signal that tells a later write by someone
// else apart.
let unpersisted: { open: boolean; stored: boolean | null } | null = null;
const advancedOpenListeners = new Set<() => void>();

/** Null until the switch is used, so an untouched panel may open the section for a model with non-default
 *  advanced values. Read straight from storage, not cached: a write from another tab while every panel was
 *  unmounted has no listener and no replayed event. */
export function readAdvancedSettingsOpen(): boolean | null {
  const stored = loadAdvancedSettingsOpen();
  if (!unpersisted) {
    return stored;
  }
  // Storage moved since the refused write, so a newer choice outranks the fallback. Checked on
  // read, not on the storage event, so it holds for an event that landed while unmounted.
  if (stored !== unpersisted.stored) {
    unpersisted = null;
    return stored;
  }
  return unpersisted.open;
}

/** True when the choice reached storage. */
function writeAdvancedSettingsOpen(open: boolean): boolean {
  if (!canUseStorage()) {
    return false;
  }
  try {
    localStorage.setItem(ADVANCED_SETTINGS_OPEN_KEY, open ? "true" : "false");
    return true;
  } catch {
    return false;
  }
}

export function saveAdvancedSettingsOpen(open: boolean): void {
  unpersisted = writeAdvancedSettingsOpen(open)
    ? null
    : { open, stored: loadAdvancedSettingsOpen() };
  // Run Settings is mounted on several surfaces at once and the sidebar copy stays mounted
  // while collapsed, so tell them all rather than let them keep a mount-time snapshot.
  for (const listener of [...advancedOpenListeners]) {
    listener();
  }
}

/** Follow the preference while mounted, including a change from another tab. */
export function subscribeAdvancedSettingsOpen(
  onChange: () => void,
): () => void {
  advancedOpenListeners.add(onChange);
  if (!canUseStorage()) {
    return () => {
      advancedOpenListeners.delete(onChange);
    };
  }
  const onStorage = (event: StorageEvent) => {
    // A null key is a clear(), which drops this preference too.
    if (event.key === null || event.key === ADVANCED_SETTINGS_OPEN_KEY) {
      onChange();
    }
  };
  window.addEventListener("storage", onStorage);
  return () => {
    advancedOpenListeners.delete(onChange);
    window.removeEventListener("storage", onStorage);
  };
}

function serializedByteLength(value: string): number {
  return typeof TextEncoder !== "undefined"
    ? new TextEncoder().encode(value).byteLength
    : value.length;
}

export function chatTemplateByteLength(value: string): number {
  return serializedByteLength(value);
}

export function isChatTemplateWithinLimit(value: string): boolean {
  return chatTemplateByteLength(value) <= MAX_CHAT_TEMPLATE_BYTES;
}

function normalizeV1(partial: RawConfig): PerModelConfig {
  const rawSpecType =
    typeof partial.speculativeType === "string"
      ? canonicalizeSpeculativeType(partial.speculativeType)
      : null;
  const speculativeType =
    rawSpecType ?? DEFAULT_PER_MODEL_CONFIG.speculativeType;
  const specDraftNMax =
    speculativeType != null &&
    DRAFT_N_MAX_SPEC_TYPES.has(speculativeType) &&
    typeof partial.specDraftNMax === "number" &&
    Number.isFinite(partial.specDraftNMax)
      ? Math.max(1, Math.min(16, Math.round(partial.specDraftNMax)))
      : null;
  // Tied to the mode like specDraftNMax: a dtype stored under a mode with no separate drafter
  // shows a row for a context that never exists.
  const specDraftCacheDtype =
    speculativeType != null &&
    SEPARATE_DRAFT_MODEL_SPEC_TYPES.has(speculativeType) &&
    typeof partial.specDraftCacheDtype === "string" &&
    VALID_KV_CACHE_DTYPES.has(partial.specDraftCacheDtype)
      ? partial.specDraftCacheDtype
      : null;
  const specDraftModel =
    speculativeType != null &&
    DRAFTER_MODEL_SPEC_TYPES.has(speculativeType) &&
    typeof partial.specDraftModel === "string" &&
    partial.specDraftModel.trim().length > 0 &&
    partial.specDraftModel.trim().length <= 1024
      ? partial.specDraftModel.trim()
      : null;
  return {
    engineParallelism: partial.engineParallelism === "pipeline" || partial.engineParallelism === "data"
      ? partial.engineParallelism : "tensor",
    enginePrecision: ["bf16", "fp16", "int4", "int8", "fp8"].includes(
      partial.enginePrecision ?? "",
    )
      ? partial.enginePrecision
      : "auto",
    engine:
      partial.engine === "vllm" || partial.engine === "sglang"
        ? partial.engine
        : "auto",
    engineKvCacheDtype: partial.engineKvCacheDtype === "fp8" ? "fp8" : "auto",
    engineSeed: normalizeEngineSeed(partial.engineSeed),
    engineGpuMemoryUtilization: normalizeEngineGpuMemoryUtilization(
      partial.engineGpuMemoryUtilization,
    ),
    loadIn4bit: partial.loadIn4bit !== false,
    customContextLength:
      typeof partial.customContextLength === "number" &&
      Number.isFinite(partial.customContextLength) &&
      partial.customContextLength > 0
        ? Math.max(CONTEXT_LENGTH_MIN, Math.floor(partial.customContextLength))
        : null,
    maxSeqLength: normalizeMaxSeqLength(partial.maxSeqLength),
    mlxKvQuant: normalizeMlxKvQuant(partial.mlxKvQuant, partial.mlxKvBits),
    mlxInt8Prefill: partial.mlxInt8Prefill === true,
    mlxPleOffload: normalizeMlxOffload(partial.mlxPleOffload),
    mlxExpertOffload: normalizeMlxOffload(partial.mlxExpertOffload),
    kvCacheDtype:
      typeof partial.kvCacheDtype === "string" &&
      VALID_KV_CACHE_DTYPES.has(partial.kvCacheDtype)
        ? partial.kvCacheDtype
        : null,
    speculativeType,
    specDraftNMax,
    specDraftCacheDtype,
    specDraftModel,
    loadMode: canonicalizeLoadMode(partial.loadMode),
    ctxCheckpoints: normalizeCtxCheckpoints(partial.ctxCheckpoints),
    cacheRam: normalizeCacheRam(partial.cacheRam),
    nParallel:
      typeof partial.nParallel === "number" &&
      Number.isFinite(partial.nParallel)
        ? Math.max(
            N_PARALLEL_MIN,
            Math.min(N_PARALLEL_MAX, Math.round(partial.nParallel)),
          )
        : null,
    nBatch:
      typeof partial.nBatch === "number" && Number.isFinite(partial.nBatch)
        ? Math.max(N_BATCH_MIN, Math.min(N_BATCH_MAX, Math.round(partial.nBatch)))
        : null,
    nUbatch:
      typeof partial.nUbatch === "number" && Number.isFinite(partial.nUbatch)
        ? Math.max(N_BATCH_MIN, Math.min(N_BATCH_MAX, Math.round(partial.nUbatch)))
        : null,
    reasoningBudget:
      typeof partial.reasoningBudget === "number" &&
      Number.isFinite(partial.reasoningBudget)
        ? Math.max(
            -1,
            Math.min(2_147_483_647, Math.trunc(partial.reasoningBudget)),
          )
        : DEFAULT_PER_MODEL_CONFIG.reasoningBudget,
    reasoningBudgetMessage:
      typeof partial.reasoningBudgetMessage === "string" &&
      isReasoningBudgetMessageValid(partial.reasoningBudgetMessage)
        ? partial.reasoningBudgetMessage
        : DEFAULT_PER_MODEL_CONFIG.reasoningBudgetMessage,
    tensorParallel:
      typeof partial.tensorParallel === "boolean"
        ? partial.tensorParallel
        : DEFAULT_PER_MODEL_CONFIG.tensorParallel,
    disableVision:
      typeof partial.disableVision === "boolean"
        ? partial.disableVision
        : DEFAULT_PER_MODEL_CONFIG.disableVision,
    chatTemplateOverride:
      typeof partial.chatTemplateOverride === "string" &&
      isChatTemplateWithinLimit(partial.chatTemplateOverride)
        ? partial.chatTemplateOverride
        : null,
    llamaExtraArgs: normalizeLlamaExtraArgs(partial.llamaExtraArgs),
    ...normalizeGpuFields(partial),
    tensorSplit: normalizeTensorSplit(partial.tensorSplit, partial.selectedGpuIds),
  };
}

/** A config in the exact shape storage keeps, so a UI value storage drops cannot read as non-default. */
export function normalizePerModelConfig(raw: unknown): PerModelConfig {
  return normalizeV1(
    raw && typeof raw === "object" && !Array.isArray(raw) ? (raw as RawConfig) : {},
  );
}

export function isDefaultConfig(config: PerModelConfig): boolean {
  return (
    (config.engine ?? "auto") === "auto" &&
    (config.enginePrecision ?? "auto") === "auto" &&
    (config.engineParallelism ?? "tensor") === "tensor" &&
    (config.engineKvCacheDtype ?? "auto") === "auto" &&
    config.engineSeed == null &&
    config.engineGpuMemoryUtilization == null &&
    config.loadIn4bit !== false &&
    config.customContextLength == null &&
    config.maxSeqLength == null &&
    (config.kvCacheDtype ?? null) === DEFAULT_PER_MODEL_CONFIG.kvCacheDtype &&
    (config.mlxKvQuant ?? null) === DEFAULT_PER_MODEL_CONFIG.mlxKvQuant &&
    !config.mlxInt8Prefill &&
    config.mlxPleOffload == null &&
    config.mlxExpertOffload == null &&
    config.speculativeType === DEFAULT_PER_MODEL_CONFIG.speculativeType &&
    config.specDraftNMax == null &&
    config.nParallel == null &&
    config.reasoningBudget === DEFAULT_PER_MODEL_CONFIG.reasoningBudget &&
    config.reasoningBudgetMessage ===
      DEFAULT_PER_MODEL_CONFIG.reasoningBudgetMessage &&
    config.nBatch == null &&
    config.nUbatch == null &&
    // Compared against null, not truth: 0 checkpoints and a 0 or -1 cache are values.
    (config.specDraftCacheDtype ?? null) === null &&
    (config.specDraftModel ?? null) === null &&
    (config.loadMode ?? null) === null &&
    config.ctxCheckpoints == null &&
    config.cacheRam == null &&
    Boolean(config.tensorParallel) ===
      Boolean(DEFAULT_PER_MODEL_CONFIG.tensorParallel) &&
    Boolean(config.disableVision) ===
      Boolean(DEFAULT_PER_MODEL_CONFIG.disableVision) &&
    (config.chatTemplateOverride ?? null) === null &&
    (config.llamaExtraArgs == null || config.llamaExtraArgs.length === 0) &&
    gpuFieldsAtDefault(config)
  );
}

// GPU knobs are "default" when mode is Auto with no explicit choice: gpuLayers < 0 or absent,
// nCpuMoe 0 or absent, selectedGpuIds null or absent.
function gpuFieldsAtDefault(config: PerModelConfig): boolean {
  return (
    (config.gpuMemoryMode ?? "auto") === "auto" &&
    (config.gpuLayers == null || config.gpuLayers < 0) &&
    (config.nCpuMoe == null || config.nCpuMoe === 0) &&
    config.selectedGpuIds == null &&
    config.tensorSplit == null
  );
}
