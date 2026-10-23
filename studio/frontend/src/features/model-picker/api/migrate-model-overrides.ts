// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// One-time import of the per-model settings older builds kept in this browser. The server's map is
// the only copy now, so each record is sent there, filling only the fields the server row lacks
// (a value already there is the newer one), and leaves this browser once the server has it.

import { sanitizeStoredExtraArgs } from "../model-config/llama-extra-args";
import {
  cachedRepoConfigId,
  ggufVariantFromStorageKey,
  modelIdFromStorageKey,
  modelStorageKey,
} from "../model-config/model-identity";
import {
  type PerModelConfig,
  isDefaultConfig,
  normalizePerModelConfig,
  storedSpeculativeAuto,
} from "../model-config/per-model-config";
import { loadManagedLlamaFlags } from "./llama-flags";
import { putModelOverride } from "./model-overrides";

const CONFIGS_KEY = "unsloth_model_configs";
// The store before that one, and the flag saying it was already merged into it.
const LEGACY_KEY = "unsloth_load_settings";
const LEGACY_MERGED_FLAG = "unsloth_model_configs_migrated";
// Left by the builds that copied these records to the server field by field.
const BACKFILL_FLAG = "unsloth_model_overrides_backfilled_v3";
const LEGACY_FIELDS = [
  "kvCacheDtype",
  "speculativeType",
  "specDraftNMax",
  "tensorParallel",
  "disableVision",
  "gpuMemoryMode",
  "gpuLayers",
  "nCpuMoe",
  "selectedGpuIds",
] as const;

type BrowserRecord = {
  modelId: string;
  ggufVariant: string | null;
  config: PerModelConfig;
};

function readObject(key: string): Record<string, unknown> {
  try {
    const parsed: unknown = JSON.parse(localStorage.getItem(key) ?? "null");
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)
      : {};
  } catch {
    return {};
  }
}

function legacyRecordConfig(raw: Record<string, unknown>): PerModelConfig {
  const fields: Record<string, unknown> = {
    customContextLength: raw.contextLength,
  };
  for (const name of LEGACY_FIELDS) {
    fields[name] = raw[name];
  }
  // Written before any MLX control, so its Auto is GGUF's.
  return storedSpeculativeAuto(normalizePerModelConfig(fields), false);
}

/** The store before unsloth_model_configs, keyed `modelId::variant`; the newer store wins a model
 *  both hold. */
function addLegacyRecords(records: Map<string, BrowserRecord>): void {
  for (const [key, raw] of Object.entries(readObject(LEGACY_KEY))) {
    const separator = key.lastIndexOf("::");
    const modelId = separator >= 0 ? key.slice(0, separator) : key;
    const ggufVariant =
      separator >= 0 ? key.slice(separator + 2) || null : null;
    const storageKey = modelStorageKey(modelId, ggufVariant);
    if (modelId && raw && typeof raw === "object" && !records.has(storageKey)) {
      records.set(storageKey, {
        modelId,
        ggufVariant,
        config: legacyRecordConfig(raw as Record<string, unknown>),
      });
    }
  }
}

function browserRecords(): BrowserRecord[] {
  const records = new Map<string, BrowserRecord>();
  for (const [key, raw] of Object.entries(readObject(CONFIGS_KEY))) {
    const modelId = modelIdFromStorageKey(key);
    if (modelId) {
      const ggufVariant = ggufVariantFromStorageKey(key) || null;
      records.set(modelStorageKey(modelId, ggufVariant), {
        modelId,
        ggufVariant,
        config: normalizePerModelConfig(raw),
      });
    }
  }
  if (!localStorage.getItem(LEGACY_MERGED_FLAG)) {
    addLegacyRecords(records);
  }
  return [...records.values()].filter(
    (record) => !isDefaultConfig(record.config),
  );
}

/** A stored list becomes an explicit one the route validates strictly, so a flag denied since it
 *  was saved would refuse the whole record. */
async function withoutRefusedFlags(
  records: BrowserRecord[],
): Promise<BrowserRecord[]> {
  if (
    !records.some((record) => (record.config.llamaExtraArgs?.length ?? 0) > 0)
  ) {
    return records;
  }
  const managed = await loadManagedLlamaFlags().catch(() => null);
  return records.map((record) => {
    const tokens = record.config.llamaExtraArgs;
    if (!tokens || tokens.length === 0) {
      return record;
    }
    const kept = sanitizeStoredExtraArgs(
      tokens,
      managed?.managed ?? new Set(),
      {
        maxBytes: managed?.maxBytes,
        windowsCommandBudget: managed?.windowsCommandBudget,
      },
    );
    return {
      ...record,
      config: {
        ...record.config,
        llamaExtraArgs: kept.length > 0 ? kept : null,
      },
    };
  });
}

/** Sends this browser's records to the server and removes them here. Never rejects: a record the
 *  server did not take stays in this browser for the next start. */
export async function importBrowserModelConfigs(): Promise<void> {
  let records: BrowserRecord[];
  try {
    records = await withoutRefusedFlags(browserRecords());
  } catch {
    // Storage denied: there is nothing this browser can hand over.
    return;
  }
  const kept: Record<string, PerModelConfig> = {};
  // A cached repo is saved under its repo id, since a snapshot-path row would outrank it on every
  // load. A record kept under the path goes last, so one already under the repo id fills first.
  const repoIdOf = (record: BrowserRecord) =>
    cachedRepoConfigId(record.modelId, record.ggufVariant);
  records.sort(
    (a, b) => Number(repoIdOf(a) !== null) - Number(repoIdOf(b) !== null),
  );
  for (const record of records) {
    const repoId = repoIdOf(record);
    try {
      await putModelOverride(
        repoId ?? record.modelId,
        repoId ? null : record.ggufVariant,
        record.config,
        { fillAbsentFields: true },
      );
    } catch {
      kept[modelStorageKey(record.modelId, record.ggufVariant)] = record.config;
    }
  }
  try {
    if (Object.keys(kept).length > 0) {
      localStorage.setItem(CONFIGS_KEY, JSON.stringify(kept));
    } else {
      localStorage.removeItem(CONFIGS_KEY);
    }
    for (const key of [LEGACY_KEY, LEGACY_MERGED_FLAG, BACKFILL_FLAG]) {
      localStorage.removeItem(key);
    }
  } catch {
    // The records are on the server; a copy left here is only sent again.
  }
}
