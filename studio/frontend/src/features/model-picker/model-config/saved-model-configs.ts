// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A model's saved load settings live in one place, the server's override map, which every load
// path applies. This module holds the last copy read from it for the synchronous readers, and
// sends each save and forget there.

import {
  type ApiModelOverrides,
  type PutModelOverrideOptions,
  fetchModelOverrides,
  fromApiOverride,
  putModelOverride,
  resolveStoredOverride,
} from "../api/model-overrides";
import {
  ggufQuantLabel,
  isStandaloneGgufPath,
  normalizeGgufVariantIdentity,
  normalizeModelIdentity,
  publicModelId,
  splitQuantSuffix,
} from "./model-identity";
import {
  DEFAULT_PER_MODEL_CONFIG,
  type PerModelConfig,
} from "./per-model-config";

export interface ResolvedPerModelConfig {
  config: PerModelConfig;
  remembered: boolean;
}

let overrides: ApiModelOverrides | null = null;
const listeners = new Set<() => void>();
// Reads and writes go out one at a time, so an older answer never replaces a newer one.
let queue: Promise<unknown> = Promise.resolve();
let firstRead: Promise<void> | null = null;

function publish(next: ApiModelOverrides): void {
  overrides = next;
  for (const listener of [...listeners]) {
    listener();
  }
}

function enqueue<T>(task: () => Promise<T>): Promise<T> {
  const run = queue.then(task, task);
  queue = run.catch(() => undefined);
  return run;
}

export function subscribeSavedModelConfigs(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

/** The map last read from the server, a new object after every change; null before the first read. */
export function readSavedModelConfigs(): ApiModelOverrides | null {
  return overrides;
}

/** Re-reads the map. Rejects when the server cannot answer, leaving the last copy in place. */
export function refreshSavedModelConfigs(): Promise<void> {
  return enqueue(async () => publish(await fetchModelOverrides()));
}

/** Settles once the map has been read, or the read failed; a later call retries a failure. */
export function ensureSavedModelConfigs(): Promise<void> {
  if (overrides !== null) {
    return Promise.resolve();
  }
  firstRead ??= refreshSavedModelConfigs().finally(() => {
    firstRead = null;
  });
  return firstRead.catch(() => undefined);
}

/** After sign-in: send what older builds kept in this browser, then read the account's map. */
export function loadSavedModelConfigs(): Promise<void> {
  return enqueue(async () => {
    // Loaded on demand: only this one-time step needs the import and its flag catalogue.
    const { importBrowserModelConfigs } = await import(
      "../api/migrate-model-overrides"
    );
    await importBrowserModelConfigs();
    publish(await fetchModelOverrides());
  }).catch(() => {
    // Readers keep the defaults until a later read succeeds.
  });
}

export async function saveModelConfig(
  modelId: string,
  ggufVariant: string | null | undefined,
  config: PerModelConfig,
  options?: PutModelOverrideOptions,
): Promise<void> {
  await enqueue(async () =>
    publish(await putModelOverride(modelId, ggufVariant, config, options)),
  );
}

/** Forgets every spelling the server keeps for this model. `modelId` may be a stored key. */
export async function forgetModelConfig(
  modelId: string,
  ggufVariant?: string | null,
): Promise<void> {
  await enqueue(async () =>
    publish(await putModelOverride(modelId, ggufVariant, null)),
  );
}

/** The keys a load tries, in order. Mirrors override_lookup_candidates in
 *  openai_auto_switch_settings.py, so a row reads here as the load that applies it reads it. */
export function overrideLookupCandidates(
  loadId: string,
  aliasId: string | null,
  ggufVariant: string | null,
): string[] {
  const fileVariant =
    !ggufVariant && loadId.toLowerCase().endsWith(".gguf")
      ? ggufQuantLabel(loadId.replace(/\\/g, "/").split("/").pop() ?? loadId)
      : null;
  const ordered = [
    ggufVariant ? `${loadId}:${ggufVariant}` : null,
    aliasId && ggufVariant ? `${aliasId}:${ggufVariant}` : null,
    loadId,
    fileVariant ? `${loadId}:${fileVariant}` : null,
    aliasId,
  ];
  return ordered.filter(
    (key, index): key is string => !!key && ordered.indexOf(key) === index,
  );
}

/** The settings a load of `loadId` (advertised as `aliasId`) applies. */
export function savedConfigFor(
  loadId: string,
  aliasId: string | null | undefined,
  ggufVariant: string | null | undefined,
): ResolvedPerModelConfig {
  const row = overrides
    ? resolveStoredOverride(
        overrides,
        overrideLookupCandidates(loadId, aliasId ?? null, ggufVariant ?? null),
      )
    : null;
  return row
    ? { config: fromApiOverride(row), remembered: true }
    : { config: { ...DEFAULT_PER_MODEL_CONFIG }, remembered: false };
}

export function resolveInitialConfig(
  modelId: string,
  ggufVariant?: string | null,
): ResolvedPerModelConfig {
  return savedConfigFor(modelId, null, ggufVariant);
}

/** For the identifier /api/inference/status reports: the snapshot path or file the load opened,
 *  while the settings may be saved under its public id. A standalone file's reported quant is a
 *  filename label, and its settings are saved without one. */
export function resolveResidentInitialConfig(
  modelId: string,
  ggufVariant?: string | null,
): ResolvedPerModelConfig {
  const alias = publicModelId(modelId);
  return savedConfigFor(
    modelId,
    alias !== modelId && alias.includes("/") ? alias : null,
    isStandaloneGgufPath(modelId) ? null : ggufVariant,
  );
}

/** The one quant of `modelId` with saved settings, or null when there are none or several. */
export function resolveOnlyRememberedGgufVariant(
  modelId: string,
): { ggufVariant: string; config: PerModelConfig } | null {
  const wanted = normalizeModelIdentity(modelId);
  const variants = new Map<string, string>();
  for (const key of Object.keys(overrides ?? {})) {
    const split = splitQuantSuffix(key);
    if (split && normalizeModelIdentity(split[0]) === wanted) {
      variants.set(normalizeGgufVariantIdentity(split[1]), split[1]);
    }
  }
  const [ggufVariant] = variants.values();
  if (variants.size !== 1 || !ggufVariant) {
    return null;
  }
  const resolved = resolveInitialConfig(modelId, ggufVariant);
  return resolved.remembered ? { ggufVariant, config: resolved.config } : null;
}

/** Stands in for a read of the server's map; null forgets it. */
export function setSavedModelConfigsForTests(
  next: ApiModelOverrides | null,
): void {
  firstRead = null;
  queue = Promise.resolve();
  if (next === null) {
    overrides = null;
  } else {
    publish(next);
  }
}
