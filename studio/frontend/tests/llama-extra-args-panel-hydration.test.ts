// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Source-level: these rules live inside component effects with no renderer here.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { fromApiOverride, toApiOverride } = await import(
  "../src/features/model-picker/api/model-overrides.ts"
);
const { DEFAULT_PER_MODEL_CONFIG } = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);
const { savedConfigFor, setSavedModelConfigsForTests } = await import(
  "../src/features/model-picker/model-config/saved-model-configs.ts"
);

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PANEL = readFileSync(
  path.join(
    HERE,
    "..",
    "src/features/model-picker/components/model-config-page.tsx",
  ),
  "utf8",
);

/** The re-read effect, from where it captures the draft to where its dependencies close. */
function reReadEffect(): string {
  const start = PANEL.indexOf(
    "const draftAtStart = readModelConfigDraft(draftKey);",
  );
  const end = PANEL.indexOf("}, [draftKey, target.id, target.isGguf]);", start);
  assert.ok(start >= 0 && end > start, "missing the re-read effect");
  return PANEL.slice(start, end);
}

test("a list typed while the re-read was in flight is not sanitized", () => {
  // Only the saved list is sanitized; a draft edited or replaced during the read is live input.
  const effect = reReadEffect();
  assert.ok(
    effect.indexOf("Promise.all([") <
      effect.indexOf("sanitizeStoredExtraArgs("),
    "the draft is captured before the read and sanitized only after it",
  );
  assert.match(
    effect,
    /cancelled \|\|\s*isModelConfigDraftEdited\(draftKey\) \|\|\s*readModelConfigDraft\(draftKey\) !== draftAtStart/,
  );
});

test("a collapsed section is re-judged once the catalogue lands", () => {
  // A verdict reached before the flag probe answered must not freeze when Advanced collapses.
  assert.match(
    PANEL,
    /if \(showAdvanced \|\| !target\.isGguf \|\| resolvedIsDiffusion\) \{/,
  );
  assert.match(PANEL, /loadLlamaFlagCatalog\(\)\.then\(\(catalog\) => \{/);
});

const ADAPTER = readFileSync(
  path.join(HERE, "..", "src/features/chat/api/chat-adapter.ts"),
  "utf8",
);

test("a background auto-load sends the saved list, sanitized", () => {
  // The startup load can run before the first read of the saved settings has answered.
  assert.match(
    ADAPTER,
    /await ensureSavedModelConfigs\(\);\s*const \{ config \} = resolveInitialConfig\(/,
  );
  // Read under the id the row was saved by, not the load path cached inventory can hand back.
  assert.match(
    ADAPTER,
    /resolveInitialConfig\(\s*candidate\.id,\s*candidate\.ggufVariant,\s*\)/,
  );
  assert.match(ADAPTER, /let resolvedExtraArgs = config\.llamaExtraArgs;/);
  // A flag managed since the list was saved would be answered with a 400.
  assert.match(
    ADAPTER,
    /sanitizeStoredExtraArgs\(\s*resolvedExtraArgs,\s*managed\?\.managed/,
  );
  assert.doesNotMatch(ADAPTER, /fetchLoadExtraArgs/);
  assert.equal(
    ADAPTER.match(/llama_extra_args: resolvedExtraArgs \?\? \[\]/g)?.length,
    2,
  );
  assert.match(ADAPTER, /candidate\.kind === "gguf" &&\s*\n?\s*!isDiffusion/);
});

test("a collapsed section stops objecting once nothing is left to object to", () => {
  // Reset with Advanced collapsed unmounts the row, so the panel must restore loadability.
  assert.match(PANEL, /setExtraArgsLoadable\(true\);\n\s*return;/);
});

const COMPOSER = readFileSync(
  path.join(HERE, "..", "src/features/chat/shared-composer.tsx"),
  "utf8",
);

test("a compare pane sanitizes its saved list", () => {
  // A list saved before a flag became managed would be rejected by /load with a 400.
  assert.match(COMPOSER, /const savedArgs = ownConfig\.llamaExtraArgs;/);
  assert.match(COMPOSER, /sanitizeStoredExtraArgs\(\s*savedArgs,/);
  assert.doesNotMatch(COMPOSER, /fetchLoadExtraArgs/);
});

test("the hidden revalidation can only tighten the verdict", () => {
  // formatExtraArgs rebalances quotes, so only lowering the verdict here is safe.
  assert.match(PANEL, /if \(!loadable\) \{\n\s*setExtraArgsLoadable\(false\);/);
});

test("a mounted row re-reads the catalogue when the binary changes", () => {
  // An in-app llama.cpp update replaces the binary while the panel stays open.
  assert.match(PANEL, /subscribeLlamaFlagCatalog\(\(\) => setCatalogEpoch/);
  assert.match(PANEL, /\}, \[catalogEpoch\]\);/);
});

test("the hidden validation re-runs when the binary changes", () => {
  // The row is unmounted while this check runs, so it cannot subscribe itself.
  assert.match(
    PANEL,
    /subscribeLlamaFlagCatalog\(\(\) =>\s*\n?\s*setHiddenCatalogEpoch/,
  );
  assert.match(PANEL, /hiddenCatalogEpoch,\n\s*\]\);/);
});

test("the hidden check knows the slot floor", () => {
  // The backend deterministically refuses a batch below the floor.
  assert.match(
    PANEL,
    /batchFloor: effectiveBatchFloor\(configState\.nParallel, catalog\)/,
  );
});

test("the panel takes the saved settings without overwriting a live edit", () => {
  const effect = reReadEffect();
  assert.match(effect, /refreshSavedModelConfigs\(\),/);
  // An edited draft, or one replaced while the read was out, keeps what it holds.
  assert.match(
    effect,
    /isModelConfigDraftEdited\(draftKey\) \|\|\s*readModelConfigDraft\(draftKey\) !== draftAtStart/,
  );
  // Remember follows whether the server holds a row for the model.
  assert.match(
    effect,
    /replaceModelConfigDraft\(\s*draftKey,\s*reconcileConfigGpuSelection\(config, resolvedIsDiffusion, gpuDevices\),\s*\{ remember: resolved\.remembered, savedRemember: resolved\.remembered \},\s*\);/,
  );
  assert.doesNotMatch(PANEL, /fetchLoadModelOverride|savePerModelConfig/);
});

test("the panel strips llama-server arguments from a non-GGUF row", () => {
  // API auto-switch applies the row to MLX and safetensors loads too, which send no such flags.
  assert.match(
    PANEL,
    /return target\.isGguf\s*\?\s*saved\s*:\s*\{ \.\.\.saved, config: \{ \.\.\.saved\.config, llamaExtraArgs: undefined \} \};/,
  );
  assert.match(
    PANEL,
    /target\.isGguf \? loadManagedLlamaFlags\(\)\.catch\(\(\) => null\) : null,/,
  );
});

test("a build that serves one slot does not raise the floor", () => {
  // Without --kv-unified the backend clamps to one slot, so an explicit Slots value is not effective.
  assert.match(PANEL, /if \(limits\?\.parallelSlotsClamped\) \{\n\s*return 2;/);
  assert.equal(
    PANEL.match(/effectiveBatchFloor\(/g)?.length,
    3,
    "one definition and two call sites",
  );
});

const ADAPTER_AUTOLOAD = ADAPTER;

test("an auto-load records what it launched with", () => {
  // The loading lease blocks the status applier, so the load itself must record the baseline.
  assert.match(
    ADAPTER_AUTOLOAD,
    /loadedLlamaExtraArgs:\s*\n?\s*loadResp\.requested_llama_extra_args !== undefined/,
  );
  assert.match(ADAPTER_AUTOLOAD, /loadedLlamaExtraArgs: null,/);
});

test("a compare pane records what it launched with", () => {
  assert.match(
    COMPOSER,
    /loadedLlamaExtraArgs:\s*\n?\s*resp\.requested_llama_extra_args !== undefined/,
  );
});

test("a stored empty list reads as a clear, not as nothing stored", () => {
  // Explicit [] is a tombstone; treating it as absent lets /load carry cleared flags over.
  const cleared = toApiOverride({
    ...DEFAULT_PER_MODEL_CONFIG,
    llamaExtraArgs: null,
  });
  assert.deepEqual(fromApiOverride(cleared).llamaExtraArgs, []);
  assert.equal(
    fromApiOverride(
      toApiOverride({ ...DEFAULT_PER_MODEL_CONFIG, nParallel: 2 }),
    ).llamaExtraArgs,
    undefined,
    "a row without the field leaves the launch flags to the load",
  );
  setSavedModelConfigsForTests({
    "org/model-GGUF:Q4_K_M": cleared,
    "org/model-GGUF": toApiOverride({
      ...DEFAULT_PER_MODEL_CONFIG,
      llamaExtraArgs: ["--no-warmup"],
    }),
  });
  try {
    // The quant's own tombstone stops the lookup before the bare repository row.
    assert.deepEqual(
      savedConfigFor("org/model-GGUF", null, "Q4_K_M").config.llamaExtraArgs,
      [],
    );
  } finally {
    setSavedModelConfigsForTests(null);
  }
});

const CHAT_PAGE = readFileSync(
  path.join(HERE, "..", "src/features/chat/chat-page.tsx"),
  "utf8",
);

test("a Chat launch that applies remembered config carries its arguments", () => {
  // /load inherits flags only from the same resident model, so cold launches must send them.
  assert.match(
    CHAT_PAGE,
    /const remembered = rememberedConfigFor\(selection\);/,
  );
  assert.match(
    CHAT_PAGE,
    /\.\.\.\(remembered \? \{ config: remembered \} : \{\}\),/,
  );
});
