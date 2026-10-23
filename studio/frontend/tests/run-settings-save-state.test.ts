// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const CONFIG_PAGE = readFileSync(
  path.join(
    HERE,
    "..",
    "src/features/model-picker/components/model-config-page.tsx",
  ),
  "utf8",
);

function bodyOf(name: string): string {
  const start = CONFIG_PAGE.indexOf(`function ${name}(`);
  assert.notEqual(start, -1, `missing ${name}`);
  const next = CONFIG_PAGE.indexOf("\nfunction ", start + 1);
  return CONFIG_PAGE.slice(start, next === -1 ? undefined : next);
}

test("the save status covers every Remember state and is always rendered", () => {
  for (const label of [
    "Saved settings could not be read",
    "Saved for this model",
    "Unsaved changes",
    "Saved settings will be forgotten",
    "Not saved yet",
    "Using default settings",
    "Not remembered for this model",
  ]) {
    assert.ok(CONFIG_PAGE.includes(`"${label}"`), `missing status: ${label}`);
  }
  // Mounted on demand it would push Load down under the cursor on the click that saves.
  assert.match(CONFIG_PAGE, /\{saveStatus\}/);
  assert.doesNotMatch(CONFIG_PAGE, /saveStatus &&/);
});

const COMPARES_WHAT_SAVE_STORES =
  /normalizePerModelConfig\(storedSpeculativeAuto\(runtimeConfig, targetIsMlx\)\),\s*storedRecord\.config/;
const STORED_RECORD_FROM_SERVER_MAP =
  /const storedRecord = useSavedTargetConfig\(target, configId\);/;
const FOLLOWS_THE_SERVER_MAP =
  /useSyncExternalStore\(\s*subscribeSavedModelConfigs,\s*readSavedModelConfigs,/;
const SAYS_WHEN_UNREAD =
  /!savedSettingsLoading && readSavedModelConfigs\(\) === null\s*\?\s*"Saved settings could not be read"/;
const FORGETS_ONLY_A_SAVED_REMEMBER =
  /const forgets = !remember && savedRemember;/;
const SAVES_OR_FORGETS =
  /if \(remember\) \{\s*await saveModelConfig\(configId, target\.ggufVariant, normalized,[\s\S]*?\} else if \(forgets\) \{\s*await forgetModelConfig\(configId, target\.ggufVariant\);\s*\}/;
const REPORTS_THE_SERVER_REASON =
  /catch \(error: unknown\) \{\s*toast\.error\(failure, \{\s*description: error instanceof Error \? error\.message : undefined,\s*\}\);\s*return null;\s*\}/;
const SAVE_STOPS_AT_A_REFUSAL = /if \(!persisted\) \{\s*return;\s*\}/;
const RUN_LOADS_ANYWAY =
  /await persistConfig\(\s*effectiveRuntimeConfig,\s*"Couldn't save these settings, loading with them anyway\.",\s*\);/;
const PRIMED_IN_A_LAYOUT_EFFECT =
  /useLayoutEffect\(\(\) => \{\s*const resolved = resolveInitial\(\);\s*primeModelConfigDraft\(/;
const READ_COMPARES_THE_PRIMED_DRAFT =
  /useEffect\(\(\) => \{\s*let cancelled = false;\s*setSavedSettingsLoading\(true\);\s*const draftAtStart = readModelConfigDraft\(draftKey\);/;
const RENDER_SNAPSHOT = /configRef/;

test("the status compares what Save would store against the saved row", () => {
  assert.match(CONFIG_PAGE, COMPARES_WHAT_SAVE_STORES);
  // Follows every change to the server's map, a save from another editor included.
  assert.match(CONFIG_PAGE, STORED_RECORD_FROM_SERVER_MAP);
  assert.match(bodyOf("useSavedTargetConfig"), FOLLOWS_THE_SERVER_MAP);
  // A failed first read says so rather than reading as a model with nothing saved.
  assert.match(CONFIG_PAGE, SAYS_WHEN_UNREAD);
});

/** persistConfig's source, empty when it is gone, which no assertion below accepts. */
function persistBody(): string {
  const start = CONFIG_PAGE.indexOf("const persistConfig = async (");
  const end = CONFIG_PAGE.indexOf("const finishPersist = ", start);
  return start === -1 || end === -1 ? "" : CONFIG_PAGE.slice(start, end);
}

test("an unticked Load forgets only a Remember that was on", () => {
  const persist = persistBody();
  assert.match(persist, FORGETS_ONLY_A_SAVED_REMEMBER);
  assert.match(persist, SAVES_OR_FORGETS);
});

test("a refused save is reported with the server's reason", () => {
  const persist = persistBody();
  assert.match(persist, REPORTS_THE_SERVER_REASON);
  // Only a write the server accepted may let a later read of the saved settings replace the draft.
  assert.ok(
    persist.indexOf("clearModelConfigDraftEdited(draftKey)") >
      persist.indexOf("return null;"),
    "the draft is marked saved before the write answered",
  );
  // Save stops at a refused write; Run says so and loads with the settings anyway.
  const save = CONFIG_PAGE.slice(
    CONFIG_PAGE.indexOf("const handleSave = async () => {"),
    CONFIG_PAGE.indexOf("const handleRun = async () => {"),
  );
  assert.match(save, SAVE_STOPS_AT_A_REFUSAL);
  assert.ok(save.indexOf("if (!persisted)") < save.indexOf("finishPersist("));
  assert.match(CONFIG_PAGE, RUN_LOADS_ANYWAY);
});

test("a fresh editor takes the saved settings when nothing changed in flight", () => {
  // The read compares against the draft the layout effect primed, never a value from a render
  // before the prime: that refused every row on a fresh editor.
  assert.match(CONFIG_PAGE, PRIMED_IN_A_LAYOUT_EFFECT);
  assert.match(CONFIG_PAGE, READ_COMPARES_THE_PRIMED_DRAFT);
  assert.doesNotMatch(CONFIG_PAGE, RENDER_SNAPSHOT);
});

test("Reasoning Budget shows unlimited as a blank box", () => {
  assert.match(
    CONFIG_PAGE,
    /value=\{config\.reasoningBudget === -1 \? "" : config\.reasoningBudget\}/,
  );
});

test("GGUF Advanced rows sit in titled groups, the llama-server ones hidden for diffusion", () => {
  const gguf = bodyOf("GgufAdvancedSettings");
  const order = ["Memory", "Speed", "Output", "Server"].map((title) =>
    gguf.indexOf(`<AdvancedGroup title="${title}">`),
  );
  assert.ok(
    order.every((at) => at !== -1),
    "a group is missing",
  );
  assert.deepEqual(
    [...order].sort((a, b) => a - b),
    order,
  );
  const server = gguf.slice(order[3] - 40, order[3]);
  assert.match(server, /!isDiffusion && \(\s*$/);
  assert.ok(
    gguf.indexOf("<ExtraArgsRow") > order[3],
    "extra args left the Server group",
  );
});
