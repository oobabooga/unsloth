// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Every unload of one chat model asks before stopping its running chats and then forces the
// unload, instead of surfacing the backend's 409.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { register } from "node:module";
import test from "node:test";

register("./helpers/loaded-models-eject-resolver.mjs", import.meta.url);

const stub = await import("./helpers/loaded-models-eject-stub.ts");
const {
  UNLOAD_DECLINED,
  unloadAfterChatStopConfirmation,
  unloadOrAbortIfDeclined,
} = await import("../src/features/loaded-models/confirmed-unload.ts");
const { unloadResident } = await import(
  "../src/features/api-monitor/unload-resident.ts"
);

const QWEN = "unsloth/Qwen3-4B-GGUF";
const HYMT = "tencent/Hy-MT1.5-1.8B-GGUF";

function source(path: string): string {
  return readFileSync(new URL(`../src/features/${path}`, import.meta.url), "utf8");
}

const unloadChat = (model: string) => ({
  action: "Unloading this model",
  effect: "unloadOne",
  model,
});

test("a model with a running chat unloads with force once the stop is confirmed", async () => {
  stub.resetStub({ active: HYMT, loaded: [HYMT], generating: true });
  stub.state.confirmAnswer = { proceed: true, forceCancelActive: true };
  assert.equal(await unloadAfterChatStopConfirmation(HYMT), true);
  assert.deepEqual(stub.state.confirmCalls, [unloadChat(HYMT)]);
  assert.deepEqual(stub.state.queueStops, [["running-thread"]]);
  assert.deepEqual(stub.state.unloads, [
    { model_path: HYMT, force_cancel_active: true },
  ]);
  assert.equal(stub.state.active, null);
});

test("declining the stop sends no unload", async () => {
  stub.resetStub({ active: HYMT, loaded: [HYMT], generating: true });
  stub.state.confirmAnswer = { proceed: false, forceCancelActive: false };
  assert.equal(await unloadAfterChatStopConfirmation(HYMT), false);
  assert.deepEqual(stub.state.unloads, []);
  assert.deepEqual(stub.state.queueStops, []);
  assert.equal(stub.state.active, HYMT);
});

// The API page's active-row loop, wired as the page wires it.
function unloadActive() {
  return unloadResident({
    readResident: async () => {
      const status = await stub.getInferenceStatus();
      const checkpoint = stub.servedInferenceCheckpointId(status);
      return checkpoint ? { checkpoint, aliases: [checkpoint] } : null;
    },
    unload: unloadOrAbortIfDeclined,
  });
}

test("the API page's unload asks and forces when a chat is running", async () => {
  stub.resetStub({ active: QWEN, loaded: [QWEN], generating: true });
  stub.state.confirmAnswer = { proceed: true, forceCancelActive: true };
  assert.deepEqual(await unloadActive(), {
    unloadedAliases: [QWEN],
    stillResident: null,
  });
  assert.deepEqual(stub.state.confirmCalls, [unloadChat(QWEN)]);
  assert.deepEqual(stub.state.unloads, [
    { model_path: QWEN, force_cancel_active: true },
  ]);
});

test("a declined confirmation ends the API page's loop without unloading", async () => {
  stub.resetStub({ active: QWEN, loaded: [QWEN], generating: true });
  stub.state.confirmAnswer = { proceed: false, forceCancelActive: false };
  await assert.rejects(unloadActive(), (error) => error === UNLOAD_DECLINED);
  assert.equal(stub.state.confirmCalls.length, 1, "asked once, not retried");
  assert.deepEqual(stub.state.unloads, []);
  assert.equal(stub.state.active, QWEN);
});

test("a model the API loads during the unload is asked about on its own", async () => {
  stub.resetStub({ active: QWEN, loaded: [QWEN], generating: true });
  stub.state.confirmAnswer = { proceed: true, forceCancelActive: true };
  stub.state.afterUnload = () => {
    stub.state.afterUnload = null;
    stub.state.active = HYMT;
    stub.state.loaded = [HYMT];
    stub.state.generating = true;
    stub.state.confirmAnswer = { proceed: false, forceCancelActive: false };
  };
  await assert.rejects(unloadActive(), (error) => error === UNLOAD_DECLINED);
  assert.deepEqual(stub.state.confirmCalls, [
    unloadChat(QWEN),
    unloadChat(HYMT),
  ]);
  assert.deepEqual(stub.state.unloads, [
    { model_path: QWEN, force_cancel_active: true },
  ]);
  assert.equal(stub.state.active, HYMT);
});

test("the API page, Translate, the panel and the picker unload through the owner", () => {
  const page = source("api-monitor/api-monitor-page.tsx");
  assert.match(page, /unload: unloadOrAbortIfDeclined,/);
  assert.match(page, /if \(err === UNLOAD_DECLINED\) return;/);
  const translate = source("translate/translate-page.tsx");
  assert.match(
    translate,
    /if \(!\(await unloadAfterChatStopConfirmation\(localModel\)\)\) return;/,
  );
  for (const [path, text] of [
    ["api-monitor-page.tsx", page],
    ["translate-page.tsx", translate],
    ["loaded-models-api.ts", source("loaded-models/loaded-models-api.ts")],
  ]) {
    assert.doesNotMatch(text, /\bunloadModel\(/, path);
    assert.doesNotMatch(text, /confirmStopRunningChatsIfNeeded/, path);
  }
  assert.match(
    source("chat/hooks/use-chat-model-runtime.ts"),
    /await unloadAfterChatStopConfirmation\(modelId\)/,
  );
});
