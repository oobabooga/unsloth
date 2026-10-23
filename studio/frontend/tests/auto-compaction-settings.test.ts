// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import ts from "@typescript/typescript6";
import {
  apiCompactionRequestFields,
  ggufCompactionRequestFields,
} from "../src/features/chat/utils/auto-compaction.ts";

import { readSrc } from "./helpers/kit.ts";

test("auto-compact on sends truncate_oldest and no policy of its own", () => {
  // the server applies UNSLOTH_CONTEXT_POLICY because Studio no longer exposes context_policy.
  assert.deepEqual(
    ggufCompactionRequestFields({ compacts: true, autoCompactEnabled: true }),
    { context_overflow: "truncate_oldest" },
  );
});

test("auto-compact off sends an explicit error overflow policy", () => {
  assert.deepEqual(
    ggufCompactionRequestFields({ compacts: true, autoCompactEnabled: false }),
    { context_overflow: "error" },
  );
});

test("a backend that does not compact gets no compaction fields", () => {
  assert.deepEqual(
    ggufCompactionRequestFields({ compacts: false, autoCompactEnabled: true }),
    {},
  );
});

test("nothing still offers the removed compaction style", () => {
  for (const file of [
    "features/settings/tabs/chat-tab.tsx",
    "features/chat/stores/chat-runtime-store.ts",
    "features/chat/utils/chat-settings-storage.ts",
    "features/chat/utils/queued-chat-run-settings.ts",
    "features/chat/api/chat-settings-api.ts",
    "features/chat/api/chat-adapter.ts",
    "features/chat/utils/auto-compaction.ts",
    "features/settings/settings-search.ts",
    "i18n/locales/en.ts",
  ]) {
    assert.doesNotMatch(
      readSrc(file),
      /contextPolicy|compactionHeadroomRatio|compactionStyle|compactionDescription/,
      `${file} still carries the removed compaction style`,
    );
  }
});

test("the chat adapter sends compaction fields through the shared helper", () => {
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  // The completion and the token count ask the loaded backend's own declaration.
  assert.match(adapter, /compacts: localBackendCompacts\(runtime\)/);
  assert.match(adapter, /compacts: localBackendCompacts\(state\)/);
  assert.match(adapter, /localBackendHonors\(state, "context_overflow", true\)/);
  // The value must derive from THIS turn's messages only (durable-gate.ts).
  assert.match(adapter, /image_base64: findLatestUserImageBase64\(currentTurnMessages\)/);
});

test("a queued run keeps its own model's compaction verdict after the picker moves on", async () => {
  const { registerBundlerResolver, installLocalStorageFake } = await import(
    "./helpers/kit.ts"
  );
  registerBundlerResolver();
  installLocalStorageFake();
  const { snapshotQueuedChatRunSettings } = await import(
    "../src/features/chat/utils/queued-chat-run-settings.ts"
  );
  const { loadedContextFields, localBackendHonors } = await import(
    "../src/features/model-picker/model-config/per-model-config.ts"
  );

  const resident = {
    params: { checkpoint: "ollama-manifest:%2Fhome%2Fu%2F.ollama%2Fmanifests%2Fq" },
    loadedChatControls: { context_overflow: true },
    loadedContextLength: 8192,
    autoCompactEnabled: true,
  };
  const queued = snapshotQueuedChatRunSettings(
    resident as unknown as Parameters<typeof snapshotQueuedChatRunSettings>[0],
  );

  const live = {
    ...resident,
    params: { checkpoint: "external::openai::gpt-5" },
    ...loadedContextFields(null),
  };
  const runtime = { ...live, ...queued };

  assert.deepEqual(
    ggufCompactionRequestFields({
      compacts: localBackendHonors(runtime, "context_overflow", true),
      autoCompactEnabled: runtime.autoCompactEnabled,
    }),
    { context_overflow: "truncate_oldest" },
  );
  // A backend that declares it cannot compact gets nothing, even while nothing else is known.
  assert.deepEqual(
    ggufCompactionRequestFields({
      compacts: localBackendHonors(
        { ...runtime, loadedChatControls: { context_overflow: false } },
        "context_overflow",
        true,
      ),
      autoCompactEnabled: true,
    }),
    {},
  );
});

test("an API model compacts a quarter short of its published window and sends that window", () => {
  assert.deepEqual(
    apiCompactionRequestFields({ autoCompactEnabled: true, contextLength: 200_000 }),
    {
      context_overflow: "truncate_oldest",
      compaction_threshold: 150_000,
      context_window: 200_000,
    },
  );
});

test("a window past the request ceiling still sends a threshold the server accepts", () => {
  assert.deepEqual(
    apiCompactionRequestFields({ autoCompactEnabled: true, contextLength: 10_000_000 }),
    {
      context_overflow: "truncate_oldest",
      compaction_threshold: 2_000_000,
      context_window: 10_000_000,
    },
  );
});

test("an API model sends nothing with auto-compact off", () => {
  assert.deepEqual(
    apiCompactionRequestFields({ autoCompactEnabled: false, contextLength: 200_000 }),
    {},
  );
  assert.deepEqual(
    apiCompactionRequestFields({ autoCompactEnabled: false, contextLength: null }),
    {},
  );
});

test("a self-hosted connection with no catalogued window still asks the server to compact", async () => {
  const { registerBundlerResolver, installLocalStorageFake } = await import(
    "./helpers/kit.ts"
  );
  registerBundlerResolver();
  installLocalStorageFake();
  const { resolveModelCatalogEntry } = await import(
    "../src/features/chat/model-catalog.ts"
  );
  for (const providerType of ["custom", "llama_cpp", "vllm"]) {
    const contextLength = resolveModelCatalogEntry(providerType, "qwen3-next")?.contextLength;
    assert.equal(contextLength ?? null, null, providerType);
    // the backend derives the threshold from the self-hosted server's reported window
    assert.deepEqual(
      apiCompactionRequestFields({ autoCompactEnabled: true, contextLength }),
      { context_overflow: "truncate_oldest" },
      providerType,
    );
  }
});

test("the external request carries the window the model catalog publishes", async () => {
  const { registerBundlerResolver, installLocalStorageFake } = await import(
    "./helpers/kit.ts"
  );
  registerBundlerResolver();
  installLocalStorageFake();
  const { resolveModelCatalogEntry } = await import(
    "../src/features/chat/model-catalog.ts"
  );
  const window = resolveModelCatalogEntry("anthropic", "claude-haiku-4-5")?.contextLength;
  assert.equal(window, 200_000);

  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  assert.match(
    adapter,
    /apiCompactionRequestFields\(\{\s*autoCompactEnabled: runtime\.autoCompactEnabled,\s*contextLength: resolveModelCatalogEntry\(\s*externalProvider\.providerType,\s*externalModelId,\s*\)\?\.contextLength,/,
  );
});

test("a provider compaction is kept on the turn and replayed only to API models", () => {
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  assert.match(adapter, /toolEvent\.type === "compaction_block"/);
  assert.match(adapter, /providerCompaction,\n/);
  assert.match(
    adapter,
    /isExternalRequest\s*\?\s*withProviderCompaction\(message, serialized, \{[\s\S]*providerType: toExternalBackendProviderType\([\s\S]*externalProvider\?\.providerType,[\s\S]*modelId: externalSelection\?\.modelId,/,
  );
});

test("provider compaction persistence keeps summary and encrypted state together", async () => {
  const {
    providerCompactionConnectionKey,
    providerCompactionForTarget,
    providerCompactionPart,
  } =
    await import("../src/features/chat/utils/provider-compaction.ts");

  assert.deepEqual(
    providerCompactionPart({
      type: "compaction_block",
      content: "Earlier conversation summary",
      encrypted_content: "opaque-compaction",
    }),
    {
      type: "compaction",
      content: "Earlier conversation summary",
      encrypted_content: "opaque-compaction",
    },
  );
  const connectionKey = providerCompactionConnectionKey(
    "provider-a",
    "https://first.openai.azure.com/openai/v1",
    "responses",
  );
  assert.ok(connectionKey);
  const origin = {
    providerCompaction: {
      type: "compaction",
      content: "summary",
      encrypted_content: "opaque-compaction",
    },
    providerCompactionProviderType: "anthropic",
    providerCompactionModelId: "claude-opus-4-7",
    providerCompactionConnectionKey: connectionKey,
  };
  assert.deepEqual(
    providerCompactionForTarget(
      origin,
      "anthropic",
      "claude-opus-4-7",
      connectionKey,
    ),
    origin.providerCompaction,
  );
  assert.equal(
    providerCompactionForTarget(origin, "openai", "gpt-5.4", connectionKey),
    null,
  );
  assert.equal(
    providerCompactionForTarget(
      origin,
      "anthropic",
      "claude-sonnet-5",
      connectionKey,
    ),
    null,
  );
  assert.equal(
    providerCompactionForTarget(
      {},
      "anthropic",
      "claude-opus-4-7",
      connectionKey,
    ),
    null,
  );
  assert.equal(
    providerCompactionForTarget(
      origin,
      "anthropic",
      "claude-opus-4-7",
      providerCompactionConnectionKey(
        "provider-b",
        "https://first.openai.azure.com/openai/v1",
        "responses",
      ),
    ),
    null,
  );
  assert.equal(
    providerCompactionForTarget(
      origin,
      "anthropic",
      "claude-opus-4-7",
      providerCompactionConnectionKey(
        "provider-a",
        "https://second.openai.azure.com/openai/v1",
        "responses",
      ),
    ),
    null,
  );
});

test("provider compaction replay stays on the tool-loop subturn that produced it", () => {
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  const start = adapter.indexOf("function providerCompactionAssistant(");
  assert.ok(start >= 0);
  const declaration = adapter.slice(start, adapter.indexOf("\n}", start) + 2);
  const providerCompactionAssistant = new Function(
    `${ts.transpileModule(declaration, {
      compilerOptions: { target: ts.ScriptTarget.ES2022 },
    }).outputText}; return providerCompactionAssistant;`,
  )() as (
    messages: Record<string, unknown>[],
    afterToolCalls: number,
  ) => Record<string, unknown> | undefined;

  const first = {
    role: "assistant",
    content: null,
    tool_calls: [{ id: "first" }],
  };
  const second = {
    role: "assistant",
    content: null,
    tool_calls: [{ id: "second" }],
  };
  const final = { role: "assistant", content: "done" };
  const replay = [
    first,
    { role: "tool", content: "one", tool_call_id: "first" },
    second,
    { role: "tool", content: "two", tool_call_id: "second" },
    final,
  ];

  assert.equal(providerCompactionAssistant(replay, 0), first);
  assert.equal(providerCompactionAssistant(replay, 1), second);
  assert.equal(providerCompactionAssistant(replay, 2), final);
  assert.match(
    adapter,
    /providerCompactionAfterToolCalls = toolCallParts\.filter\([\s\S]*toolCallPartSurvivesOpenAIReplay\(part\)[\s\S]*\)\.length/,
  );
});
