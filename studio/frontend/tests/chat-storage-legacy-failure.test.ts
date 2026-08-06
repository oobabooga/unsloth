// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What the chat history reader does when the legacy Dexie database cannot be opened at
// all. That is the state a Linux user lands in after switching between the .deb build
// (system WebKit) and the AppImage build (older bundled WebKit) on one profile: the
// older engine refuses the newer engine's IndexedDB file, so every Dexie call rejects
// while studio.db is untouched. Chat history must still come back from the backend.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import type {
  MessageRecord,
  ThreadRecord,
} from "../src/features/chat/types.ts";
import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();

const DB_URL = new URL("../src/features/chat/db.ts", import.meta.url).href;
const API_URL = new URL("../src/features/chat/api/chat-api.ts", import.meta.url)
  .href;

// chat-api reaches the whole app (auth, hub, native intents) and Dexie needs a real
// IndexedDB, so both are replaced with test doubles read from globalThis at call time.
const CHAT_API_EXPORTS = [
  "buildBackendChatExport",
  "CHAT_HISTORY_UPDATED_EVENT",
  "clearBackendChats",
  "deleteChatProject",
  "deleteChatThreads",
  "getChatProject",
  "getChatMessage",
  "getChatThread",
  "batchListChatMessages",
  "listChatProjects",
  "listChatImportLedger",
  "listChatMessages",
  "listChatThreads",
  "notifyChatHistoryUpdated",
  "recordChatImportLedger",
  "saveChatProject",
  "saveChatMessage",
  "saveChatThread",
  "syncChatMessages",
  "updateChatProject",
  "updateChatThread",
];

register("./chat-storage-fakes-loader.mjs", import.meta.url, {
  data: {
    dbUrl: DB_URL,
    apiUrl: API_URL,
    apiExports: CHAT_API_EXPORTS,
  },
});

const { store, storage: localStorageFake } = installLocalStorageFake();
// The engine is present and the legacy database exists; only reading it fails. That is
// what keeps the reader on its slow path instead of the "no Dexie at all" fast path.
Object.assign(globalThis, {
  indexedDB: { databases: async () => [{ name: "unsloth-chat", version: 3 }] },
});

const LEGACY_IMPORT_HINT = "unsloth_chat_legacy_imported_to_studio_db";
const MESSAGE_TOMBSTONES_KEY = "unsloth_chat_deleted_messages";

function thread(
  id: string,
  overrides: Partial<ThreadRecord> = {},
): ThreadRecord {
  return {
    id,
    title: `Thread ${id}`,
    modelType: "base",
    archived: false,
    createdAt: 1,
    updatedAt: 2,
    ...overrides,
  };
}

function message(
  id: string,
  threadId: string,
  role: "user" | "assistant",
  text: string,
  createdAt: number,
): MessageRecord {
  return {
    id,
    threadId,
    parentId: null,
    role,
    content: [{ type: "text", text }],
    createdAt,
  };
}

/** studio.db, in memory: the side that stays healthy in the reported bug. */
function backendFake(
  threads: ThreadRecord[],
  messages: MessageRecord[],
): {
  api: Record<string, (...args: never[]) => unknown>;
  exportCalls: () => number;
  threads: Map<string, ThreadRecord>;
  messages: Map<string, MessageRecord[]>;
  importedThreadIds: Set<string>;
} {
  const threadsById = new Map(threads.map((t) => [t.id, t]));
  const messagesByThread = new Map<string, MessageRecord[]>();
  for (const m of messages) {
    const existing = messagesByThread.get(m.threadId);
    if (existing) existing.push(m);
    else messagesByThread.set(m.threadId, [m]);
  }
  let exportCalls = 0;
  const importedThreadIds = new Set<string>();
  const api = {
    listChatThreads: async () => Array.from(threadsById.values()),
    getChatThread: async (id: string) => threadsById.get(id) ?? null,
    saveChatThread: async (record: ThreadRecord) => {
      threadsById.set(record.id, record);
      return record;
    },
    updateChatThread: async (id: string, patch: Partial<ThreadRecord>) => {
      const merged = { ...(threadsById.get(id) as ThreadRecord), ...patch };
      threadsById.set(id, merged);
      return merged;
    },
    saveChatMessage: async (record: MessageRecord) => {
      const records = messagesByThread.get(record.threadId) ?? [];
      messagesByThread.set(record.threadId, [
        ...records.filter((message) => message.id !== record.id),
        record,
      ]);
      return record;
    },
    listChatMessages: async (threadId: string) =>
      messagesByThread.get(threadId) ?? [],
    getChatMessage: async (threadId: string, messageId: string) =>
      (messagesByThread.get(threadId) ?? []).find((m) => m.id === messageId) ??
      null,
    batchListChatMessages: async (threadIds: string[]) =>
      new Map(
        threadIds.map((id) => [id, messagesByThread.get(id) ?? []] as const),
      ),
    syncChatMessages: async (
      threadId: string,
      records: MessageRecord[],
      options: { pruneMissing?: boolean } = {},
    ) => {
      const next = options.pruneMissing
        ? records
        : Array.from(
            new Map(
              [...(messagesByThread.get(threadId) ?? []), ...records].map(
                (record) => [record.id, record] as const,
              ),
            ).values(),
          );
      messagesByThread.set(threadId, next);
      return next;
    },
    deleteChatThreads: async (threadIds: string[]) => {
      for (const id of threadIds) {
        threadsById.delete(id);
        messagesByThread.delete(id);
      }
    },
    clearBackendChats: async () => {
      threadsById.clear();
      messagesByThread.clear();
    },
    listChatImportLedger: async () => new Set(importedThreadIds),
    recordChatImportLedger: async (threadIds: string[]) => {
      for (const id of threadIds) importedThreadIds.add(id);
      return { supported: true };
    },
    buildBackendChatExport: async () => {
      exportCalls += 1;
      return {
        projects: [],
        threads: Array.from(threadsById.values()),
        messages: Array.from(messagesByThread.values()).flat(),
      };
    },
    notifyChatHistoryUpdated: () => {},
  } as unknown as Record<string, (...args: never[]) => unknown>;
  return {
    api,
    exportCalls: () => exportCalls,
    threads: threadsById,
    messages: messagesByThread,
    importedThreadIds,
  };
}

/** A Dexie whose every read rejects, the way an unreadable IndexedDB file behaves. */
function unreadableDexie(): { fake: unknown; reads: () => number } {
  let reads = 0;
  const reject = () => {
    reads += 1;
    return Promise.reject(
      new Error("UnknownError: Internal error opening backing store"),
    );
  };
  const collection = { toArray: reject, delete: reject, modify: reject };
  const table = {
    get: reject,
    toArray: reject,
    count: reject,
    bulkDelete: reject,
    clear: reject,
    toCollection: () => collection,
    where: () => ({ equals: () => collection, anyOf: () => collection }),
  };
  return {
    fake: { threads: table, messages: table, transaction: reject },
    reads: () => reads,
  };
}

/** A working Dexie holding pre-studio.db chats, for the unchanged legacy import path. */
function readableDexie(
  threads: ThreadRecord[],
  messages: MessageRecord[],
  options: { failBulkMessageRead?: () => boolean } = {},
) {
  const table = <T extends { id: string }>(
    rows: T[],
    field: (row: T) => string,
    failAnyOf: () => boolean = () => false,
  ) => {
    const collection = (matches: (row: T) => boolean) => ({
      toArray: async () => rows.filter(matches),
    });
    return {
      get: async (id: string) => rows.find((row) => row.id === id),
      toArray: async () => [...rows],
      count: async () => rows.length,
      toCollection: () => collection(() => true),
      where: () => ({
        equals: (value: string) => collection((row) => field(row) === value),
        anyOf: (values: string[]) => ({
          toArray: async () => {
            if (failAnyOf()) {
              throw new Error("UnknownError: targeted messages read failure");
            }
            return rows.filter((row) => values.includes(field(row)));
          },
        }),
      }),
    };
  };
  return {
    threads: table(threads, (row) => row.modelType),
    messages: table(
      messages,
      (row) => row.threadId,
      options.failBulkMessageRead,
    ),
  };
}

function setFakes(dexie: unknown, api: unknown): void {
  Object.assign(globalThis, { __dexieFake: dexie, __chatApiFake: api });
}

const storage = await import(
  "../src/features/chat/utils/chat-history-storage.ts"
);
const messageTombstones = await import(
  "../src/features/chat/utils/chat-message-tombstones.ts"
);
const { deleteThreadMessage } = await import(
  "../src/features/chat/utils/delete-thread-message.ts"
);
const { updateThreadMessage } = await import(
  "../src/features/chat/utils/update-thread-message.ts"
);
const { buildChatSearchIndex } = await import(
  "../src/features/chat/hooks/use-chat-search-index.ts"
);
const { ExportedMessageRepository } = await import("@assistant-ui/core");

test("an unreadable legacy database still serves backend chat history", async () => {
  store.clear();
  messageTombstones.__resetChatMessageTombstonesForTests();
  const backend = backendFake(
    [thread("t1"), thread("t2", { updatedAt: 5 })],
    [
      message("m1", "t1", "user", "hello", 10),
      message("m2", "t1", "assistant", "hi", 20),
    ],
  );
  const dexie = unreadableDexie();
  setFakes(dexie.fake, backend.api);
  const warnings: unknown[] = [];
  const originalWarn = console.warn;
  console.warn = (...args: unknown[]) => {
    warnings.push(args);
  };

  try {
    // Recents: the whole list must survive, not just fail to throw.
    const threads = await storage.listStoredChatThreads();
    assert.deepEqual(
      threads.map((t) => t.id),
      ["t2", "t1"],
    );

    // Opening a thread, and the autosave path that runs on every message.
    assert.equal((await storage.getStoredChatThread("t1"))?.id, "t1");
    assert.equal((await storage.ensureStoredChatThread("t1"))?.id, "t1");

    // Its messages.
    const messages = await storage.listStoredChatMessages("t1");
    assert.deepEqual(
      messages.map((m) => m.id),
      ["m1", "m2"],
    );
    assert.equal((await storage.getStoredChatMessage("t1", "m2"))?.id, "m2");

    // The way out: an export must carry the backend's data, and must build it once.
    const exported = await storage.buildStoredChatExport();
    assert.equal(exported.threadCount, 2);
    assert.deepEqual(
      (exported.threads as ThreadRecord[]).map((t) => t.id).sort(),
      ["t1", "t2"],
    );
    assert.deepEqual(
      (exported.messages as MessageRecord[]).map((m) => m.id).sort(),
      ["m1", "m2"],
    );
    assert.equal(backend.exportCalls(), 1);

    // The reads really were attempted, so this exercises the failure, not a bypass.
    assert.ok(dexie.reads() > 0);
    // One diagnostic for the session, not one per read.
    assert.equal(warnings.length, 1);
    // The migration hint is untouched: a later launch on the build that can read
    // Dexie must still be able to import those chats.
    assert.equal(store.get(LEGACY_IMPORT_HINT), undefined);
  } finally {
    console.warn = originalWarn;
  }
});

test("a selective legacy message read failure retries without finalizing import", async () => {
  store.clear();
  messageTombstones.__resetChatMessageTombstonesForTests();
  const legacyThread = thread("legacy1", { title: "Old chat" });
  const kept = message("keep", "legacy1", "user", "keep me", 30);
  const removed = message("remove", "legacy1", "assistant", "delete me", 40);
  const otherThread = thread("other", { updatedAt: 10 });
  const sameIdElsewhere = message(
    "remove",
    "other",
    "user",
    "same id, different thread",
    50,
  );
  const backend = backendFake([legacyThread], [kept, removed]);
  let failBulkMessageRead = true;
  const dexie = readableDexie([legacyThread], [kept, removed], {
    failBulkMessageRead: () => failBulkMessageRead,
  });
  setFakes(dexie, backend.api);

  // Thread reads and direct message reads work. Only the import's selective bulk
  // message query fails, so it must not treat the thread as an empty success.
  await storage.listStoredChatThreads();
  assert.equal(backend.importedThreadIds.has("legacy1"), false);
  assert.equal(backend.importedThreadIds.size, 0);
  assert.equal(store.get(LEGACY_IMPORT_HINT), undefined);

  // Export retries the failed import, then reads the still-readable legacy tables
  // directly. The incomplete import cannot hide the legacy messages.
  const exportBeforeRecovery = await storage.buildStoredChatExport();
  assert.equal(backend.importedThreadIds.size, 0);
  assert.equal(store.get(LEGACY_IMPORT_HINT), undefined);
  assert.deepEqual(
    (exportBeforeRecovery.messages as MessageRecord[])
      .map((record) => `${record.threadId}:${record.id}`)
      .sort(),
    ["legacy1:keep", "legacy1:remove"],
  );

  // Add a backend-only thread with the same message id to prove that the
  // deletion recorded below remains scoped to its original thread.
  backend.threads.set(otherThread.id, otherThread);
  backend.messages.set(otherThread.id, [sameIdElsewhere]);

  // Run the real single-message UI flow through its repository conversion,
  // durable tombstone, and backend-only prune.
  let repository = ExportedMessageRepository.fromBranchableArray(
    [
      {
        message: {
          id: kept.id,
          role: "user",
          content: [{ type: "text", text: "keep me" }],
          createdAt: new Date(kept.createdAt),
        },
        parentId: null,
      },
      {
        message: {
          id: removed.id,
          role: "assistant",
          content: [{ type: "text", text: "delete me" }],
          createdAt: new Date(removed.createdAt),
        },
        parentId: kept.id,
      },
    ],
    { headId: removed.id },
  );
  await deleteThreadMessage({
    thread: {
      export: () => repository,
      import: (next) => {
        repository = next;
      },
    },
    messageId: removed.id,
    remoteId: "legacy1",
  });
  assert.deepEqual(
    repository.messages.map(({ message: record }) => record.id),
    ["keep"],
  );
  assert.deepEqual(
    backend.messages.get("legacy1")?.map((record) => record.id),
    ["keep"],
  );
  assert.equal(
    messageTombstones.isChatMessageDeleted("legacy1", "remove"),
    true,
  );
  assert.equal(
    messageTombstones.isChatMessageDeleted("other", "remove"),
    false,
  );
  const persisted = JSON.parse(store.get(MESSAGE_TOMBSTONES_KEY) ?? "[]");
  assert.deepEqual(
    persisted.map((entry: { threadId: string; messageId: string }) => [
      entry.threadId,
      entry.messageId,
    ]),
    [["legacy1", "remove"]],
  );
  const hydratedTombstones = await import(
    `${
      new URL(
        "../src/features/chat/utils/chat-message-tombstones.ts",
        import.meta.url,
      ).href
    }?hydrate=restart`
  );
  assert.equal(
    hydratedTombstones.isChatMessageDeleted("legacy1", "remove"),
    true,
  );
  assert.equal(
    hydratedTombstones.isChatMessageDeleted("other", "remove"),
    false,
  );

  // The selective message read recovers. The rejected import promise must have
  // reset so this call retries, records its ledger row, and does not merge the
  // deleted legacy record back into studio.db.
  failBulkMessageRead = false;

  const threads = await storage.listStoredChatThreads();
  assert.deepEqual(
    threads.map((t) => t.id),
    ["other", "legacy1"],
  );
  assert.equal(backend.threads.get("legacy1")?.title, "Old chat");
  const messages = await storage.listStoredChatMessages("legacy1");
  assert.deepEqual(
    messages.map((m) => m.id),
    ["keep"],
  );
  assert.equal(
    await storage.getStoredChatMessage("legacy1", "remove"),
    undefined,
  );
  assert.equal(
    (await storage.getStoredChatMessage("other", "remove"))?.threadId,
    "other",
  );
  assert.deepEqual(
    (await storage.listStoredChatThreadsWithMessages()).map(
      (record) => record.id,
    ),
    ["other", "legacy1"],
  );
  const exported = await storage.buildStoredChatExport();
  assert.deepEqual(
    (exported.messages as MessageRecord[])
      .map((record) => `${record.threadId}:${record.id}`)
      .sort(),
    ["legacy1:keep", "other:remove"],
  );
  assert.equal(backend.importedThreadIds.has("legacy1"), true);
  assert.equal(store.get(LEGACY_IMPORT_HINT), "true");
  assert.deepEqual(
    backend.messages.get("legacy1")?.map((record) => record.id),
    ["keep"],
  );

  // A stale save or hydration snapshot cannot clear the durable deletion.
  await storage.saveStoredChatMessage(removed);
  await storage.syncStoredChatMessages("legacy1", [kept, removed]);
  assert.equal(
    messageTombstones.isChatMessageDeleted("legacy1", "remove"),
    true,
  );
  assert.deepEqual(
    (await storage.listStoredChatMessages("legacy1")).map(
      (record) => record.id,
    ),
    ["keep"],
  );
  assert.deepEqual(
    backend.messages.get("legacy1")?.map((record) => record.id),
    ["keep"],
  );
});

test("a failed backend prune rolls its message tombstone back", async () => {
  store.clear();
  messageTombstones.__resetChatMessageTombstonesForTests();
  const record = message("rollback", "t3", "user", "still here", 60);
  const backend = backendFake([thread("t3")], [record]);
  backend.api.syncChatMessages = async () => {
    throw new Error("backend prune failed");
  };
  setFakes(unreadableDexie().fake, backend.api);

  await assert.rejects(
    storage.syncStoredChatMessageDeletion("t3", [], [record.id]),
    /backend prune failed/,
  );
  assert.equal(messageTombstones.isChatMessageDeleted("t3", record.id), false);
  assert.deepEqual(JSON.parse(store.get(MESSAGE_TOMBSTONES_KEY) ?? "[]"), []);
});

test("a tombstone persistence failure aborts the backend prune", async () => {
  store.clear();
  messageTombstones.__resetChatMessageTombstonesForTests();
  const record = message(
    "not-persisted",
    "persist-failure",
    "user",
    "still here",
    62,
  );
  const backend = backendFake([thread("persist-failure")], [record]);
  let pruneCalls = 0;
  const syncChatMessages = backend.api.syncChatMessages;
  backend.api.syncChatMessages = (...args: never[]) => {
    pruneCalls += 1;
    return syncChatMessages(...args);
  };
  setFakes(unreadableDexie().fake, backend.api);
  const originalSetItem = localStorageFake.setItem;
  localStorageFake.setItem = () => {
    throw new Error("localStorage is full");
  };

  try {
    await assert.rejects(
      storage.syncStoredChatMessageDeletion("persist-failure", [], [record.id]),
      /Could not persist deleted chat messages/,
    );
  } finally {
    localStorageFake.setItem = originalSetItem;
  }

  assert.equal(pruneCalls, 0);
  assert.equal(
    messageTombstones.isChatMessageDeleted("persist-failure", record.id),
    false,
  );
  assert.equal(store.has(MESSAGE_TOMBSTONES_KEY), false);
  assert.deepEqual(
    backend.messages.get("persist-failure")?.map((entry) => entry.id),
    [record.id],
  );

  const restartedTombstones = await import(
    `${
      new URL(
        "../src/features/chat/utils/chat-message-tombstones.ts",
        import.meta.url,
      ).href
    }?hydrate=persist-failure`
  );
  assert.equal(
    restartedTombstones.isChatMessageDeleted("persist-failure", record.id),
    false,
  );
});

test("direct legacy thread hydration excludes deleted messages", async () => {
  store.clear();
  messageTombstones.__resetChatMessageTombstonesForTests();
  const legacyThread = thread("direct");
  const removed = message("direct-remove", "direct", "user", "gone", 65);
  const backend = backendFake([], []);
  messageTombstones.markChatMessagesDeleted("direct", [removed.id]);
  setFakes(readableDexie([legacyThread], [removed]), backend.api);

  assert.equal((await storage.getStoredChatThread("direct"))?.id, "direct");
  assert.equal(backend.threads.has("direct"), true);
  assert.deepEqual(backend.messages.get("direct") ?? [], []);
  assert.deepEqual(await storage.listStoredChatMessages("direct"), []);
  assert.equal(
    await storage.getStoredChatMessage("direct", removed.id),
    undefined,
  );
});

test("clearing a thread discards redundant message tombstones", async () => {
  store.clear();
  messageTombstones.__resetChatMessageTombstonesForTests();
  const record = message("clear-me", "t4", "user", "gone", 70);
  const backend = backendFake([thread("t4")], [record]);
  setFakes(unreadableDexie().fake, backend.api);
  messageTombstones.markChatMessagesDeleted("t4", [record.id]);

  const originalError = console.error;
  console.error = () => undefined;
  const result = await storage.clearStoredChats().finally(() => {
    console.error = originalError;
  });

  assert.equal(result.backend, "cleared");
  assert.equal(result.legacy, "failed");
  assert.deepEqual(result.deletedThreadIds, ["t4"]);
  assert.equal(messageTombstones.isChatMessageDeleted("t4", record.id), false);
  assert.deepEqual(JSON.parse(store.get(MESSAGE_TOMBSTONES_KEY) ?? "[]"), []);
});

test("editing a tombstoned stale snapshot does not restore it", async () => {
  store.clear();
  messageTombstones.__resetChatMessageTombstonesForTests();
  const backend = backendFake([thread("stale-thread")], []);
  setFakes(unreadableDexie().fake, backend.api);
  messageTombstones.markChatMessagesDeleted("stale-thread", ["stale-message"]);

  const staleExport = {
    messages: [
      {
        parentId: null,
        message: {
          id: "stale-message",
          role: "assistant",
          content: [{ type: "text", text: "deleted text" }],
          createdAt: new Date(10),
        },
      },
    ],
  };
  let currentExport = staleExport;

  await updateThreadMessage({
    thread: {
      export: () => currentExport as never,
      import: (value) => {
        currentExport = value as unknown as typeof staleExport;
      },
    },
    messageId: "stale-message",
    remoteId: "stale-thread",
    newText: "edited stale text",
    isIncognito: false,
  });

  assert.deepEqual(backend.messages.get("stale-thread") ?? [], []);
});

test("a tombstoned backend message is excluded from chat search", async () => {
  store.clear();
  messageTombstones.__resetChatMessageTombstonesForTests();
  const backend = backendFake(
    [thread("search-thread", { title: "Visible title" })],
    [message("deleted-message", "search-thread", "user", "secret needle", 10)],
  );
  setFakes(unreadableDexie().fake, backend.api);
  messageTombstones.markChatMessagesDeleted("search-thread", ["deleted-message"]);

  const index = await buildChatSearchIndex();

  assert.equal(index.some((item) => item.searchText.includes("secret needle")), false);
});
