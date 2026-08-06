// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Durable exclusions for messages deleted from studio.db while the legacy
 * Dexie database is unavailable. Tombstones are scoped by both thread and
 * message id so an id reused by another thread remains visible.
 */

interface MessageTombstone {
  threadId: string;
  messageId: string;
  deletedAt: number;
}

const TOMBSTONES_KEY = "unsloth_chat_deleted_messages";
const TOMBSTONE_MAX_AGE_MS = 90 * 24 * 60 * 60 * 1000; // 90 days
const TOMBSTONE_MAX_COUNT = 10000;

const deletedMessages = new Map<string, MessageTombstone>();

function canUseStorage(): boolean {
  return typeof window !== "undefined";
}

function tombstoneKey(threadId: string, messageId: string): string {
  return JSON.stringify([threadId, messageId]);
}

function isMessageTombstone(value: unknown): value is MessageTombstone {
  return (
    typeof value === "object" &&
    value !== null &&
    typeof (value as MessageTombstone).threadId === "string" &&
    typeof (value as MessageTombstone).messageId === "string" &&
    typeof (value as MessageTombstone).deletedAt === "number"
  );
}

function loadTombstones(): MessageTombstone[] {
  if (!canUseStorage()) return [];
  try {
    const raw = JSON.parse(localStorage.getItem(TOMBSTONES_KEY) ?? "[]");
    return Array.isArray(raw) ? raw.filter(isMessageTombstone) : [];
  } catch {
    return [];
  }
}

function gc(): void {
  const cutoff = Date.now() - TOMBSTONE_MAX_AGE_MS;
  for (const [key, tombstone] of deletedMessages) {
    if (tombstone.deletedAt < cutoff) deletedMessages.delete(key);
  }
  if (deletedMessages.size > TOMBSTONE_MAX_COUNT) {
    const sorted = Array.from(deletedMessages.entries()).sort(
      (a, b) => a[1].deletedAt - b[1].deletedAt,
    );
    const drop = sorted.slice(0, deletedMessages.size - TOMBSTONE_MAX_COUNT);
    for (const [key] of drop) deletedMessages.delete(key);
  }
}

function persist(): boolean {
  if (!canUseStorage()) return false;
  try {
    localStorage.setItem(
      TOMBSTONES_KEY,
      JSON.stringify(Array.from(deletedMessages.values())),
    );
    return true;
  } catch {
    return false;
  }
}

for (const tombstone of loadTombstones()) {
  deletedMessages.set(
    tombstoneKey(tombstone.threadId, tombstone.messageId),
    tombstone,
  );
}
gc();

export function isChatMessageDeleted(
  threadId: string,
  messageId: string,
): boolean {
  return deletedMessages.has(tombstoneKey(threadId, messageId));
}

export function markChatMessagesDeleted(
  threadId: string,
  messageIds: Iterable<string>,
): void {
  const previous = new Map(deletedMessages);
  const deletedAt = Date.now();
  let changed = false;
  for (const messageId of messageIds) {
    const key = tombstoneKey(threadId, messageId);
    if (deletedMessages.has(key)) continue;
    deletedMessages.set(key, {
      threadId,
      messageId,
      deletedAt,
    });
    changed = true;
  }
  if (!changed) return;
  gc();
  if (!persist()) {
    deletedMessages.clear();
    for (const [key, tombstone] of previous) {
      deletedMessages.set(key, tombstone);
    }
    throw new Error("Could not persist deleted chat messages");
  }
}

export function removeChatMessageTombstones(
  threadId: string,
  messageIds: Iterable<string>,
): void {
  let changed = false;
  for (const messageId of messageIds) {
    if (deletedMessages.delete(tombstoneKey(threadId, messageId))) {
      changed = true;
    }
  }
  if (changed) persist();
}

export function removeChatMessageTombstonesForThreads(
  threadIds: Iterable<string>,
): void {
  const ids = new Set(threadIds);
  let changed = false;
  for (const [key, tombstone] of deletedMessages) {
    if (ids.has(tombstone.threadId)) {
      deletedMessages.delete(key);
      changed = true;
    }
  }
  if (changed) persist();
}

export function __resetChatMessageTombstonesForTests(): void {
  deletedMessages.clear();
}
