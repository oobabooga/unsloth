// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

let dbUrl;
let apiUrl;
let apiExports;

export function initialize(data) {
  dbUrl = data.dbUrl;
  apiUrl = data.apiUrl;
  apiExports = data.apiExports;
}

export function load(url, context, nextLoad) {
  if (url === dbUrl) {
    return {
      format: "module",
      shortCircuit: true,
      source: [
        "export const DEXIE_DB_NAME = 'unsloth-chat';",
        "export const db = {",
        "  get threads() { return globalThis.__dexieFake.threads; },",
        "  get messages() { return globalThis.__dexieFake.messages; },",
        "  transaction(...args) { return globalThis.__dexieFake.transaction(...args); },",
        "};",
      ].join("\n"),
    };
  }
  if (url === apiUrl) {
    return {
      format: "module",
      shortCircuit: true,
      source: apiExports
        .map(
          (name) =>
            `export function ${name}(...args) {
  const fn = globalThis.__chatApiFake.${name};
  if (typeof fn !== "function") {
    throw new Error("unexpected chat-api call: ${name}");
  }
  return fn(...args);
}`,
        )
        .join("\n"),
    };
  }
  return nextLoad(url, context);
}
