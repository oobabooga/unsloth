// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// bundler-resolver plus stand-ins for the three modules that stop a feature store
// from loading under bare node: the auth and hub barrels re-export .tsx components,
// and config/env reads import.meta.env (through lib/api-base) which only vite can
// supply. Everything else, including the store under test, is the real module.
import { resolve as resolveBundler } from "./bundler-resolver.mjs";

const STUBS = new Map([
  ["@/features/auth", "./helpers/store-stubs/auth.ts"],
  ["@/features/hub", "./helpers/store-stubs/hub.ts"],
  ["@/config/env", "./helpers/store-stubs/env.ts"],
]);

export function resolve(specifier, context, next) {
  const stub = STUBS.get(specifier);
  if (stub) {
    return next(new URL(stub, import.meta.url).href, context);
  }
  return resolveBundler(specifier, context, next);
}
