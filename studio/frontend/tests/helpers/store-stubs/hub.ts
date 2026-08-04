// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Stand-in for @/features/hub: only the HF-token surface the store reaches for. */
let token = "";

export function getHfToken(): string {
  return token;
}

export function hubTokenHeader(): Record<string, string> {
  return {};
}

export function mirrorHfTokenInto(): void {
  // Nothing to mirror into: the stub keeps the token in this module.
}

export const useHfTokenStore = {
  getState: () => ({
    setToken: (next: string) => {
      token = next;
    },
  }),
};
