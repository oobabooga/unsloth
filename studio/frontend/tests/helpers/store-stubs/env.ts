// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Stand-in for @/config/env: the platform store, with a settable device type. */
let deviceType: string | null = null;

export const usePlatformStore = {
  getState: () => ({ deviceType }),
  setState: (next: { deviceType: string | null }) => {
    deviceType = next.deviceType;
  },
};
