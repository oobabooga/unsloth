// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  MAX_AUDIO_SIZE,
  MAX_AUDIO_SIZE_LABEL,
} from "../src/lib/audio-utils.ts";

test("chat audio uses the backend's raw upload limit", () => {
  const backendLimits = readFileSync(
    new URL("../../backend/utils/upload_limits.py", import.meta.url),
    "utf8",
  );
  const limitMb = Number(
    backendLimits.match(
      /STT_AUDIO_RAW_MAX_BYTES\s*=\s*(\d+)\s*\*\s*_BYTES_PER_MB/,
    )?.[1],
  );

  assert.ok(Number.isInteger(limitMb));
  assert.equal(MAX_AUDIO_SIZE, limitMb * 1024 * 1024);
  assert.equal(MAX_AUDIO_SIZE_LABEL, `${limitMb}MB`);
});
