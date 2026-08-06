// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  DEFAULT_APP_WINDOW_SIZE_BOUNDS,
  type LogicalWindowSize,
  type WindowSizeBounds,
  calculateWindowSizeBounds,
} from "./window-layout.ts";

export type WindowLayoutGuard = () => boolean;

type WorkAreaMonitor = {
  scaleFactor: number;
  workArea: {
    size: {
      toLogical: (scaleFactor: number) => LogicalWindowSize;
    };
  };
};

export type MeasuredWindowLayout<Monitor extends WorkAreaMonitor> = {
  bounds: WindowSizeBounds;
  monitor: Monitor | null;
};

type WindowMonitorReader<Monitor extends WorkAreaMonitor> = {
  currentMonitor: () => Promise<Monitor | null>;
  primaryMonitor: () => Promise<Monitor | null>;
};

/** Size bounds the window has to stay within on its current monitor. */
export async function measureWindowLayout<Monitor extends WorkAreaMonitor>(
  { currentMonitor, primaryMonitor }: WindowMonitorReader<Monitor>,
  isCurrent: WindowLayoutGuard,
): Promise<MeasuredWindowLayout<Monitor> | null> {
  // A hidden window has no position to resolve a monitor from on some
  // platforms, so fall back to the primary one before giving up.
  const monitor = (await currentMonitor()) ?? (await primaryMonitor());
  if (!isCurrent()) return null;

  const bounds = monitor
    ? calculateWindowSizeBounds(
        monitor.workArea.size.toLogical(monitor.scaleFactor),
      )
    : DEFAULT_APP_WINDOW_SIZE_BOUNDS;
  return { bounds, monitor };
}

type FinalizeAppWindowLayoutOptions<Monitor extends WorkAreaMonitor> = {
  restored: boolean;
  measured: MeasuredWindowLayout<Monitor>;
  show: () => Promise<void>;
  measure: () => Promise<MeasuredWindowLayout<Monitor> | null>;
  setMinimumConstraints: (minimum: LogicalWindowSize) => Promise<void>;
  enforceBounds: (bounds: WindowSizeBounds) => Promise<void>;
  isCurrent: WindowLayoutGuard;
};

/** Shows the app window, then applies bounds from the visible monitor. */
export async function finalizeAppWindowLayout<Monitor extends WorkAreaMonitor>({
  restored,
  measured,
  show,
  measure,
  setMinimumConstraints,
  enforceBounds,
  isCurrent,
}: FinalizeAppWindowLayoutOptions<Monitor>): Promise<void> {
  await show();
  if (!isCurrent()) return;

  // A restored hidden window can resolve only the primary monitor even after
  // its saved position targets a secondary. Once visible, currentMonitor can
  // resolve the monitor that actually owns the restored window.
  if (restored) {
    measured = (await measure()) ?? measured;
    if (!isCurrent()) return;
  }

  await setMinimumConstraints(measured.bounds.minimum);
  if (!isCurrent()) return;
  // Restored geometry keeps the user's size and is only raised to the visible
  // monitor's minimum. A first layout is also capped to its measured work area.
  await enforceBounds(
    restored ? { minimum: measured.bounds.minimum } : measured.bounds,
  );
}
