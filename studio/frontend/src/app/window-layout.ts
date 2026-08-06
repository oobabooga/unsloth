// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type LogicalWindowSize = {
  width: number;
  height: number;
};

export type PhysicalWindowRect = {
  position: { x: number; y: number };
  size: { width: number; height: number };
};

export type WindowSizeBounds = {
  minimum: LogicalWindowSize;
  maximum?: LogicalWindowSize;
};

export const MINIMUM_APP_WINDOW_SIZE: LogicalWindowSize = {
  width: 900,
  height: 600,
};

export const PREFERRED_SETUP_WINDOW_SIZE: LogicalWindowSize = {
  width: 760,
  height: 560,
};

// Used only when no monitor can be read, where the nominal minimum is the same
// blind guess the app has always made.
export const DEFAULT_APP_WINDOW_SIZE_BOUNDS: WindowSizeBounds = {
  minimum: MINIMUM_APP_WINDOW_SIZE,
};

// Share of the work area a relaxed minimum may claim. A minimum equal to the
// maximum would leave the window with no resize range at all.
const RELAXED_MINIMUM_RATIO = 0.85;
const FIRST_WINDOW_WIDTH_RATIO = 0.75;
const FIRST_WINDOW_HEIGHT_RATIO = 0.85;
const FIRST_WINDOW_ASPECT_RATIO = 1.618;

function relaxMinimum(preferred: number, maximum: number): number {
  if (preferred <= maximum) return preferred;
  return Math.max(1, Math.floor(maximum * RELAXED_MINIMUM_RATIO));
}

/**
 * Size bounds for a window that must stay inside the monitor work area, i.e.
 * the panel minus taskbars and docks. The window carries no native frame:
 * `setup_custom_titlebar` undecorates it on Windows and Linux and macOS uses an
 * overlay titlebar, so its outer size is its inner size.
 */
export function calculateWindowSizeBounds(
  workAreaSize: LogicalWindowSize,
): WindowSizeBounds {
  const maximum = {
    width: Math.max(1, Math.floor(workAreaSize.width)),
    height: Math.max(1, Math.floor(workAreaSize.height)),
  };
  return {
    minimum: {
      width: relaxMinimum(MINIMUM_APP_WINDOW_SIZE.width, maximum.width),
      height: relaxMinimum(MINIMUM_APP_WINDOW_SIZE.height, maximum.height),
    },
    maximum,
  };
}

export function fitWindowSize(
  size: LogicalWindowSize,
  maximum?: LogicalWindowSize,
): LogicalWindowSize {
  if (!maximum) return size;
  return {
    width: Math.min(size.width, maximum.width),
    height: Math.min(size.height, maximum.height),
  };
}

export function calculateFirstAppWindowSize(
  { minimum, maximum }: WindowSizeBounds,
  cssSafeLogicalWidth?: number,
): LogicalWindowSize {
  if (!maximum) return minimum;

  const width = Math.max(
    minimum.width,
    Math.round(maximum.width * FIRST_WINDOW_WIDTH_RATIO),
    Math.min(cssSafeLogicalWidth ?? 0, maximum.width),
  );
  // Leave a margin around a roomy work area, but never trade away height the
  // app actually wants on a short panel.
  const heightCap = Math.max(
    MINIMUM_APP_WINDOW_SIZE.height,
    Math.round(maximum.height * FIRST_WINDOW_HEIGHT_RATIO),
  );
  const height = Math.max(
    minimum.height,
    Math.min(Math.round(width / FIRST_WINDOW_ASPECT_RATIO), heightCap),
  );
  return fitWindowSize({ width, height }, maximum);
}

export function constrainWindowSize(
  currentSize: LogicalWindowSize,
  requestedSize: LogicalWindowSize,
  { minimum, maximum }: WindowSizeBounds,
): LogicalWindowSize {
  return fitWindowSize(
    {
      width: Math.max(currentSize.width, minimum.width, requestedSize.width),
      height: Math.max(
        currentSize.height,
        minimum.height,
        requestedSize.height,
      ),
    },
    maximum,
  );
}

/**
 * Physical position that centers `windowSize` inside the work area. Tauri's own
 * `center()` reads the size from the window instead, which on Linux is the GTK
 * configure-event cache, so right after a resize it centers the previous size.
 */
export function calculateCenteredPosition(
  workArea: PhysicalWindowRect,
  windowSize: { width: number; height: number },
): { x: number; y: number } {
  return {
    x:
      workArea.position.x +
      Math.max(0, Math.floor((workArea.size.width - windowSize.width) / 2)),
    y:
      workArea.position.y +
      Math.max(0, Math.floor((workArea.size.height - windowSize.height) / 2)),
  };
}
