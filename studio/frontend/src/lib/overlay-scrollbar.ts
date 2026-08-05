// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Overlay scrollbars (WebKitGTK on Linux, WKWebView on macOS) take no layout
 * width, so `offsetWidth - clientWidth` is 0 and nothing in the layout moves
 * out of their way. They still hit-test though: a strip along the scroller's
 * inner right edge swallows pointer events before they reach the content
 * underneath, which is why a button flush to that edge cannot be clicked.
 * `scrollbar-gutter: stable` is defined as a no-op for overlay scrollbars, so
 * it does not help.
 *
 * The width of that strip is decided by the engine and the platform theme, not
 * by our CSS, so it is measured rather than assumed. The result is published as
 * `--overlay-scrollbar-gutter` on the root element for `.overlay-scrollbar-gutter`
 * to reserve; on engines whose scrollbars take layout space it stays unset and
 * the utility resolves to 0px.
 */

export const OVERLAY_SCROLLBAR_GUTTER_VAR = "--overlay-scrollbar-gutter";

/** Widest strip we will believe in. Real values are ~0-30px; anything past this
 *  means the sweep is measuring something other than a scrollbar. */
const MAX_GUTTER_PX = 48;

/** Re-measuring is a forced layout, so resize bursts are coalesced. */
const RESIZE_SETTLE_MS = 200;

/**
 * Measures how far inward from a scroller's padding-box right edge pointer
 * events are taken by the scrollbar instead of reaching the content beneath it.
 *
 * Returns 0 when the scrollbar takes layout width (classic scrollbars, so no
 * overlap is possible), when the document is not ready, or when the sweep finds
 * no content at all, which keeps the caller on today's layout rather than
 * reserving space on a guess.
 */
export function measureOverlayScrollbarGutter(doc: Document): number {
  const body = doc.body;
  if (!body) {
    return 0;
  }

  const probe = doc.createElement("div");
  // Inside the viewport because elementFromPoint only answers for points that
  // are in it, invisible because it is torn down before the frame is painted,
  // and above everything so an open dialog cannot answer the sweep for it.
  // `pointer-events:auto` opts the probe (and, by inheritance, its content)
  // back into hit-testing: a Radix modal layer sets `pointer-events:none` on
  // the body while it is open, which would otherwise leave the sweep unable to
  // see either of them and drop the gutter for as long as the modal is up.
  probe.style.cssText =
    "position:fixed;top:0;left:0;width:60px;height:60px;margin:0;border:0;padding:0;opacity:0;overflow-y:scroll;pointer-events:auto;z-index:2147483647";
  const content = doc.createElement("div");
  content.style.cssText = "width:100%;height:300px";
  probe.appendChild(content);
  body.appendChild(probe);

  try {
    if (probe.offsetWidth - probe.clientWidth > 0) {
      return 0;
    }

    // Where the scrollbar only appears while scrolling (macOS "Show scroll bars:
    // When scrolling"), measuring a resting scroller reports the rail as absent.
    // Scrolling the probe first asks the platform for the state the user is
    // actually in when they reach for a row action.
    probe.scrollTop = 1;

    const rect = probe.getBoundingClientRect();
    const right = Math.round(rect.right);
    const y = Math.round(rect.top + rect.height / 2);

    let gutter = 0;
    for (let offset = 1; offset <= MAX_GUTTER_PX; offset++) {
      if (doc.elementFromPoint(right - offset, y) === content) {
        return gutter;
      }
      gutter = offset;
    }
    return 0;
  } finally {
    body.removeChild(probe);
  }
}

/**
 * Measures and publishes the gutter on the root element. The variable is only
 * written when there is something to reserve, so classic-scrollbar engines fall
 * through to the 0px default and keep byte-identical layout.
 */
export function applyOverlayScrollbarGutter(doc: Document): number {
  const gutter = measureOverlayScrollbarGutter(doc);
  const root = doc.documentElement;
  if (gutter > 0) {
    root.style.setProperty(OVERLAY_SCROLLBAR_GUTTER_VAR, `${gutter}px`);
  } else {
    root.style.removeProperty(OVERLAY_SCROLLBAR_GUTTER_VAR);
  }
  return gutter;
}

/**
 * Keeps the published gutter in step with the things that can change it while
 * the app is open: display scaling and browser zoom (resize), and the macOS
 * "Show scroll bars" setting, which is changed in System Settings and so is
 * always followed by the window regaining focus.
 */
export function watchOverlayScrollbarGutter(win: Window): () => void {
  const doc = win.document;
  let resizeTimer: ReturnType<typeof setTimeout> | undefined;

  const remeasure = () => applyOverlayScrollbarGutter(doc);
  const onResize = () => {
    if (resizeTimer !== undefined) {
      clearTimeout(resizeTimer);
    }
    resizeTimer = setTimeout(remeasure, RESIZE_SETTLE_MS);
  };

  remeasure();
  win.addEventListener("resize", onResize);
  win.addEventListener("focus", remeasure);
  doc.addEventListener("visibilitychange", remeasure);

  return () => {
    if (resizeTimer !== undefined) {
      clearTimeout(resizeTimer);
    }
    win.removeEventListener("resize", onResize);
    win.removeEventListener("focus", remeasure);
    doc.removeEventListener("visibilitychange", remeasure);
  };
}
