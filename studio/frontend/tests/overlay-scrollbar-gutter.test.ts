// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  OVERLAY_SCROLLBAR_GUTTER_VAR,
  applyOverlayScrollbarGutter,
  measureOverlayScrollbarGutter,
  watchOverlayScrollbarGutter,
} from "../src/lib/overlay-scrollbar.ts";

const PROBE_WIDTH = 60;

type Node = {
  style: { cssText: string };
  scrollTop: number;
  children: Node[];
  appendChild: (child: Node) => Node;
  offsetWidth: number;
  clientWidth: number;
  getBoundingClientRect: () => { top: number; right: number; height: number };
};

/**
 * `pointer-events` is inherited, so a probe that declares `auto` makes both
 * itself and its content hit-testable again under a `none` body.
 */
function optsIntoHitTesting(node: Node): boolean {
  return /(^|;)\s*pointer-events\s*:\s*auto\s*(;|$)/.test(node.style.cssText);
}

/**
 * Stands in for an engine whose scrollbar eats `railPx` of the scroller's inner
 * right edge while taking `layoutPx` of layout width. WebKitGTK 4.1 measures
 * (rail 21, layout 0); Chromium with this app's scrollbar CSS measures
 * (rail 0, layout 10). `bodyPointerEventsNone` stands in for an open Radix
 * modal, which sets `pointer-events: none` on the body.
 */
function fakeDocument({
  railPx,
  layoutPx,
  contentReachable = true,
  bodyPointerEventsNone = false,
}: {
  railPx: number;
  layoutPx: number;
  contentReachable?: boolean;
  bodyPointerEventsNone?: boolean;
}) {
  const vars = new Map<string, string>();
  const bodyChildren: Node[] = [];
  const documentElement = {
    style: {
      setProperty: (name: string, value: string) => vars.set(name, value),
      removeProperty: (name: string) => vars.delete(name),
    },
  };

  function createElement(): Node {
    const node: Node = {
      style: { cssText: "" },
      scrollTop: 0,
      children: [],
      appendChild: (child) => {
        node.children.push(child);
        return child;
      },
      offsetWidth: PROBE_WIDTH,
      clientWidth: PROBE_WIDTH - layoutPx,
      getBoundingClientRect: () => ({
        top: 0,
        right: PROBE_WIDTH,
        height: PROBE_WIDTH,
      }),
    };
    return node;
  }

  const doc = {
    createElement,
    documentElement,
    body: {
      appendChild: (child: Node) => {
        bodyChildren.push(child);
        return child;
      },
      removeChild: (child: Node) => {
        bodyChildren.splice(bodyChildren.indexOf(child), 1);
        return child;
      },
    },
    elementFromPoint: (x: number) => {
      const probe = bodyChildren[0];
      if (!probe) {
        return null;
      }
      // A Radix modal layer sets `pointer-events: none` on the body while it is
      // open, so hit-testing skips everything mounted under it and the point
      // resolves to the root element instead. Only a probe that opts back in
      // with `pointer-events: auto` stays answerable.
      if (bodyPointerEventsNone && !optsIntoHitTesting(probe)) {
        return documentElement;
      }
      // The rail sits on the trailing `railPx` columns of the padding box.
      if (x >= PROBE_WIDTH - railPx) {
        return probe;
      }
      return contentReachable ? probe.children[0] : probe;
    },
    addEventListener: () => undefined,
    removeEventListener: () => undefined,
  };

  return { doc: doc as unknown as Document, vars, bodyChildren };
}

test("an overlay scrollbar's hit strip is measured, not assumed", () => {
  // The number WebKitGTK 4.1 actually returns, and the reason the model
  // picker's gear was only clickable on its leftmost 3 of 20 px.
  const { doc, bodyChildren } = fakeDocument({ railPx: 21, layoutPx: 0 });

  assert.equal(measureOverlayScrollbarGutter(doc), 21);
  // The probe is torn down whatever the outcome, so it never paints or lingers
  // in front of the app.
  assert.deepEqual(bodyChildren, []);
});

test("a scrollbar that takes layout width reserves nothing", () => {
  // Chromium and WebView2: the scrollbar already displaced the content, so a
  // gutter on top of it would be a second, visible one.
  const { doc, vars } = fakeDocument({ railPx: 0, layoutPx: 10 });

  assert.equal(measureOverlayScrollbarGutter(doc), 0);
  applyOverlayScrollbarGutter(doc);
  assert.equal(vars.has(OVERLAY_SCROLLBAR_GUTTER_VAR), false);
});

test("an unreadable sweep leaves the layout alone rather than guessing", () => {
  const { doc, vars, bodyChildren } = fakeDocument({
    railPx: 0,
    layoutPx: 0,
    contentReachable: false,
  });

  assert.equal(applyOverlayScrollbarGutter(doc), 0);
  assert.equal(vars.has(OVERLAY_SCROLLBAR_GUTTER_VAR), false);
  assert.deepEqual(bodyChildren, []);
});

test("an open modal's pointer-events:none does not erase the gutter", () => {
  // A modal is up while the app is refocused or resized, so the re-measure runs
  // under it. Without the probe opting back into hit-testing the sweep reads
  // the root element everywhere, reports no gutter, and the rows snap back
  // under the rail.
  const { doc, vars } = fakeDocument({
    railPx: 21,
    layoutPx: 0,
    bodyPointerEventsNone: true,
  });

  assert.equal(measureOverlayScrollbarGutter(doc), 21);
  assert.equal(applyOverlayScrollbarGutter(doc), 21);
  assert.equal(vars.get(OVERLAY_SCROLLBAR_GUTTER_VAR), "21px");
});

test("the measured width is published in px for the CSS utility", () => {
  const { doc, vars } = fakeDocument({ railPx: 21, layoutPx: 0 });

  assert.equal(applyOverlayScrollbarGutter(doc), 21);
  assert.equal(vars.get(OVERLAY_SCROLLBAR_GUTTER_VAR), "21px");
});

test("regaining focus re-measures, so a changed scrollbar setting is picked up", () => {
  // macOS "Show scroll bars" is changed in System Settings, so the app is
  // always refocused afterwards. A boot-time-only probe would stay stale.
  let railPx = 0;
  const { doc, vars } = fakeDocument({ railPx: 0, layoutPx: 0 });
  const live = doc as unknown as {
    elementFromPoint: (x: number) => unknown;
    body: { appendChild: (c: unknown) => unknown };
  };
  const bodyProbes: { children: unknown[] }[] = [];
  const appendChild = live.body.appendChild;
  live.body.appendChild = (child: unknown) => {
    bodyProbes.push(child as { children: unknown[] });
    return appendChild(child);
  };
  live.elementFromPoint = (x: number) => {
    const probe = bodyProbes[bodyProbes.length - 1];
    if (x >= PROBE_WIDTH - railPx) {
      return probe;
    }
    return probe.children[0];
  };

  const handlers = new Map<string, () => void>();
  const win = {
    document: doc,
    addEventListener: (type: string, fn: () => void) => handlers.set(type, fn),
    removeEventListener: (type: string) => handlers.delete(type),
  } as unknown as Window;

  const stop = watchOverlayScrollbarGutter(win);
  assert.equal(vars.has(OVERLAY_SCROLLBAR_GUTTER_VAR), false);

  railPx = 15;
  handlers.get("focus")?.();
  assert.equal(vars.get(OVERLAY_SCROLLBAR_GUTTER_VAR), "15px");

  // And back again, so turning overlay scrollbars off drops the gutter rather
  // than leaving a permanent gap.
  railPx = 0;
  handlers.get("focus")?.();
  assert.equal(vars.has(OVERLAY_SCROLLBAR_GUTTER_VAR), false);

  stop();
  assert.equal(handlers.size, 0);
});

test("right-edge action lists reserve the gutter they publish", async () => {
  const css = await readFile(
    new URL("../src/index.css", import.meta.url),
    "utf8",
  );
  // The utility has to read the same variable the probe writes, or the gutter
  // silently stays at the 0px fallback on every platform.
  assert.match(
    css,
    new RegExp(
      `\\.overlay-scrollbar-gutter\\s*\\{[^}]*padding-right:\\s*var\\(${OVERLAY_SCROLLBAR_GUTTER_VAR},\\s*0px\\)`,
    ),
  );

  const pickers = await readFile(
    new URL(
      "../src/features/model-picker/components/model-selector/pickers.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  // One match rather than two greps over a 5000-line file: the utility is only
  // worth anything on the wrapper inside the list scroller, since the scroller
  // is the element the rail overlays and the wrapper is the one every row, and
  // so every gear, is laid out inside. The scroller is matched by the class
  // that names it plus the overflow that makes it a scroller, so editing the
  // rest of its class list does not fail this.
  assert.match(
    pickers,
    /"model-list-scroll[^"]*overflow-y-auto[^"]*"[\s\S]{0,800}"overlay-scrollbar-gutter",/,
  );

  const apiKeysTab = await readFile(
    new URL(
      "../src/features/settings/tabs/api-keys-tab.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  // Keep the scroller's existing padding for classic scrollbars, while an inner
  // wrapper moves every API-key row action clear of an overlay scrollbar.
  assert.match(
    apiKeysTab,
    /"hover-scrollbar[^"]*overflow-y-auto[^"]*\bpr-1\b[^"]*"[\s\S]{0,200}<div className="overlay-scrollbar-gutter">[\s\S]{0,300}<ApiKeyRow/,
  );

  const projectSourceDropzone = await readFile(
    new URL(
      "../src/features/rag/components/project-source-dropzone.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  // Staged-source rows put a small remove action at the list's right edge. The
  // scrolling list itself must reserve the measured strip so every row action
  // remains reachable once enough files are staged for the height cap to apply.
  assert.match(
    projectSourceDropzone,
    /<ul className="[^"]*overlay-scrollbar-gutter[^"]*max-h-52[^"]*overflow-y-auto[^"]*">[\s\S]{0,1000}aria-label={`Remove \${entry\.file\.name}`}/,
  );
});
