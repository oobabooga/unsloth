// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A thinking block folds when the answer starts and drops a few hundred px at once. Without a hold, a
// reader at or just above the bottom has scrollTop clamped and the whole conversation jumps down.
// The padding maths is tested on numbers; the browser behaviour is in
// tests/studio/playwright_chat_autoscroll.py, and the .tsx wiring is pinned from source.

import assert from "node:assert/strict";
import test from "node:test";

import {
  REPLY_VISIBLE_PX,
  collapsePaddingCap,
  naturalContentHeight,
  reconcilePadding,
} from "../src/components/assistant-ui/collapse-hold.ts";
import { readSrc } from "./helpers/kit.ts";

const HOOK = readSrc("components/assistant-ui/use-intent-aware-autoscroll.tsx");
const REASONING = readSrc("components/assistant-ui/reasoning.tsx");

const MAX = 64;
const BIG_CAP = 10_000;

type Step = { content: number; followMax?: number | null; holdFloor?: number | null; capPx?: number };

/** Runs frames through reconcilePadding the way the hook does, returning the padding after each. */
function run(start: { content: number; padding?: number }, steps: Step[]): number[] {
  let lastContent = start.content;
  let padding = start.padding ?? 0;
  const out: number[] = [];
  for (const step of steps) {
    const next = reconcilePadding({
      content: step.content,
      lastContent,
      padding,
      followMax: step.followMax ?? null,
      holdFloor: step.holdFloor ?? null,
      stabilizerMaxPx: MAX,
      capPx: step.capPx ?? BIG_CAP,
    });
    lastContent = step.content;
    padding = next.padding;
    out.push(padding);
  }
  return out;
}

test("a fold in the hold keeps the thread's height, and the answer uses the padding up", () => {
  // 3000 px of content when the block starts to fold; it loses 320, then the answer grows 120 and 300.
  const floor = 3000;
  assert.deepEqual(
    run({ content: 3000 }, [
      { content: 2680, holdFloor: floor },
      { content: 2800, holdFloor: floor },
      { content: 3100, holdFloor: floor },
    ]),
    [320, 200, 0],
  );
});

test("an animated fold is held frame by frame", () => {
  const floor = 3000;
  assert.deepEqual(
    run({ content: 3000 }, [
      { content: 2900, holdFloor: floor },
      { content: 2780, holdFloor: floor },
      { content: 2680, holdFloor: floor },
    ]),
    [100, 220, 320],
  );
});

test("growth uses up padding for a reader who is not following, so scrolling back down never goes backward", () => {
  // 200 px kept after a fold, the hold over; the answer grows 400 px while the reader is away.
  assert.deepEqual(run({ content: 2800, padding: 200 }, [{ content: 3000 }, { content: 3200 }]), [0, 0]);
  // Outside a hold and a follow window nothing adds padding: a removal is not padded.
  assert.deepEqual(run({ content: 3000, padding: 50 }, [{ content: 2700 }]), [50]);
});

test("the follow window pads small shrinks only, as before", () => {
  assert.deepEqual(run({ content: 3000 }, [{ content: 2960, followMax: 3000 }]), [40]);
  assert.deepEqual(run({ content: 3000 }, [{ content: 2900, followMax: 3000 }]), [0]);
  const removal = reconcilePadding({
    content: 2900, lastContent: 3000, padding: 0, followMax: 3000, holdFloor: null,
    stabilizerMaxPx: MAX, capPx: BIG_CAP,
  });
  assert.equal(removal.followMax, 2900, "a removal re-anchors the follow window");
});

test("a tall fold is held only while the reply's last lines stay below the header", () => {
  // A 245 px composer spacer under the last message and a 48 px header, in a 600 px window.
  const cap = collapsePaddingCap({ clientHeight: 600, tailPx: 245, topInsetPx: 48, stabilizerMaxPx: MAX });
  assert.equal(cap, 600 - 245 - 48 - REPLY_VISIBLE_PX);
  assert.deepEqual(run({ content: 3000 }, [{ content: 2450, holdFloor: 3000, capPx: cap }]), [cap]);
  assert.ok(600 - cap - 245 >= 48 + REPLY_VISIBLE_PX, "pinned with that padding, the reply ends below the header");
  assert.equal(collapsePaddingCap({ clientHeight: 300, tailPx: 245, topInsetPx: 48, stabilizerMaxPx: MAX }), MAX);
});

test("a thread the fold leaves shorter than the viewport is measured from its children", () => {
  // scrollHeight reads 1000 for this thread; its children end at 800 (the last block, then a sticky
  // footer that keeps its flow space; the absolute one takes none).
  const content = naturalContentHeight([
    { top: 48, height: 126, position: "flow" },
    { top: 174, height: 448, position: "flow" },
    { top: 47, height: 1, position: "out" },
    { top: 622, height: 150, position: "flow" },
    { top: 940, height: 28, position: "sticky" },
  ]);
  assert.equal(content, 800);
  // It was 1080 tall (scrollTop 80 at the bottom); holding 1080 keeps scrollTop 80 reachable.
  assert.deepEqual(run({ content: 1080 }, [{ content, holdFloor: 1080 }]), [280]);
});

test("a closing thinking block holds the thread from a layout effect, while it is still open", () => {
  assert.match(
    REASONING,
    /useLayoutEffect\(\(\) => \{\s*if \(wasOpenRef\.current && !isOpen\) \{\s*holdThroughCollapse\(\);\s*\}\s*wasOpenRef\.current = isOpen;\s*\}, \[isOpen, holdThroughCollapse\]\);/,
  );
  assert.match(
    HOOK,
    /holdImplRef\.current = \(\) => \{\s*holdRef\.current = \{\s*floor: contentHeight\(\) \+ stabilizerPx,\s*until: performance\.now\(\) \+ COLLAPSE_HOLD_MS,/,
    "the floor is the padded height before the fold, so what folds with the block is held too",
  );
  assert.match(HOOK, /const holdRef = useRef</, "the hold lives outside the attach closure, so a rebind keeps it");
  assert.match(
    HOOK,
    /let stabilizerPx = Number\.parseFloat\(el\.style\.getPropertyValue\("--aui-scroll-stabilizer"\)\) \|\| 0;/,
    "a rebind takes over the padding on the element",
  );
});

test("padding is dropped only once dropping it cannot clamp the reader", () => {
  const start = HOOK.indexOf("const detach = (): void => {");
  const detach = HOOK.slice(start, HOOK.indexOf("\n      };", start));
  assert.match(detach, /releaseStabilizerIfSafe\(\);/);
  assert.doesNotMatch(detach, /releaseStabilizer\(\);/);
  assert.match(HOOK, /if \(stabilizerPx > 0 && distanceFromBottom\(\) >= stabilizerPx\) \{\s*releaseStabilizer\(\);/);
});
