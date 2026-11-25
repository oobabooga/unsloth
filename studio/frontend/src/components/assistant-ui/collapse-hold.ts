// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Bottom padding for the thread viewport (use-intent-aware-autoscroll.tsx). Pure, so each case is
// tested on numbers.

/** How much of the last message has to stay on screen below the header: its action bar and about
 *  three lines of text above it. */
export const REPLY_VISIBLE_PX = 144;

/**
 * Hold padding past this would push the reply out of view: a short answer under a tall fold leaves
 * a reader at the bottom looking at blank space. `tailPx` is what sits below the last message (the
 * composer's spacer), `topInsetPx` the header the thread scrolls under.
 */
export function collapsePaddingCap(input: {
  clientHeight: number;
  tailPx: number;
  topInsetPx: number;
  stabilizerMaxPx: number;
}): number {
  return Math.max(
    input.stabilizerMaxPx,
    input.clientHeight - input.tailPx - input.topInsetPx - REPLY_VISIBLE_PX,
  );
}

export type ChildBox = { top: number; height: number; position: "flow" | "sticky" | "out" };

/**
 * The thread's own height. scrollHeight never reads below the viewport, so a thread that fits is
 * measured from its children: the last in-flow bottom, plus sticky children after it (they keep
 * their flow space). Absolutely positioned children take none.
 */
export function naturalContentHeight(children: readonly ChildBox[]): number {
  let bottom = 0;
  let trailingSticky = 0;
  for (const child of children) {
    if (child.position === "flow") {
      bottom = Math.max(bottom, child.top + child.height);
      trailingSticky = 0;
    } else if (child.position === "sticky") {
      trailingSticky += child.height;
    }
  }
  return bottom + trailingSticky;
}

/**
 * The padding the viewport keeps below its content, recomputed every frame.
 *
 * - Growth uses padding up first, so the total height holds still while the reply fills the space.
 * - In a follow window (`followMax`, the tallest content in it), a shrink of up to `stabilizerMaxPx`
 *   is padded (a code block re-rendering); a larger one is a removal and re-anchors, as before.
 * - In a fold hold (`holdFloor`, the padded height when a thinking block began to fold), the
 *   padding keeps that height.
 *
 * Nothing else adds padding, so outside both it can only shrink. It never exceeds `capPx`, the cap
 * for the viewport as it is now: a window made shorter after a fold lowers it too.
 */
export function reconcilePadding(input: {
  content: number;
  lastContent: number;
  padding: number;
  followMax: number | null;
  holdFloor: number | null;
  stabilizerMaxPx: number;
  capPx: number;
}): { padding: number; followMax: number | null } {
  const { content } = input;
  let padding = Math.max(0, input.padding - Math.max(0, content - input.lastContent));
  let followMax = input.followMax;
  if (followMax !== null) {
    followMax = Math.max(followMax, content);
    const shrink = followMax - content;
    if (shrink <= input.stabilizerMaxPx) {
      padding = Math.max(padding, shrink);
    } else {
      followMax = content;
    }
  }
  if (input.holdFloor !== null) {
    padding = Math.max(padding, input.holdFloor - content);
  }
  return { padding: Math.min(padding, input.capPx), followMax };
}
