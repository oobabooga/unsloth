// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import {
  type ChildBox,
  collapsePaddingCap,
  naturalContentHeight,
  reconcilePadding,
} from "@/components/assistant-ui/collapse-hold";
import { ROW_DISTANT_ATTRIBUTE } from "@/components/assistant-ui/distant-rows";
import { useChatPreferencesStore } from "@/features/chat/stores/chat-preferences-store";
import { useAui, useAuiEvent } from "@assistant-ui/react";
import {
  type ReactNode,
  type RefCallback,
  createContext,
  useCallback,
  useContext,
  useMemo,
  useRef,
  useSyncExternalStore,
} from "react";

/**
 * Replaces assistant-ui's autoscroll, whose observers write `isAtBottom` on every layout change
 * and race our correction. A rAF loop pins until a follow deadline; upward intent detaches, and
 * scrolling back within 24px of the bottom re-attaches.
 */

// 2px, not 1: HiDPI subpixel rounding can leave a fractional gap.
const AT_BOTTOM_THRESHOLD_PX = 2;
const RE_ATTACH_THRESHOLD_PX = 24;
const TOUCH_MOVE_THRESHOLD_PX = 4;
// Summed, not per event, so 1px-per-event sources still detach.
const UPWARD_DETACH_THRESHOLD_PX = 2;
const SCROLL_KEY_WINDOW_MS = 500;
// Window the viewport stays pinned through layout/content races. Extends on every resize/mutation,
// so streaming keeps it pinned; settles this long after the last change.
const FOLLOW_SETTLE_MS = 600;
// Content can grow with no mutation or border-box resize (image decode, fonts, KaTeX), so a
// quiet pinned frame re-checks layout on this timer.
const SETTLE_CHECK_MS = 100;
// Absorbs sub-frame transients (shiki re-renders); larger shrinks are real removals and re-pin.
const STABILIZER_MAX_PX = 64;
// A thinking block folding as the answer starts drops a few hundred px at once. For this long after
// it starts to fold, the thread keeps its height in bottom padding (reconcilePadding).
const COLLAPSE_HOLD_MS = 1000;

export type ScrollToBottom = (behavior?: ScrollBehavior) => void;

type AutoScrollContextValue = {
  scrollToBottom: ScrollToBottom;
  getIsAtBottom: () => boolean;
  subscribe: (listener: () => void) => () => void;
  /** Detach as if scrolled up: composer growth must not let observer pins shove the chat up. */
  detachFromBottom: () => void;
  /**
   * Called on every widening commit, zero included, since this hook owns scrollTop and resyncs
   * intent.
   */
  adjustForContentInsertedAbove: (deltaPx: number) => boolean;
  notifyContentResized: () => void;
  /** A block is about to fold (called while it is still open): hold the thread's height. */
  holdThroughCollapse: () => void;
};

const noopContext: AutoScrollContextValue = {
  scrollToBottom: () => {
    /* no viewport mounted */
  },
  getIsAtBottom: () => true,
  subscribe: () => () => {
    /* no-op */
  },
  detachFromBottom: () => {
    /* no viewport mounted */
  },
  adjustForContentInsertedAbove: () => false,
  notifyContentResized: () => undefined,
  holdThroughCollapse: () => undefined,
};

const AutoScrollContext = createContext<AutoScrollContextValue>(noopContext);

export function IntentAwareScrollProvider({
  value,
  children,
}: {
  value: AutoScrollContextValue;
  children: ReactNode;
}) {
  return (
    <AutoScrollContext.Provider value={value}>
      {children}
    </AutoScrollContext.Provider>
  );
}

export function useScrollThreadToBottom(): ScrollToBottom {
  return useContext(AutoScrollContext).scrollToBottom;
}

/** See AutoScrollContextValue.adjustForContentInsertedAbove. */
export function useAdjustForContentInsertedAbove(): (deltaPx: number) => boolean {
  return useContext(AutoScrollContext).adjustForContentInsertedAbove;
}

/** Opening a collapsible near the bottom must grow it downward, not pin and shove the header. */
export function useDetachThreadFromBottom(): () => void {
  return useContext(AutoScrollContext).detachFromBottom;
}

/** See AutoScrollContextValue.holdThroughCollapse. */
export function useHoldThreadThroughCollapse(): () => void {
  return useContext(AutoScrollContext).holdThroughCollapse;
}

export function useIsThreadAtBottom(): boolean {
  const ctx = useContext(AutoScrollContext);
  return useSyncExternalStore(ctx.subscribe, ctx.getIsAtBottom, () => true);
}

export function useIntentAwareAutoScroll(): {
  ref: RefCallback<HTMLElement>;
  context: AutoScrollContextValue;
} {
  const aui = useAui();
  const cleanupRef = useRef<(() => void) | null>(null);

  const userDetachedRef = useRef(false);
  const followUntilRef = useRef(0);
  const runStartedHereRef = useRef(false);
  // runEnd fires before the last chunk commits, so the hold outlives it briefly.
  const runEndAtRef = useRef(Number.NEGATIVE_INFINITY);

  const isAtBottomRef = useRef(true);
  const listenersRef = useRef<Set<() => void>>(new Set());

  const scrollImplRef = useRef<ScrollToBottom>(() => {
    /* no viewport mounted */
  });
  const detachImplRef = useRef<() => void>(() => {
    /* no viewport mounted */
  });
  const adjustImplRef = useRef<(deltaPx: number) => void>(() => {
    /* no viewport mounted */
  });
  const tickRequestRef = useRef<() => void>(() => undefined);
  // Outside the closure, so a rebind to the same element keeps them: the fold hold, and the content
  // height the padding last saw (growth since then uses the padding up).
  const holdRef = useRef<{ floor: number; until: number } | null>(null);
  const lastContentRef = useRef<number | null>(null);
  const holdImplRef = useRef<() => void>(() => undefined);

  const getIsAtBottom = useCallback(() => isAtBottomRef.current, []);

  const subscribe = useCallback((listener: () => void) => {
    listenersRef.current.add(listener);
    return () => {
      listenersRef.current.delete(listener);
    };
  }, []);

  const setIsAtBottom = useCallback((value: boolean) => {
    if (isAtBottomRef.current === value) {
      return;
    }
    isAtBottomRef.current = value;
    for (const listener of listenersRef.current) {
      listener();
    }
  }, []);

  const scrollToBottom = useCallback<ScrollToBottom>((behavior) => {
    runStartedHereRef.current = false;
    scrollImplRef.current(behavior);
  }, []);

  const detachFromBottom = useCallback(() => {
    detachImplRef.current();
  }, []);

  const adjustForContentInsertedAbove = useCallback((deltaPx: number) => {
    adjustImplRef.current(deltaPx);
    return userDetachedRef.current;
  }, []);

  const notifyContentResized = useCallback(() => {
    tickRequestRef.current();
  }, []);

  const holdThroughCollapse = useCallback(() => {
    holdImplRef.current();
  }, []);

  const attach = useCallback(
    (el: HTMLElement, isRebind: boolean) => {
      let rafId: number | null = null;
      let settleTimer: number | null = null;
      let settleCheckDue = false;
      let layoutChanged = true;
      let lastScrollTop = el.scrollTop;
      let lastClientWidth = el.clientWidth;
      let lastClientHeight = el.clientHeight;
      let upwardAccumulator = 0;
      let touchStartY = 0;
      let scrollKeyAt = Number.NEGATIVE_INFINITY;
      let pointerHeld = false;
      let insertionRemainder = 0;
      // Set when distant-rows.ts skips or releases a row, so a clamped scrollTop is not a scroll
      // up.
      let rowsToggledSinceScroll = false;

      const distanceFromBottom = (): number => {
        if (el.scrollHeight <= el.clientHeight) {
          return 0;
        }
        return el.scrollHeight - el.scrollTop - el.clientHeight;
      };
      let lastDistanceFromBottom = distanceFromBottom();
      let lastScrollHeight = el.scrollHeight;

      const atBottomStrict = (): boolean =>
        distanceFromBottom() <= AT_BOTTOM_THRESHOLD_PX;

      // On the viewport itself, matched as the subject: reaching it through the hidden host made
      // WebKit restyle the whole thread on every flip (40-70 ms each at 150 turns).
      const setFollowing = (on: boolean): void => {
        el.toggleAttribute("data-aui-following", on);
      };

      // Upward wheel/touch detaches only with room above, else auto-follow stays dead for the
      // session.
      const canScrollUp = (): boolean => el.scrollTop > 0;

      // An inner scroller with room above consumes the upward delta, so it must not detach.
      const innerScrollWillConsumeUpward = (
        target: EventTarget | null,
      ): boolean => {
        let node =
          target instanceof Element ? (target as Element | null) : null;
        while (node && node !== el) {
          if (node.scrollTop > 0) {
            const { overflowY } = window.getComputedStyle(node);
            if (
              overflowY === "auto" ||
              overflowY === "scroll" ||
              overflowY === "overlay"
            ) {
              return true;
            }
          }
          node = node.parentElement;
        }
        return false;
      };

      // Stabilizer state lives in this closure so it resets when the viewport remounts. A rebind to
      // the same element takes over the padding on it; starting from zero would leave that padding
      // unowned, read as content and never released.
      let stabilizerPx = Number.parseFloat(el.style.getPropertyValue("--aui-scroll-stabilizer")) || 0;
      let maxContentHeight = 0;

      const setStabilizer = (px: number): void => {
        if (px === stabilizerPx) {
          return;
        }
        stabilizerPx = px;
        if (px === 0) {
          el.style.removeProperty("--aui-scroll-stabilizer");
        } else {
          el.style.setProperty("--aui-scroll-stabilizer", `${px}px`);
        }
      };

      const releaseStabilizer = (): void => {
        setStabilizer(0);
      };

      // Dropping the padding with the reader closer to the bottom than its height would clamp them.
      // A reader that far up is past any fold too, so the hold ends with it rather than re-pad.
      const releaseStabilizerIfSafe = (): void => {
        if (stabilizerPx > 0 && distanceFromBottom() >= stabilizerPx) {
          releaseStabilizer();
          holdRef.current = null;
        }
      };

      // See naturalContentHeight: scrollHeight floors at the viewport height.
      const contentHeight = (): number => {
        if (el.scrollHeight > el.clientHeight + 1) {
          return el.scrollHeight - stabilizerPx;
        }
        const origin = el.getBoundingClientRect().top - el.scrollTop;
        const children: ChildBox[] = [];
        for (const child of el.children) {
          const style = getComputedStyle(child);
          const box = child.getBoundingClientRect();
          children.push({
            top: box.top - origin,
            // The last child's bottom margin is part of the scroll height.
            height: box.height + (Number.parseFloat(style.marginBottom) || 0),
            position:
              style.position === "sticky"
                ? "sticky"
                : style.position === "absolute" || style.position === "fixed"
                  ? "out"
                  : "flow",
          });
        }
        return naturalContentHeight(children);
      };
      // A different element is another thread: neither the hold nor the height carries over.
      if (!isRebind || lastContentRef.current === null) {
        holdRef.current = null;
        lastContentRef.current = contentHeight();
      }

      const holdActive = (): boolean => {
        const hold = holdRef.current;
        if (hold !== null && performance.now() >= hold.until) {
          holdRef.current = null;
        }
        return holdRef.current !== null;
      };

      // See collapsePaddingCap: the last message's last lines stay below the header.
      const collapseCap = (): number => {
        const rows = el.querySelectorAll("[data-role]");
        const last = rows[rows.length - 1];
        const box = el.getBoundingClientRect();
        const lastBottom = last ? last.getBoundingClientRect().bottom - box.top + el.scrollTop : 0;
        return collapsePaddingCap({
          clientHeight: el.clientHeight,
          tailPx: Math.max(0, contentHeight() - lastBottom),
          topInsetPx: Number.parseFloat(getComputedStyle(el).paddingTop) || 0,
          stabilizerMaxPx: STABILIZER_MAX_PX,
        });
      };

      const extendFollow = (): void => {
        if (userDetachedRef.current) {
          return;
        }
        followUntilRef.current = performance.now() + FOLLOW_SETTLE_MS;
      };

      const holdStill = (): boolean =>
        runStartedHereRef.current &&
        !useChatPreferencesStore.getState().autoScrollWhileGenerating &&
        (aui.thread().getState().isRunning ||
          performance.now() - runEndAtRef.current < FOLLOW_SETTLE_MS);

      const holdCeiling = (): number | null => {
        const rows = el.querySelectorAll<HTMLElement>("[data-role]");
        let reply: HTMLElement | null = null;
        let user: HTMLElement | null = null;
        for (let i = rows.length - 1; i >= 0 && !user; i--) {
          if (rows[i].dataset.role === "user") {
            user = rows[i];
          } else if (rows[i].dataset.role === "assistant") {
            reply = rows[i];
          }
        }
        if (!user || !reply) {
          return null;
        }
        const origin =
          el.getBoundingClientRect().top -
          el.scrollTop +
          (Number.parseFloat(getComputedStyle(el).paddingTop) || 0);
        return Math.max(
          0,
          user.getBoundingClientRect().top - origin,
          reply.getBoundingClientRect().top - origin - el.clientHeight / 2,
        );
      };

      const clearSettleCheck = (): void => {
        if (settleTimer !== null) {
          clearTimeout(settleTimer);
          settleTimer = null;
        }
        settleCheckDue = false;
      };

      let parked = false;
      let pendingInitialPin = false;

      const detach = (): void => {
        parked = false;
        userDetachedRef.current = true;
        followUntilRef.current = 0;
        // Hygiene: `following` checks userDetached first, so a queued check cannot re-pin anyway.
        clearSettleCheck();
        // Drop residual stabilizer padding once the user is far enough up that it cannot clamp them;
        // onScroll drops it later otherwise.
        releaseStabilizerIfSafe();
        maxContentHeight = contentHeight();
      };

      const parkIfHeld = (): boolean => {
        if (userDetachedRef.current || !holdStill()) {
          return false;
        }
        const ceiling = holdCeiling();
        if (ceiling === null || el.scrollHeight - el.clientHeight < ceiling) {
          return false;
        }
        pendingInitialPin = false;
        el.scrollTo({ top: ceiling, behavior: "instant" });
        detach();
        parked = true;
        return true;
      };

      const requestTick = (): void => {
        if (rafId === null) {
          rafId = requestAnimationFrame(tick);
        }
      };
      tickRequestRef.current = requestTick;

      // A quiet pinned frame hands the window to this self-rearming timer. See SETTLE_CHECK_MS.
      const scheduleSettleCheck = (): void => {
        if (settleTimer !== null) {
          return;
        }
        const remaining = followUntilRef.current - performance.now();
        if (remaining <= 0) {
          requestTick();
          return;
        }
        settleTimer = window.setTimeout(
          () => {
            settleTimer = null;
            // The window may have closed by then; grant that frame one last follow pass.
            settleCheckDue = true;
            requestTick();
          },
          Math.min(SETTLE_CHECK_MS, remaining),
        );
      };

      // Edge-triggered rAF loop: re-arms only while layout is still moving.
      const tick = (): void => {
        rafId = null;
        const settling = settleCheckDue;
        settleCheckDue = false;
        // Park first so a frame without an observer record can't overshoot.
        parkIfHeld();
        if (pendingInitialPin && el.querySelector("[data-role]") !== null) {
          pendingInitialPin = false;
          if (el.scrollHeight > el.clientHeight) {
            parked = false;
            userDetachedRef.current = false;
            extendFollow();
          }
        }
        const following =
          !userDetachedRef.current &&
          (settling || performance.now() < followUntilRef.current);

        if (following) {
          setFollowing(true);
          const scrollHeight = stabilize();
          const pinned = atBottomStrict();
          if (!pinned && scrollHeight > el.clientHeight) {
            el.scrollTo({ top: scrollHeight, behavior: "instant" });
          }
          setIsAtBottom(true);
          // A fold can shrink with no mutation to wake the loop (an animated close), so it runs on.
          if (layoutChanged || !pinned || holdActive()) {
            layoutChanged = false;
            requestTick();
            return;
          }
          scheduleSettleCheck();
          return;
        }

        setFollowing(false);
        // Padding keeps reconciling for a reader who is not following: growth uses it up, and a
        // fold near the bottom is held as for a following reader.
        if (stabilizerPx > 0 || holdActive()) {
          stabilize();
          if (holdActive()) {
            requestTick();
          }
        }
        setIsAtBottom(atBottomStrict());
      };

      scrollImplRef.current = (behavior = "auto") => {
        parked = false;
        userDetachedRef.current = false;
        followUntilRef.current = performance.now() + FOLLOW_SETTLE_MS;
        if (el.scrollHeight > el.clientHeight) {
          pendingInitialPin = false;
          el.scrollTo({ top: el.scrollHeight, behavior });
        }
        setIsAtBottom(true);
        requestTick();
      };

      detachImplRef.current = () => {
        detach();
        requestTick();
      };

      // Correct prepends before paint: a queued follow tick can run a frame late.
      // This hook owns scrollTop and resyncs intent after either kind of correction.
      // Use instant scrolling despite the viewport's scroll-smooth class.
      adjustImplRef.current = (deltaPx: number) => {
        if (!userDetachedRef.current) {
          insertionRemainder = 0;
          if (!parkIfHeld()) {
            extendFollow();
            pinIfFollowing(stabilize());
          }
        } else if (deltaPx !== 0) {
          const top = Math.max(0, Math.min(
            el.scrollTop + deltaPx + insertionRemainder,
            el.scrollHeight - el.clientHeight,
          ));
          el.scrollTo({ top, behavior: "instant" });
          // WebKit rounds scrollTop; carry the fraction across background prepends.
          insertionRemainder = top - el.scrollTop;
        }
        // Resync lastScrollTop even with no write: native anchoring may have moved scrollTop and its
        // scroll event would otherwise read as a downward scroll that re-attaches the user.
        lastScrollTop = el.scrollTop;
        lastDistanceFromBottom = distanceFromBottom();
        lastScrollHeight = el.scrollHeight;
      };

      const onWheel = (e: WheelEvent) => {
        if (
          e.deltaY < 0 &&
          canScrollUp() &&
          !innerScrollWillConsumeUpward(e.target)
        ) {
          pendingInitialPin = false;
          detach();
        }
      };

      const onKeyDown = (e: KeyboardEvent) => {
        if (
          e.defaultPrevented ||
          !(
            e.key === "ArrowUp" ||
            e.key === "PageUp" ||
            e.key === "Home" ||
            (e.key === " " && e.shiftKey)
          )
        ) {
          return;
        }
        const target = e.target instanceof HTMLElement ? e.target : null;
        if (
          target?.isContentEditable ||
          target?.closest("input, textarea, select")
        ) {
          return;
        }
        scrollKeyAt = performance.now();
      };

      const onPointerDown = () => {
        pointerHeld = true;
      };
      const onPointerUp = () => {
        pointerHeld = false;
      };

      const onTouchStart = (e: TouchEvent) => {
        touchStartY = e.touches[0]?.clientY ?? 0;
      };

      const onTouchMove = (e: TouchEvent) => {
        const y = e.touches[0]?.clientY ?? 0;
        // Finger moves DOWN on the screen = content scrolls UP.
        if (
          y - touchStartY > TOUCH_MOVE_THRESHOLD_PX &&
          canScrollUp() &&
          !innerScrollWillConsumeUpward(e.target)
        ) {
          pendingInitialPin = false;
          detach();
        }
      };

      const onScroll = () => {
        if (userDetachedRef.current) {
          // A fold's clamp can be reported here before the frame's tick: pad and restore first, so the
          // clamped position never becomes the baseline.
          if (holdActive()) {
            stabilize();
          }
          releaseStabilizerIfSafe();
        }
        const scrollTop = el.scrollTop;
        const clientWidth = el.clientWidth;
        const clientHeight = el.clientHeight;
        const sizeChanged =
          clientWidth !== lastClientWidth || clientHeight !== lastClientHeight;

        const delta = scrollTop - lastScrollTop;
        const distanceNow = distanceFromBottom();
        const scrollHeight = el.scrollHeight;

        // Resizes can clamp scrollTop and fake direction; only deliberate scrolls flip intent.
        if (sizeChanged) {
          upwardAccumulator = 0;
        } else if (delta > 0) {
          upwardAccumulator = 0;
          if (
            userDetachedRef.current &&
            distanceNow <= RE_ATTACH_THRESHOLD_PX &&
            !holdStill()
          ) {
            userDetachedRef.current = false;
            extendFollow();
          }
        } else if (delta < 0 && !userDetachedRef.current) {
          // Count distance-from-bottom growth, not scrollTop: anchoring moves scrollTop, distance stays put.
          const distanceDelta = distanceNow - lastDistanceFromBottom;
          const keyed =
            performance.now() - scrollKeyAt < SCROLL_KEY_WINDOW_MS;
          // Leaving a pinned bottom in the same event the content changed, with no key or pointer
          // behind it, is a clamp (something below shrank as a new row landed), not the user.
          const clamped =
            lastDistanceFromBottom <= AT_BOTTOM_THRESHOLD_PX &&
            (scrollHeight !== lastScrollHeight || rowsToggledSinceScroll) &&
            !keyed &&
            !pointerHeld;
          if (
            distanceDelta > 0 &&
            !clamped &&
            (keyed || scrollHeight >= lastScrollHeight)
          ) {
            upwardAccumulator += distanceDelta;
            if (upwardAccumulator >= UPWARD_DETACH_THRESHOLD_PX) {
              pendingInitialPin = false;
              detach();
              upwardAccumulator = 0;
            }
          }
        }

        rowsToggledSinceScroll = false;
        lastScrollTop = scrollTop;
        lastClientWidth = clientWidth;
        lastClientHeight = clientHeight;
        lastDistanceFromBottom = distanceNow;
        lastScrollHeight = scrollHeight;
        requestTick();
      };

      // Keeps the thread's height through a shrink that is not the reader's doing (reconcilePadding): a
      // finalizing code block in the follow window, a thinking block folding in a hold. The shortfall
      // goes into `--aui-scroll-stabilizer` padding.
      const stabilize = (): number => {
        const followActive =
          !userDetachedRef.current &&
          performance.now() < followUntilRef.current;
        const hold = holdActive() ? holdRef.current : null;
        const before = stabilizerPx;
        // Read before the padding changes: a reader at the very end was clamped there, not scrolling.
        const clamped = el.scrollTop >= el.scrollHeight - el.clientHeight - 1;
        const content = contentHeight();
        const next = reconcilePadding({
          content,
          lastContent: lastContentRef.current ?? content,
          padding: stabilizerPx,
          followMax: followActive ? maxContentHeight : null,
          holdFloor: hold ? hold.floor : null,
          stabilizerMaxPx: STABILIZER_MAX_PX,
          // Only padding past the ordinary stabilizer can push the reply out of view; skip the row query
          // otherwise.
          capPx: hold || stabilizerPx > STABILIZER_MAX_PX ? collapseCap() : STABILIZER_MAX_PX,
        });
        lastContentRef.current = content;
        maxContentHeight = next.followMax ?? content;
        setStabilizer(next.padding);
        // A reader who is not pinned was clamped when the shrink was laid out; the padding makes their
        // place reachable again, so put them back before paint.
        if (stabilizerPx > before && !followActive && clamped && el.scrollTop < lastScrollTop - 0.5) {
          el.scrollTo({ top: lastScrollTop, behavior: "instant" });
        }
        return content + stabilizerPx;
      };

      // Observer callbacks run after layout and before paint, so this lands in the same frame.
      const pinIfFollowing = (scrollHeight: number): void => {
        if (userDetachedRef.current) {
          return;
        }
        if (performance.now() >= followUntilRef.current) {
          return;
        }
        if (scrollHeight <= el.clientHeight) {
          return;
        }
        el.scrollTo({ top: scrollHeight, behavior: "instant" });
      };

      // Order matters: extend first so the stabilizer sees follow active, then stabilize, then pin.
      const onLayoutChange = (): void => {
        layoutChanged = true;
        if (!parkIfHeld()) {
          extendFollow();
        }
        const scrollHeight = stabilize();
        pinIfFollowing(scrollHeight);
        requestTick();
      };

      const onMutation = (records: MutationRecord[]): void => {
        if (records.some((record) => record.attributeName === ROW_DISTANT_ATTRIBUTE)) {
          rowsToggledSinceScroll = true;
        }
        if (el.ownerDocument.hidden) {
          onLayoutChange();
          return;
        }
        layoutChanged = true;
        if (!parkIfHeld()) {
          extendFollow();
        }
        requestTick();
      };

      // Measured while the block is still open: its content shrinks after this commit.
      holdImplRef.current = () => {
        holdRef.current = {
          floor: contentHeight() + stabilizerPx,
          until: performance.now() + COLLAPSE_HOLD_MS,
        };
        requestTick();
      };

      const resizeObserver = new ResizeObserver(onLayoutChange);
      const mutationObserver = new MutationObserver(onMutation);
      const onViewportResize = onLayoutChange;

      maxContentHeight = lastContentRef.current;

      // Only a fresh attach pins: React re-runs the composed ref on unrelated renders for the same element.
      if (!isRebind) {
        userDetachedRef.current = false;

        // Pin on first attach, covering thread.initialize firing before the ref is bound.
        extendFollow();
        if (el.scrollHeight > el.clientHeight) {
          el.scrollTo({ top: el.scrollHeight, behavior: "instant" });
        } else {
          pendingInitialPin = true;
        }
        setIsAtBottom(true);
      }
      requestTick();

      // Border box: the stabilizer's padding writes would otherwise echo back as resizes.
      resizeObserver.observe(el, { box: "border-box" });
      mutationObserver.observe(el, {
        childList: true,
        subtree: true,
        characterData: true,
        // Excludes `style` to avoid a feedback loop; `class` and the rest catch collapsibles.
        attributes: true,
        attributeFilter: [
          "class",
          "hidden",
          "aria-hidden",
          "aria-expanded",
          "data-state",
          ROW_DISTANT_ATTRIBUTE,
        ],
      });
      el.addEventListener("wheel", onWheel, { passive: true });
      el.addEventListener("touchstart", onTouchStart, { passive: true });
      el.addEventListener("touchmove", onTouchMove, { passive: true });
      el.addEventListener("scroll", onScroll, { passive: true });
      el.ownerDocument.addEventListener("keydown", onKeyDown, {
        passive: true,
      });
      el.addEventListener("pointerdown", onPointerDown, { passive: true });
      el.ownerDocument.addEventListener("pointerup", onPointerUp, true);
      el.ownerDocument.addEventListener("pointercancel", onPointerUp, true);
      // visualViewport resize is the only signal for iOS keyboard changes, where clientHeight stays
      // put.
      window.visualViewport?.addEventListener("resize", onViewportResize);

      const unsubscribePreferences = useChatPreferencesStore.subscribe(
        (state, prev) => {
          if (
            !parked ||
            !state.autoScrollWhileGenerating ||
            prev.autoScrollWhileGenerating ||
            !aui.thread().getState().isRunning
          ) {
            return;
          }
          parked = false;
          userDetachedRef.current = false;
          extendFollow();
          requestTick();
        },
      );

      return () => {
        unsubscribePreferences();
        if (rafId !== null) {
          cancelAnimationFrame(rafId);
          rafId = null;
        }
        clearSettleCheck();
        setFollowing(false);
        resizeObserver.disconnect();
        mutationObserver.disconnect();
        el.removeEventListener("wheel", onWheel);
        el.removeEventListener("touchstart", onTouchStart);
        el.removeEventListener("touchmove", onTouchMove);
        el.removeEventListener("scroll", onScroll);
        el.ownerDocument.removeEventListener("keydown", onKeyDown);
        el.removeEventListener("pointerdown", onPointerDown);
        el.ownerDocument.removeEventListener("pointerup", onPointerUp, true);
        el.ownerDocument.removeEventListener("pointercancel", onPointerUp, true);
        window.visualViewport?.removeEventListener("resize", onViewportResize);
        scrollImplRef.current = () => {
          /* no viewport mounted */
        };
        detachImplRef.current = () => {
          /* no viewport mounted */
        };
        adjustImplRef.current = () => {
          /* no viewport mounted */
        };
        tickRequestRef.current = () => undefined;
        holdImplRef.current = () => undefined;
      };
    },
    [aui, setIsAtBottom],
  );

  // Thread lifecycle moments that always pin, regardless of detach state.
  const pinToBottom = useCallback((behavior: ScrollBehavior) => {
    userDetachedRef.current = false;
    scrollImplRef.current(behavior);
  }, []);

  useAuiEvent("thread.runStart", () => {
    runStartedHereRef.current = true;
    runEndAtRef.current = Number.NEGATIVE_INFINITY;
    pinToBottom("instant");
  });
  useAuiEvent("thread.runEnd", () => {
    runEndAtRef.current = performance.now();
  });
  useAuiEvent("thread.initialize", () => {
    runStartedHereRef.current = false;
    pinToBottom("instant");
  });
  useAuiEvent("threadListItem.switchedTo", () => {
    runStartedHereRef.current = false;
    pinToBottom("instant");
  });

  const lastElRef = useRef<HTMLElement | null>(null);
  const ref = useCallback<RefCallback<HTMLElement>>(
    (el) => {
      if (cleanupRef.current) {
        cleanupRef.current();
        cleanupRef.current = null;
      }
      if (el) {
        const isRebind = lastElRef.current === el;
        lastElRef.current = el;
        cleanupRef.current = attach(el, isRebind);
      }
      // On null, keep lastElRef so a rebind to the same element is recognized.
    },
    [attach],
  );

  const context = useMemo<AutoScrollContextValue>(
    () => ({
      scrollToBottom,
      getIsAtBottom,
      subscribe,
      detachFromBottom,
      adjustForContentInsertedAbove,
      notifyContentResized,
      holdThroughCollapse,
    }),
    [
      scrollToBottom,
      getIsAtBottom,
      subscribe,
      detachFromBottom,
      adjustForContentInsertedAbove,
      notifyContentResized,
      holdThroughCollapse,
    ],
  );

  return { ref, context };
}
