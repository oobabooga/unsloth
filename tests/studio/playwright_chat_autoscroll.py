# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-frame cost and scroll-intent checks of the chat autoscroll loop, under Playwright."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import (  # noqa: E402
    chromium_launch_args,
    echo_browser_errors,
    start_vite,
    stop_process,
    wait_for_smoke_page,
)

PORT = int(os.environ.get("SMOKE_PORT", "5193"))
_EXTERNAL = os.environ.get("SMOKE_BASE_URL", "").strip()
BASE = _EXTERNAL or f"http://127.0.0.1:{PORT}"
OWNS_SERVER = not _EXTERNAL
LABEL = os.environ.get("SMOKE_LABEL", "tree")
OUT = Path(os.environ.get("PW_ART_DIR", "logs/playwright-chat-autoscroll"))
OUT.mkdir(parents = True, exist_ok = True)

# 250ms per token is the research synthesis cadence where the old loop was wasteful.
TOKEN_COUNT = int(os.environ.get("SMOKE_TOKEN_COUNT", "32"))
TOKEN_GAP_MS = int(os.environ.get("SMOKE_TOKEN_GAP_MS", "250"))
# One frame per token is 4/s here; the old loop ran ~62/s. 25/s leaves room for React frames.
MAX_STREAM_RAF_PER_SECOND = float(os.environ.get("SMOKE_MAX_RAF_PER_S", "25"))
FOLLOW_SETTLE_MS = 600
# Silent growth is followed on a timer; measured 115ms.
SILENT_GROWTH_REPIN_BUDGET_MS = int(os.environ.get("SMOKE_REPIN_BUDGET_MS", "250"))

PUMP_INIT = """
(() => {
  window.__longTasks = [];
  try {
    new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        window.__longTasks.push({ start: entry.startTime, duration: entry.duration });
      }
    }).observe({ type: "longtask", buffered: true });
  } catch (e) { /* longtask unsupported: the CDP metrics still apply */ }
  window.__rafCount = 0;
  let nextHandle = 1;
  const pending = new Map();
  window.requestAnimationFrame = (cb) => {
    const handle = nextHandle++;
    pending.set(
      handle,
      setTimeout(() => {
        pending.delete(handle);
        window.__rafCount += 1;
        cb(performance.now());
      }, 16),
    );
    return handle;
  };
  window.cancelAnimationFrame = (handle) => {
    const timer = pending.get(handle);
    if (timer !== undefined) {
      clearTimeout(timer);
      pending.delete(handle);
    }
  };
})();
"""


# A thinking block folding as the answer starts (collapse-hold.ts). Reduced motion folds in one step;
# the viewport is short so a first turn can come to fit inside it.
FOLD_VIEWPORT = {"width": 1440, "height": 600}

FOLD_CASES = (
    "pinnedFits",
    "pinnedFitsSettled",
    "detachedFits",
    "sibling",
    "reuse",
    "rebind",
    "detachedGrowth",
)

# One case per page load, so padding one case leaves cannot reach the next.
FOLD_CASE = """
async (name) => {
  const a = window.__autoscroll;
  const frames = (n) => new Promise((resolve) => {
    const step = (left) => (left <= 0 ? resolve() : requestAnimationFrame(() => step(left - 1)));
    step(n);
  });
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  // A first turn that overflows by `overflow` px, sized with the reply's other content so the block
  // keeps its height. Sized from an overflowing layout: scrollHeight never reads below the viewport.
  const scene = async (blockCount, overflow, base) => {
    a.foldScene(blockCount, { ...base, extraPx: 3000 });
    await frames(3);
    const m = a.metrics();
    const extraPx = 3000 - (m.scrollHeight - m.clientHeight - overflow);
    a.foldScene(blockCount, { ...base, extraPx });
    await frames(3);
    a.scrollToBottom();
    await wait(100);
  };
  const foldAndRead = async (collapsedPx) => {
    const before = { ...a.foldTops(), top: a.metrics().scrollTop };
    a.fold(collapsedPx);
    await frames(4);
    const after = { ...a.foldTops(), top: a.metrics().scrollTop, clientHeight: a.metrics().clientHeight };
    return { before, after };
  };
  const firstTurn = { foldPx: 188, siblingPx: 0, extraPx: 0 };
  if (name === "pinnedFits") {
    // A first turn at the bottom: 80 px of overflow, a 160 px fold, and the thread then fits.
    await scene(0, 80, firstTurn);
    return foldAndRead(28);
  }
  if (name === "pinnedFitsSettled") {
    await scene(0, 80, firstTurn);
    await wait(900);
    return foldAndRead(28);
  }
  if (name === "detachedFits") {
    await scene(0, 80, firstTurn);
    a.scrollUpBy(24);
    await wait(200);
    return foldAndRead(28);
  }
  // A long thread for these two, sized by its messages, not by the block.
  const longThread = async (base) => {
    a.foldScene(30, base);
    await frames(3);
    a.scrollToBottom();
    await wait(100);
  };
  if (name === "sibling") {
    // A tool run folding with the block.
    await longThread({ foldPx: 100, siblingPx: 160, extraPx: 0 });
    return foldAndRead(28);
  }
  if (name === "rebind") {
    // A one-word answer re-renders the page, whose inline ref rebinds the hook to the same element.
    await longThread({ foldPx: 348, siblingPx: 0, extraPx: 0 });
    const out = await foldAndRead(28);
    a.token("OK");
    await frames(4);
    out.afterRebind = { ...a.foldTops(), top: a.metrics().scrollTop, clientHeight: a.metrics().clientHeight };
    return out;
  }
  if (name === "detachedGrowth") {
    // 100 px up when the block folds, then the answer grows past the fold while the reader is away;
    // scrolling back down must never move backward.
    await longThread({ foldPx: 348, siblingPx: 0, extraPx: 0 });
    a.scrollUpBy(100);
    await wait(200);
    const out = await foldAndRead(28);
    for (let i = 0; i < 30; i += 1) a.token("A sentence of the answer that wraps across the column. ");
    await frames(4);
    out.paddingAfterAnswer = a.foldTops().padding;
    const tops = [a.metrics().scrollTop];
    for (let i = 0; i < 40 && a.distanceFromBottom() > 0; i += 1) {
      a.scrollDownBy(60);
      await frames(2);
      tops.push(a.metrics().scrollTop);
    }
    out.downTops = tops;
    return out;
  }
  // The answer uses up the fold's padding, then something else is removed within the same second.
  await longThread({ foldPx: 348, siblingPx: 0, extraPx: 180 });
  const out = await foldAndRead(28);
  out.paddingAfterFold = a.foldTops().padding;
  for (let i = 0; i < 40; i += 1) a.token("A sentence of the answer that wraps across the column. ");
  await frames(4);
  out.paddingAfterAnswer = a.foldTops().padding;
  a.removeExtra();
  await frames(4);
  out.paddingAfterRemoval = a.foldTops().padding;
  return out;
}
"""


def info(message: str) -> None:
    print(f"[chat-autoscroll] {message}", flush = True)


def metrics(cdp) -> dict[str, float]:
    got = cdp.send("Performance.getMetrics")
    return {m["name"]: m["value"] for m in got["metrics"]}


def delta(before: dict[str, float], after: dict[str, float], name: str) -> float:
    return round(after.get(name, 0.0) - before.get(name, 0.0), 4)


def run() -> dict:
    results: dict = {"label": LABEL, "base": BASE}
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless = os.environ.get("SMOKE_HEADLESS", "1") == "1",
            args = chromium_launch_args(),
        )
        context = browser.new_context(viewport = {"width": 1440, "height": 900})
        context.add_init_script(PUMP_INIT)
        context.add_init_script(
            "localStorage.setItem('unsloth_auth_token', 'chat-autoscroll-smoke');"
        )
        context.route(
            re.compile(rf"^{re.escape(BASE)}/api/"),
            lambda route: route.fulfill(status = 200, content_type = "application/json", body = "{}"),
        )
        page = context.new_page()
        echo_browser_errors(page, info)
        page.goto(f"{BASE}/smoke-autoscroll.html", wait_until = "domcontentloaded")
        page.wait_for_function("() => Boolean(window.__autoscroll)", timeout = 30_000)
        cdp = context.new_cdp_session(page)
        cdp.send("Performance.enable")

        page.evaluate("window.__autoscroll.seed(40)")
        page.wait_for_timeout(800)
        results["seeded"] = page.evaluate("window.__autoscroll.metrics()")

        # Clock and counter must start and stop inside the page or the rate drifts across round trips.
        before = metrics(cdp)
        streamed = page.evaluate(
            """async ([count, gap, tailMs]) => {
                window.__longTasks.length = 0;
                window.__rafCount = 0;
                const started = performance.now();
                for (let i = 0; i < count; i += 1) {
                    window.__autoscroll.token("token " + i + " ");
                    // A finished message every 20 tokens, so childList mutations are in the mix
                    // alongside the characterData ones.
                    if (i > 0 && i % 20 === 0) window.__autoscroll.block();
                    await new Promise((r) => setTimeout(r, gap));
                }
                // The tail is part of the measurement: re-arming after the last token is the
                // failure mode, so count those frames rather than stopping the clock early.
                await new Promise((r) => setTimeout(r, tailMs));
                return { wallMs: performance.now() - started, rafCallbacks: window.__rafCount };
            }""",
            [TOKEN_COUNT, TOKEN_GAP_MS, 1200],
        )
        after = metrics(cdp)
        long_tasks = page.evaluate("window.__longTasks")
        results["stream"] = {
            "tokens": TOKEN_COUNT,
            "wall_ms": round(streamed["wallMs"], 1),
            "raf_callbacks": streamed["rafCallbacks"],
            "long_tasks": len(long_tasks),
            "worst_long_task_ms": round(max((t["duration"] for t in long_tasks), default = 0.0), 1),
            "layout_count": delta(before, after, "LayoutCount"),
            "recalc_style_count": delta(before, after, "RecalcStyleCount"),
            "layout_ms": round(delta(before, after, "LayoutDuration") * 1000, 1),
            "recalc_style_ms": round(delta(before, after, "RecalcStyleDuration") * 1000, 1),
            "task_ms": round(delta(before, after, "TaskDuration") * 1000, 1),
            "pinned_after_stream": page.evaluate("window.__autoscroll.distanceFromBottom()"),
            "is_at_bottom": page.evaluate("window.__autoscroll.isAtBottom()"),
        }

        page.evaluate("window.__rafCount = 0")
        page.wait_for_timeout(2000)
        results["idle_raf_per_2s"] = page.evaluate("window.__rafCount")

        # Growth as an inline style reaches neither observer.
        page.evaluate("window.__autoscroll.resetGrowth()")
        page.wait_for_timeout(900)
        settled = page.evaluate(
            """async ([followMs]) => {
                window.__autoscroll.token("x ");
                // 100ms in: inside the follow window, which is where a decoding image lands.
                await new Promise((r) => setTimeout(r, 100));
                const start = performance.now();
                const heightBefore = window.__autoscroll.metrics().scrollHeight;
                window.__autoscroll.growSilently(240);
                const grewBy = window.__autoscroll.metrics().scrollHeight - heightBefore;
                let repinnedAfterMs = null;
                // Watch past the window's end plus a margin, so a settle check at the deadline
                // is still counted rather than reported as a permanent failure to follow.
                while (performance.now() - start < followMs + 600) {
                    if (window.__autoscroll.distanceFromBottom() <= 2) {
                        repinnedAfterMs = performance.now() - start;
                        break;
                    }
                    await new Promise((r) => setTimeout(r, 8));
                }
                return {
                    grewBy,
                    repinnedAfterMs,
                    distanceAtEnd: window.__autoscroll.distanceFromBottom(),
                    isAtBottom: window.__autoscroll.isAtBottom(),
                };
            }""",
            [FOLLOW_SETTLE_MS],
        )
        results["silent_growth"] = settled

        page.evaluate("window.__autoscroll.resetGrowth()")
        page.evaluate("window.__autoscroll.scrollToBottom()")
        page.wait_for_timeout(900)
        intent = page.evaluate(
            """async () => {
                window.__autoscroll.scrollUpBy(400);
                await new Promise((r) => setTimeout(r, 200));
                const detached = !window.__autoscroll.isAtBottom();
                const distanceAfterDetach = window.__autoscroll.distanceFromBottom();
                for (let i = 0; i < 12; i += 1) {
                    window.__autoscroll.token("more " + i + " ");
                    await new Promise((r) => setTimeout(r, 40));
                }
                await new Promise((r) => setTimeout(r, 900));
                const stillDetached = !window.__autoscroll.isAtBottom();
                const grewWhileDetached =
                    window.__autoscroll.distanceFromBottom() > distanceAfterDetach;
                // Scroll back to within the re-attach threshold.
                const before = window.__autoscroll.distanceFromBottom();
                window.__autoscroll.scrollDownBy(before);
                await new Promise((r) => setTimeout(r, 400));
                return {
                    detached,
                    stillDetached,
                    grewWhileDetached,
                    reattached: window.__autoscroll.isAtBottom(),
                };
            }"""
        )
        results["intent"] = intent

        context.close()

        fold_context = browser.new_context(viewport = FOLD_VIEWPORT, reduced_motion = "reduce")
        fold_context.add_init_script(
            "localStorage.setItem('unsloth_auth_token', 'chat-autoscroll-smoke');"
        )
        fold_context.route(
            re.compile(rf"^{re.escape(BASE)}/api/"),
            lambda route: route.fulfill(status = 200, content_type = "application/json", body = "{}"),
        )
        fold_page = fold_context.new_page()
        echo_browser_errors(fold_page, info)
        results["fold"] = {}
        for name in FOLD_CASES:
            fold_page.goto(f"{BASE}/smoke-autoscroll.html", wait_until = "domcontentloaded")
            fold_page.wait_for_function("() => Boolean(window.__autoscroll)", timeout = 30_000)
            results["fold"][name] = fold_page.evaluate(FOLD_CASE, name)
        fold_context.close()
        browser.close()
    return results


def fold_failures(fold: dict) -> list[str]:
    failures: list[str] = []

    def still(name: str, case: dict) -> None:
        before, after = case["before"], case["after"]
        moved = after["block"] - before["block"]
        if abs(moved) > 1:
            failures.append(f"fold {name}: the folding block moved {moved:+.0f} px on screen")
        if not (0 <= after["tailBottom"] <= after["clientHeight"]):
            failures.append(
                f"fold {name}: the reply is out of view (bottom at {after['tailBottom']:.0f} px)"
            )

    for name in ("pinnedFits", "pinnedFitsSettled", "detachedFits", "sibling"):
        still(name, fold[name])
    for name in ("pinnedFits", "pinnedFitsSettled", "detachedFits"):
        case = fold[name]
        if case["before"]["top"] < 1:
            failures.append(
                f"fold {name}: the thread did not overflow before the fold; nothing was measured"
            )
    rebind = fold["rebind"]
    still("rebind", rebind)
    moved = rebind["afterRebind"]["block"] - rebind["after"]["block"]
    if abs(moved) > 1:
        failures.append(f"fold rebind: rebinding the viewport moved the thread {moved:+.0f} px")
    if rebind["after"]["padding"] < 300:
        failures.append("fold rebind: the fold was not padded; the rebind check proved nothing")
    growth = fold["detachedGrowth"]
    still("detachedGrowth", growth)
    if growth["after"]["padding"] < 1:
        failures.append("fold detachedGrowth: the fold was not padded; the check proved nothing")
    if growth["paddingAfterAnswer"] > 1:
        failures.append(
            f"fold detachedGrowth: {growth['paddingAfterAnswer']:.0f} px of padding outlived the answer"
        )
    tops = growth["downTops"]
    backward = [b - a for a, b in zip(tops, tops[1:]) if b < a - 1]
    if backward:
        failures.append(
            f"fold detachedGrowth: scrolling down moved backward by {-min(backward):.0f} px"
        )
    reuse = fold["reuse"]
    still("reuse", reuse)
    if reuse["paddingAfterFold"] < 300:
        failures.append(
            f"fold reuse: the fold was padded {reuse['paddingAfterFold']:.0f} px, not its 320"
        )
    if reuse["paddingAfterAnswer"] > 1:
        failures.append(
            "fold reuse: the answer did not use up the fold's padding; the check proved nothing"
        )
    if reuse["paddingAfterRemoval"] > 64:
        failures.append(
            f"fold reuse: a removal after the answer used the padding was padded "
            f"{reuse['paddingAfterRemoval']:.0f} px"
        )
    return failures


def main() -> int:
    vite = None
    if OWNS_SERVER:
        info(f"starting vite dev server on port {PORT}")
        vite = start_vite(PORT)
    try:
        wait_for_smoke_page(
            f"{BASE}/smoke-autoscroll.html", "smoke-autoscroll-main.tsx", proc = vite, info = info
        )
        results = run()
    finally:
        if vite is not None:
            stop_process(vite)
            info("vite stopped")
    stream_seconds = max(0.001, results["stream"]["wall_ms"] / 1000)
    raf_rate = results["stream"]["raf_callbacks"] / stream_seconds
    results["stream"]["raf_per_second"] = round(raf_rate, 1)
    out = OUT / f"{LABEL}.json"
    out.write_text(json.dumps(results, indent = 2), encoding = "utf-8")
    info(json.dumps(results, indent = 2))
    info(f"wrote {out}")

    failures: list[str] = []
    if results["seeded"]["scrollHeight"] <= results["seeded"]["clientHeight"]:
        failures.append("the seeded viewport does not overflow; nothing was measured")
    if results["stream"]["pinned_after_stream"] > 2:
        failures.append("the viewport did not stay pinned through the stream")
    if not results["stream"]["is_at_bottom"]:
        failures.append("isAtBottom went false while following")
    if raf_rate > MAX_STREAM_RAF_PER_SECOND:
        failures.append(
            f"the follow loop ran at {raf_rate:.1f} rAF/s during the stream, over the "
            f"{MAX_STREAM_RAF_PER_SECOND} rAF/s budget"
        )
    if results["idle_raf_per_2s"] > 8:
        failures.append(
            f"the loop still runs when idle ({results['idle_raf_per_2s']} frames in 2s)"
        )
    intent = results["intent"]
    if not intent["detached"]:
        failures.append("scrolling up did not detach")
    # Without this the check above is vacuous.
    if not intent["grewWhileDetached"]:
        failures.append("the content did not grow while detached; the re-pin check proved nothing")
    if not intent["stillDetached"]:
        failures.append("streaming re-pinned a detached reader")
    if not intent["reattached"]:
        failures.append("scrolling back to the bottom did not re-attach")
    if results["silent_growth"]["grewBy"] < 200:
        failures.append(
            f"silent growth grew the page {results['silent_growth']['grewBy']} px, not 240; "
            "the follow check proved nothing"
        )
    repinned_after_ms = results["silent_growth"]["repinnedAfterMs"]
    if repinned_after_ms is None:
        failures.append("silent growth was never followed inside the window")
    elif repinned_after_ms > SILENT_GROWTH_REPIN_BUDGET_MS:
        failures.append(
            f"silent growth took {repinned_after_ms:.0f}ms to follow, over the "
            f"{SILENT_GROWTH_REPIN_BUDGET_MS}ms budget"
        )
    failures.extend(fold_failures(results["fold"]))
    for problem in failures:
        info(f"FAIL {problem}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
