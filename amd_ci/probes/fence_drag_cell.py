#!/usr/bin/env python3
"""Windows copy of scripts/fence_drag/drag.py, with no studiobench import. Scrollbar-thumb drag, first pass and theme change on a long thread of big code fences, in the real Studio app.

Each arm is a running Studio (one production dist each). Arms are interleaved per repetition, each repetition is
a fresh browser context (cold renderer), and the same seeded thread is reopened. Writes one JSON row per cell.

  python scripts/fence_drag/drag.py --arm A0=http://127.0.0.1:5411 --arm G=http://127.0.0.1:5412 \
      --reps 3 --throttle 4 --out outputs/fence_drag/run.jsonl
An arm may carry a fence-mode override: --arm PW=http://127.0.0.1:5413@window
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import uuid
from pathlib import Path

import base64  # noqa: E402
import urllib.error  # noqa: E402
import urllib.request  # noqa: E402


class _Auth:
    def __init__(self, access_token: str, refresh_token: str):
        self.access_token, self.refresh_token = access_token, refresh_token


def _request(url: str, method: str = "GET", body: dict | None = None, token: str | None = None, timeout: float = 60.0):
    data = None if body is None else json.dumps(body).encode("utf-8")
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data = data, headers = headers, method = method)
    with urllib.request.urlopen(req, timeout = timeout) as r:
        raw = r.read().decode("utf-8")
    return json.loads(raw) if raw.strip() else None


def authenticate(base: str, username: str, password: str) -> _Auth:
    body = _request(f"{base}/api/auth/login", "POST", {"username": username, "password": password})
    auth = _Auth(body["access_token"], body.get("refresh_token", ""))
    status = _request(f"{base}/api/auth/status", token = auth.access_token) or {}
    if status.get("requires_password_change"):
        new = password + "-2"
        body = _request(f"{base}/api/auth/change-password", "POST",
                        {"current_password": password, "new_password": new}, token = auth.access_token)
        auth = _Auth(body["access_token"], body.get("refresh_token", ""))
    return auth


def auth_request_json(auth: _Auth, url: str, method: str = "GET", body: dict | None = None, timeout: float = 60.0):
    return _request(url, method, body, auth.access_token, timeout)


def seed_init_script(auth: _Auth, _providers) -> str:
    seed = {"unsloth_auth_token": auth.access_token, "unsloth_auth_refresh_token": auth.refresh_token}
    return ("(() => { const s = " + json.dumps(seed) + "; for (const k of Object.keys(s)) {"
            " try { window.localStorage.setItem(k, s[k]); } catch (e) {} } })();")


PASSWORD = "fdbench-Passw0rd!"
FENCE_LINES = [280, 350, 420, 480, 3200, 540, 600, 650, 700, 300]
BIG = 4

PAGE_PROBE = r"""
(() => {
  const fd = { rec: false, frames: [], tasks: [], muts: 0, downs: [] };
  window.__fd = fd;
  const loop = (t) => { if (fd.rec) fd.frames.push(t); requestAnimationFrame(loop); };
  requestAnimationFrame(loop);
  try {
    new PerformanceObserver((list) => {
      if (!fd.rec) return;
      for (const e of list.getEntries()) fd.tasks.push(e.duration);
    }).observe({ type: "longtask" });
  } catch (e) {}
  addEventListener("pointerdown", (e) => { if (fd.rec) fd.downs.push(e.target && e.target.className ? String(e.target.className).slice(0, 40) : String(e.target && e.target.nodeName)); }, true);
  const watch = () => {
    const vp = document.body;
    if (!vp) { setTimeout(watch, 200); return; }
    new MutationObserver((records) => {
      if (!fd.rec) return;
      for (const r of records) {
        if (r.type === "childList" && r.target instanceof Element && r.target.closest("pre")) fd.muts += r.addedNodes.length + r.removedNodes.length;
      }
    }).observe(vp, { childList: true, subtree: true });
  };
  if (document.readyState === "loading") addEventListener("DOMContentLoaded", watch); else watch();
})();
"""


def code_line(fence: int, i: int) -> str:
    kinds = (
        f"    result_{fence}_{i} = transform_{i % 13}(items[{i}], scale={i * 0.5:.1f}, label=\"row {i}\")",
        f"    if counter_{i % 7} > {i} and not flags.get(\"f{fence}_{i}\"):  # check {i}",
        f"        total += weights[{i % 31}] * math.sqrt({i}) - offsets.pop({i % 5}, 0)",
        f"    log.debug(\"fence %d line %d -> %s\", {fence}, {i}, repr(state_{i % 11}))",
        f"def helper_{fence}_{i}(x: int, y: float = {i}.0) -> list[int]:",
        f"        return [v for v in range({i % 97}) if v % {1 + i % 9} == 0]",
    )
    return kinds[i % len(kinds)]


def build_thread(thread_id: str) -> list[dict]:
    messages, parent, now = [], None, int(time.time() * 1000) - 100_000
    for f, n in enumerate(FENCE_LINES):
        tag = "BIGFENCE" if f == BIG else f"FENCE{f}"
        body = "\n".join([f"# {tag} start"] + [code_line(f, i) for i in range(1, n)])
        uid, aid = str(uuid.uuid4()), str(uuid.uuid4())
        messages.append({"id": uid, "threadId": thread_id, "parentId": parent, "role": "user",
                         "content": [{"type": "text", "text": f"fd turn {f}: show fence {f}"}],
                         "attachments": None, "metadata": None, "createdAt": now + f * 2000})
        text = f"Here is fence {f}, {n} lines.\n\n```python\n{body}\n```\n\nThat is the whole of fence {f}."
        messages.append({"id": aid, "threadId": thread_id, "parentId": uid, "role": "assistant",
                         "content": [{"type": "text", "text": text}],
                         "attachments": None, "metadata": None, "createdAt": now + f * 2000 + 1000})
        parent = aid
    return messages


def seed(base: str) -> tuple[object, str]:
    auth = authenticate(base, "unsloth", PASSWORD)
    tid = str(uuid.uuid4())
    auth_request_json(auth, f"{base}/api/chat/threads", method = "POST", timeout = 60,
                      body = {"id": tid, "title": "fence drag", "modelType": "base", "modelId": "bench",
                              "createdAt": int(time.time() * 1000)})
    auth_request_json(auth, f"{base}/api/chat/threads/{tid}/messages", method = "PUT", timeout = 300,
                      body = {"messages": build_thread(tid), "pruneMissing": True})
    return auth, tid


def frame_stats(frames: list[float], tasks: list[float]) -> dict:
    iv = [b - a for a, b in zip(frames, frames[1:])]
    return {
        "frames": len(iv),
        "worst_frame_ms": round(max(iv), 1) if iv else None,
        "frames_over_50": sum(1 for x in iv if x > 50),
        "jank_ms": round(sum(x - 16.7 for x in iv if x > 50), 1),
        "p99_frame_ms": round(sorted(iv)[int(0.99 * (len(iv) - 1))], 1) if iv else None,
        "longest_task_ms": round(max(tasks), 1) if tasks else 0.0,
        "long_tasks": len(tasks),
    }


def start(page) -> None:
    page.evaluate("() => { const f = window.__fd; f.frames = []; f.tasks = []; f.muts = 0; f.downs = []; f.rec = true; }")


def stop(page) -> dict:
    got = page.evaluate("() => { const f = window.__fd; f.rec = false; return {frames: f.frames, tasks: f.tasks, muts: f.muts, downs: f.downs}; }")
    out = frame_stats(got["frames"], got["tasks"])
    out["pre_mutations"] = got["muts"]
    out["pointerdown_targets"] = sorted(set(got["downs"]))[:5]
    return out


def counts(page) -> dict:
    return page.evaluate("""() => {
      const pres = [...document.querySelectorAll('.aui-thread-viewport pre')];
      let spans = 0; for (const p of pres) spans += p.getElementsByTagName('span').length;
      return {dom_nodes: document.getElementsByTagName('*').length, code_spans: spans, pres: pres.length,
              windowed: document.querySelectorAll('[data-unsloth-fence-windowed]').length,
              painted: document.querySelectorAll('[data-unsloth-fence-paint]').length};
    }""")


def settle(page, timeout_s: float = 30.0) -> float:
    """Wait until the code span count holds still for 1.5 s; returns seconds waited."""
    t0, last, still = time.monotonic(), -1, 0.0
    while time.monotonic() - t0 < timeout_s:
        n = counts(page)["code_spans"]
        if n == last:
            still += 0.5
            if still >= 1.5:
                break
        else:
            still, last = 0.0, n
        page.wait_for_timeout(500)
    return round(time.monotonic() - t0, 1)


def viewport_geom(page) -> dict:
    return page.evaluate("""() => {
      const v = document.querySelector('.aui-thread-viewport'); const r = v.getBoundingClientRect();
      return {top: r.top, left: r.left, width: r.width, height: r.height, clientWidth: v.clientWidth,
              clientHeight: v.clientHeight, scrollHeight: v.scrollHeight, scrollTop: v.scrollTop,
              borderLeft: v.clientLeft, sbw: v.offsetWidth - v.clientWidth - v.clientLeft};
    }""")


def wheel_traverse(page, direction: int, budget_s: float) -> dict:
    g = viewport_geom(page)
    page.mouse.move(g["left"] + g["clientWidth"] / 2, g["top"] + g["clientHeight"] / 2)
    t0, n = time.monotonic(), 0
    while time.monotonic() - t0 < budget_s:
        page.mouse.wheel(0, direction * 500)
        n += 1
        page.wait_for_timeout(16)
        g = viewport_geom(page) if n % 10 == 0 else None
        if g and ((direction < 0 and g["scrollTop"] <= 0) or
                  (direction > 0 and g["scrollTop"] + g["clientHeight"] >= g["scrollHeight"] - 2)):
            break
    g = viewport_geom(page)
    return {"wheel_events": n, "end_scrollTop": g["scrollTop"], "seconds": round(time.monotonic() - t0, 1)}


def drag(page, seconds: float) -> dict:
    # Put the big fence's first line at the top of the viewport, then drag the thumb across the fence.
    fence = page.evaluate("""() => {
      const v = document.querySelector('.aui-thread-viewport');
      const pre = [...v.querySelectorAll('pre')].find(p => (p.textContent || '').startsWith('# BIGFENCE'));
      if (!pre) return null;
      v.style.scrollBehavior = 'auto';
      const top = pre.getBoundingClientRect().top - v.getBoundingClientRect().top + v.scrollTop;
      v.scrollTop = top - 50;
      return {top, height: pre.getBoundingClientRect().height};
    }""")
    if not fence:
        return {"ran": False, "reason": "big fence not found"}
    page.wait_for_timeout(1500)
    g = viewport_geom(page)
    if g["sbw"] <= 0:
        return {"ran": False, "reason": f"no scrollbar gutter (sbw={g['sbw']})"}
    ratio = g["clientHeight"] / g["scrollHeight"]
    thumb_h = max(20.0, g["clientHeight"] * ratio)
    x = g["left"] + g["borderLeft"] + g["clientWidth"] + g["sbw"] / 2
    y0 = g["top"] + g["scrollTop"] * ratio + thumb_h / 2
    span = max(20.0, (fence["height"] - g["clientHeight"]) * ratio)
    page.mouse.move(x, y0)
    start(page)
    page.mouse.down()
    tops, t0, k = [], time.monotonic(), 0
    while time.monotonic() - t0 < seconds:
        phase = (k % 120) / 60.0
        frac = phase if phase <= 1 else 2 - phase
        page.mouse.move(x, y0 + frac * span)
        k += 1
        if k % 15 == 0:
            tops.append(page.evaluate("() => document.querySelector('.aui-thread-viewport').scrollTop"))
    page.mouse.up()
    page.wait_for_timeout(300)
    out = stop(page)
    travel = (max(tops) - min(tops)) if tops else 0
    out.update({"ran": travel > 0.5 * (fence["height"] - g["clientHeight"]), "travel_px": round(travel),
                "fence_px": round(fence["height"]), "moves": k, "sbw": g["sbw"]})
    return out


def theme(page, n: int = 4) -> dict:
    ms = page.evaluate("""async (n) => {
      const out = []; const html = document.documentElement;
      for (let i = 0; i < n; i++) {
        await new Promise(r => setTimeout(r, 400));
        const t0 = performance.now();
        html.classList.toggle('dark');
        void html.getBoundingClientRect().height;
        await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
        out.push(performance.now() - t0);
      }
      return out;
    }""", n)
    return {"theme_ms": [round(x, 1) for x in ms], "theme_ms_median": round(statistics.median(ms), 1)}


def run_cell(browser, arm: str, base: str, override: str | None, auth, tid: str, throttle: float, drag_s: float) -> dict:
    ctx = browser.new_context(viewport = {"width": 1440, "height": 900})
    ctx.add_init_script(seed_init_script(auth, []))
    if override:
        ctx.add_init_script(f"window.__UNSLOTH_DEFER_FENCE_HIGHLIGHT__ = {json.dumps(override)};")
    ctx.add_init_script(PAGE_PROBE)
    page = ctx.new_page()
    cdp = ctx.new_cdp_session(page)
    row: dict = {"arm": arm, "throttle": throttle, "override": override}
    crashed = []
    page.on("crash", lambda *_: crashed.append(time.time()))
    try:
        if throttle != 1:
            cdp.send("Emulation.setCPUThrottlingRate", {"rate": throttle})
        t0 = time.monotonic()
        page.goto(f"{base}/chat?thread={tid}", wait_until = "domcontentloaded", timeout = 180_000)
        page.wait_for_function(
            f"() => document.body.innerText.includes('fd turn {len(FENCE_LINES) - 1}:')", timeout = 180_000)
        row["open_s"] = round(time.monotonic() - t0, 2)
        page.wait_for_timeout(1500)
        start(page)
        row["first_pass"] = {**wheel_traverse(page, -1, 25.0)}
        row["first_pass"].update(stop(page))
        row["settle_s"] = settle(page)
        start(page)
        row["warm_pass"] = {**wheel_traverse(page, 1, 25.0)}
        row["warm_pass"].update(stop(page))
        settle(page)
        row["after_traversal"] = counts(page)
        row["drag"] = drag(page, drag_s)
        row["after_drag"] = counts(page)
        row["theme"] = theme(page)
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}"[:400]
    row["crashed"] = bool(crashed)
    ctx.close()
    return row


def stress(browser, arm: str, base: str, override: str | None, auth, tid: str, n: int) -> dict:
    """Rapid back-and-forth thumb drags at full speed; a renderer or browser crash is the observation."""
    ctx = browser.new_context(viewport = {"width": 1440, "height": 900})
    ctx.add_init_script(seed_init_script(auth, []))
    if override:
        ctx.add_init_script(f"window.__UNSLOTH_DEFER_FENCE_HIGHLIGHT__ = {json.dumps(override)};")
    ctx.add_init_script(PAGE_PROBE)
    page = ctx.new_page()
    crashed, disconnected = [], []
    page.on("crash", lambda *_: crashed.append(time.time()))
    browser.on("disconnected", lambda *_: disconnected.append(time.time()))
    row = {"arm": arm, "stress": n, "override": override, "drags": 0}
    try:
        page.goto(f"{base}/chat?thread={tid}", wait_until = "domcontentloaded", timeout = 180_000)
        page.wait_for_function(
            f"() => document.body.innerText.includes('fd turn {len(FENCE_LINES) - 1}:')", timeout = 180_000)
        page.wait_for_timeout(1500)
        for _ in range(n):
            if crashed or disconnected:
                break
            d = drag(page, 3.0)
            row["drags"] += 1
            row.setdefault("worst", []).append(d.get("worst_frame_ms"))
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}"[:300]
    row["renderer_crashes"] = len(crashed)
    row["browser_disconnects"] = len(disconnected)
    try:
        ctx.close()
    except Exception:  # noqa: BLE001
        pass
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action = "append", required = True, help = "LABEL=URL[@mode]")
    ap.add_argument("--reps", type = int, default = 3)
    ap.add_argument("--throttle", type = float, action = "append")
    ap.add_argument("--drag-seconds", type = float, default = 8.0)
    ap.add_argument("--channel", default = "chromium")
    ap.add_argument("--headed", action = "store_true")
    ap.add_argument("--stress", type = int, default = 0, help = "extra rapid drags per arm, counting crashes")
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()
    from playwright.sync_api import sync_playwright

    arms = []
    for spec in args.arm:
        label, rest = spec.split("=", 1)
        url, _, mode = rest.partition("@")
        arms.append((label, url, mode or None))
    seeded = {}
    for _, url, _ in arms:
        if url not in seeded:
            seeded[url] = seed(url)
    args.out.parent.mkdir(parents = True, exist_ok = True)
    with sync_playwright() as pw, open(args.out, "a") as fh:
        browser = pw.chromium.launch(channel = args.channel, headless = not args.headed,
                                     ignore_default_args = ["--hide-scrollbars"])
        print(f"browser {args.channel} {browser.version} headed={args.headed}", flush = True)
        for throttle in args.throttle or [1.0]:
            for rep in range(args.reps):
                order = arms[rep % len(arms):] + arms[:rep % len(arms)]
                if rep % 2:
                    order = order[::-1]
                for label, url, mode in order:
                    auth, tid = seeded[url]
                    row = run_cell(browser, label, url, mode, auth, tid, throttle, args.drag_seconds)
                    row["rep"] = rep
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    d = row.get("drag") or {}
                    print(f"[{label} x{throttle:g} r{rep}] drag worst={d.get('worst_frame_ms')} >50={d.get('frames_over_50')} "
                          f"ran={d.get('ran')} muts={d.get('pre_mutations')} | first worst={(row.get('first_pass') or {}).get('worst_frame_ms')} "
                          f"| spans={(row.get('after_traversal') or {}).get('code_spans')} dom={(row.get('after_traversal') or {}).get('dom_nodes')} "
                          f"| theme={(row.get('theme') or {}).get('theme_ms_median')} {row.get('error', '')}", flush = True)
            if args.stress:
                for label, url, mode in arms:
                    auth, tid = seeded[url]
                    row = stress(browser, label, url, mode, auth, tid, args.stress)
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    print(f"[{label} stress] {row}", flush = True)
                    if not browser.is_connected():
                        browser = pw.chromium.launch(channel = args.channel, headless = not args.headed,
                                                     ignore_default_args = ["--hide-scrollbars"])
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
