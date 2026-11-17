"""Blog stills: Unsloth Studio on the NPU writes a three.js voxel kingdom that renders in the HTML preview.

env: BASE, PW, OUT, MODEL (catalog id), NEW_NAME (the picker label it should show)
Up to 3 attempts, each in a fresh chat; failed chats are deleted so Recents holds one.
"""

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

BASE = os.environ["BASE"].rstrip("/")
OUT = Path(os.environ["OUT"])
OUT.mkdir(parents = True, exist_ok = True)
MODEL = os.environ.get("MODEL", "qwen3.6-moe-35b-a3b-FLM")
NEW_NAME = os.environ.get("NEW_NAME", "Qwen3.6-35B-A3B-NPU2")
PROMPT = (
    "using html with embedded JavaScript with 3js make a voxel art model of a medieval kingdom, please dont just "
    "make it out of blocks i want real voxels and a nice scene to demo with good lighting, good grass and trees "
    "and paths and the kingdom buildings."
)
W, H = 1440, 900
TRIES = 3


def call(method, path, body = None, token = None, timeout = 60):
    req = urllib.request.Request(
        BASE + path,
        json.dumps(body).encode() if body is not None else None,
        {"Content-Type": "application/json"},
        method = method,
    )
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout = timeout) as r:
        raw = r.read()
    return json.loads(raw) if raw else {}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush = True)


login = call("POST", "/api/auth/login", {"username": "unsloth", "password": os.environ["PW"]})
TOKEN = login["access_token"]


def seed():
    store = {
        "unsloth_auth_token": TOKEN,
        "unsloth_refresh_token": login.get("refresh_token", ""),
        "theme": "light",
        "palette": "mint",
        "unsloth_window_material": json.dumps(
            {"state": {"material": "liquid-3d", "style": "liquid-3d", "menus": True,
                       "strength": {"light": 40, "dark": 15}, "lens": "off", "scope": "all"}, "version": 0}
        ),
        "unsloth_chat_allow_artifact_network_access": "true",
    }
    return "Object.entries(" + json.dumps(store) + ").forEach(([k,v])=>{if(localStorage.getItem(k)===null||k.startsWith('unsloth_auth')||k==='unsloth_refresh_token')localStorage.setItem(k,v)})"


def threads():
    return call("GET", "/api/chat/threads", token = TOKEN).get("threads", [])


def delete_threads(ids):
    if ids:
        log("deleting chats", ids)
        call("DELETE", "/api/chat/threads", {"ids": ids, "delete_files": True}, TOKEN)


def dismiss(page):
    for label in ("Later", "Not now", "Dismiss"):
        b = page.get_by_role("button", name = label)
        if b.count():
            try:
                b.first.click(timeout = 1500)
                page.wait_for_timeout(300)
            except Exception:
                pass


def shot(page, name, **kw):
    try:
        page.screenshot(path = str(OUT / f"{name}.png"), **kw)
        log("shot", name)
    except Exception as exc:
        log("screenshot failed", name, str(exc)[:200])


def network_setting(page):
    """Settings > Browser > Allow HTML pages network access, on."""
    page.goto(f"{BASE}/settings")
    page.wait_for_timeout(2500)
    tab = page.get_by_role("tab", name = "Browser")
    if not tab.count():
        tab = page.get_by_role("button", name = "Browser")
    if tab.count():
        tab.first.click()
        page.wait_for_timeout(1000)
    sw = page.get_by_role("switch", name = "Allow HTML pages network access")
    if not sw.count():
        log("network switch not found; relying on the seeded setting")
        shot(page, "00-settings-missing")
        return
    sw.first.scroll_into_view_if_needed()
    if sw.first.get_attribute("aria-checked") != "true":
        sw.first.click()
        page.wait_for_timeout(500)
    log("network access switch:", sw.first.get_attribute("aria-checked"))
    shot(page, "00-settings-network-on")
    page.keyboard.press("Escape")
    page.wait_for_timeout(600)


def open_picker(page):
    for loc in (page.locator('button[aria-label^="Model:"]'), page.get_by_role("button", name = "Select model"),
                page.locator("button", has_text = "FLM"), page.locator("button", has_text = NEW_NAME)):
        if loc.count() and loc.first.is_visible():
            loc.first.focus()
            page.keyboard.press("Enter")
            page.wait_for_timeout(1500)
            return True
    return False


def find_row(page):
    for name in (NEW_NAME, MODEL):
        row = page.get_by_text(name, exact = True)
        if row.count():
            return name, row.first
    footer = page.get_by_test_id("picker-npu-footer")
    if footer.count():
        footer.first.click()
        page.wait_for_timeout(2000)
        for name in (NEW_NAME, MODEL):
            row = page.get_by_text(name, exact = True)
            if row.count():
                return name, row.first
    return None, None


def load_model(page):
    page.goto(f"{BASE}/chat")
    page.wait_for_timeout(4000)
    dismiss(page)
    if not open_picker(page):
        raise RuntimeError("no model picker button")
    name, row = find_row(page)
    shot(page, "01-picker-open-before-load")
    log("PICKER_LABEL", name)
    if row is None:
        raise RuntimeError("the model is not in the picker")
    row.click()
    deadline = time.time() + 900
    while time.time() < deadline:
        st = call("GET", "/api/npu/status", token = TOKEN)
        if st.get("loaded_model") == f"lemonade:{MODEL}" and not st.get("loading_model"):
            log("loaded", json.dumps({k: st.get(k) for k in ("loaded_model", "context_length")}))
            break
        page.wait_for_timeout(2000)
    else:
        shot(page, "load-timeout")
        raise RuntimeError("the model did not load in 15 minutes")
    page.wait_for_timeout(2500)
    dismiss(page)


def sidebar_state(page):
    return page.evaluate(
        "()=>{const s=document.querySelector('[data-slot=sidebar][data-state]');return s?s.getAttribute('data-state'):'?'}"
    )


def set_sidebar(page, want):
    before = sidebar_state(page)
    if before == want:
        return
    page.mouse.move(W / 2, H / 2)
    page.keyboard.press("Control+b")
    page.wait_for_timeout(900)
    if sidebar_state(page) != want:
        for b in page.get_by_role("button", name = "Toggle Sidebar").all():
            if b.is_visible():
                b.click()
                page.wait_for_timeout(900)
                break
    log("sidebar", before, "->", sidebar_state(page), "want", want)


def collapse_sidebar(page):
    set_sidebar(page, "collapsed")


def expand_sidebar(page):
    set_sidebar(page, "expanded")


def preview_frame(page):
    """The frame drawing the scene: the one holding a canvas, deepest first."""
    for f in reversed(page.frames):
        try:
            if f.evaluate("()=>!!document.querySelector('canvas')"):
                return f
        except Exception:
            pass
    return None


def frame_state(page):
    out = {"frames": len(page.frames)}
    for f in reversed(page.frames[1:]):
        try:
            s = f.evaluate(
                """()=>{const c=document.querySelector('canvas');
                const vis=[...document.querySelectorAll('body *')].filter(e=>{const r=e.getBoundingClientRect();const st=getComputedStyle(e);
                  return r.width>0&&r.height>0&&st.visibility!=='hidden'&&st.display!=='none'&&parseFloat(st.opacity||'1')>0.05&&e.children.length===0&&e.textContent.trim()});
                return {canvas: c?[c.width,c.height]:null, three: typeof THREE!=='undefined'?(THREE.REVISION||'module'):null,
                        text: vis.map(e=>e.textContent.trim()).join(' | ').slice(0,300)}}"""
            )
            s["url"] = f.url[:80]
            out.setdefault("inner", []).append(s)
        except Exception as exc:
            out.setdefault("inner", []).append({"err": str(exc)[:120]})
    return out


def flat_share(path):
    """Share of the frame in its single most common colour: a loading screen is mostly one flat colour."""
    from PIL import Image
    im = Image.open(path).convert("RGB").resize((240, 150))
    px = list(im.getdata())
    counts = {}
    for r, g, b in px:
        k = (r // 8, g // 8, b // 8)
        counts[k] = counts.get(k, 0) + 1
    return max(counts.values()) / len(px)


def colour_spread(path):
    from PIL import Image
    im = Image.open(path).convert("RGB").resize((240, 150))
    px = list(im.getdata())
    distinct = len(set((r // 16, g // 16, b // 16) for r, g, b in px))
    return distinct


def drag_splitter(page, frac = 0.58):
    handle = page.locator(".chat-artifact-split > [data-slot=resizable-handle]")
    panel = page.locator("#chat-artifact")
    if not handle.count() or not panel.count():
        log("no splitter found")
        return
    for _ in range(3):
        hb = handle.first.bounding_box()
        pb = panel.first.bounding_box()
        group = page.locator(".chat-artifact-split").first.bounding_box()
        cur = pb["width"] / group["width"]
        log(f"canvas share {cur:.3f} of the split ({pb['width']:.0f}/{group['width']:.0f}px), page {pb['width'] / W:.3f}")
        if abs(pb["width"] / W - frac) < 0.015:
            return
        target_x = W - frac * W
        x0, y0 = hb["x"] + hb["width"] / 2, hb["y"] + hb["height"] / 2
        page.mouse.move(x0, y0)
        page.mouse.down()
        steps = 25
        for i in range(1, steps + 1):
            page.mouse.move(x0 + (target_x - x0) * i / steps, y0)
            page.wait_for_timeout(16)
        page.mouse.up()
        page.wait_for_timeout(1200)
    pb = panel.first.bounding_box()
    log(f"canvas width after drag {pb['width']:.0f}px = {pb['width'] / W:.3f} of the page")


def orbit(page, dx = 140, dy = -30):
    f = page.locator("iframe").last.bounding_box()
    cx, cy = f["x"] + f["width"] / 2, f["y"] + f["height"] * 0.6
    page.mouse.move(cx, cy)
    page.mouse.down()
    for i in range(1, 41):
        page.mouse.move(cx + dx * i / 40, cy + dy * i / 40)
        page.wait_for_timeout(30)
    page.mouse.up()
    page.wait_for_timeout(2500)


def attempt(page, n):
    tag = f"try{n}"
    page.goto(f"{BASE}/chat")
    page.wait_for_timeout(3500)
    dismiss(page)
    collapse_sidebar(page)
    box = page.get_by_placeholder("Ask anything")
    box.click()
    page.keyboard.type(PROMPT, delay = 8)
    page.wait_for_timeout(400)
    page.keyboard.press("Enter")
    log(tag, "sent")
    t0 = time.time()
    page.get_by_text("tok/s").first.wait_for(timeout = 1800 * 1000)
    log(tag, f"reply done in {time.time() - t0:.0f}s, badge", page.get_by_text("tok/s").first.inner_text())
    page.wait_for_timeout(800)
    shot(page, f"{tag}-reply-done")

    btn = page.get_by_role("button", name = "Open HTML preview preview")
    if not btn.count():
        btn = page.get_by_role("button", name = "Open in")
    if not btn.count():
        log(tag, "no HTML preview card")
        return False
    btn.first.click()
    page.wait_for_timeout(2500)
    allow = page.get_by_role("button", name = "Allow for this page")
    if allow.count() and allow.first.is_visible():
        log(tag, "WARNING: the preview still asked for network access; the setting did not take. Allowing for this page.")
        allow.first.click()
        page.wait_for_timeout(1500)
    drag_splitter(page)
    # SwiftShader WebGL is slow: wait for the loading screen to give way to the scene
    ok = False
    state = {}
    t1 = time.time()
    k = 0
    while time.time() - t1 < 480:
        page.wait_for_timeout(10000)
        k += 1
        probe = OUT / f"{tag}-canvas-probe.png"
        page.locator("iframe").last.screenshot(path = str(probe))
        flat, spread = flat_share(probe), colour_spread(probe)
        state = frame_state(page)
        has_canvas = any(s.get("canvas") for s in state.get("inner", []))
        log(tag, f"t+{time.time() - t1:.0f}s canvas={has_canvas} flat={flat:.2f} buckets={spread}")
        if k % 6 == 0:
            page.locator("iframe").last.screenshot(path = str(OUT / f"{tag}-probe-{k:02d}.png"))
        if has_canvas and flat < 0.45 and spread > 60:
            ok = True
            break
    log(tag, "PREVIEW_STATE", json.dumps(state)[:900])
    log(tag, f"RENDER_CHECK ok={ok}")
    if not ok:
        shot(page, f"{tag}-not-rendered")
        try:
            for i, f in enumerate(page.frames[1:]):
                (OUT / f"{tag}-frame{i}.html").write_text(f.content()[:600000], encoding = "utf-8")
        except Exception as exc:
            log(tag, "frame dump failed", str(exc)[:200])
        return False
    page.wait_for_timeout(4000)

    # the finished frame: chat with tok/s on the left, kingdom on the right
    orbit(page)
    page.mouse.move(5, H - 5)
    page.wait_for_timeout(1500)
    shot(page, "1-final-chat-and-kingdom")
    page.locator("iframe").last.screenshot(path = str(OUT / "2-canvas-closeup.png"))
    log("shot 2-canvas-closeup")
    orbit(page, dx = -220, dy = 20)
    page.mouse.move(5, H - 5)
    page.wait_for_timeout(1500)
    shot(page, "1b-final-alt-angle")
    page.locator("iframe").last.screenshot(path = str(OUT / "2b-canvas-closeup-alt-angle.png"))
    # the picker open over the finished chat, showing the Hub name
    if open_picker(page):
        name, _ = find_row(page)
        log("PICKER_LABEL_FINAL", name)
        page.wait_for_timeout(800)
        shot(page, "3-picker-open-new-name")
        page.keyboard.press("Escape")
        page.wait_for_timeout(600)
    # the same frame with the sidebar out, to show Recents holds one chat
    expand_sidebar(page)
    shot(page, "4-sidebar-recents-one-chat")
    collapse_sidebar(page)
    return True


def main():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless = True,
            args = ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist", "--enable-webgl"],
        )
        ctx = browser.new_context(viewport = {"width": W, "height": H}, device_scale_factor = 2, color_scheme = "light")
        ctx.add_init_script(seed())
        page = ctx.new_page()
        page.on("console", lambda m: m.type in ("error", "warning") and log("console", m.type, m.text[:200]))
        try:
            try:
                network_setting(page)
            except Exception as exc:
                log("settings step failed, relying on the seeded value", str(exc)[:200])
            delete_threads([t["id"] for t in threads()])
            load_model(page)
            good = False
            for n in range(1, TRIES + 1):
                before = {t["id"] for t in threads()}
                try:
                    good = attempt(page, n)
                except Exception as exc:
                    log(f"try{n}", "FAILED", type(exc).__name__, str(exc)[:400])
                    shot(page, f"try{n}-failure")
                if good:
                    break
                delete_threads([t["id"] for t in threads()])
                log(f"try{n}", "did not render; chats left", [t["id"] for t in threads()], "had", sorted(before))
            left = threads()
            log("RECENTS", json.dumps([[t["id"], t.get("title")] for t in left]))
            log("RESULT", "rendered" if good else "not rendered")
        finally:
            ctx.close()
            browser.close()
    sys.exit(0 if good else 1)


main()
