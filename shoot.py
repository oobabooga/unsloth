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
    "Recreate this spinning 2D triangle lattice as one self-contained HTML file. Keep the same look (cream "
    "background, black sticks and dots, the SVG goo filter) and the same physics and constants, and make sure it runs "
    "without errors. Send the complete HTML file.\n\n```html\n"
    + (Path(__file__).parent / "ref_lattice.html").read_text()
    + "\n```"
)
W, H = 1440, 900
FLAT2D = True  # 2D canvas: no orbit drags or zoom, looser coverage checks
ERRORS = []
BENIGN = ("localStorage", "Failed to load resource")
TRIES = int(os.environ.get("TRIES", "6"))


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
        page.screenshot(path = str(OUT / f"{name}.png"), timeout = 180000, **kw)
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
    st = call("GET", "/api/npu/status", token = TOKEN)
    if st.get("loaded_model") == f"lemonade:{MODEL}" and not st.get("loading_model"):
        log("model already loaded")
        page.goto(f"{BASE}/chat")
        page.wait_for_timeout(4000)
        dismiss(page)
        return
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


def drag(page, dx, dy, button = "left"):
    if FLAT2D:
        return
    f = page.locator("iframe").last.bounding_box()
    cx, cy = f["x"] + f["width"] / 2, f["y"] + f["height"] * 0.55
    page.mouse.move(cx, cy)
    page.mouse.down(button = button)
    for i in range(1, 31):
        page.mouse.move(cx + dx * i / 30, cy + dy * i / 30)
        page.wait_for_timeout(25)
    page.mouse.up(button = button)
    page.wait_for_timeout(1500)


def wheel(page, dy):
    if FLAT2D:
        return
    f = page.locator("iframe").last.bounding_box()
    page.mouse.move(f["x"] + f["width"] / 2, f["y"] + f["height"] * 0.55)
    for _ in range(3):
        page.mouse.wheel(0, dy / 3)
        page.wait_for_timeout(120)
    page.wait_for_timeout(1500)


def overlays(page):
    """Rects (css px, frame coords) of the page's own text overlays."""
    f = preview_frame(page)
    if not f:
        return []
    try:
        return f.evaluate("""()=>[...document.querySelectorAll('body *')].filter(e=>{if(e.tagName==='CANVAS')return false;
            const r=e.getBoundingClientRect(),st=getComputedStyle(e);return r.width>0&&r.height>0&&st.display!=='none'&&
            st.visibility!=='hidden'&&parseFloat(st.opacity||'1')>0.05&&e.textContent.trim()&&r.width<innerWidth*0.95})
            .map(e=>{const r=e.getBoundingClientRect();return [r.left,r.top,r.right,r.bottom]})""")
    except Exception:
        return []


def measure(page, path):
    """Scene box and coverage in a preview screenshot: background is whatever matches the row's edge pixels."""
    from PIL import Image
    page.locator("iframe").last.screenshot(path = str(path), timeout = 180000)
    im = Image.open(path).convert("RGB")
    fw, fh = page.locator("iframe").last.bounding_box()["width"], page.locator("iframe").last.bounding_box()["height"]
    sw, sh = 160, 160
    sm = im.resize((sw, sh))
    px = sm.load()
    masks = [[False] * sw for _ in range(sh)]
    for (l, t, r, b) in overlays(page):
        for y in range(max(0, int(t / fh * sh)), min(sh, int(b / fh * sh) + 1)):
            for x in range(max(0, int(l / fw * sw)), min(sw, int(r / fw * sw) + 1)):
                masks[y][x] = True
    black = 0
    xs, ys, n = [], [], 0
    for y in range(sh):
        refs = [px[0, y], px[1, y], px[sw - 1, y], px[sw - 2, y]]
        for x in range(sw):
            c = px[x, y]
            if sum(c) < 30:
                black += 1
            if masks[y][x]:
                continue
            if min(sum(abs(c[i] - r[i]) for i in range(3)) for r in refs) > 40:
                xs.append(x); ys.append(y); n += 1
    cov = n / (sw * sh)
    fam = {}
    for y in range(sh):
        for x in range(sw):
            c = px[x, y]
            k = (c[0] // 40, c[1] // 40, c[2] // 40)
            fam[k] = fam.get(k, 0) + 1
    rich = sum(1 for v in fam.values() if v > sw * sh * 0.003)
    if n < 50:
        return {"cov": cov, "black": black / (sw * sh), "box": None, "rich": rich}
    xs.sort(); ys.sort()
    q = lambda v, f: v[min(len(v) - 1, int(len(v) * f))]
    box = (q(xs, 0.02) / sw, q(ys, 0.02) / sh, q(xs, 0.98) / sw, q(ys, 0.98) / sh)
    return {"cov": cov, "black": black / (sw * sh), "box": box, "rich": rich}


def frame_scene(page, tag):
    """Zoom and pan until the kingdom fills most of the panel, centred."""
    if FLAT2D:
        return measure(page, OUT / f"{tag}-measure.png")
    m = {}
    zin = 0
    f = page.locator("iframe").last.bounding_box()
    for step in range(8):
        m = measure(page, OUT / f"{tag}-measure.png")
        box = m["box"]
        log(tag, f"frame step {step}: cov={m['cov']:.2f} black={m['black']:.3f} box={box and [round(v, 2) for v in box]}")
        if not box:
            return m
        bw, bh = box[2] - box[0], box[3] - box[1]
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        clipped = box[0] < 0.01 and box[2] > 0.99
        if abs(cx - 0.5) > 0.06 or abs(cy - 0.55) > 0.08:
            drag(page, (0.5 - cx) * f["width"], (0.55 - cy) * f["height"], button = "right")
            continue
        if clipped or bw > 0.97 or bh > 0.9:
            wheel(page, 240)
            continue
        if bw < 0.82 and bh < 0.75 and zin < 4:
            zin += 1
            wheel(page, -240 if bw < 0.6 else -120)
            continue
        break
    return measure(page, OUT / f"{tag}-measure.png")


def open_preview(page, tag):
    btn = page.get_by_role("button", name = "Open HTML preview preview")
    if not btn.count():
        btn = page.get_by_role("button", name = "Open in")
    if not btn.count():
        return False
    btn.last.click()
    page.wait_for_timeout(2500)
    allow = page.get_by_role("button", name = "Allow for this page")
    if allow.count() and allow.first.is_visible():
        log(tag, "WARNING: the preview still asked for network access; allowing for this page")
        allow.first.click()
        page.wait_for_timeout(1500)
    return True


def present(page, tag):
    """Open the preview at ~58% width with the page loaded at that size, wait for the scene, check it is clean."""
    if not open_preview(page, tag):
        log(tag, "no HTML preview card")
        return None
    drag_splitter(page)
    # close and reopen: the page then starts at the new width, so a broken resize handler never runs
    for b in page.get_by_role("button", name = "Close tab").all():
        if b.is_visible():
            b.click()
            break
    page.wait_for_timeout(1500)
    e0 = len(ERRORS)
    open_preview(page, tag)
    pb = page.locator("#chat-artifact").first.bounding_box()
    log(tag, f"reopened preview at {pb['width'] / W:.3f} of the page")
    if abs(pb["width"] / W - 0.58) > 0.03:
        drag_splitter(page)
    ok = False
    t1 = time.time()
    while time.time() - t1 < 300:
        page.wait_for_timeout(8000)
        if any(not any(b in e for b in BENIGN) for e in ERRORS[e0:]):
            log(tag, "the page threw; not waiting further")
            break
        probe = OUT / f"{tag}-canvas-probe.png"
        page.locator("iframe").last.screenshot(path = str(probe), timeout = 180000)
        flat, spread = flat_share(probe), colour_spread(probe)
        st = frame_state(page)
        has_canvas = any(x.get("canvas") for x in st.get("inner", []))
        log(tag, f"t+{time.time() - t1:.0f}s canvas={has_canvas} flat={flat:.2f} buckets={spread}")
        texts = " ".join(x.get("text", "") for x in st.get("inner", [])).lower()
        loading = any(w in texts for w in ("loading", "building", "generating", "%"))
        if has_canvas and spread > (3 if FLAT2D else 50) and not loading and measure(page, OUT / f"{tag}-measure.png")["cov"] > (0.01 if FLAT2D else 0.08):
            ok = True
            break
    page.wait_for_timeout(5000)  # let title cards fade
    try:
        for i, fr in enumerate(page.frames[1:]):
            (OUT / f"{tag}-frame{i}.html").write_text(fr.content()[:600000], encoding = "utf-8")
    except Exception as exc:
        log(tag, "frame dump failed", str(exc)[:200])
    errs = [e for e in ERRORS[e0:] if not any(b in e for b in BENIGN)]
    banner = page.get_by_text("This page hit", exact = False)
    if banner.count() and banner.first.is_visible():
        errs.append("banner: " + " ".join(banner.first.locator("xpath=..").inner_text().split())[:200])
    m = measure(page, OUT / f"{tag}-measure.png")
    clean = ok and not errs and m["black"] < 0.02
    log(tag, f"CHECK rendered={ok} errors={errs[:3]} black={m['black']:.3f} cov={m['cov']:.2f} rich={m['rich']} clean={clean}")
    return {"ok": ok, "clean": clean, "errs": errs, "cov": m["cov"], "rich": m["rich"]}


def scroll_chat_end(page):
    page.evaluate(
        "()=>{document.querySelectorAll('#chat-thread *').forEach(e=>{const o=getComputedStyle(e).overflowY;"
        "if(e.scrollHeight>e.clientHeight+10&&(o==='auto'||o==='scroll'))e.scrollTop=e.scrollHeight})}"
    )
    page.wait_for_timeout(800)


def park_mouse(page):
    page.mouse.move(W * 0.2, 12)
    page.wait_for_timeout(1200)


VIEWS = [0] if FLAT2D else [0, 70, -70]  # orbit offsets (px of drag) tried per candidate


def pick_view(page, tag):
    """Try a few orbit angles, frame each, keep the one with the fullest frame."""
    best, cur = None, 0
    for v in VIEWS:
        drag(page, v - cur, 0)
        cur = v
        m = frame_scene(page, f"{tag}-v{v}")
        score = m["cov"] if m["box"] else 0
        page.locator("iframe").last.screenshot(path = str(OUT / f"{tag}-view{v:+d}.png"), timeout = 180000)
        log(tag, f"view {v:+d}: cov={m['cov']:.2f}")
        if best is None or score > best[1]:
            best = (v, score)
    drag(page, best[0] - cur, 0)
    m = frame_scene(page, f"{tag}-best")
    return best[0], m


def attempt(page, n):
    tag = f"try{n}"
    page.goto(f"{BASE}/chat")
    page.wait_for_timeout(3500)
    dismiss(page)
    collapse_sidebar(page)
    box = page.get_by_placeholder("Ask anything")
    box.click()
    if "\n" in PROMPT:
        page.keyboard.insert_text(PROMPT)  # typing a newline would send early
    else:
        page.keyboard.type(PROMPT, delay = 6)
    page.wait_for_timeout(400)
    page.keyboard.press("Enter")
    log(tag, "sent")
    t0 = time.time()
    page.get_by_text("tok/s").first.wait_for(timeout = 1800 * 1000)
    log(tag, f"reply done in {time.time() - t0:.0f}s, badge", page.get_by_text("tok/s").first.inner_text())
    page.wait_for_timeout(800)
    r = present(page, tag)
    fixes = 0
    while r and not r["clean"] and fixes < 2:
        # most failures are one typo; ask the model to fix it in the same chat, like a user would
        fixes += 1
        shot(page, f"{tag}-fix{fixes}-before")
        for b in page.get_by_role("button", name = "Close tab").all():
            if b.is_visible():
                b.click()
                break
        page.wait_for_timeout(1500)
        cards = lambda: page.get_by_text("HTML preview", exact = True).count()
        ncards = cards()
        errs = r.get("errs") or []
        err = ("; ".join(e for e in errs if not e.startswith("banner:"))[:200] or errs[0][:200]) if errs else ""
        ask = (f"The page throws this error: {err}. Please fix it and send the complete corrected HTML file."
               if err else
               "The preview shows almost nothing: the triangle lattice is not visible. Fix it so the whole lattice is "
               "drawn and spinning in the middle of the canvas, and send the complete corrected HTML file.")
        page.get_by_placeholder("Ask anything").click()
        page.keyboard.type(ask, delay = 6)
        page.wait_for_timeout(400)
        page.keyboard.press("Enter")
        log(tag, f"fix{fixes} sent:", err)
        t0 = time.time()
        # only the newest reply shows a tok/s badge: wait for a new preview card, then a badge below it
        while time.time() - t0 < 1500:
            page.wait_for_timeout(5000)
            if cards() <= ncards:
                continue
            badge = page.get_by_text("tok/s").last
            try:
                cb = page.get_by_text("HTML preview", exact = True).last.bounding_box()
                bb = badge.bounding_box() if badge.count() and badge.is_visible() else None
            except Exception:
                continue
            if bb and cb and bb["y"] > cb["y"]:
                break
        log(tag, f"fix{fixes} reply done in {time.time() - t0:.0f}s, badge", page.get_by_text("tok/s").last.inner_text())
        page.wait_for_timeout(800)
        r = present(page, f"{tag}-fix{fixes}")
    if not r or not r["clean"]:
        shot(page, f"{tag}-rejected")
        try:
            page.locator("iframe").last.screenshot(path = str(OUT / f"{tag}-rejected-canvas.png"), timeout = 60000)
        except Exception:
            pass
        return {"clean": False, **(r or {})}
    view, m = pick_view(page, tag)
    scroll_chat_end(page)
    park_mouse(page)
    shot(page, f"{tag}-1-final")
    page.locator("iframe").last.screenshot(path = str(OUT / f"{tag}-2-closeup.png"), timeout = 180000)
    drag(page, 45, 0)
    frame_scene(page, f"{tag}-b")
    scroll_chat_end(page)
    park_mouse(page)
    shot(page, f"{tag}-1b-alt")
    return {"clean": True, "cov": m["cov"], "rich": m.get("rich", r["rich"]), "view": view}


# Inside the preview iframe only: real time until __arm(), then time moves only on __step(ms), so a
# recording of the whole UI gets evenly spaced frames however slow each screenshot is.
FRAME_CLOCK = """(() => { if (window.top === window) return;
  const rr = window.requestAnimationFrame.bind(window), rn = performance.now.bind(performance), rd = Date.now;
  let armed = false, t = 0, d0 = 0, q = [];
  performance.now = () => armed ? t : rn();
  Date.now = () => armed ? d0 + t : rd();
  window.requestAnimationFrame = cb => armed ? (q.push(cb), q.length) : rr(cb);
  window.__arm = () => { t = rn(); d0 = rd() - t; armed = true; };
  window.__step = ms => { t += ms; const run = q; q = []; run.forEach(cb => { try { cb(t); } catch (e) {} }); };
})();"""


def record_ui(page, seconds = 22, fps = 30, warmup = 1.5):
    """Base page shot plus preview-only PNG frames on a fixed clock (OUT/ui-rec), long enough to cut a clean loop."""
    fr = None
    for f in page.frames[1:]:
        try:
            if f.evaluate("typeof window.__arm === 'function' && !!document.querySelector('canvas')"):
                fr = f
        except Exception:
            pass
    if fr is None:
        log("record_ui: no preview frame with a canvas")
        return
    rec = OUT / "ui-rec"
    rec.mkdir(exist_ok = True)
    ifr = page.locator("iframe").last
    box = ifr.bounding_box()
    page.screenshot(path = str(rec / "base.png"))
    (rec / "box.json").write_text(json.dumps({"box": box, "viewport": {"width": W, "height": H}, "fps": fps}))
    fr.evaluate("__arm()")
    dt = 1000 / fps
    for _ in range(int(warmup * fps)):
        fr.evaluate(f"__step({dt})")
    t0 = time.time()
    for i in range(int(seconds * fps)):
        fr.evaluate(f"__step({dt})")
        ifr.screenshot(path = str(rec / f"f{i:04d}.png"), timeout = 60000)
    log("record_ui", int(seconds * fps), "frames", f"in {time.time() - t0:.0f}s", json.dumps(box))


def final_shots(page, title, view):
    page.goto(f"{BASE}/chat")
    page.wait_for_timeout(3000)
    dismiss(page)
    expand_sidebar(page)
    page.get_by_text(title[:30], exact = False).first.click()
    page.wait_for_timeout(3500)
    collapse_sidebar(page)
    scroll_chat_end(page)
    r = present(page, "final")
    if not r or not r["clean"]:
        log("final: the kept chat did not come back clean", json.dumps(r)[:300])
        return False
    drag(page, view, 0)
    m = frame_scene(page, "final")
    scroll_chat_end(page)
    park_mouse(page)
    shot(page, "1-final-chat-and-kingdom")
    page.locator("iframe").last.screenshot(path = str(OUT / "2-canvas-closeup.png"), timeout = 180000)
    log("shot 2-canvas-closeup", f"cov={m['cov']:.2f}")
    try:
        record_ui(page)
    except Exception as exc:
        log("record_ui failed", type(exc).__name__, str(exc)[:300])
    drag(page, 45, 0)
    frame_scene(page, "final-b")
    scroll_chat_end(page)
    park_mouse(page)
    shot(page, "1b-final-alt-angle")
    page.locator("iframe").last.screenshot(path = str(OUT / "2b-canvas-closeup-alt-angle.png"), timeout = 180000)
    if open_picker(page):
        name, _ = find_row(page)
        log("PICKER_LABEL_FINAL", name)
        page.wait_for_timeout(800)
        shot(page, "3-picker-open-new-name")
        page.keyboard.press("Escape")
        page.wait_for_timeout(600)
    expand_sidebar(page)
    scroll_chat_end(page)
    park_mouse(page)
    shot(page, "4-sidebar-recents-one-chat")
    collapse_sidebar(page)
    return True


GL_CANDIDATES = [
    ("chromium", ["--enable-gpu", "--ignore-gpu-blocklist", "--use-angle=vulkan", "--enable-features=Vulkan,DefaultANGLEVulkan,VulkanFromANGLE"]),
    ("chromium", ["--enable-gpu", "--ignore-gpu-blocklist", "--use-gl=angle", "--use-angle=gl-egl"]),
    ("chromium", ["--enable-gpu", "--ignore-gpu-blocklist", "--use-angle=gl"]),
    (None, ["--enable-gpu", "--ignore-gpu-blocklist", "--use-angle=vulkan", "--enable-features=Vulkan"]),
    (None, ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]),
]
GL_JS = """()=>{const c=document.createElement('canvas');const g=c.getContext('webgl2')||c.getContext('webgl');
if(!g)return 'no webgl';const e=g.getExtension('WEBGL_debug_renderer_info');
return e?g.getParameter(e.UNMASKED_RENDERER_WEBGL):g.getParameter(g.RENDERER)}"""


def pick_browser(p):
    """First launch whose WebGL runs on the real GPU; SwiftShader as the fallback."""
    for channel, args in GL_CANDIDATES:
        kw = {"headless": True, "args": args}
        if channel:
            kw["channel"] = channel
        try:
            b = p.chromium.launch(**kw)
            pg = b.new_page()
            pg.goto("about:blank")
            r = pg.evaluate(GL_JS)
            log("GL probe", channel, args, "->", r)
            if r and "SwiftShader" not in r and "llvmpipe" not in r and r != "no webgl":
                return b, kw, False
            b.close()
        except Exception as exc:
            log("GL probe failed", channel, args, str(exc)[:200])
    b = p.chromium.launch(**kw)
    return b, kw, True


def main():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser, kw, software = pick_browser(p)
        # software GL draws a shadowed scene at 2x too slowly to screenshot
        dsf = 1 if software else 2
        log("browser", kw, "software GL" if software else "GPU GL", "scale", dsf)
        ctx = browser.new_context(viewport = {"width": W, "height": H}, device_scale_factor = dsf, color_scheme = "light")
        ctx.add_init_script(seed())
        ctx.add_init_script(FRAME_CLOCK)
        def fresh(old = None):
            # a generated page can exhaust the renderer; every try starts in a new tab
            if old is not None:
                try:
                    old.close()
                except Exception:
                    pass
            pg = ctx.new_page()
            pg.on("console", lambda m: m.type in ("error", "warning") and log("console", m.type, m.text[:200]))
            pg.on("pageerror", lambda e: (ERRORS.append(str(e)), log("pageerror", str(e)[:300])))
            pg.on("crash", lambda *_: log("PAGE CRASHED"))
            return pg

        page = fresh()
        try:
            try:
                network_setting(page)
            except Exception as exc:
                log("settings step failed, relying on the seeded value", str(exc)[:200])
            delete_threads([t["id"] for t in threads()])
            load_model(page)
            results = {}
            for n in range(1, TRIES + 1):
                before = {t["id"] for t in threads()}
                try:
                    if n > 1:
                        page = fresh(page)
                    r = attempt(page, n)
                except Exception as exc:
                    log(f"try{n}", "FAILED", type(exc).__name__, str(exc)[:400])
                    shot(page, f"try{n}-failure")
                    r = {"clean": False}
                new = [t for t in threads() if t["id"] not in before]
                for t in new:
                    try:
                        msgs = call("GET", f"/api/chat/threads/{t['id']}/messages", token = TOKEN)
                        (OUT / f"try{n}-messages.json").write_text(json.dumps(msgs, indent = 1), encoding = "utf-8")
                    except Exception as exc:
                        log("messages dump failed", str(exc)[:200])
                log(f"try{n}", "RESULT", json.dumps(r)[:300], "chats", [t["id"] for t in new])
                if r.get("clean") and new:
                    results[new[0]["id"]] = (r, new[0].get("title") or "")
                else:
                    delete_threads([t["id"] for t in new])
            good = False
            if results:
                best = max(results, key = lambda k: (results[k][0]["rich"], results[k][0]["cov"]))
                log("BEST", best, json.dumps(results[best][0]))
                delete_threads([t["id"] for t in threads() if t["id"] != best])
                page = fresh(page)
                good = final_shots(page, results[best][1], results[best][0]["view"])
            left = threads()
            log("RECENTS", json.dumps([[t["id"], t.get("title")] for t in left]))
            log("RESULT", "rendered" if good else "not rendered")
        finally:
            ctx.close()
            browser.close()
    sys.exit(0 if good else 1)


main()
