"""A/B of DC-018 on a real NPU: the reply's speed readout and pre-load run settings.

Usage: ab_ui.py BASE_URL PASSWORD OUT_DIR SIDE
Drives one running Studio. Results go to OUT_DIR/results.json, screenshots to OUT_DIR/*.png.
"""

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE, PASSWORD, OUT, SIDE = sys.argv[1], sys.argv[2], Path(sys.argv[3]), sys.argv[4]
OUT.mkdir(parents = True, exist_ok = True)
MODEL = "qwen3-0.6b-FLM"
PATH = "lemonade:" + MODEL
PROMPT = "Write two sentences about the moon."
PIN = 4096
SIDEBAR_PIN = 2048
RESULTS: dict = {"side": SIDE}
TOKEN = None


def save():
    (OUT / "results.json").write_text(json.dumps(RESULTS, indent = 1))


def log(*a):
    print(f"[{SIDE}]", *a, flush = True)


def call(method, path, body = None, timeout = 900, raw = False):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data = data, method = method)
    if TOKEN:
        req.add_header("Authorization", "Bearer " + TOKEN)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        resp = urllib.request.urlopen(req, timeout = timeout)
        if raw:
            return resp.status, resp
        txt = resp.read().decode("utf-8", "replace")
        code = resp.status
    except urllib.error.HTTPError as e:
        txt, code = e.read().decode("utf-8", "replace"), e.code
    try:
        return code, json.loads(txt)
    except ValueError:
        return code, txt


def api_login():
    global TOKEN
    for pw in (PASSWORD, PASSWORD + "-x9"):
        code, body = call("POST", "/api/auth/login", {"username": "unsloth", "password": pw})
        if code == 200:
            TOKEN = body["access_token"]
            if body.get("must_change_password"):
                code, body = call(
                    "POST",
                    "/api/auth/change-password",
                    {"current_password": pw, "new_password": PASSWORD + "-x9"},
                )
                TOKEN = body.get("access_token", TOKEN)
                return PASSWORD + "-x9"
            return pw
    raise SystemExit(f"login failed: {body}")


def stream_lines(body):
    code, resp = call("POST", "/v1/chat/completions", body, raw = True)
    assert code == 200, code
    return [l.decode().strip() for l in resp if l.strip()]


password = api_login()
code, status = call("POST", "/api/npu/enable", timeout = 1800)
log("enable", code, status.get("state") if isinstance(status, dict) else status)
code, resp = call("POST", f"/api/npu/models/{MODEL}/download", raw = True, timeout = 3600)
events = [json.loads(l[5:]) for l in (x.decode().strip() for x in resp) if l.startswith("data:")]
log("download", events[-1] if events else None)
save()

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(
        viewport = {"width": 1440, "height": 900}, device_scale_factor = 2, locale = "en-US"
    )
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))

    def shot(name, **kw):
        page.screenshot(path = str(OUT / f"{name}.png"), animations = "disabled", **kw)

    def open_picker():
        page.wait_for_load_state("networkidle")
        time.sleep(2)
        # A click while the page is still hydrating can be lost, so retry it.
        for attempt in range(4):
            page.locator('[data-tour="chat-model-selector"]').first.click()
            try:
                page.get_by_role("tab").first.wait_for(state = "visible", timeout = 10_000)
                return
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2)

    row = lambda: page.locator("[data-model-picker-option]", has_text = MODEL).first
    gear = lambda: page.get_by_role("button", name = f"Inference settings for {MODEL}")
    try:
        page.goto(f"{BASE}/login", wait_until = "domcontentloaded", timeout = 60_000)
        page.locator("#password").wait_for(state = "visible", timeout = 60_000)
        if page.locator("#username").count():
            page.locator("#username").fill("unsloth")
        page.locator("#password").fill(password)
        page.locator('button[type="submit"]').click()
        composer = page.locator('textarea[aria-label="Message input"]')
        composer.wait_for(state = "visible", timeout = 90_000)

        # 1. The NPU row in Recommended > NPU, hovered so its row actions show.
        open_picker()
        page.get_by_role("tab", name = "Recommended").click()
        page.get_by_label("Filter by format").click()
        page.get_by_role("option", name = "NPU").click()
        row().wait_for(state = "visible", timeout = 60_000)
        row().hover()
        time.sleep(1)
        shot("01-recommended-npu-row")
        RESULTS["recommended_gear"] = gear().count()

        # 2. The same model under On Device.
        page.get_by_role("tab", name = "On Device").click()
        row().wait_for(state = "visible", timeout = 30_000)
        row().hover()
        time.sleep(1)
        shot("02-on-device-npu-row")
        RESULTS["on_device_gear"] = gear().count()
        log("gears", RESULTS["recommended_gear"], RESULTS["on_device_gear"])
        save()

        if RESULTS["on_device_gear"]:
            # 3. The gear opens run settings: one Context Length control, set before loading.
            gear().first.hover()
            time.sleep(0.5)
            RESULTS["gear_tooltip"] = page.get_by_role("tooltip").first.inner_text()
            gear().first.click()
            page.get_by_text("Run settings").first.wait_for(state = "visible", timeout = 30_000)
            time.sleep(1)
            panel = page.get_by_label(f"Run settings for {MODEL}")
            RESULTS["settings_text"] = panel.inner_text()
            RESULTS["settings_engine_picker"] = panel.get_by_text("Inference engine").count()
            shot("03-npu-run-settings-auto")
            ctx = panel.locator('input[aria-label="Context Length"]')
            RESULTS["settings_context_shown"] = ctx.input_value()
            ctx.click()
            ctx.fill(str(PIN))
            ctx.press("Tab")
            time.sleep(0.5)
            shot("04-npu-run-settings-pinned")
            panel.get_by_role("button", name = "Load model").click()
        else:
            row().click()
        page.get_by_text("loaded on the NPU").first.wait_for(state = "visible", timeout = 300_000)
        code, st = call("GET", "/api/inference/status")
        RESULTS["loaded_context_length"] = st.get("context_length")
        RESULTS["loaded_model"] = st.get("model_identifier")
        log("loaded", RESULTS["loaded_model"], "ctx", RESULTS["loaded_context_length"])
        save()
        page.keyboard.press("Escape")
        time.sleep(2)

        # 5. A reply, then its speed readout.
        composer.fill(PROMPT)
        composer.press("Enter")
        trigger = page.locator('[data-slot="message-timing-trigger"]').last
        trigger.wait_for(state = "visible", timeout = 300_000)
        time.sleep(2)
        RESULTS["badge"] = trigger.inner_text()
        trigger.hover()
        popover = page.locator('[data-slot="message-timing-popover"]').last
        popover.wait_for(state = "visible", timeout = 30_000)
        time.sleep(1)
        RESULTS["metrics"] = popover.inner_text()
        log("badge", RESULTS["badge"])
        log("metrics", RESULTS["metrics"].replace("\n", " | "))
        shot("05-reply-metrics")
        save()
        page.mouse.move(5, 5)
        time.sleep(1)

        # 6. The loaded model's own run settings, in the side panel.
        page.get_by_role("button", name = "Open run settings").first.click()
        time.sleep(3)
        shot("06-loaded-model-settings")
        field = page.locator(
            'input[aria-label="Context Length"], input[aria-label="Max Seq Length"]'
        ).first
        RESULTS["sidebar_label"] = field.get_attribute("aria-label")
        RESULTS["sidebar_value"] = field.input_value()
        field.click()
        field.fill(str(SIDEBAR_PIN))
        field.press("Tab")
        time.sleep(0.5)
        shot("07-loaded-model-settings-edited")
        page.get_by_role("button", name = "Reload model").first.click()
        time.sleep(5)
        ctx = None
        for _ in range(60):
            code, st = call("GET", "/api/inference/status")
            ctx = st.get("context_length") if isinstance(st, dict) else None
            if ctx == SIDEBAR_PIN:
                break
            time.sleep(1)
        RESULTS["sidebar_reload_context_length"] = ctx
        log("sidebar", RESULTS["sidebar_label"], RESULTS["sidebar_value"], "->", SIDEBAR_PIN, "loaded", ctx)
        time.sleep(2)
        shot("08-after-sidebar-reload")
        save()
    except Exception as exc:
        RESULTS["ui_error"] = f"{type(exc).__name__}: {exc}"
        log("UI ERROR", RESULTS["ui_error"])
        try:
            shot("99-failure")
        except Exception:
            pass
    RESULTS["page_errors"] = errors[:3]
    browser.close()

# 7. The same reply over the OpenAI API, as any client sees it.
lines = stream_lines(
    {
        "model": PATH,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": True,
        "max_tokens": 200,
        "stream_options": {"include_usage": True},
    }
)
closing = [json.loads(l[5:]) for l in lines if l.startswith("data:") and '"usage"' in l]
RESULTS["api_closing_chunk"] = closing[-1] if closing else None
log("api closing chunk", json.dumps(RESULTS["api_closing_chunk"])[:900])
save()
