"""Drive Studio's chat UI onto the NPU: log in, pick the NPU format on Recommended, pick a
FastFlowLM model, chat, then find it under On Device. Screenshots go to OUT_DIR.

Usage: npu_ui.py BASE_URL PASSWORD OUT_DIR
"""

import json
import sys
import time
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

BASE, PASSWORD, OUT = sys.argv[1], sys.argv[2], Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = []
MODEL = "qwen3-0.6b-FLM"


def record(name, ok, detail=""):
    RESULTS.append({"name": name, "ok": bool(ok), "detail": str(detail)[:800]})
    print(("PASS " if ok else "FAIL ") + name + (f" :: {detail}" if detail else ""), flush=True)
    (OUT / "ui-results.json").write_text(json.dumps(RESULTS, indent=1))


def open_picker(page):
    """Open the model picker once the page has settled; a click mid-hydration can be lost."""
    page.wait_for_load_state("networkidle")
    time.sleep(2)
    page.locator('[data-tour="chat-model-selector"]').first.click()
    page.get_by_role("tab").first.wait_for(state="visible", timeout=30_000)


with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2, locale="en-US")
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    shot = lambda name: page.screenshot(path=str(OUT / f"{name}.png"), animations="disabled")
    row = lambda: page.locator("[data-model-picker-option]", has_text=MODEL).first
    try:
        page.goto(f"{BASE}/login", wait_until="domcontentloaded", timeout=60_000)
        page.locator("#password").wait_for(state="visible", timeout=60_000)
        # The username field only appears in multi-user mode.
        if page.locator("#username").count():
            page.locator("#username").fill("unsloth")
        page.locator("#password").fill(PASSWORD)
        page.locator('button[type="submit"]').click()
        composer = page.locator('textarea[aria-label="Message input"]')
        composer.wait_for(state="visible", timeout=90_000)
        record("logged in, chat mounted", True)

        open_picker(page)
        record("no separate NPU tab", page.get_by_role("tab", name="NPU").count() == 0)
        page.get_by_role("tab", name="Recommended").click()
        page.get_by_label("Filter by format").click()
        page.get_by_role("option", name="NPU").click()
        row().wait_for(state="visible", timeout=60_000)
        time.sleep(1)
        shot("01-recommended-npu-format")
        record("Recommended lists FastFlowLM models under the NPU format", True)

        row().click()
        page.get_by_text("loaded on the NPU").first.wait_for(state="visible", timeout=180_000)
        time.sleep(1)
        shot("02-npu-model-loaded")
        record("picking a model loads it on the NPU", True)

        composer.fill("What is 17*23? Answer with just the number.")
        composer.press("Enter")
        expect(page.get_by_text("391").last).to_be_visible(timeout=120_000)
        # Let the reply finish before the screenshot.
        time.sleep(10)
        shot("03-npu-chat")
        record("chat answers from the NPU model", True)

        open_picker(page)
        page.get_by_role("tab", name="On Device").click()
        row().wait_for(state="visible", timeout=30_000)
        time.sleep(1)
        shot("04-on-device-npu")
        record("On Device lists the downloaded NPU model", True)
        page.keyboard.press("Escape")
    except Exception as exc:
        record("ui step", False, f"{type(exc).__name__}: {exc}")
        try:
            shot("99-failure")
        except Exception:
            pass
    if errors:
        record("no page errors", False, errors[:3])
    browser.close()

failed = [r["name"] for r in RESULTS if not r["ok"]]
print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} UI checks passed" + (f"; failed: {failed}" if failed else ""))
