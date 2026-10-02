"""Drive the model picker through an NPU download and record what each step shows.

env: BASE, TOKENS (json file), OUT (dir), TAG, MODEL, MOCK_NPU=1 to fake the NPU API locally.
Prints one JSON line per observation and saves screenshots as OUT/<TAG>_<step>.png.
"""

import json
import os
import re
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = os.environ["BASE"]
TOKENS = json.loads(Path(os.environ["TOKENS"]).read_text())
OUT = Path(os.environ["OUT"])
TAG = os.environ["TAG"]
MODEL = os.environ.get("MODEL", "qwen3.5-9b-FLM")
MOCK = os.environ.get("MOCK_NPU") == "1"
OUT.mkdir(parents = True, exist_ok = True)
T0 = time.monotonic()
results: dict = {"tag": TAG, "model": MODEL, "steps": []}


def api(page, path, method = "GET"):
    # The page's token: setting the password replaces the minted one.
    token = page.evaluate("localStorage.getItem('unsloth_auth_token')")
    req = urllib.request.Request(BASE + path, method = method, headers = {"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout = 60) as r:
        return json.load(r)


def record(step, **data):
    row = {"t": round(time.monotonic() - T0, 1), "step": step, **data}
    results["steps"].append(row)
    print(json.dumps(row), flush = True)


INIT = f"""
if (!localStorage.getItem("unsloth_auth_token")) {{
  localStorage.setItem("unsloth_auth_token", {json.dumps(TOKENS["access_token"])});
  localStorage.setItem("unsloth_auth_refresh_token", {json.dumps(TOKENS["refresh_token"])});
}}
"""


def setup_account(page):
    """A fresh Studio home asks for a password before anything else."""
    page.wait_for_timeout(3000)
    if "change-password" not in page.url:
        return
    page.get_by_label("New password").fill("npu-ab-test-pw")
    page.get_by_label("Confirm password").fill("npu-ab-test-pw")
    page.get_by_role("button", name = "Change password").click()
    page.wait_for_url(lambda url: "change-password" not in url, timeout = 60000)
    page.goto(BASE + "/chat")


def mock_routes(ctx):
    status = {
        "supported": True,
        "hardware": {"present": True, "supported": True, "family": "XDNA2", "name": "NPU Strix Halo"},
        "runtime_installed": True, "runtime_running": True, "state": "ready", "ready": True,
        "error": None, "validation": {"ready": True, "problems": []}, "help_url": None,
        "loaded_model": None, "context_length": None, "loading_model": None,
    }
    model = {
        "id": MODEL, "model_path": f"lemonade:{MODEL}", "checkpoint": "qwen3.5:9b", "size_gb": 8.94,
        "downloaded": False, "labels": ["chat"], "supports_vision": False, "supports_reasoning": True,
        "supports_tools": False, "max_context_length": 32768,
    }
    ctx.route("**/api/npu/status", lambda r: r.fulfill(json = status))
    ctx.route("**/api/npu/models", lambda r: r.fulfill(json = {"models": [model]}))
    ctx.route("**/api/npu/downloads", lambda r: r.fulfill(json = {"downloads": []}))
    # Left pending, like a pull that is still running.
    ctx.route("**/api/npu/models/*/download", lambda r: None)


def open_picker(page):
    page.locator(".unsloth-model-selector-trigger").first.click()
    search = page.locator("[data-model-picker-search-input]").first
    search.wait_for(timeout = 30000)
    # NPU rows live under Recommended; a reopened picker may land on On Device.
    page.locator(POP).get_by_text("Recommended", exact = True).first.click()
    search.fill(MODEL)
    row = page.locator(POP).get_by_text(MODEL, exact = True).first
    try:
        row.wait_for(timeout = 60000)
    except Exception:
        page.screenshot(path = OUT / f"{TAG}_picker_debug.png")
        print(page.locator("[data-radix-popper-content-wrapper]").first.inner_text()[:2000])
        raise
    page.wait_for_timeout(1500)
    return row


POP = "[data-radix-popper-content-wrapper]"


def row_text(page):
    """The picker's text from the model's label up to the next section."""
    text = page.locator(POP).first.inner_text()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    i = lines.index(MODEL)
    return " | ".join(lines[i : i + 3])


def shot(page, step):
    path = OUT / f"{TAG}_{step}.png"
    pop = page.locator("[data-radix-popper-content-wrapper]").first
    try:
        pop.screenshot(path = path)
    except Exception:  # noqa: BLE001
        page.screenshot(path = path)
    return path.name


def close_picker(page):
    page.keyboard.press("Escape")
    page.wait_for_timeout(500)
    if page.locator("[data-model-picker-search-input]").count():
        page.keyboard.press("Escape")
        page.wait_for_timeout(500)


def nav(page, label, href):
    page.get_by_text(label, exact = True).first.click()
    page.wait_for_url(f"**{href}*", timeout = 30000)
    page.wait_for_timeout(1500)


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport = {"width": 1440, "height": 900}, locale = "en-US")
        ctx.add_init_script(INIT)
        if MOCK:
            mock_routes(ctx)
        page = ctx.new_page()
        page.goto(BASE + "/chat")
        setup_account(page)
        page.locator(".unsloth-model-selector-trigger").first.wait_for(timeout = 120000)
        page.wait_for_timeout(2000)
        page.screenshot(path = OUT / f"{TAG}_landing.png")
        record("landing", lang = page.evaluate("navigator.language"), url = page.url)

        row = open_picker(page)
        record("picker_before_download", row = row_text(page), shot = shot(page, "0_before"))
        row.click()
        page.wait_for_function(
            """([sel, floor]) => {
                const text = document.querySelector(sel)?.innerText || "";
                if (floor < 0) return text.includes("Downloading");
                const m = /Downloading (\\d+)%/.exec(text);
                return m && Number(m[1]) >= floor;
            }""",
            arg = [POP, -1 if MOCK else 3], timeout = 120000,
        )
        record("downloading", row = row_text(page), shot = shot(page, "1_downloading"))

        close_picker(page)
        page.wait_for_timeout(3000)
        row = open_picker(page)
        record("reopened_picker", row = row_text(page), shot = shot(page, "2_reopened"))
        close_picker(page)

        nav(page, "Model hub", "/hub")
        page.screenshot(path = OUT / f"{TAG}_hub_tab.png")
        nav(page, "New chat", "/chat")
        row = open_picker(page)
        record("after_switching_tabs", row = row_text(page), shot = shot(page, "3_tabs"))
        close_picker(page)

        page.reload()
        page.locator(".unsloth-model-selector-trigger").first.wait_for(timeout = 120000)
        page.wait_for_timeout(2000)
        row = open_picker(page)
        record("after_reload", row = row_text(page), shot = shot(page, "4_reload"))

        if not MOCK:
            # Keep the picker open and wait for the backend to finish the pull.
            deadline = time.monotonic() + 1800
            shown = []
            while time.monotonic() < deadline:
                models = api(page, "/api/npu/models")["models"]
                if next(m for m in models if m["id"] == MODEL)["downloaded"]:
                    break
                m = re.search(r"Downloading (\d+)%", row_text(page))
                if m and (not shown or shown[-1] != int(m.group(1))):
                    shown.append(int(m.group(1)))
                time.sleep(3)
            record("backend_finished", percents_shown = shown)
            page.wait_for_timeout(8000)
            on_device = page.locator(POP).locator('[aria-label="On device"]').count()
            record(
                "picker_after_finish", row = row_text(page), on_device_badges = on_device,
                shot = shot(page, "5_finished"),
            )
            close_picker(page)
            open_picker(page)
            on_device = page.locator(POP).locator('[aria-label="On device"]').count()
            record("picker_reopened_after_finish", on_device_badges = on_device, shot = shot(page, "6_reopened_finished"))
        ctx.close()
        browser.close()
    (OUT / f"{TAG}_ui.json").write_text(json.dumps(results, indent = 1))


if __name__ == "__main__":
    main()
