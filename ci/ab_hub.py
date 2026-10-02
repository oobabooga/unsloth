"""The Model Hub's NPU format: list, download with progress, leave and return, then Run in chat.

env: BASE, TOKENS (json), OUT, TAG, MODEL, MOCK_NPU=1 to fake the NPU API locally (stops after
the download starts). Prints one JSON line per observation; screenshots go to OUT/<TAG>_hub_*.png.
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
MODEL = os.environ.get("MODEL", "llama3.2-1b-FLM")
MOCK = os.environ.get("MOCK_NPU") == "1"
OUT.mkdir(parents = True, exist_ok = True)
T0 = time.monotonic()
results: dict = {"tag": TAG, "model": MODEL, "steps": []}

INIT = f"""
if (!localStorage.getItem("unsloth_auth_token")) {{
  localStorage.setItem("unsloth_auth_token", {json.dumps(TOKENS["access_token"])});
  localStorage.setItem("unsloth_auth_refresh_token", {json.dumps(TOKENS["refresh_token"])});
}}
"""


def record(step, **data):
    row = {"t": round(time.monotonic() - T0, 1), "step": step, **data}
    results["steps"].append(row)
    print(json.dumps(row), flush = True)


def api(page, path, body = None):
    token = page.evaluate("localStorage.getItem('unsloth_auth_token')")
    req = urllib.request.Request(
        BASE + path,
        data = None if body is None else json.dumps(body).encode(),
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout = 60) as r:
        return json.load(r)


def setup_account(page):
    page.wait_for_timeout(3000)
    if "change-password" not in page.url:
        return
    page.get_by_label("New password").fill("npu-ab-test-pw")
    page.get_by_label("Confirm password").fill("npu-ab-test-pw")
    page.get_by_role("button", name = "Change password").click()
    page.wait_for_url(lambda url: "change-password" not in url, timeout = 60000)


def mock_routes(ctx):
    status = {
        "supported": True,
        "hardware": {"present": True, "supported": True, "family": "XDNA2", "name": "NPU Strix Halo"},
        "runtime_installed": True, "runtime_running": True, "state": "ready", "ready": True,
        "error": None, "validation": {"ready": True, "problems": []}, "help_url": None,
        "loaded_model": None, "context_length": None, "loading_model": None,
    }

    def model(mid, downloaded, vision = False):
        return {
            "id": mid, "model_path": f"lemonade:{mid}", "checkpoint": mid.replace("-FLM", ""),
            "size_gb": 1.3, "downloaded": downloaded, "labels": ["chat"], "supports_vision": vision,
            "supports_reasoning": False, "supports_tools": True, "max_context_length": 131072,
        }

    models = [model(MODEL, False), model("gemma3-4b-FLM", True, True), model("qwen3-0.6b-FLM", False)]
    ctx.route("**/api/npu/status", lambda r: r.fulfill(json = status))
    ctx.route("**/api/npu/models", lambda r: r.fulfill(json = {"models": models}))
    ctx.route("**/api/npu/downloads", lambda r: r.fulfill(json = {"downloads": []}))
    ctx.route("**/api/npu/models/*/download", lambda r: None)


def row(page):
    return page.locator(f'[data-testid="hub-npu-row"][data-model="{MODEL}"]')


def row_text(page):
    loc = row(page)
    return re.sub(r"\s+", " ", loc.inner_text()).strip() if loc.count() else None


def select_npu_format(page):
    page.get_by_role("button", name = "Format filter").first.click()
    page.get_by_role("menuitemradio", name = "NPU").or_(page.get_by_role("option", name = "NPU")).first.click()
    page.locator('[data-testid="hub-npu-catalog"]').wait_for(timeout = 60000)
    search = page.get_by_placeholder(re.compile("Search (all|on-device) models")).first
    search.fill(MODEL)
    row(page).wait_for(timeout = 60000)
    page.wait_for_timeout(1000)


def shot(page, step):
    path = OUT / f"{TAG}_hub_{step}.png"
    page.screenshot(path = path)
    return path.name


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport = {"width": 1440, "height": 900}, locale = "en-US")
        ctx.add_init_script(INIT)
        if MOCK:
            mock_routes(ctx)
        page = ctx.new_page()
        page.goto(BASE + "/hub")
        setup_account(page)
        page.goto(BASE + "/hub")
        page.get_by_role("button", name = "Format filter").first.wait_for(timeout = 120000)
        page.wait_for_timeout(2000)
        page.get_by_role("button", name = "Format filter").first.click()
        page.wait_for_timeout(500)
        menu = page.locator("[role=menu], [role=listbox]").first
        options = menu.inner_text().split("\n") if menu.count() else []
        record("format_menu", options = options, shot = shot(page, "0_menu"))
        page.keyboard.press("Escape")
        if "NPU" not in options:
            ctx.close()
            browser.close()
            (OUT / f"{TAG}_hub.json").write_text(json.dumps(results, indent = 1))
            return

        select_npu_format(page)
        record("npu_list", row = row_text(page), rows = page.locator('[data-testid="hub-npu-row"]').count(), shot = shot(page, "1_list"))

        row(page).get_by_role("button", name = "Download").click()
        page.wait_for_function(
            """([sel, floor]) => {
                const text = document.querySelector(sel)?.innerText || "";
                if (floor < 0) return text.includes("Downloading");
                const m = /Downloading (\\d+)%/.exec(text);
                return m && Number(m[1]) >= floor;
            }""",
            arg = [f'[data-testid="hub-npu-row"][data-model="{MODEL}"]', -1 if MOCK else 2],
            timeout = 120000,
        )
        record("downloading", row = row_text(page), shot = shot(page, "2_downloading"))

        page.get_by_text("New chat", exact = True).first.click()
        page.wait_for_url("**/chat*", timeout = 30000)
        page.wait_for_timeout(1500)
        page.get_by_text("Model hub", exact = True).first.click()
        page.wait_for_url("**/hub*", timeout = 30000)
        page.get_by_role("button", name = "Format filter").first.wait_for(timeout = 60000)
        select_npu_format(page)
        record("back_in_hub", row = row_text(page), shot = shot(page, "3_back"))
        if MOCK:
            ctx.close()
            browser.close()
            (OUT / f"{TAG}_hub.json").write_text(json.dumps(results, indent = 1))
            return

        row(page).get_by_role("button", name = "Run").wait_for(timeout = 1800000)
        page.wait_for_timeout(1000)
        record("downloaded", row = row_text(page), shot = shot(page, "4_downloaded"))

        row(page).get_by_role("button", name = "Run").click()
        page.wait_for_url("**/chat*", timeout = 30000)
        load = page.get_by_role("button", name = "Load model")
        load.wait_for(timeout = 60000)
        page.wait_for_timeout(1500)
        record("run_settings", url = page.url, shot = shot(page, "5_run_settings"))
        load.click()
        deadline = time.monotonic() + 600
        status = {}
        while time.monotonic() < deadline:
            status = api(page, "/api/npu/status")
            if status.get("loaded_model") == f"lemonade:{MODEL}" or status.get("loaded_model") == MODEL:
                break
            time.sleep(2)
        page.wait_for_timeout(2000)
        record("loaded", loaded_model = status.get("loaded_model"), shot = shot(page, "6_loaded"))

        composer = page.get_by_placeholder(re.compile("Ask anything|Message")).first
        composer.fill("Reply with the single word: ready")
        composer.press("Enter")
        page.wait_for_timeout(15000)
        record("chat", url = page.url, transcript = page.locator("body").inner_text()[-1200:], shot = shot(page, "7_chat"))
        ctx.close()
        browser.close()
    (OUT / f"{TAG}_hub.json").write_text(json.dumps(results, indent = 1))


if __name__ == "__main__":
    main()
