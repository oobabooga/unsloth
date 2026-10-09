"""Drive one running Studio: make sure the NPU runtime is enabled, then read the NPU credit line
in the Hub's NPU list and in the chat model picker, with screenshots.

usage: npu_versions_ab.py BASE_URL STUDIO_HOME LABEL OUT_DIR [--mock] [--expect=TEXT]
--mock replaces /api/npu/* with a canned ready runtime (local selector check, no NPU needed).
--expect waits, without a reload, for the Hub credit to contain TEXT (a pin bump's upgrade).
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

base, home, label, out = sys.argv[1], Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
mock = "--mock" in sys.argv
expect = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--expect=")), None)
out.mkdir(parents = True, exist_ok = True)


def api(method, path, body = None, token = None, timeout = 60):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data = data, method = method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout = timeout) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors = "replace")


secret = (home / "auth" / ".desktop_secret").read_text().strip()
status, tokens = api("POST", "/api/auth/desktop-login", {"secret": secret})
assert status == 200, (status, tokens)
token = tokens["access_token"]

report = {"label": label}
if not mock:
    _, npu = api("GET", "/api/npu/status", token = token)
    report["status_before"] = npu
    if not (npu.get("runtime_installed") and npu.get("ready")):
        t0 = time.time()
        code, enabled = api("POST", "/api/npu/enable", {}, token = token, timeout = 1800)
        report["enable"] = {"http": code, "seconds": round(time.time() - t0, 1), "body": enabled}
    _, report["status"] = api("GET", "/api/npu/status", token = token)
    print(json.dumps({k: report[k] for k in report if k != "status_before"}, indent = 1)[:4000])

MOCK_STATUS = {
    "supported": True,
    "hardware": {"present": True, "supported": True, "family": "XDNA2", "name": "NPU Strix Halo"},
    "runtime_installed": True,
    "runtime_running": True,
    "state": "ready",
    "ready": True,
    "error": None,
    "validation": {"ready": True, "problems": []},
    "help_url": None,
    "loaded_model": None,
    "context_length": None,
    "loading_model": None,
    "versions": {"lemonade": "11.9.0", "fastflowlm": "1.0.7"},
}
MOCK_MODELS = {
    "models": [
        {
            "id": "qwen3.8-mtp:27b",
            "model_path": "lemonade:qwen3.8-mtp:27b",
            "checkpoint": "FastFlowLM/Qwen3.8-27B-MTP-NPU2",
            "size_gb": 17.0,
            "downloaded": False,
            "labels": ["reasoning"],
            "supports_vision": False,
            "supports_reasoning": True,
            "supports_tools": True,
            "max_context_length": 32768,
            "resume_percent": None,
        }
    ]
}


def credit_text(page):
    line = page.get_by_text("Powered by", exact = False).last
    line.wait_for(timeout = 60_000)
    return " ".join(line.inner_text().split())


with sync_playwright() as p:
    browser = p.chromium.launch()
    ctx = browser.new_context(viewport = {"width": 1400, "height": 900})
    ctx.add_init_script(
        f"""
        localStorage.setItem("unsloth_auth_token", {json.dumps(token)});
        localStorage.setItem("unsloth_auth_refresh_token", {json.dumps(tokens["refresh_token"])});
        localStorage.setItem("unsloth_auth_session_mark", "ab");
        localStorage.setItem("unsloth_locale", "en");
        """
    )
    if mock:
        ctx.route(
            "**/api/npu/status",
            lambda r: r.fulfill(json = MOCK_STATUS),
        )
        ctx.route("**/api/npu/models", lambda r: r.fulfill(json = MOCK_MODELS))
        ctx.route("**/api/npu/downloads", lambda r: r.fulfill(json = {"downloads": []}))
    page = ctx.new_page()
    seen = []
    page.on(
        "response",
        lambda r: seen.append(r.url) if "/api/npu/status" in r.url else None,
    )

    # Hub > Discover > format NPU.
    page.goto(f"{base}/hub")
    page.locator('[data-tour="hub-tabs"]').wait_for(timeout = 60_000)
    page.get_by_role("radio", name = "Discover").click()
    page.get_by_role("button", name = "Format filter").click()
    page.get_by_role("option", name = "NPU").click()
    page.locator('[data-testid="hub-npu-catalog"]').wait_for(timeout = 120_000)
    page.wait_for_timeout(3000)
    hub = page.locator('[data-testid="hub-npu-catalog"]')
    report["hub_text_tail"] = " ".join(hub.inner_text().split())[-300:]
    credit = hub.get_by_text("Powered by", exact = False)
    report["hub_credit"] = " ".join(credit.inner_text().split()) if credit.count() else None
    page.screenshot(path = str(out / f"{label}-hub-npu.png"))
    if expect:
        report["hub_credit_on_open"] = report["hub_credit"]
        t0 = time.time()
        credit.filter(has_text = expect).wait_for(timeout = 1_200_000)
        report["expect_seconds"] = round(time.time() - t0, 1)
        report["hub_credit"] = " ".join(credit.inner_text().split())
        report["hub_text_tail"] = " ".join(hub.inner_text().split())[-300:]
        page.screenshot(path = str(out / f"{label}-hub-npu-after.png"))

    # Chat > model picker > NPU section.
    page.goto(f"{base}/chat")
    page.wait_for_timeout(3000)
    page.locator(".unsloth-model-selector-trigger:visible").first.click()
    page.wait_for_timeout(2000)
    # The picker's NPU group starts folded.
    if not page.get_by_text("Powered by", exact = False).count():
        page.get_by_role("button", name = "Expand section").first.click()
    report["picker_credit"] = credit_text(page)
    page.wait_for_timeout(1500)
    page.screenshot(path = str(out / f"{label}-picker-npu.png"))
    report["status_requests"] = len(seen)
    browser.close()

keys = ("label", "hub_credit_on_open", "expect_seconds", "hub_credit", "picker_credit", "hub_text_tail")
print(json.dumps({k: report[k] for k in keys if k in report}, indent = 1))
(out / f"{label}.json").write_text(json.dumps(report, indent = 1))
