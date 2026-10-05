"""Drive Studio's UI the way a user would to install vLLM and load a model on it.

Log in, open the model picker, search MODEL, open its run settings, pick vLLM as the inference
engine, install it, load the model and chat. On a build without AMD support the engine picker is
hidden or vLLM is disabled, which is recorded rather than treated as a crash.

Usage: vllm_ui.py BASE_URL PASSWORD OUT_DIR ARM MODEL
"""

import json
import sys
import time
import traceback
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

BASE, PASSWORD, OUT, ARM, MODEL = sys.argv[1], sys.argv[2], Path(sys.argv[3]), sys.argv[4], sys.argv[5]
OUT.mkdir(parents = True, exist_ok = True)
RESULTS = []
FACTS = {}


def record(name, ok, detail = ""):
    RESULTS.append({"name": name, "ok": bool(ok), "detail": str(detail)[:1500]})
    print(("PASS " if ok else "FAIL ") + name + (f" :: {detail}" if detail else ""), flush = True)
    (OUT / "ui-results.json").write_text(json.dumps(RESULTS, indent = 1))


def fact(name, value):
    FACTS[name] = value
    print(f"FACT {name} = {json.dumps(value)[:1500]}", flush = True)
    (OUT / "facts.json").write_text(json.dumps(FACTS, indent = 1))


def api(page, path, method = "GET", body = None):
    return page.evaluate(
        """async ([path, method, body]) => {
            const token = localStorage.getItem('unsloth_auth_token');
            const response = await fetch(path, {
                method,
                headers: {Authorization: `Bearer ${token}`, 'Content-Type': 'application/json'},
                body: body === null ? undefined : JSON.stringify(body),
            });
            let data = null;
            try { data = await response.json(); } catch (e) {}
            return {status: response.status, data};
        }""",
        [path, method, body],
    )


def open_run_settings(page):
    # Best effort: a page that keeps polling never goes idle, and the click below is retried anyway.
    try:
        page.wait_for_load_state("networkidle", timeout = 15_000)
    except Exception:
        pass
    # The model selector waits for the backend's torch warm-up.
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        health = page.evaluate("async () => (await fetch('/api/health')).json()")
        if not health.get("torch_warm_in_progress"):
            break
        time.sleep(5)
    search = page.get_by_placeholder("Search Unsloth models")
    # A click while the page is still hydrating (first polls) can be lost.
    for attempt in range(6):
        time.sleep(3)
        page.locator('[data-tour="chat-model-selector"]').first.click(timeout = 120_000)
        try:
            search.wait_for(state = "visible", timeout = 10_000)
            break
        except Exception:
            page.keyboard.press("Escape")
    search.wait_for(state = "visible", timeout = 10_000)
    search.fill(MODEL)
    button = page.locator(f"button[aria-label='Inference settings for {MODEL}']").first
    button.wait_for(state = "visible", timeout = 90_000)
    button.click(timeout = 60_000)
    page.get_by_text("Run settings").first.wait_for(state = "visible", timeout = 60_000)
    time.sleep(2)


def open_run_settings_retrying(page, shot):
    for attempt in range(3):
        try:
            open_run_settings(page)
            return
        except Exception:
            fact(f"open_run_settings_attempt_{attempt}", traceback.format_exc()[-1500:])
            shot(f"00-open-attempt-{attempt}")
            page.keyboard.press("Escape")
            page.keyboard.press("Escape")
    open_run_settings(page)


with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport = {"width": 1440, "height": 1000}, device_scale_factor = 1, locale = "en-US")
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    shot = lambda name: page.screenshot(path = str(OUT / f"{name}.png"), animations = "disabled")
    try:
        page.goto(f"{BASE}/login", wait_until = "domcontentloaded", timeout = 60_000)
        page.locator("#password").wait_for(state = "visible", timeout = 60_000)
        if page.locator("#username").count():
            page.locator("#username").fill("unsloth")
        page.locator("#password").fill(PASSWORD)
        page.locator('button[type="submit"]').click()
        composer = page.locator('textarea[aria-label="Message input"]')
        composer.wait_for(state = "visible", timeout = 90_000)
        record("logged in, chat mounted", True)

        engines = api(page, "/api/engines")
        fact("engines_before", engines)
        vllm = next((e for e in engines["data"] or [] if e.get("engine") == "vllm"), {})
        fact("vllm_unsupported_reason", vllm.get("unsupported_reason"))
        fact("vllm_download_bytes", vllm.get("download_bytes"))
        system = api(page, "/api/system")
        fact("device_backend", (system["data"] or {}).get("device_backend"))

        open_run_settings_retrying(page, shot)
        shot("01-run-settings")
        select = page.get_by_label("Inference engine")
        fact("engine_select_visible", select.count() > 0 and select.first.is_visible())
        if not (select.count() and select.first.is_visible()):
            record("vLLM offered in run settings", False, "the Inference engine select is not shown")
            fact("api_install_response", api(page, "/api/engines/vllm/install", "POST"))
            time.sleep(10)
            vllm = next(e for e in api(page, "/api/engines")["data"] if e["engine"] == "vllm")
            fact("api_install_job", vllm["job"])
            raise SystemExit
        select.first.click()
        time.sleep(1)
        options = {o.inner_text(): o.get_attribute("data-disabled") is not None for o in page.get_by_role("option").all()}
        fact("engine_options_disabled", options)
        shot("02-engine-options")
        vllm_option = page.get_by_role("option", name = "vLLM").first
        if vllm_option.get_attribute("data-disabled") is not None:
            record("vLLM offered in run settings", False, "the vLLM option is disabled")
            page.keyboard.press("Escape")
            # What the backend does with an install the UI would not offer.
            fact("api_install_response", api(page, "/api/engines/vllm/install", "POST"))
            time.sleep(10)
            vllm = next(e for e in api(page, "/api/engines")["data"] if e["engine"] == "vllm")
            fact("api_install_job", vllm["job"])
            raise SystemExit
        vllm_option.click()
        time.sleep(1)
        record("vLLM offered in run settings", True)
        gpus = [s.get_attribute("aria-label") for s in page.locator("[role=group][aria-label='GPUs'] [role=switch]").all()]
        fact("engine_gpu_switches", gpus)
        page.get_by_label("Precision").first.click()
        time.sleep(1)
        fact(
            "precision_options_disabled",
            {o.inner_text(): o.get_attribute("data-disabled") is not None for o in page.get_by_role("option").all()},
        )
        page.keyboard.press("Escape")
        time.sleep(1)
        shot("03-vllm-selected")

        install = page.get_by_role("button", name = "Install engine")
        if not install.count() or install.first.is_disabled():
            record("Install engine button enabled", False, page.locator("[role=alert]").all_inner_texts())
            raise SystemExit
        record("Install engine button enabled", True)
        install.first.click()
        prompt = page.get_by_role("region", name = "Install vLLM")
        prompt.wait_for(state = "visible", timeout = 30_000)
        fact("install_prompt", prompt.inner_text())
        shot("04-install-prompt")
        prompt.get_by_role("button", name = "Install and load").click()
        started = time.monotonic()

        # Install, then the automatic load: poll the backend the UI is driving.
        last_shot = 0.0
        state = None
        loaded_at = None
        while time.monotonic() - started < 60 * 60:
            engines = api(page, "/api/engines")["data"] or []
            job = next((e for e in engines if e.get("engine") == "vllm"), {}).get("job", {})
            if job.get("state") != state:
                state = job.get("state")
                fact(f"install_job_{state}_at_s", round(time.monotonic() - started))
            if state in ("error", "cancelled", "waiting"):
                fact("install_job", job)
                break
            status = api(page, "/api/inference/status")["data"] or {}
            if status.get("active_model") and status.get("engine") == "vllm":
                break
            if state == "success":
                loaded_at = loaded_at or time.monotonic()
                alerts = [t for t in page.locator("[role=alert], [data-sonner-toast]").all_inner_texts() if t.strip()]
                if alerts:
                    fact("alerts_while_loading", alerts)
                # The load starts right after the install; nothing loading for a while means it failed.
                if not status.get("loading") and time.monotonic() - loaded_at > 180:
                    fact("load_status", status)
                    break
            if time.monotonic() - last_shot > 300:
                shot(f"05-installing-{int(time.monotonic() - started)}s")
                last_shot = time.monotonic()
            time.sleep(10)
        fact("install_and_load_s", round(time.monotonic() - started))
        status = api(page, "/api/inference/status")["data"] or {}
        fact("inference_status", {k: status.get(k) for k in ("active_model", "engine", "engine_precision", "loaded", "loading", "context_length")})
        engines = api(page, "/api/engines")["data"] or []
        fact("engines_after", engines)
        shot("06-after-install")
        if state in ("error", "cancelled", "waiting"):
            record("vLLM installs", False, job.get("message"))
            raise SystemExit
        record("vLLM installs", next((e for e in engines if e.get("engine") == "vllm"), {}).get("installed"))
        if not (status.get("active_model") and status.get("engine") == "vllm"):
            record("model loads on vLLM", False, status)
            raise SystemExit
        record("model loads on vLLM", True, status.get("active_model"))

        page.keyboard.press("Escape")
        time.sleep(2)
        composer.fill("What is the capital of France? Answer in one word.")
        composer.press("Enter")
        expect(page.get_by_text("Paris").last).to_be_visible(timeout = 180_000)
        time.sleep(5)
        shot("07-chat")
        record("chat answers through vLLM", True)
    except SystemExit:
        pass
    except Exception as exc:
        record("ui step", False, traceback.format_exc()[-1500:])
        try:
            shot("99-failure")
        except Exception:
            pass
    if errors:
        fact("page_errors", errors[:5])
    browser.close()

failed = [r["name"] for r in RESULTS if not r["ok"]]
print(f"[{ARM}] {len(RESULTS) - len(failed)}/{len(RESULTS)} UI checks passed" + (f"; failed: {failed}" if failed else ""))
