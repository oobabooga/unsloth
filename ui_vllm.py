"""Drive Unsloth Studio as a user: install vLLM from Settings, load a model with it, chat.

usage: ui_vllm.py BASE OUTDIR MODE [MODEL]
  MODE "inspect": record the Settings > System > Inference engines row and the model's engine
                  picker without changing anything (the A side, and a pre-check on B).
  MODE "full":    inspect, then install vLLM from Settings if needed, load MODEL with vLLM from
                  its run settings, and chat with it.
Auth: UNSLOTH_TOKEN, or the bootstrap password file in BOOTSTRAP_FILE (changed to NEW_PASSWORD).
Writes OUTDIR/result.json and numbered screenshots.
"""

import json
import os
import re
import sys
import time
import urllib.request

from playwright.sync_api import expect, sync_playwright

BASE, OUT, MODE = sys.argv[1], sys.argv[2], sys.argv[3]
MODEL = sys.argv[4] if len(sys.argv) > 4 else "Qwen/Qwen2.5-0.5B-Instruct"
os.makedirs(OUT, exist_ok = True)
result = {"mode": MODE, "base": BASE, "model": MODEL, "steps": []}
shot_no = [0]
current_page = [None]


def log(_text, **data):
    entry = {"t": round(time.time()), "msg": _text, **data}
    result["steps"].append(entry)
    print("[ui]", _text, json.dumps(data) if data else "", flush = True)


def save():
    with open(os.path.join(OUT, "result.json"), "w") as fh:
        json.dump(result, fh, indent = 1)


def api(path, token, method = "GET", body = None, timeout = 60):
    req = urllib.request.Request(
        BASE + path,
        data = json.dumps(body).encode() if body is not None else None,
        method = method,
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout = timeout) as r:
        return json.load(r)


def post(path, body, token = None):
    req = urllib.request.Request(
        BASE + path,
        data = json.dumps(body).encode(),
        headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {token}"} if token else {})},
    )
    with urllib.request.urlopen(req, timeout = 60) as r:
        return json.load(r)


def token():
    if os.environ.get("UNSLOTH_TOKEN"):
        return os.environ["UNSLOTH_TOKEN"]
    new = os.environ.get("NEW_PASSWORD", "AmdVllmTest-2026!")
    # Studio deletes the bootstrap file once the password has been changed.
    try:
        boot = open(os.environ["BOOTSTRAP_FILE"]).read().strip()
    except OSError:
        boot = None
    for password in (new, boot):
        if password is None:
            continue
        try:
            resp = post("/api/auth/login", {"username": "unsloth", "password": password})
        except urllib.error.HTTPError:
            continue
        if resp.get("must_change_password"):
            resp = post(
                "/api/auth/change-password",
                {"current_password": password, "new_password": new},
                resp["access_token"],
            )
        return resp["access_token"]
    raise SystemExit("could not log in")


def main():
    tok = token()
    result["engines_api_before"] = api("/api/engines", tok)
    log("engines api", engines = [
        {k: e.get(k) for k in ("engine", "installed", "unsupported_reason", "precisions", "platform", "version")}
        for e in result["engines_api_before"]
    ])
    with sync_playwright() as p:
        browser = p.chromium.launch(args = ["--disable-gpu"])
        ctx = browser.new_context(viewport = {"width": 1440, "height": 1000})
        ctx.add_init_script(f"try{{localStorage.setItem('unsloth_auth_token', {json.dumps(tok)});}}catch(e){{}}")
        if os.environ.get("FORCE_SYSTEM_FONT"):
            # The AMD runner's headless Chromium lays out but never paints the bundled web fonts.
            ctx.add_init_script(
                "document.addEventListener('DOMContentLoaded', () => { const s = document.createElement('style');"
                " s.textContent = '*:not(svg):not(svg *) { font-family: \"DejaVu Sans\", sans-serif !important; }';"
                " document.head.appendChild(s); });"
            )
        page = ctx.new_page()
        current_page[0] = page
        page.set_default_timeout(60_000)

        def shoot(name):
            shot_no[0] += 1
            path = os.path.join(OUT, f"{shot_no[0]:02d}-{name}.png")
            try:
                page.screenshot(path = path, full_page = False)
            except Exception as exc:
                log("screenshot failed", name = name, error = str(exc)[:200])

        try:
            journey(page, browser, tok, shoot)
        except BaseException:
            try:
                page.screenshot(path = os.path.join(OUT, "99-failure.png"))
                open(os.path.join(OUT, "99-failure.html"), "w", encoding = "utf-8").write(page.content())
            except Exception:
                pass
            raise
        browser.close()


def journey(page, browser, tok, shoot):
    if True:
        page.goto(BASE, wait_until = "domcontentloaded")
        page.locator('textarea[aria-label="Message input"]').wait_for(state = "visible", timeout = 120_000)
        page.wait_for_timeout(1500)
        shoot("chat-loaded")
        result["text_probe"] = page.evaluate(
            "() => { const e = [...document.querySelectorAll('a,button,span,h1,h2,p')].find(x => x.textContent.trim().length > 3 && x.getBoundingClientRect().height > 0 && x.getBoundingClientRect().width > 8 && !x.closest('[aria-hidden=true]'));"
            " if (!e) return null; const r = e.getBoundingClientRect(); const s = getComputedStyle(e);"
            " return {text: e.textContent.trim().slice(0, 40), w: r.width, h: r.height, font: s.fontFamily}; }"
        )
        log("text probe", probe = result["text_probe"])

        # ── Settings > System > Inference engines ──
        def open_engines_settings():
            page.locator("[aria-label=Settings]").filter(visible = True).first.click(force = True)
            dialog = page.get_by_role("dialog").first
            dialog.wait_for(state = "visible")
            dialog.get_by_role("button", name = "System", exact = True).first.click()
            # The vLLM row, not the section heading: on the AMD runner headings never render visible.
            row = dialog.locator('section[data-settings-label="Inference engines"] [data-settings-label="vLLM"]')
            row.wait_for(state = "attached", timeout = 60_000)
            # Off-screen settings sections are not rendered until scrolled to, so they read as hidden.
            row.evaluate("e => e.scrollIntoView({block: 'center'})")
            row.wait_for(state = "visible", timeout = 60_000)
            return dialog

        def engines_text(dialog):
            return dialog.locator('section[data-settings-label="Inference engines"]').inner_text()

        dialog = open_engines_settings()
        page.wait_for_timeout(1500)
        shoot("settings-engines")
        section = engines_text(dialog)
        result["settings_engines_text"] = section
        log("settings engines section", text = section[:1500])
        install_button = dialog.locator('[data-settings-label="vLLM"]').get_by_role(
            "button", name = re.compile(r"^(Install engine|Update engine|Repair)$")
        ).first
        result["vllm_install_button"] = {
            "present": install_button.count() > 0,
            "enabled": install_button.count() > 0 and install_button.is_enabled(),
            "label": install_button.inner_text() if install_button.count() else None,
        }
        log("vllm install button", **result["vllm_install_button"])
        save()

        if MODE == "full":
            status = next(e for e in api("/api/engines", tok) if e["engine"] == "vllm")
            if not (status["installed"] and status["current"]):
                # The first engine row's button: vLLM is listed before SGLang.
                install_button.click()
                confirm = dialog.get_by_role("region", name = "Install vLLM")
                if confirm.count() == 0:
                    confirm = dialog.locator("section").filter(has_text = "Install vLLM").first
                confirm.wait_for(state = "visible")
                result["install_prompt"] = confirm.inner_text()
                log("install prompt", text = result["install_prompt"])
                shoot("install-confirm")
                t0 = time.time()
                confirm.get_by_role("button", name = "Install engine").click()
                last_phase = None
                while time.time() - t0 < 3600:
                    status = next(e for e in api("/api/engines", tok) if e["engine"] == "vllm")
                    job = status["job"]
                    if job.get("phase") != last_phase:
                        last_phase = job.get("phase")
                        log("install phase", phase = last_phase, state = job.get("state"), message = job.get("message"))
                        shoot(f"install-{last_phase}")
                    if job.get("state") in ("success", "error", "cancelled"):
                        break
                    time.sleep(5)
                result["install_seconds"] = round(time.time() - t0)
                result["install_job"] = status["job"]
                log("install finished", seconds = result["install_seconds"], state = status["job"].get("state"),
                    message = status["job"].get("message"), log_tail = (status["job"].get("log") or [])[-8:])
                page.wait_for_timeout(3000)
                shoot("install-finished")
                result["settings_after_install_text"] = engines_text(dialog)
                save()
                if status["job"].get("state") != "success":
                    raise SystemExit("vLLM install failed")
            expect(dialog.get_by_text(re.compile(r"Installed 0\.30\.0")).first).to_be_visible(timeout = 60_000)
            shoot("settings-installed")

        page.keyboard.press("Escape")
        page.wait_for_timeout(500)

        # ── Model picker > run settings > Inference engine ──
        page.locator('[data-tour="chat-model-selector"]').first.click()
        popover = page.locator('[data-tour="chat-model-selector-popover"]').first
        popover.wait_for(state = "visible")
        search = popover.locator("[data-model-picker-search-input]").first
        search.click()
        search.fill(MODEL)
        page.wait_for_timeout(2500)
        shoot("picker-search")
        gear = popover.get_by_role("button", name = re.compile(r"^Inference settings for .*" + re.escape(MODEL.split("/")[-1]), re.I)).first
        try:
            gear.wait_for(state = "visible", timeout = 30_000)
        except Exception:
            rows = popover.locator("[data-model-picker-option]")
            result["picker_rows"] = [rows.nth(i).inner_text()[:200] for i in range(min(rows.count(), 10))]
            labels = popover.get_by_role("button")
            result["picker_buttons"] = [labels.nth(i).get_attribute("aria-label") for i in range(min(labels.count(), 40))]
            save()
            shoot("picker-no-gear")
            raise
        gear.click()
        popover.get_by_role("button", name = "Back to model list").wait_for(state = "visible")
        page.wait_for_timeout(1500)
        engine_select = popover.get_by_role("combobox", name = "Inference engine")
        result["engine_picker_present"] = engine_select.count() > 0
        log("engine picker present", present = result["engine_picker_present"])
        shoot("model-config")
        if not result["engine_picker_present"]:
            save()
            return
        engine_select.scroll_into_view_if_needed()
        engine_select.click()
        listbox = page.get_by_role("listbox").last
        listbox.wait_for(state = "visible")
        options = listbox.get_by_role("option")
        for _ in range(50):
            if options.count() > 1:
                break
            page.wait_for_timeout(200)
        result["engine_options"] = [
            {"text": options.nth(i).inner_text(), "disabled": options.nth(i).get_attribute("data-disabled") is not None}
            for i in range(options.count())
        ]
        log("engine options", options = result["engine_options"])
        shoot("engine-options")
        vllm = listbox.get_by_role("option", name = re.compile(r"^vLLM"))
        if MODE != "full" or result["engine_options"] and any(
            o["text"].startswith("vLLM") and o["disabled"] for o in result["engine_options"]
        ):
            page.keyboard.press("Escape")
            save()
            return
        vllm.click()
        page.wait_for_timeout(1000)
        precision = popover.get_by_role("combobox", name = "Precision")
        precision.scroll_into_view_if_needed()
        precision.click()
        plist = page.get_by_role("listbox").last
        plist.wait_for(state = "visible")
        popts = plist.get_by_role("option")
        result["precision_options"] = [
            {"text": popts.nth(i).inner_text(), "disabled": popts.nth(i).get_attribute("data-disabled") is not None}
            for i in range(popts.count())
        ]
        log("precision options", options = result["precision_options"])
        shoot("precision-options")
        page.keyboard.press("Escape")
        engine_panel_text = engine_select.locator("xpath=ancestor::div[contains(@class,'rounded-lg')][1]").inner_text()
        result["engine_panel_text"] = engine_panel_text
        log("engine panel", text = engine_panel_text)
        shoot("engine-panel")
        # Context length small enough for any card.
        ctx_input = popover.get_by_role("spinbutton", name = "Context Length")
        if ctx_input.count() == 0:
            ctx_input = popover.get_by_role("textbox", name = "Context Length")
        if ctx_input.count():
            ctx_input.first.fill("4096")
        load = popover.get_by_role("button", name = re.compile(r"^(Load model|Reload model)$")).first
        t0 = time.time()
        load.click()
        log("load clicked")
        loaded = False
        while time.time() - t0 < 1800:
            st = api("/api/inference/status", tok)
            if st.get("active_model") and not st.get("loading"):
                loaded = True
                break
            # A failed load leaves nothing loading and shows its error as a toast.
            failure = page.get_by_text(re.compile(r"Failed to load model", re.I))
            if time.time() - t0 > 20 and not st.get("loading") and failure.count():
                result["load_error"] = failure.first.inner_text()[:2000]
                break
            if int(time.time() - t0) % 30 < 5:
                shoot("loading")
            time.sleep(5)
        result["load_seconds"] = round(time.time() - t0)
        result["status_after_load"] = api("/api/inference/status", tok)
        log("load finished", loaded = loaded, seconds = result["load_seconds"],
            status = {k: result["status_after_load"].get(k) for k in ("active_model", "engine", "inference_engine", "loading", "error")})
        page.wait_for_timeout(2000)
        shoot("loaded")
        save()
        if not loaded:
            raise SystemExit("model did not load")

        composer = page.locator('textarea[aria-label="Message input"]')
        composer.click()
        composer.fill("What is the capital of France? Answer in one word.")
        composer.press("Enter")
        deadline = time.time() + 300
        reply = ""
        while time.time() < deadline:
            msgs = page.locator('[data-role="assistant"]')
            if msgs.count():
                reply = msgs.last.inner_text().strip()
                if "Paris" in reply:
                    break
            time.sleep(2)
        if not reply:
            reply = page.locator("main").inner_text()[-2000:]
        result["reply"] = reply[:1000]
        log("reply", text = reply[:300])
        shoot("chat-reply")
        result["chat_ok"] = "Paris" in reply
        save()


try:
    main()
    result["ok"] = True
except BaseException as exc:
    result["ok"] = False
    result["error"] = f"{type(exc).__name__}: {str(exc)[:2000]}"
    print("[ui] FAILED", result["error"], flush = True)

finally:
    save()
sys.exit(0 if result.get("ok") else 1)
