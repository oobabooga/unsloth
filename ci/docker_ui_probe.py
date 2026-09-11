"""Drive the Unsloth Docker image's two web UIs the way a user would, with screenshots.

Studio (port 8000): open the root URL, sign in (or do the first-boot password change),
pick a GGUF model through the model picker, send chat messages, read the reply.
JupyterLab (port 8888): sign in on the branded login page, check the notebook view,
open a new notebook, run a cell, read its output.

Env:
  STUDIO_URL        http://127.0.0.1:8000   (empty string skips Studio)
  JUPYTER_URL       http://127.0.0.1:8888   (empty string skips Jupyter)
  STUDIO_PW         the password to sign in with (current password)
  STUDIO_NEW_PW     the password to set if Studio asks for a change
  JUPYTER_PW        JupyterLab password
  GGUF_QUERY        text typed into the model picker search (default gemma-3-270m)
  GGUF_REPO         repo the picker should end up loading (default unsloth/gemma-3-270m-it-GGUF)
  EXPECT_GPU        "cuda" | "cpu" | "" : what the Jupyter cell should report
  OUT               screenshot + log dir (default ./ui_out)
  TURN_TIMEOUT_S    per chat turn (default 300)
  LOAD_TIMEOUT_S    model download + load (default 900)
"""

import json
import os
import re
import sys
import time
import traceback
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright, expect

STUDIO_URL = os.environ.get("STUDIO_URL", "http://127.0.0.1:8000").rstrip("/")
JUPYTER_URL = os.environ.get("JUPYTER_URL", "http://127.0.0.1:8888").rstrip("/")
STUDIO_PW = os.environ.get("STUDIO_PW", "")
STUDIO_NEW_PW = os.environ.get("STUDIO_NEW_PW", "")
JUPYTER_PW = os.environ.get("JUPYTER_PW", "")
GGUF_QUERY = os.environ.get("GGUF_QUERY", "gemma-3-270m")
GGUF_REPO = os.environ.get("GGUF_REPO", "unsloth/gemma-3-270m-it-GGUF")
EXPECT_GPU = os.environ.get("EXPECT_GPU", "")
OUT = Path(os.environ.get("OUT", "ui_out"))
TURN_TIMEOUT_S = float(os.environ.get("TURN_TIMEOUT_S", "300"))
LOAD_TIMEOUT_S = float(os.environ.get("LOAD_TIMEOUT_S", "900"))
OUT.mkdir(parents = True, exist_ok = True)
PW_CHANNEL = os.environ.get("PW_CHANNEL") or None

RESULTS = {}
_shot_n = [0]


def log(msg):
    print(f"[ui {time.strftime('%H:%M:%S')}] {msg}", flush = True)


def shoot(page, name):
    _shot_n[0] += 1
    path = OUT / f"{_shot_n[0]:02d}-{name}.png"
    try:
        page.screenshot(path = str(path), full_page = False, timeout = 60_000)
        log(f"screenshot {path}")
    except Exception as e:
        log(f"screenshot {name} failed: {e}")


def record(key, ok, detail = ""):
    RESULTS[key] = {"ok": bool(ok), "detail": str(detail)[:2000]}
    log(f"{'PASS' if ok else 'FAIL'} {key}: {detail}")


def studio_token(page):
    return page.evaluate("() => localStorage.getItem('unsloth_auth_token')")


def api(page, path, method = "GET", body = None):
    tok = studio_token(page)
    return page.evaluate(
        """async ([url, method, body, tok]) => {
            const r = await fetch(url, {method, headers: {'Authorization': 'Bearer ' + tok,
                'Content-Type': 'application/json'}, body: body ? JSON.stringify(body) : undefined});
            let j = null; try { j = await r.json(); } catch (e) {}
            return {status: r.status, body: j};
        }""",
        [STUDIO_URL + path, method, body, tok],
    )


def studio_sign_in(page):
    page.goto(STUDIO_URL, wait_until = "domcontentloaded", timeout = 120_000)
    page.wait_for_url(re.compile(r"/(login|change-password|chat)"), timeout = 60_000)
    try:
        page.wait_for_load_state("networkidle", timeout = 20_000)
    except Exception:
        pass
    shoot(page, "studio-landing")
    log(f"landing url: {page.url}")
    if "/login" in page.url:
        page.locator("#password").fill(STUDIO_PW)
        shoot(page, "studio-login-filled")
        page.locator('button[type="submit"]').click()
        page.wait_for_url(re.compile(r"/(chat|change-password|studio|$)"), timeout = 60_000)
        page.wait_for_timeout(1500)
        log(f"after login: {page.url}")
    if "/change-password" in page.url:
        cur = page.locator("#current-password")
        if cur.count():
            cur.fill(STUDIO_PW)
        page.locator("#new-password").fill(STUDIO_NEW_PW)
        page.locator("#confirm-password").fill(STUDIO_NEW_PW)
        shoot(page, "studio-change-password-filled")
        page.locator('button[type="submit"]').click()
        page.wait_for_url(lambda u: "/change-password" not in u, timeout = 60_000)
        record("studio_change_password", True, f"now at {page.url}")
    composer = page.locator('textarea[aria-label="Message input"]')
    composer.wait_for(state = "visible", timeout = 120_000)
    shoot(page, "studio-chat-shell")
    record("studio_sign_in", True, page.url)


def dismiss_overlays(page):
    for sel in ('[data-testid="llama-update-snooze-button"]', '[data-testid="web-update-snooze-button"]'):
        b = page.locator(sel)
        if b.count():
            try:
                b.first.click(timeout = 2000)
            except Exception:
                pass
    # guided tour / dialogs
    for name in ("Skip", "Skip tour", "Close", "Got it", "Dismiss", "Not now"):
        b = page.get_by_role("button", name = name, exact = True)
        if b.count():
            try:
                b.first.click(timeout = 1500)
                page.wait_for_timeout(300)
            except Exception:
                pass


def studio_pick_model(page):
    """Open the picker, search, click the repo, pick a quant if asked, wait for load."""
    dismiss_overlays(page)
    picker = page.locator('[data-tour="chat-model-selector"]').first
    picker.wait_for(state = "visible", timeout = 60_000)
    log(f"picker text before: {picker.inner_text().strip()!r}")
    picker.click()
    page.wait_for_timeout(800)
    shoot(page, "studio-picker-open")
    search = page.get_by_placeholder(re.compile(r"search", re.I)).first
    search.fill(GGUF_QUERY)
    page.wait_for_timeout(3000)
    shoot(page, "studio-picker-search")
    repo_short = GGUF_REPO.split("/")[-1]
    # Candidates: a row that names the repo
    row = page.locator(f'[role="dialog"] >> text=/{re.escape(repo_short)}/i').first
    if not row.count():
        row = page.get_by_text(re.compile(re.escape(repo_short), re.I)).first
    row.wait_for(state = "visible", timeout = 60_000)
    row.click()
    page.wait_for_timeout(2000)
    shoot(page, "studio-picker-after-repo-click")
    # A GGUF repo opens a quant list; prefer a small quant if a choice is shown.
    for q in ("Q4_K_M", "UD-Q4_K_XL", "Q4_0", "Q8_0"):
        qb = page.get_by_text(re.compile(rf"\b{re.escape(q)}\b")).first
        if qb.count() and qb.is_visible():
            log(f"choosing quant {q}")
            qb.click()
            break
    page.wait_for_timeout(1500)
    shoot(page, "studio-picker-after-quant")
    # Wait for the load to finish: /api/inference/status reports the active model.
    t0 = time.time()
    last = None
    while time.time() - t0 < LOAD_TIMEOUT_S:
        st = api(page, "/api/inference/status")
        body = st.get("body") or {}
        last = body
        am = body.get("active_model") or body.get("model") or ""
        if am and repo_short.lower().replace("-gguf", "") in str(am).lower():
            break
        if int(time.time() - t0) % 30 == 0:
            shoot(page, "studio-loading")
        page.wait_for_timeout(3000)
    else:
        shoot(page, "studio-load-timeout")
        raise AssertionError(f"model never became active; last status={json.dumps(last)[:800]}")
    record("studio_model_loaded_via_ui", True, f"{time.time() - t0:.0f}s status={json.dumps(last)[:400]}")
    page.wait_for_timeout(1500)
    shoot(page, "studio-model-loaded")
    return last


def studio_chat(page, prompt, idx):
    composer = page.locator('textarea[aria-label="Message input"]')
    before = page.locator('[data-role="assistant"]').count()
    dismiss_overlays(page)
    composer.click()
    composer.fill(prompt)
    page.locator('button[aria-label="Send message"]').click()
    page.wait_for_function(
        "(n) => document.querySelectorAll('[data-role=\"assistant\"]').length > n",
        arg = before, timeout = TURN_TIMEOUT_S * 1000)
    # streaming done when Send is back and Stop is gone
    try:
        page.wait_for_selector('button[aria-label="Stop generating"]', state = "attached", timeout = 5000)
    except Exception:
        pass
    page.wait_for_selector('button[aria-label="Stop generating"]', state = "detached",
                           timeout = TURN_TIMEOUT_S * 1000)
    page.wait_for_timeout(800)
    text = page.locator('[data-role="assistant"]').last.inner_text().strip()
    shoot(page, f"studio-chat-turn{idx}")
    return text


def run_studio(p):
    browser = p.chromium.launch(channel = PW_CHANNEL)
    ctx = browser.new_context(viewport = {"width": 1440, "height": 900}, locale = "en-US")
    page = ctx.new_page()
    page.set_default_timeout(60_000)
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    try:
        studio_sign_in(page)
        sysinfo = api(page, "/api/system")
        record("studio_api_system", sysinfo.get("status") == 200,
               json.dumps((sysinfo.get("body") or {}).get("gpu"))[:600])
        status = studio_pick_model(page)
        texts = []
        t0 = time.time()
        for i, prompt in enumerate(["What is the capital of France? Answer in one word.",
                                    "Now say hello in Spanish."], 1):
            texts.append(studio_chat(page, prompt, i))
            log(f"turn {i} reply: {texts[-1][:200]!r}")
        record("studio_chat_replies", all(t for t in texts),
               f"{time.time() - t0:.0f}s replies={[t[:120] for t in texts]}")
        # which backend device served it
        st = api(page, "/api/inference/status")
        record("studio_inference_status_after_chat", st.get("status") == 200,
               json.dumps(st.get("body"))[:800])
    except Exception as e:
        shoot(page, "studio-error")
        record("studio_flow", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}")
    finally:
        if errors:
            log(f"page errors: {errors[:5]}")
        browser.close()


def run_jupyter(p):
    browser = p.chromium.launch(channel = PW_CHANNEL)
    ctx = browser.new_context(viewport = {"width": 1440, "height": 900}, locale = "en-US")
    page = ctx.new_page()
    page.set_default_timeout(60_000)
    try:
        page.goto(JUPYTER_URL, wait_until = "domcontentloaded", timeout = 120_000)
        page.wait_for_timeout(1500)
        shoot(page, "jupyter-landing")
        if page.locator("#password_input").count():
            record("jupyter_branded_login", "unsloth" in page.content().lower(), page.url)
            page.locator("#password_input").fill(JUPYTER_PW)
            page.locator("#login_submit").click()
        page.wait_for_url(re.compile(r"/lab"), timeout = 60_000)
        page.locator(".jp-DirListing-content, .jp-DirListing").first.wait_for(timeout = 120_000)
        page.wait_for_timeout(4000)
        shoot(page, "jupyter-lab-home")
        record("jupyter_login", True, page.url)
        listing = page.locator(".jp-DirListing-content").first.inner_text()
        record("jupyter_notebook_view", len(listing.strip()) > 0, f"url={page.url} listing={listing[:300]!r}")
        # New notebook through the launcher/menu, the way a user does it.
        page.keyboard.press("Escape")
        page.locator('li.lm-MenuBar-item:has-text("File")').first.click()
        page.wait_for_timeout(300)
        page.locator('li.lm-Menu-item:has-text("New")').first.hover()
        page.wait_for_timeout(300)
        page.locator('li.lm-Menu-item[data-command="notebook:create-new"]').first.click()
        # JupyterLab may ask "Select Kernel" a moment after the notebook opens (slow links make
        # it late); accept it like a user would, for up to 60 s.
        t_d = time.time()
        while time.time() - t_d < 60:
            sel = page.locator('.jp-Dialog button.jp-mod-accept')
            if sel.count() and sel.first.is_visible():
                shoot(page, "jupyter-kernel-dialog")
                sel.first.click()
                log("accepted the Select Kernel dialog")
                break
            if page.locator(".jp-NotebookPanel:not(.lm-mod-hidden) .jp-Cell .cm-content").count() and time.time() - t_d > 8:
                break
            page.wait_for_timeout(1000)
        NB = ".jp-NotebookPanel:not(.lm-mod-hidden)"
        cell = page.locator(f"{NB} .jp-Cell .cm-content").first
        cell.wait_for(timeout = 120_000)
        # A user waits for the kernel indicator to settle before running anything.
        t_k = time.time()
        kstat = ""
        while time.time() - t_k < 240:
            kstat = page.locator(".jp-StatusBar-Component, .jp-StatusBar-TextItem").all_inner_texts()
            kstat = " | ".join(t.strip() for t in kstat if t.strip())
            dlg = page.locator('.jp-Dialog button.jp-mod-accept')
            if dlg.count() and dlg.first.is_visible():
                dlg.first.click()
                log("accepted a late Select Kernel dialog")
            if re.search(r"\bIdle\b", kstat):
                break
            page.wait_for_timeout(2000)
        log(f"kernel status after {time.time() - t_k:.0f}s: {kstat[:200]!r}")
        cell.click()
        code = ("import torch, platform, subprocess\n"
                "print('TORCH', torch.__version__, 'CUDA', torch.cuda.is_available(), "
                "torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])\n"
                "print('ARCH', platform.machine())")
        page.keyboard.type(code)
        shoot(page, "jupyter-cell-typed")
        page.keyboard.press("Shift+Enter")
        got = False
        for attempt in range(3):
            try:
                page.wait_for_function(
                    "() => [...document.querySelectorAll('.jp-NotebookPanel:not(.lm-mod-hidden) .jp-OutputArea-output')].some(e => /ARCH|Error/.test(e.innerText))",
                    timeout = 120_000)
                got = True
                break
            except Exception:
                kstat = " | ".join(t.strip() for t in page.locator(".jp-StatusBar-Component, .jp-StatusBar-TextItem").all_inner_texts() if t.strip())
                prompt = page.locator(f"{NB} .jp-InputPrompt").first.inner_text()
                log(f"no output after attempt {attempt + 1}; prompt={prompt!r} kernel={kstat[:200]!r}")
                shoot(page, f"jupyter-no-output-{attempt + 1}")
                RESULTS.setdefault("jupyter_retries", {"ok": True, "detail": ""})["detail"] += f"attempt{attempt + 1}: prompt={prompt!r} kernel={kstat[:120]!r}; "
                # re-run the first cell the way a user would: click it, Run menu
                page.locator(f"{NB} .jp-Cell").first.click()
                page.locator('li.lm-MenuBar-item:has-text("Run")').first.click()
                page.wait_for_timeout(300)
                page.locator('li.lm-Menu-item[data-command="notebook:run-cell"]').first.click()
        if not got:
            raise AssertionError("cell never produced output after 3 attempts")
        text = page.locator(f"{NB} .jp-OutputArea-output").first.inner_text()
        shoot(page, "jupyter-cell-output")
        ok = "TORCH" in text
        if EXPECT_GPU == "cuda":
            ok = ok and "CUDA True" in text
        elif EXPECT_GPU == "cpu":
            ok = ok and "CUDA False" in text
        record("jupyter_run_cell", ok, text[:400])
    except Exception as e:
        shoot(page, "jupyter-error")
        record("jupyter_flow", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}")
    finally:
        browser.close()


def main():
    with sync_playwright() as p:
        if STUDIO_URL:
            run_studio(p)
        if JUPYTER_URL:
            run_jupyter(p)
    (OUT / "results.json").write_text(json.dumps(RESULTS, indent = 2))
    bad = [k for k, v in RESULTS.items() if not v["ok"]]
    log(f"SUMMARY: {len(RESULTS) - len(bad)}/{len(RESULTS)} passed; failed={bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
