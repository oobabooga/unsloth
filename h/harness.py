"""Drive the Unsloth Desktop (debug build of main) like a user, in the interactive session.

Usage: python harness.py <phase>
Env: H_APP (exe), H_ART (artifact dir), H_PICK (folder to choose in the native dialog, or empty)
"""
import json
import os
import re
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path

ART = Path(os.environ["H_ART"])
ART.mkdir(parents=True, exist_ok=True)
APP = os.environ["H_APP"]
PHASE = sys.argv[1]
SHOTS = ART / "shots"
SHOTS.mkdir(exist_ok=True)
REPORT = ART / f"report-{PHASE}.json"
report = {"phase": PHASE, "events": []}
T0 = time.time()
CDP_PORT = 9333


def log(msg):
    line = f"[{time.time() - T0:7.1f}s] {msg}"
    print(line, flush=True)
    report["events"].append(line)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")


def rec(key, value):
    report[key] = value
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"REC {key} = {json.dumps(value)[:1500]}")


def desktop_shot(name):
    try:
        from PIL import ImageGrab

        ImageGrab.grab(all_screens=True).save(SHOTS / f"{PHASE}-{name}-desktop.png")
    except Exception as e:
        log(f"desktop shot failed: {e}")


def shot(page, name, full=False):
    try:
        page.screenshot(path=str(SHOTS / f"{PHASE}-{name}.png"), full_page=full, timeout=20000)
    except Exception as e:
        log(f"page shot {name} failed: {e}")
        desktop_shot(name)


def body_text(page, limit=1500):
    try:
        t = page.evaluate("() => document.body ? document.body.innerText : ''")
        return re.sub(r"\s+", " ", t)[:limit]
    except Exception as e:
        return f"<err {e}>"


def launch_app():
    env = dict(os.environ)
    env["WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS"] = f"--remote-debugging-port={CDP_PORT}"
    for k in list(env):
        if k.startswith("H_"):
            env.pop(k)
    out = open(ART / f"app-stdout-{PHASE}.log", "ab")
    p = subprocess.Popen([APP], env=env, stdout=out, stderr=subprocess.STDOUT, cwd=str(Path(APP).parent))
    log(f"launched app pid={p.pid}")
    return p


def wait_cdp(timeout=180):
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{CDP_PORT}/json/version", timeout=3) as r:
                return json.loads(r.read())
        except Exception:
            time.sleep(1)
    raise RuntimeError("CDP endpoint never came up")


def main_page(browser, timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        pages = [p for c in browser.contexts for p in c.pages]
        for p in pages:
            if "tauri.localhost" in p.url or p.url.startswith("tauri://"):
                return p
        if pages and time.time() > end - timeout / 2:
            log(f"no tauri page yet; pages: {[p.url for p in pages]}")
        time.sleep(1)
    pages = [p for c in browser.contexts for p in c.pages]
    if pages:
        return pages[0]
    raise RuntimeError("no page")


def kill_app(proc):
    subprocess.run(["taskkill", "/F", "/T", "/IM", Path(APP).name], capture_output=True)
    time.sleep(3)
    # The managed backend and llama-server can outlive the shell.
    subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'unsloth|llama-server' -and $_.Name -match 'python|llama|unsloth' } | ForEach-Object { Write-Output ('killing ' + $_.ProcessId + ' ' + $_.Name); Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"],
        capture_output=True,
    )
    log("app and backend killed")


def visible(loc):
    try:
        return loc.count() > 0 and loc.first.is_visible()
    except Exception:
        return False


def wait_ready(page, timeout_s, label):
    """Wait until the main Studio shell shows (chat composer or sidebar); snapshot progress."""
    end = time.time() + timeout_s
    last = ""
    n = 0
    while time.time() < end:
        if visible(page.locator('textarea[aria-label="Message input"]')) or visible(
            page.locator('[data-sidebar="menu-button"]')
        ):
            log(f"{label}: Studio shell is up")
            return True
        txt = body_text(page, 600)
        if txt != last:
            log(f"{label} screen: {txt}")
            last = txt
        if n % 6 == 0:
            shot(page, f"{label}-progress-{n:03d}")
        n += 1
        if re.search(r"(Installation failed|Repair failed|Update failed|could not|Something went wrong)", txt, re.I):
            shot(page, f"{label}-failure")
            log(f"{label}: failure text on screen")
        time.sleep(10)
    shot(page, f"{label}-timeout")
    return False


def dismiss_overlays(page):
    for _ in range(3):
        try:
            page.keyboard.press("Escape")
            time.sleep(0.3)
        except Exception:
            pass


def pick_folder_native(folder):
    """Answer the native 'Choose where to install Unsloth' folder dialog with pywinauto."""
    from pywinauto import Desktop
    from pywinauto.keyboard import send_keys

    d = Desktop(backend="uia")
    end = time.time() + 30
    dlg = None
    while time.time() < end:
        try:
            dlg = d.window(title_re=".*Choose where to install Unsloth.*")
            if dlg.exists(timeout=1):
                break
        except Exception:
            pass
        time.sleep(1)
    if dlg is None or not dlg.exists():
        raise RuntimeError("folder dialog never appeared")
    desktop_shot("folder-dialog")
    dlg.set_focus()
    # The 'Folder:' box takes a full path; Select Folder then commits it.
    try:
        edit = dlg.child_window(auto_id="1152", control_type="Edit")
        edit.set_edit_text(folder)
    except Exception as e:
        log(f"Folder edit by id failed ({e}); typing into the focused box")
        send_keys("%d")  # address bar
        time.sleep(0.5)
        send_keys(folder.replace(" ", "{SPACE}") + "{ENTER}")
        time.sleep(1)
    desktop_shot("folder-dialog-filled")
    try:
        dlg.child_window(title="Select Folder", control_type="Button").click_input()
    except Exception as e:
        log(f"Select Folder click failed ({e}); pressing Enter")
        send_keys("{ENTER}")
    time.sleep(2)


def token_for(root):
    secret_file = Path(root) / "auth" / ".desktop_secret"
    return secret_file.read_text().strip() if secret_file.exists() else None


def api(port, path, token=None, body=None, timeout=60, method=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method or ("POST" if body is not None else "GET"))
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    data = json.dumps(body).encode() if body is not None else None
    try:
        with urllib.request.urlopen(req, data=data, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, e.read()[:2000].decode("utf-8", "replace")
    except Exception as e:
        return None, str(e)


def backend_port():
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-NetTCPConnection -State Listen | ForEach-Object { $p = Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue; '{0} {1} {2}' -f $_.LocalPort, $p.Name, $_.OwningProcess }"],
        capture_output=True, text=True,
    ).stdout
    cands = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1].lower().startswith("python"):
            cands.append(int(parts[0]))
    for port in sorted(set(cands)):
        st, body = api(port, "/api/health", timeout=5)
        if st == 200:
            return port, body
    return None, None


def api_login(port, root):
    secret = token_for(root)
    if not secret:
        return None
    st, body = api(port, "/api/auth/desktop-login", body={"secret": secret})
    if st == 200 and isinstance(body, dict):
        return body.get("access_token")
    log(f"desktop-login -> {st} {str(body)[:300]}")
    return None


def sidebar_inventory(page):
    items = page.evaluate(
        """() => Array.from(document.querySelectorAll('[data-sidebar="menu-button"]')).map(b => ({
            text: (b.innerText || '').trim().replace(/\\s+/g, ' ').slice(0, 60),
            aria: b.getAttribute('aria-label'),
            disabled: b.getAttribute('aria-disabled') || b.getAttribute('data-disabled') || (b.disabled ? 'true' : null),
            title: b.getAttribute('title'),
            visible: !!(b.offsetWidth || b.offsetHeight),
        }))"""
    )
    return items


def nav_click(page, label):
    btn = page.locator('[data-sidebar="menu-button"]').filter(has_text=re.compile(rf"^\s*{label}\b", re.I)).first
    if btn.count() == 0:
        btn = page.get_by_role("link", name=re.compile(rf"^{label}", re.I)).first
    if btn.count() == 0:
        return False
    btn.click(timeout=10000)
    time.sleep(3)
    return True


def hover_hint(page, label):
    btn = page.locator('[data-sidebar="menu-button"]').filter(has_text=re.compile(rf"^\s*{label}\b", re.I)).first
    if btn.count() == 0:
        return None
    btn.hover(timeout=10000)
    time.sleep(2)
    tips = page.locator('[role="tooltip"], [data-radix-popper-content-wrapper]')
    texts = []
    for i in range(tips.count()):
        try:
            t = tips.nth(i).inner_text().strip()
            if t:
                texts.append(t)
        except Exception:
            pass
    shot(page, f"hover-{label.lower()}")
    page.mouse.move(5, 5)
    return texts


def open_settings(page):
    attempts = [
        lambda: page.keyboard.press("Control+Comma"),
        lambda: page.get_by_role("button", name=re.compile(r"^Settings", re.I)).first.click(timeout=5000),
        lambda: page.locator('[aria-label*="Settings" i]').first.click(timeout=5000),
    ]
    for i, a in enumerate(attempts):
        try:
            a()
            time.sleep(2)
            if visible(page.get_by_role("dialog")):
                log(f"settings opened via attempt {i}")
                return True
        except Exception as e:
            log(f"settings attempt {i} failed: {e}")
    # Profile menu route
    try:
        page.locator('[data-sidebar="footer"] button').last.click(timeout=5000)
        time.sleep(1)
        page.get_by_role("menuitem", name=re.compile("Settings", re.I)).first.click(timeout=5000)
        time.sleep(2)
        return visible(page.get_by_role("dialog"))
    except Exception as e:
        log(f"profile-menu settings failed: {e}")
    return False


def find_repair_row(page):
    """Open Settings and locate the Repair installation row. Returns the dialog locator or None."""
    if not open_settings(page):
        shot(page, "settings-not-opened")
        return None
    dlg = page.get_by_role("dialog").last
    shot(page, "settings-open")
    # Try the search box first, then walk the tabs.
    try:
        search = dlg.locator('input[type="search"], input[placeholder*="Search" i]').first
        if search.count():
            search.fill("Repair")
            time.sleep(1.5)
            shot(page, "settings-search-repair")
    except Exception as e:
        log(f"settings search failed: {e}")
    if dlg.get_by_text("Repair installation").count() == 0:
        tabs = dlg.get_by_role("tab")
        n = tabs.count()
        names = []
        for i in range(n):
            try:
                names.append(tabs.nth(i).inner_text().strip())
            except Exception:
                pass
        rec("settings_tabs", names)
        for i in range(n):
            try:
                tabs.nth(i).click(timeout=3000)
                time.sleep(1)
                if dlg.get_by_text("Repair installation").count():
                    break
            except Exception:
                pass
    row = dlg.get_by_text("Repair installation").first
    if row.count() == 0:
        shot(page, "settings-no-repair")
        return None
    row.scroll_into_view_if_needed()
    shot(page, "settings-repair-row")
    return dlg


def check_all_pages(page):
    inv = sidebar_inventory(page)
    rec("sidebar", inv)
    hints = {}
    for label in ["Train", "Video", "Images", "Audio", "Export", "Recipes", "Data"]:
        try:
            hints[label] = hover_hint(page, label)
        except Exception as e:
            hints[label] = f"err {e}"
    rec("hover_hints", hints)
    pages = {}
    for label in ["Hub", "Images", "Train", "Video", "Audio", "Recipes", "Data", "Export", "API", "Library", "Projects"]:
        try:
            ok = nav_click(page, label)
            if not ok:
                pages[label] = "no nav item"
                continue
            time.sleep(2)
            pages[label] = {"url": page.url, "text": body_text(page, 900)}
            shot(page, f"page-{label.lower()}")
            dismiss_overlays(page)
        except Exception as e:
            pages[label] = f"err {e}"
    rec("pages", pages)


def load_model_via_ui(page, port, token):
    """Pick a small GGUF in the chat model picker like a user would."""
    nav_click(page, "Chat") or nav_click(page, "New chat")
    time.sleep(2)
    trig = page.locator('[data-tour="chat-model-selector"]').first
    if trig.count() == 0:
        log("no model selector trigger")
        shot(page, "no-model-trigger")
        return False
    trig.click()
    pop = page.locator('[data-tour="chat-model-selector-popover"]').first
    pop.wait_for(state="visible", timeout=30000)
    shot(page, "picker-open")
    rec("picker_text_initial", body_text(page, 2500))
    tabs = pop.get_by_role("tab")
    rec("picker_tabs", [tabs.nth(i).inner_text().strip() for i in range(tabs.count())])
    search = pop.locator("[data-model-picker-search-input]").first
    if search.count() == 0:
        search = pop.locator("input").first
    search.click()
    search.fill("gemma-3-270m-it")
    time.sleep(6)
    shot(page, "picker-search")
    opts = pop.locator("[data-model-picker-option]")
    rows = [opts.nth(i).inner_text().strip().replace("\n", " ")[:120] for i in range(min(opts.count(), 15))]
    rec("picker_search_rows", rows)
    target = None
    for i in range(opts.count()):
        t = opts.nth(i).inner_text()
        if re.search(r"gemma-3-270m-it-GGUF", t, re.I):
            target = opts.nth(i)
            break
    if target is None and opts.count():
        target = opts.first
    if target is None:
        return False
    target.click()
    time.sleep(4)
    shot(page, "picker-after-click")
    # A quant list may open; prefer Q8_0 then Q4_K_M.
    for q in ["Q8_0", "Q4_K_M", "Q4_0"]:
        cand = page.locator("[data-model-picker-option]").filter(has_text=re.compile(rf"\b{q}\b"))
        if visible(cand):
            cand.first.click()
            log(f"picked quant {q}")
            time.sleep(3)
            break
    shot(page, "picker-after-quant")
    # Accept any download / load confirmation.
    for name in ["Download", "Download and load", "Load", "Continue"]:
        b = page.get_by_role("button", name=re.compile(rf"^{name}$", re.I))
        if visible(b):
            b.first.click()
            log(f"clicked {name}")
            time.sleep(2)
    end = time.time() + 900
    while time.time() < end:
        st, body = api(port, "/api/inference/status", token)
        if st == 200 and isinstance(body, dict) and body.get("active_model"):
            rec("ui_loaded_model", {k: body.get(k) for k in ("active_model", "backend", "gguf_variant", "is_gguf") if k in body})
            shot(page, "model-loaded")
            return True
        time.sleep(10)
    shot(page, "model-load-timeout")
    return False


def chat_via_ui(page):
    comp = page.locator('textarea[aria-label="Message input"]').first
    comp.wait_for(state="visible", timeout=60000)
    comp.click()
    comp.fill("Reply with exactly three words: hello from Windows")
    page.keyboard.press("Enter")
    end = time.time() + 240
    while time.time() < end:
        txt = body_text(page, 6000)
        msgs = page.locator('[data-message-role="assistant"], [data-role="assistant"]')
        if msgs.count():
            t = msgs.last.inner_text().strip()
            if t and len(t) > 3 and not re.search(r"Thinking|Generating", t):
                time.sleep(4)
                t = msgs.last.inner_text().strip()
                rec("ui_chat_reply", t[:500])
                shot(page, "chat-reply")
                return True
        time.sleep(3)
    rec("ui_chat_page_text", body_text(page, 3000))
    shot(page, "chat-timeout")
    return False


def api_server_checks(port, token):
    st, models = api(port, "/v1/models", token)
    rec("v1_models", {"status": st, "body": models if st == 200 else str(models)[:400]})
    mid = None
    if st == 200 and isinstance(models, dict) and models.get("data"):
        mid = models["data"][0]["id"]
    st, resp = api(
        port,
        "/v1/chat/completions",
        token,
        body={"model": mid or "default", "messages": [{"role": "user", "content": "What is 2+2? Answer with a number."}], "max_tokens": 32, "stream": False},
        timeout=180,
    )
    content = None
    if st == 200 and isinstance(resp, dict):
        try:
            content = resp["choices"][0]["message"]["content"]
        except Exception:
            pass
    rec("v1_chat", {"status": st, "content": content, "raw": None if content else str(resp)[:600]})


def health_summary(port):
    st, h = api(port, "/api/health")
    keep = {}
    if isinstance(h, dict):
        for k in ("chat_only", "chat_only_reason", "chat_only_detail", "chat_only_hint", "device_type", "version", "video_cpu_opt_in"):
            if k in h:
                keep[k] = h[k]
    rec("health", {"status": st, **keep})
    return h


def phase_firstrun(pw, root_hint):
    proc = launch_app()
    try:
        wait_cdp()
        browser = pw.chromium.connect_over_cdp(f"http://127.0.0.1:{CDP_PORT}")
        page = main_page(browser)
        page.set_default_timeout(20000)
        log(f"page url {page.url}")
        page.get_by_role("button", name="Get Started").wait_for(state="visible", timeout=120000)
        time.sleep(2)
        shot(page, "01-first-run")
        desktop_shot("01-first-run")
        rec("first_run_text", body_text(page))
        cb = page.locator("#install-chat-only")
        rec("checkbox_present", cb.count() > 0)
        if cb.count():
            rec("checkbox_default_state", cb.get_attribute("data-state") or cb.get_attribute("aria-checked"))
        # Order on screen: is the checkbox under the button?
        try:
            gb = page.get_by_role("button", name="Get Started").bounding_box()
            cbb = cb.bounding_box()
            rec("layout", {"get_started": gb, "checkbox": cbb})
        except Exception as e:
            log(f"layout err {e}")
        # Custom install location through the real dialog.
        pick = os.environ.get("H_PICK", "")
        if pick:
            change = page.get_by_role("button", name="Change")
            rec("change_location_button", change.count() > 0)
            if change.count():
                change.first.click()
                try:
                    pick_folder_native(pick)
                except Exception as e:
                    log(f"native dialog failed: {e}")
                    desktop_shot("folder-dialog-failed")
                time.sleep(2)
                rec("location_text_after_pick", body_text(page))
                shot(page, "02-location-picked")
        # Tick Chat only by clicking its label, like a user.
        page.get_by_text("Chat only (GGUF): smaller install, no training").click()
        time.sleep(0.5)
        rec("checkbox_after_click", cb.get_attribute("data-state") or cb.get_attribute("aria-checked"))
        shot(page, "03-chat-only-ticked")
        page.get_by_role("button", name="Get Started").click()
        time.sleep(3)
        txt = body_text(page)
        if "Windows will ask for permission" in txt:
            shot(page, "04-long-paths-explainer")
            rec("long_paths_explainer", txt)
            skip = page.get_by_role("button", name=re.compile("skip|not now|later", re.I))
            if skip.count():
                skip.first.click()
            else:
                page.get_by_role("button").last.click()
        t_install = time.time()
        ok = wait_ready(page, 60 * 45, "install")
        rec("install_seconds", round(time.time() - t_install))
        rec("install_ready", ok)
        shot(page, "05-after-install")
        if not ok:
            return
        time.sleep(10)
        dismiss_overlays(page)
        shot(page, "06-home")
        port, h = backend_port()
        rec("backend_port", port)
        if port:
            health_summary(port)
        check_all_pages(page)
        token = api_login(port, root_hint) if port else None
        rec("have_token", bool(token))
        # Settings / repair discoverability
        dlg = find_repair_row(page)
        rec("repair_row_found", dlg is not None)
        if dlg is not None:
            rec("repair_row_text", re.sub(r"\s+", " ", dlg.inner_text())[:1500])
        dismiss_overlays(page)
        # Model load + chat
        if port and token:
            ok = False
            try:
                ok = load_model_via_ui(page, port, token)
            except Exception as e:
                log(f"UI load failed: {e}")
                shot(page, "ui-load-exception")
            rec("ui_model_load_ok", ok)
            if not ok:
                st, body = api(port, "/api/inference/load", token, body={"model_path": "unsloth/gemma-3-270m-it-GGUF", "gguf_variant": "Q8_0"}, timeout=900)
                rec("api_model_load", {"status": st, "body": str(body)[:500]})
                page.reload()
                time.sleep(8)
            dismiss_overlays(page)
            try:
                rec("ui_chat_ok", chat_via_ui(page))
            except Exception as e:
                log(f"UI chat failed: {e}")
                shot(page, "ui-chat-exception")
            api_server_checks(port, token)
    finally:
        kill_app(proc)


def phase_relaunch(pw, root_hint, label):
    """Start the app on an existing install; report what the user sees."""
    proc = launch_app()
    try:
        wait_cdp()
        browser = pw.chromium.connect_over_cdp(f"http://127.0.0.1:{CDP_PORT}")
        page = main_page(browser)
        page.set_default_timeout(20000)
        t = time.time()
        ok = wait_ready(page, 60 * 45, label)
        rec("ready", ok)
        rec("ready_seconds", round(time.time() - t))
        if not ok:
            return None, None, None
        time.sleep(8)
        dismiss_overlays(page)
        shot(page, "home")
        port, _ = backend_port()
        rec("backend_port", port)
        if port:
            health_summary(port)
        rec("sidebar", sidebar_inventory(page))
        rec("train_hint", hover_hint(page, "Train"))
        return proc, page, port
    except Exception:
        kill_app(proc)
        raise


def phase_update(pw, root_hint):
    """The Desktop update banner's first step: invoke start_backend_update from the page."""
    proc, page, port = phase_relaunch(pw, root_hint, "pre-update")
    try:
        if page is None:
            return
        res = page.evaluate(
            """() => new Promise(async (resolve) => {
                const T = window.__TAURI__;
                const logs = [];
                const un1 = await T.event.listen('update-progress', e => logs.push(String(e.payload)));
                const un2 = await T.event.listen('update-complete', () => resolve({result: 'complete', logs}));
                const un3 = await T.event.listen('update-failed', e => resolve({result: 'failed: ' + e.payload, logs}));
                T.core.invoke('start_backend_update').catch(e => resolve({result: 'invoke error: ' + e, logs}));
            })"""
        )
        rec("backend_update_result", res.get("result"))
        (ART / "update-progress.log").write_text("\n".join(res.get("logs", [])), encoding="utf-8")
        log(f"update log tail: {res.get('logs', [])[-15:]}")
        shot(page, "after-backend-update")
    finally:
        kill_app(proc)
    # The shell relaunches after an update; do the same.
    proc, page, port = phase_relaunch(pw, root_hint, "post-update")
    if proc:
        kill_app(proc)


def phase_autorepair(pw, root_hint):
    proc, page, port = phase_relaunch(pw, root_hint, "autorepair")
    if proc:
        kill_app(proc)


def phase_upgrade(pw, root_hint):
    proc, page, port = phase_relaunch(pw, root_hint, "pre-upgrade")
    try:
        if page is None:
            return
        dlg = find_repair_row(page)
        if dlg is None:
            rec("upgrade_started", False)
            return
        btn = dlg.get_by_role("button", name=re.compile(r"^Repair install$", re.I)).first
        btn.scroll_into_view_if_needed()
        btn.click()
        time.sleep(2)
        confirm = page.get_by_role("alertdialog")
        rec("repair_confirm_text", re.sub(r"\s+", " ", confirm.inner_text())[:800] if confirm.count() else None)
        shot(page, "repair-confirm")
        page.get_by_role("button", name=re.compile(r"^Repair now$", re.I)).first.click()
        rec("upgrade_started", True)
        t = time.time()
        time.sleep(20)
        ok = wait_ready_after_repair(page, 60 * 60)
        rec("upgrade_seconds", round(time.time() - t))
        rec("upgrade_ready", ok)
        shot(page, "after-repair")
    finally:
        kill_app(proc)
    proc, page, port = phase_relaunch(pw, root_hint, "post-upgrade")
    try:
        if page is not None:
            nav_click(page, "Train")
            time.sleep(4)
            rec("train_page_text", body_text(page, 1200))
            shot(page, "train-page-after-upgrade")
    finally:
        if proc:
            kill_app(proc)


def wait_ready_after_repair(page, timeout_s):
    end = time.time() + timeout_s
    last = ""
    n = 0
    while time.time() < end:
        txt = body_text(page, 600)
        if txt != last:
            log(f"repair screen: {txt}")
            last = txt
        if n % 6 == 0:
            shot(page, f"repair-progress-{n:03d}")
        n += 1
        port, h = backend_port()
        if port and n > 3 and visible(page.locator('[data-sidebar="menu-button"]')) and not re.search(r"Repairing|Installing|repair", txt, re.I):
            return True
        time.sleep(15)
    return False


def main():
    from playwright.sync_api import sync_playwright

    root = os.environ.get("H_ROOT", "")
    try:
        with sync_playwright() as pw:
            if PHASE == "firstrun":
                phase_firstrun(pw, root)
            elif PHASE == "update":
                phase_update(pw, root)
            elif PHASE == "autorepair":
                phase_autorepair(pw, root)
            elif PHASE == "upgrade":
                phase_upgrade(pw, root)
            elif PHASE == "relaunch":
                p, _, _ = phase_relaunch(pw, root, "relaunch")
                if p:
                    kill_app(p)
    except Exception:
        log("EXCEPTION " + traceback.format_exc())
        desktop_shot("exception")
        sys.exit(1)


if __name__ == "__main__":
    main()
