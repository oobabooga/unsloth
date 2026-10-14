"""A first-time user on a Ryzen AI machine: start Studio, find the NPU, download, load, chat,
change settings, reload, use the Hub, restart Studio. Then the same with a GGUF for comparison."""
import base64, json, os, re, shutil, subprocess, sys, time

sys.path.insert(0, os.path.dirname(__file__))
import uxlib as U
from uxlib import step, note, shot, api, poll
from playwright.sync_api import sync_playwright

PASSWORD = "Ux-user-pass-2026"
(U.ROOT / "password.txt").write_text(PASSWORD)
SMALL = os.environ.get("UX_SMALL", "qwen3:0.6b")      # checkpoint of the first model a user tries
VISION = os.environ.get("UX_VISION", "qwen3.5:2b")    # vision + tools model, used through the Hub
GGUF = os.environ.get("UX_GGUF", "unsloth/Qwen3-0.6B-GGUF")
studio = {"proc": None, "n": 0}


def unsloth_exe():
    exe = shutil.which("unsloth")
    if not exe:
        raise RuntimeError("`unsloth` is not on PATH after the install")
    return exe


def start_studio(tag):
    studio["n"] += 1
    log = open(U.OUT / f"studio-{studio['n']}-{tag}.log", "w", encoding="utf-8")
    cmd = [unsloth_exe(), "studio", "-H", "127.0.0.1", "-p", str(U.PORT)]
    t = time.monotonic()
    kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if U.WIN else {"start_new_session": True}
    studio["proc"] = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, **kw)
    poll(lambda: U.requests.get(U.BASE + "/api/health", timeout=5).status_code == 200, 600)
    log.flush()
    return {"cmd": " ".join(cmd), "seconds_to_health": round(time.monotonic() - t, 1),
            "banner": (U.OUT / f"studio-{studio['n']}-{tag}.log").read_text(encoding="utf-8", errors="replace")[-3000:]}


def stop_studio():
    p = studio["proc"]
    if not p:
        return
    if U.WIN:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True)
    else:
        # Ctrl+C in the user's terminal signals the whole foreground group.
        print(f"sending SIGINT to studio group {p.pid} at {time.strftime('%H:%M:%S')}", flush=True)
        os.killpg(p.pid, 2)
        try:
            p.wait(60)
        except subprocess.TimeoutExpired:
            note("`unsloth studio` did not exit within 60 s of Ctrl+C.")
            os.killpg(p.pid, 9)
    # Lemonade/FastFlowLM children must not outlive Studio.
    time.sleep(5)
    left = U.run("tasklist" if U.WIN else "ps -eo pid,ppid,etime,cmd", name=f"procs-after-stop-{studio['n']}")[1]
    # /usr/bin/lemond is the runner image's own system service, not Studio's.
    stray = [l for l in left.splitlines() if re.search(r"lemond|lemonade|flm(\.exe)?\b|llama-server", l, re.I)
             and "/usr/bin/lemond" not in l]
    if stray:
        note(f"After stopping Studio #{studio['n']} these runtime processes were still alive: {stray[:6]}")
    studio["proc"] = None
    return {"stray_runtime_processes": stray}


def npu_status():
    return api("GET", "/api/npu/status")["body"]


def inference_status():
    return api("GET", "/api/inference/status")["body"]


def body_text():
    return U.page.locator("body").inner_text(timeout=10000)


def wait_text(pattern, timeout=60):
    rx = re.compile(pattern, re.I)
    return poll(lambda: (m := rx.search(body_text())) and m.group(0), timeout, 1)


def picker_trigger():
    page = U.page
    for name in ("Select model",):
        loc = page.get_by_role("button", name=name, exact=True)
        if loc.count():
            return loc.first
    # Once a model is chosen the trigger shows its name; it is the first popup button in the chat header.
    loc = page.locator("main button[aria-haspopup]").first
    return loc


def open_picker(tab="Recommended"):
    page = U.page
    if page.locator(".unsloth-model-selector-menu").count() and page.locator(".unsloth-model-selector-menu").first.is_visible():
        pass
    else:
        picker_trigger().click()
    page.locator(".unsloth-model-selector-menu").first.wait_for(timeout=20000)
    if tab:
        t = page.get_by_role("tab", name=tab, exact=True)
        if t.count():
            t.first.click()
    time.sleep(1.5)


def close_picker():
    U.page.keyboard.press("Escape")
    time.sleep(0.8)


def picker_row(model_id):
    menu = U.page.locator(".unsloth-model-selector-menu").first
    opt = menu.get_by_role("option").filter(has_text=re.compile(r"^\s*" + re.escape(model_id)))
    if opt.count():
        return opt.first
    return menu.get_by_text(model_id, exact=True).first


def send_chat(text, timeout=600):
    page = U.page
    box = page.locator("textarea[aria-label='Message input']").first
    box.click()
    box.fill(text)
    msgs = page.locator(".aui-assistant-message-root")
    before = msgs.count()
    t = time.monotonic()
    page.keyboard.press("Enter")
    # A new assistant turn appears, then the stop button goes away when it is done.
    poll(lambda: msgs.count() > before or page.locator(".aui-composer-cancel").count() > 0, 120, 0.5)
    poll(lambda: page.locator(".aui-composer-cancel").count() == 0 and msgs.count() > before, timeout, 1)
    time.sleep(1.5)
    secs = round(time.monotonic() - t, 1)
    reply = msgs.last.inner_text() if msgs.count() else body_text()[-1500:]
    return {"prompt": text, "seconds": secs, "reply": reply[-2000:]}


def open_run_settings():
    btn = U.page.get_by_role("button", name="Open run settings", exact=True)
    if btn.count() and btn.first.is_visible():
        btn.first.click()
        time.sleep(1.5)


def set_number(box, value):
    box.scroll_into_view_if_needed()
    box.click()
    box.press("Control+a")
    box.type(value, delay=60)
    box.press("Tab")
    time.sleep(0.8)


def goto_hub():
    page = U.page
    nav = page.get_by_test_id("nav-row-hub")
    if nav.count() and nav.first.is_visible():
        nav.first.click()
    else:
        page.goto(U.BASE + "/hub", wait_until="networkidle")
    page.wait_for_url(re.compile(r"/hub"), timeout=30000)
    time.sleep(3)


def catalog():
    return api("GET", "/api/npu/models")["body"]["models"]


def model_for(checkpoint):
    for m in catalog():
        if m["checkpoint"].lower() == checkpoint.lower():
            return m
    raise RuntimeError(f"{checkpoint} not in the NPU catalog")


def ensure_login():
    page = U.page
    if "/change-password" in page.url or page.get_by_text("Setup your account").count():
        note("Fresh install on localhost: Studio asks the user to create a password before anything else.")
        page.get_by_label("New password", exact=True).fill(PASSWORD)
        page.get_by_label("Confirm password", exact=True).fill(PASSWORD)
        page.get_by_role("button", name="Change password", exact=True).click()
    elif "/login" in page.url:
        page.get_by_label("Password", exact=True).fill(PASSWORD)
        page.get_by_role("button", name="Login", exact=True).click()
    page.wait_for_url(re.compile(r"/chat"), timeout=60000)
    token = page.evaluate("localStorage.getItem('unsloth_auth_token')")
    U.session.headers["Authorization"] = "Bearer " + token
    (U.ROOT / "token.txt").write_text(token)
    return page.url


def main():
    step("start_studio_first_time", lambda: start_studio("first"), shoot=False)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=["--no-sandbox"])
        ctx = browser.new_context(locale="en-US", viewport={"width": 1440, "height": 1000})
        if not U.WIN:
            ctx.add_init_script("document.addEventListener('DOMContentLoaded',()=>{const s=document.createElement('style');s.textContent='*{font-family:\"DejaVu Sans\",sans-serif!important}';document.head.appendChild(s)})")
        ctx.tracing.start(screenshots=True, snapshots=True)
        page = ctx.new_page()
        page.set_default_timeout(30000)
        U.page = page
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append("console." + m.type + ": " + m.text) if m.type == "error" else None)
        net = []
        page.on("response", lambda r: net.append({"t": round(time.time(), 1), "url": r.url.replace(U.BASE, ""), "status": r.status})
                if ("/api/npu" in r.url or "/api/inference" in r.url or "/v1/" in r.url) else None)
        try:
            run_ui(page)
        finally:
            (U.OUT / "browser-errors.json").write_text(json.dumps(errors, indent=1))
            (U.OUT / "network.json").write_text(json.dumps(net, indent=1))
            try:
                ctx.tracing.stop(path=str(U.OUT / "trace.zip"))
            except Exception:
                pass
            browser.close()
            stop_studio()
    fails = [r["test"] for r in U.RESULTS if not r["ok"]]
    print("FAILED STEPS:", fails)
    print("NOTES:", json.dumps(U.NOTES, indent=1))


def run_ui(page):
    step("open_studio", lambda: (page.goto(U.BASE + "/", wait_until="networkidle"), page.url)[1])
    step("first_login", ensure_login)
    st0 = step("npu_status_initial", npu_status, shoot=False) or {}
    if not st0.get("supported"):
        note(f"Studio does not consider this NPU supported: {json.dumps(st0)[:800]}")

    # --- Discovery: what does a user see in the chat picker before doing anything? ---
    def picker_first_look():
        open_picker("Recommended")
        txt = U.page.locator(".unsloth-model-selector-menu").first.inner_text()
        found = {k: (k in txt) for k in ("NPU", "Enable NPU", "Start NPU runtime", "Powered by", "FastFlowLM")}
        if not found["NPU"]:
            note("Chat model picker (Recommended tab) shows no NPU section on a supported Ryzen AI machine.")
        return {"found": found, "picker_text_head": txt[:1500]}
    step("chat_picker_first_look", picker_first_look)

    # --- Enable NPU from where the user sees it ---
    def enable():
        menu = U.page.locator(".unsloth-model-selector-menu").first
        btn = menu.get_by_role("button", name=re.compile(r"^(Enable NPU|Start NPU runtime|Try again)$"))
        if not btn.count():
            # The NPU group starts folded: only an "NPU >" heading. A user has to think to open it.
            note("Chat picker on a fresh Ryzen AI install: the NPU group is folded to a bare 'NPU' heading; Enable NPU is hidden until the user expands it.")
            heading = menu.get_by_role("button", name=re.compile(r"^NPU$", re.I))
            (heading.first if heading.count() else menu.get_by_text("NPU", exact=True).first).click()
            time.sleep(1.5)
            shot("npu-group-expanded")
        btn.first.scroll_into_view_if_needed()
        label = btn.first.inner_text()
        t = time.monotonic()
        btn.first.click()
        seen = []
        def ready():
            s = npu_status()
            if not seen or seen[-1] != s["state"]:
                seen.append(s["state"])
            return s if s["state"] in ("ready", "failed") else None
        s = poll(ready, 900, 1)
        secs = round(time.monotonic() - t, 1)
        shot("enable-finished")
        if not s["ready"]:
            raise RuntimeError(f"enable failed: {json.dumps(s)[:2000]}")
        # The picker should now list NPU models without the user reopening it.
        time.sleep(3)
        return {"button": label, "seconds": secs, "states": seen, "versions": s.get("versions"),
                "picker_after": menu.inner_text()[:1500]}
    step("enable_npu_from_chat_picker", enable)

    small = step("catalog_small_model", lambda: model_for(SMALL), shoot=False)
    vision = step("catalog_vision_model", lambda: model_for(VISION), shoot=False)
    step("catalog_overview", lambda: [{k: m[k] for k in ("id", "checkpoint", "size_gb", "downloaded", "max_context_length")} for m in catalog()], shoot=False)
    if not small:
        return

    # --- Click the model in the picker, as a user would, and see what happens ---
    def click_small():
        open_picker("Recommended")
        row = picker_row(small["id"])
        row.scroll_into_view_if_needed()
        shot("picker-before-click-small")
        row.click()
        t = time.monotonic()
        samples = []
        def done():
            m = model_for(SMALL)
            dl = api("GET", "/api/npu/downloads")["body"]
            samples.append({"t": round(time.monotonic() - t), "downloaded": m["downloaded"], "dl": dl})
            if len(samples) % 15 == 1:
                shot("downloading-small", full=False)
            return m if m["downloaded"] else None
        poll(done, 1800, 2)
        secs = round(time.monotonic() - t, 1)
        # Per the code a finished download selects the model; does it also load it?
        time.sleep(8)
        st = inference_status()
        shot("after-small-download")
        return {"download_seconds": secs, "samples_head": samples[:5], "inference_status_after": st,
                "picker_open": U.page.locator(".unsloth-model-selector-menu").count() > 0,
                "visible_text": body_text()[:1200]}
    after_dl = step("pick_small_npu_model_in_chat", click_small) or {}

    def ensure_loaded():
        st = inference_status()
        if st.get("active_model") == small["model_path"] or (st.get("is_npu") and st.get("loaded")):
            return {"already_loaded": True, "status": st}
        note("Downloading an NPU model from the chat picker did not load it; the user must act again.")
        # Look for the obvious next action.
        txt = body_text()
        load_btn = U.page.get_by_role("button", name=re.compile(r"^(Load model|Load)$"))
        if load_btn.count():
            load_btn.first.click()
        else:
            open_picker("On Device")
            row = picker_row(small["id"])
            row.click()
        t = time.monotonic()
        st = poll(lambda: (s := inference_status()) and (s.get("active_model") == small["model_path"] or s.get("is_npu")) and s.get("loaded") and s, 600, 1)
        return {"already_loaded": False, "load_seconds": round(time.monotonic() - t, 1), "status": st, "text_before": txt[:800]}
    step("small_model_loaded", ensure_loaded)
    step("npu_status_after_load", npu_status, shoot=False)

    # --- Chat ---
    close_picker()
    step("chat_first_message", lambda: send_chat("What is 17 + 25? Reply with just the number."))
    step("chat_follow_up", lambda: send_chat("Now double that number. Reply with just the number."))
    def speed_visible():
        txt = body_text()
        m = re.findall(r"[\d.]+\s*tok(?:ens)?/s", txt)
        if not m:
            note("No tokens/s figure visible after an NPU reply.")
        return m[-3:]
    step("reply_speed_shown", speed_visible)

    # --- Thinking toggle (qwen3 is a reasoning model) ---
    def thinking():
        btn = U.page.get_by_role("button", name=re.compile(r"think", re.I))
        sw = U.page.get_by_role("switch", name=re.compile(r"think", re.I))
        target = btn if btn.count() else sw
        if not target.count():
            note("No Thinking control visible in the composer for a reasoning NPU model.")
            return {"control": None}
        before = target.first.get_attribute("aria-pressed") or target.first.get_attribute("aria-checked") or target.first.get_attribute("data-state")
        target.first.click()
        after = target.first.get_attribute("aria-pressed") or target.first.get_attribute("aria-checked") or target.first.get_attribute("data-state")
        r = send_chat("Name one primary color. One word.")
        return {"before": before, "after": after, "reply": r}
    step("thinking_toggle_and_chat", thinking)

    # --- Sampling settings in the Run settings panel ---
    def sampling():
        page = U.page
        out = {}
        open_run_settings()
        for label, value in (("Temperature", "0.2"), ("Max Tokens", "64")):
            box = page.get_by_label(label, exact=True)
            if not box.count():
                box = page.get_by_role("spinbutton", name=label)
            if not box.count():
                out[label] = "not found"
                continue
            set_number(box.first, value)
            out[label] = box.first.input_value()
        r = send_chat("Write a long story about a lighthouse keeper. Make it at least 500 words.")
        out["reply_words"] = len(r["reply"].split())
        out["reply_tail"] = r["reply"][-400:]
        out["hit_limit_banner"] = "hit the Max Tokens limit" in r["reply"]
        out["seconds"] = r["seconds"]
        return out
    step("sampling_settings_apply", sampling)

    # --- Change the context length and reload, from the model's settings ---
    def reload_ctx():
        open_picker("On Device")
        gear = U.page.get_by_role("button", name=f"Inference settings for {small['id']}")
        if not gear.count():
            open_picker("Recommended")
            gear = U.page.get_by_role("button", name=f"Inference settings for {small['id']}")
        gear.first.click()
        time.sleep(1.5)
        shot("npu-model-settings")
        panel = U.page.locator(".unsloth-model-selector-menu").first
        fields = panel.inner_text()[:2000]
        ctx_box = U.page.get_by_label("Context Length", exact=True)
        before = ctx_box.first.input_value() if ctx_box.count() else None
        if ctx_box.count():
            set_number(ctx_box.first, "4096")
        btn = U.page.get_by_role("button", name=re.compile(r"^(Reload model|Load model)$"))
        label = btn.first.inner_text()
        t = time.monotonic()
        btn.first.click()
        st = poll(lambda: (s := npu_status()) and s.get("context_length") == 4096 and s.get("loaded_model") and s, 300, 1)
        return {"panel_fields": fields, "ctx_before": before, "button": label, "reload_seconds": round(time.monotonic() - t, 1), "status": st}
    step("change_context_and_reload", reload_ctx)
    close_picker()
    step("chat_after_reload", lambda: send_chat("Say hello in French. One word."))

    # --- Browser refresh: is the model still there, and the conversation? ---
    def refresh():
        U.page.reload(wait_until="networkidle")
        time.sleep(4)
        trig = picker_trigger().inner_text()
        st = inference_status()
        return {"trigger_text": trig, "loaded": st.get("loaded"), "active": st.get("active_model"), "url": U.page.url}
    step("browser_refresh_keeps_model", refresh)

    # --- The Model hub, as a user browsing for more NPU models ---
    def hub_default():
        goto_hub()
        txt = body_text()
        visible = small["id"] in txt or "NPU" in txt
        if small["id"] not in txt:
            note("Model hub default view does not show NPU models; the user must find the Format filter and pick NPU.")
        return {"npu_word_visible": "NPU" in txt, "model_listed": small["id"] in txt, "head": txt[:1200]}
    step("hub_default_view", hub_default)

    def hub_npu():
        page = U.page
        ff = page.get_by_label("Format filter", exact=True)
        ff.first.click()
        page.get_by_role("option", name="NPU", exact=True).click()
        page.get_by_test_id("hub-npu-catalog").wait_for(timeout=60000)
        time.sleep(2)
        rows = page.get_by_test_id("hub-npu-row")
        return {"rows": rows.count(), "first_rows": [rows.nth(i).inner_text().replace("\n", " | ") for i in range(min(rows.count(), 8))]}
    step("hub_npu_filter", hub_npu)

    if vision:
        def hub_download_vision():
            page = U.page
            row = page.get_by_test_id("hub-npu-row").and_(page.locator(f"[data-model='{vision['id']}']"))
            row.get_by_role("button", name=re.compile(r"^(Download|Resume)$")).click()
            t = time.monotonic()
            time.sleep(10)
            shot("hub-vision-downloading")
            # Leave and come back mid-download: the progress should still be there.
            page.reload(wait_until="networkidle")
            time.sleep(3)
            txt = body_text()
            kept = vision["id"] in txt and "Downloading" in txt
            if not kept:
                ff = page.get_by_label("Format filter", exact=True)
                ff.first.click(); page.get_by_role("option", name="NPU", exact=True).click(); time.sleep(3)
                if "Downloading" not in body_text():
                    note("After a browser refresh during an NPU download the Hub reset to its default format; the user had to choose NPU again to see the progress.")
                else:
                    note("After a browser refresh the Hub format filter fell back from NPU (download progress still there once NPU was chosen again).")
            poll(lambda: model_for(VISION)["downloaded"], 2400, 3)
            time.sleep(3)
            return {"seconds": round(time.monotonic() - t, 1), "kept_after_refresh": kept}
        step("hub_download_vision_model", hub_download_vision)

        def hub_run():
            page = U.page
            row = page.get_by_test_id("hub-npu-row").and_(page.locator(f"[data-model='{vision['id']}']"))
            row.get_by_role("button", name="Run", exact=True).click()
            time.sleep(5)
            shot("after-hub-run")
            url = page.url
            settings_open = page.locator(".unsloth-model-selector-menu").count() > 0
            btn = page.get_by_role("button", name=re.compile(r"^(Load model|Reload model)$"))
            info = {"url": url, "settings_open": settings_open, "load_button": btn.count()}
            if not btn.count():
                note("Hub Run on an NPU model did not show a Load button.")
                return info
            t = time.monotonic()
            btn.first.click()
            poll(lambda: (s := inference_status()) and s.get("active_model") == vision["model_path"] and s.get("loaded"), 600, 1)
            info["load_seconds"] = round(time.monotonic() - t, 1)
            return info
        step("hub_run_vision_model", hub_run)
        close_picker()

        def image_chat():
            page = U.page
            png = U.ROOT / "red.png"
            png.write_bytes(base64.b64decode(RED_PNG))
            inp = page.locator("input[type=file]").first
            inp.set_input_files(str(png))
            time.sleep(2)
            return send_chat("What color is this image? Answer with one word.")
        step("vision_chat_with_image", image_chat)

        def tool_chat():
            page = U.page
            tools = page.get_by_role("button", name="Tools", exact=True)
            info = {}
            if tools.count():
                tools.first.click(); time.sleep(1); shot("tools-menu")
                info["menu"] = body_text()[-800:]
                page.keyboard.press("Escape")
            r = send_chat("Use Python to compute the sum of the integers from 1 to 100, and tell me the result.", timeout=900)
            info["reply"] = r
            return info
        step("tool_use_chat", tool_chat)

    # --- Restart Studio: does the NPU come back without clicking Enable again? ---
    close_picker()
    step("stop_studio", stop_studio, shoot=False)
    step("restart_studio", lambda: start_studio("restart"), shoot=False)

    def after_restart():
        page = U.page
        page.goto(U.BASE + "/chat", wait_until="networkidle")
        if "/login" in page.url:
            ensure_login()
        time.sleep(3)
        st = npu_status()
        open_picker("Recommended")
        txt = U.page.locator(".unsloth-model-selector-menu").first.inner_text()
        if "Enable NPU" in txt or "Start NPU runtime" in txt:
            note(f"After a Studio restart the picker shows '{'Enable NPU' if 'Enable NPU' in txt else 'Start NPU runtime'}' again.")
        recents = body_text()
        return {"npu_status": st, "picker_head": txt[:1000], "chat_history_listed": "17 + 25" in recents or "lighthouse" in recents.lower()}
    step("after_restart_state", after_restart)

    def reload_after_restart():
        open_picker("On Device")
        row = picker_row(small["id"])
        row.click()
        t = time.monotonic()
        st = poll(lambda: (s := inference_status()) and s.get("active_model") == small["model_path"] and s.get("loaded") and s, 600, 1)
        close_picker()
        r = send_chat("What is 3 times 4? Reply with just the number.")
        return {"load_seconds": round(time.monotonic() - t, 1), "context_length": npu_status().get("context_length"), "chat": r}
    step("load_small_after_restart", reload_after_restart)

    # --- Normal unsloth for comparison: a small GGUF on the same machine's GPU ---
    def gguf_flow():
        page = U.page
        open_picker("Recommended")
        search = page.locator("[data-model-picker-search-input]").first
        search.fill("Qwen3-0.6B")
        time.sleep(4)
        shot("gguf-search")
        opt = page.locator(".unsloth-model-selector-menu").first.get_by_text(re.compile(r"^Qwen3-0\.6B-GGUF$")).first
        opt.click()
        time.sleep(3)
        shot("gguf-after-click")
        # GGUF repos expand into quants; pick the recommended/first one if asked.
        q = page.get_by_text(re.compile(r"^(Q4_K_M|UD-Q4_K_XL)$"))
        if q.count():
            q.first.click()
        t = time.monotonic()
        st = poll(lambda: (s := inference_status()) and s.get("loaded") and not s.get("is_npu") and s, 1200, 2)
        close_picker()
        r = send_chat("What is 17 + 25? Reply with just the number.")
        return {"load_seconds": round(time.monotonic() - t, 1), "status": st, "chat": r, "npu_after": npu_status()}
    step("gguf_comparison_flow", gguf_flow)

    def switch_back():
        open_picker("On Device")
        picker_row(small["id"]).click()
        t = time.monotonic()
        st = poll(lambda: (s := inference_status()) and s.get("active_model") == small["model_path"] and s.get("loaded") and s, 600, 1)
        close_picker()
        return {"switch_seconds": round(time.monotonic() - t, 1), "chat": send_chat("Reply with the word ok.")}
    step("switch_gguf_to_npu", switch_back)



RED_PNG = ("iVBORw0KGgoAAAANSUhEUgAAAQAAAAEACAIAAADTED8xAAACwElEQVR4nO3TMQEAIAzAMEDD/CtDDDI4mijo031nFlSd3wHwkwFIMwBpBiDNAKQZgDQDkGYA0gxAmgFIMwBpBiDNAKQZgDQDkGYA0gxAmgFIMwBpBiDNAKQZgDQDkGYA0gxAmgFIMwBpBiDNAKQZgDQDkGYA0gxAmgFIMwBpBiDNAKQZgDQDkGYA0gxAmgFIMwBpBiDNAKQZgDQDkGYA0gxAmgFIMwBpBiDNAKQZgDQDkGYA0gxAmgFIMwBpBiDNAKQZgDQDkGYA0gxAmgFIMwBpBiDNAKQZgDQDkGYA0gxAmgFIMwBpBiDNAKQZgDQDkGYA0gxAmgFIMwBpBiDNAKQZgDQDkGYA0gxAmgFIMwBpBiDNAKQZgDQDkGYA0gxAmgFIMwBpBiDNAKQZgDQDkGYA0gxAmgFIMwBpBiDNAKQZgDQDkGYA0gxAmgFIMwBpBiDNAKQZgDQDkGYA0gxAmgFIMwBpBiDNAKQZgDQDkGYA0gxAmgFIMwBpBiDNAKQZgDQDkGYA0gxAmgFIMwBpBiDNAKQZgDQDkGYA0gxAmgFIMwBpBiDNAKQZgDQDkGYA0gxAmgFIMwBpBiDNAKQZgDQDkGYA0gxAmgFIMwBpBiDNAKQZgDQDkGYA0gxAmgFIMwBpBiDNAKQZgDQDkGYA0gxAmgFIMwBpBiDNAKQZgDQDkGYA0gxAmgFIMwBpBiDNAKQZgDQDkGYA0gxAmgFIMwBpBiDNAKQZgDQDkGYA0gxAmgFIMwBpBiDNAKQZgDQDkGYA0gxAmgFIMwBpBiDNAKQZgDQDkGYA0gxAmgFIMwBpBiDNAKQZgDQDkGYA0gxAmgFIMwBpBiDNAKQZgDQDkGYA0gxAmgFIMwBpBiDNAKQZgDQDkGYA0gxAmgFIMwBpBiDNAKQZgDQDkGYA0gxAmgFIMwBpBiDNAKQZgDQDkPYAXr8DBG1eXNAAAAAASUVORK5CYII=")

if __name__ == "__main__":
    main()
