"""A/B for the NPU fixes on real hardware. UX_SIDE=A runs the merge base, B the base plus the patch;
both sides run exactly the same steps and record what a user sees."""
import json, os, re, shutil, subprocess, sys, threading, time

sys.path.insert(0, os.path.dirname(__file__))
import uxlib as U
from uxlib import step, note, shot, api, poll
import ux_ui as X
from playwright.sync_api import sync_playwright

SIDE = os.environ.get("UX_SIDE", "?")
SMALL, TOOLS = "qwen3:0.6b", "qwen3.5:2b"
LOOP_PROMPT = ("Think very carefully, step by step, and list every prime number below 3000 "
               "with a one-line justification for each.")


def run_cli(args, name, *, ready=None, timeout=900, stdin=None, then=None):
    """Run `unsloth <args>` like a user in a terminal; stop it once it serves or exits."""
    exe = shutil.which("unsloth")
    log = U.OUT / "cmd" / f"{name}.log"
    log.parent.mkdir(exist_ok=True)
    kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if U.WIN else {"start_new_session": True}
    env = dict(os.environ, PYTHONUNBUFFERED="1", COLUMNS="160", NO_COLOR="1")
    t0 = time.monotonic()
    p = subprocess.Popen([exe, *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                         text=True, encoding="utf-8", errors="replace", env=env, **kw)
    lines = []
    th = threading.Thread(target=lambda: [lines.append(l.rstrip("\n")) for l in p.stdout], daemon=True)
    th.start()
    if stdin is not None:
        try:
            p.stdin.write(stdin)
            p.stdin.close()
        except OSError:
            pass
    hit, extra = None, None
    while time.monotonic() - t0 < timeout and p.poll() is None:
        if ready and (m := re.search(ready, "\n".join(lines), re.I | re.M)):
            hit = m.group(0)
            break
        time.sleep(1)
    if hit and then:
        try:
            extra = then("\n".join(lines))
        except Exception as exc:  # noqa: BLE001
            extra = {"error": str(exc)[:500]}
    rc = p.poll()
    if rc is None and not (then and extra and extra.get("keep")):
        if U.WIN:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True)
        else:
            os.killpg(p.pid, 9)
        time.sleep(3)
    th.join(5)
    text = "\n".join(lines)
    log.write_text(f"$ unsloth {' '.join(args)}\nrc={rc} seconds={time.monotonic()-t0:.1f} ready={hit!r}\n\n{text}", encoding="utf-8")
    return {"rc": rc, "ready": hit, "then": extra, "tail": text[-1800:], "proc": p}


def last_events(lines):
    finishes, tool_events, usage = [], [], None
    for line in lines:
        if not line.startswith("data:") or "[DONE]" in line:
            continue
        try:
            ev = json.loads(line[5:])
        except ValueError:
            continue
        if ev.get("_toolEvent"):
            tool_events.append(ev["_toolEvent"])
        for ch in ev.get("choices") or []:
            if ch.get("finish_reason"):
                finishes.append(ch["finish_reason"])
        usage = ev.get("usage") or usage
    return {"finish_reasons": finishes, "tool_events": tool_events,
            "usage": {k: (usage or {}).get(k) for k in ("total_tokens", "max_kv_token_capacity")}}


def main():
    step("start_studio", lambda: X.start_studio("ab"), shoot=False)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=["--no-sandbox"])
        ctx = browser.new_context(locale="en-US", viewport={"width": 1440, "height": 1000})
        if not U.WIN:
            ctx.add_init_script("document.addEventListener('DOMContentLoaded',()=>{const s=document.createElement('style');s.textContent='*{font-family:\"DejaVu Sans\",sans-serif!important}';document.head.appendChild(s)})")
        page = ctx.new_page()
        page.set_default_timeout(30000)
        U.page = page
        try:
            ui(page)
        finally:
            browser.close()
    X.stop_studio()
    cli()
    print("NOTES:", json.dumps(U.NOTES, indent=1))


def ui(page):
    page.goto(U.BASE + "/", wait_until="networkidle")
    step("login", X.ensure_login)

    # U1 + copy: the picker on a fresh Ryzen AI install, before anything is enabled.
    def picker_first_look():
        X.open_picker("Recommended")
        menu = page.locator(".unsloth-model-selector-menu").first
        time.sleep(2)
        shot("picker-first-look", full=False)
        text = menu.inner_text()
        return {"enable_visible": menu.get_by_role("button", name="Enable NPU").count() > 0,
                "copy": next((l for l in text.splitlines() if "NPU" in l and len(l) > 20), None)}
    step("u1_picker_first_look", picker_first_look, shoot=False)

    def label_click():
        menu = page.locator(".unsloth-model-selector-menu").first
        before = menu.get_by_role("button", name="Enable NPU").count() > 0
        menu.get_by_text("NPU", exact=True).first.click()
        time.sleep(1)
        after = menu.get_by_role("button", name="Enable NPU").count() > 0
        shot("picker-after-label-click", full=False)
        return {"enable_visible_before": before, "after_label_click": after}
    step("u1_click_the_npu_label", label_click, shoot=False)

    def enable_from_picker():
        menu = page.locator(".unsloth-model-selector-menu").first
        btn = menu.get_by_role("button", name="Enable NPU")
        if not btn.count():
            menu.get_by_role("button", name="Expand section").first.click()
            time.sleep(1)
        btn.first.click()
        st = poll(lambda: (s := X.npu_status()) and s["state"] in ("ready", "failed") and s, 900, 1)
        time.sleep(3)
        shot("picker-after-enable", full=False)
        return {"ready": st["ready"], "enable_still_open": menu.locator("text=Powered by").count() > 0,
                "rows_visible": menu.get_by_text("qwen3-0.6b-FLM", exact=True).count()}
    step("u1_enable_keeps_group_open", enable_from_picker, shoot=False)
    X.close_picker()

    # H4: the Hub feed, the NPU entry point, and a refresh on the NPU list.
    def hub():
        X.goto_hub()
        time.sleep(3)
        shot("hub-default", full=False)
        out = {"prompt_shown": page.get_by_test_id("hub-npu-prompt").count() > 0}
        if out["prompt_shown"]:
            page.get_by_role("button", name="Browse NPU models").click()
        else:
            page.get_by_label("Format filter", exact=True).first.click()
            page.get_by_role("option", name="NPU", exact=True).click()
        page.get_by_test_id("hub-npu-catalog").wait_for(timeout=60000)
        out["url_after_choosing_npu"] = page.url.replace(U.BASE, "")
        page.reload(wait_until="networkidle")
        time.sleep(4)
        out["npu_list_after_refresh"] = page.get_by_test_id("hub-npu-catalog").count() > 0
        shot("hub-after-refresh", full=False)
        return out
    step("h4_hub_npu_entry_and_refresh", hub, shoot=False)

    small = step("download_small", lambda: dl(SMALL), shoot=False)
    if not small:
        return

    # R1: reasoning fills a 2048-token context. What does the chat show?
    def context_full():
        api("POST", "/api/inference/load", json={"model_path": small["model_path"], "max_seq_length": 2048}, timeout=600)
        page.goto(U.BASE + "/chat", wait_until="networkidle")
        time.sleep(4)
        r = X.send_chat(LOOP_PROMPT, timeout=600, new_chat=True)
        shot("context-full-reply", full=False)
        r["banners"] = [b for b in ("filled the model's context window", "hit the Max Tokens limit", "empty response")
                        if b in r["reply"]]
        r.pop("expanded_thinking", None)
        return r
    step("r1_context_full_in_chat", context_full, shoot=False)

    def context_full_api():
        r = U.session.post(U.BASE + "/v1/chat/completions", timeout=900, stream=True,
                           json={"model": small["model_path"], "stream": True, "max_tokens": 4096,
                                 "messages": [{"role": "user", "content": LOOP_PROMPT}]})
        return {"status": r.status_code, **last_events([l for l in r.iter_lines(decode_unicode=True) if l])}
    step("r1_context_full_api", context_full_api, shoot=False)

    # A1: Anthropic Messages with an NPU model loaded.
    def messages():
        r = U.session.post(U.BASE + "/v1/messages", timeout=120,
                           json={"model": small["model_path"], "max_tokens": 50, "messages": [{"role": "user", "content": "hi"}]})
        return {"status": r.status_code, "body": r.text[:500]}
    step("a1_messages_with_npu_loaded", messages, shoot=False)

    # A2: name an NPU model in a request with "Switch model by request" on and nothing loaded.
    def auto_switch():
        api("POST", "/api/inference/unload", json={"model_path": small["model_path"]}, ok=None)
        api("PUT", "/api/settings/openai-auto-switch", json={"enabled": True})
        t = time.monotonic()
        r = U.session.post(U.BASE + "/v1/chat/completions", timeout=600,
                           json={"model": small["model_path"], "max_tokens": 200,
                                 "messages": [{"role": "user", "content": "What is 3 times 4? Reply with just the number."}],
                                 "chat_template_kwargs": {"enable_thinking": False}})
        return {"status": r.status_code, "seconds": round(time.monotonic() - t, 1), "body": r.text[:600]}
    step("a2_auto_switch_loads_npu_model", auto_switch, shoot=False)
    api("PUT", "/api/settings/openai-auto-switch", json={"enabled": False}, ok=None)

    def off_message():
        api("POST", "/api/inference/unload", json={"model_path": small["model_path"]}, ok=None)
        r = U.session.post(U.BASE + "/v1/chat/completions", timeout=120,
                           json={"model": small["model_path"], "messages": [{"role": "user", "content": "hi"}]})
        return {"status": r.status_code, "body": r.text[:500]}
    step("a2_switch_off_message", off_message, shoot=False)

    # Minor: whitespace content beside a tool call.
    tools_model = step("download_tool_model", lambda: dl(TOOLS), shoot=False)
    if tools_model:
        def tool_call():
            api("POST", "/api/inference/load", json={"model_path": tools_model["model_path"]}, timeout=600)
            tools = [{"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
                      "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
            r = api("POST", "/v1/chat/completions", json={"model": tools_model["model_path"], "tools": tools, "max_tokens": 600,
                    "messages": [{"role": "user", "content": "What's the weather in Paris? Use the tool."}]})["body"]
            m = r["choices"][0]["message"]
            return {"finish": r["choices"][0]["finish_reason"], "content": m.get("content"),
                    "tool_calls": [c["function"] for c in m.get("tool_calls") or []]}
        step("minor_tool_call_content", tool_call, shoot=False)
        api("POST", "/api/inference/unload", json={"model_path": tools_model["model_path"]}, ok=None)


def dl(checkpoint):
    m = X.model_for(checkpoint)
    if not m["downloaded"]:
        list(U.session.post(U.BASE + f"/api/npu/models/{m['id']}/download", stream=True, timeout=3600).iter_lines())
    return X.model_for(checkpoint)


def cli():
    ready = r"OpenAI / Anthropic SDK base URL:\s*\n\s*\S+"

    def serve_and_chat(text):
        url = re.search(r"Unsloth Studio running at (https?://\S+)", text)
        key = re.search(r"API Key:\s+(\S+)", text)
        base = url.group(1).rstrip("/") if url else None
        h = {"Authorization": "Bearer " + key.group(1)} if key else {}
        r = U.requests.post(base + "/v1/chat/completions", headers=h, timeout=300,
                            json={"messages": [{"role": "user", "content": "What is 17 + 25? Reply with just the number."}],
                                  "max_tokens": 300, "chat_template_kwargs": {"enable_thinking": False}})
        loaded = re.search(r"Model loaded:\s*(.+)", text)
        out = {"model_loaded_line": loaded.group(1) if loaded else None, "chat_status": r.status_code, "chat": r.text[:400]}
        # While this Studio serves, `unsloth chat` should attach to it.
        attach = dict(os.environ, UNSLOTH_STUDIO_URL=base or "")
        rc, chat_text = U.run([shutil.which("unsloth"), "chat", "qwen3-0.6b-FLM"], name="chat-npu-attached", timeout=600,
                              input="What is 2 + 2? Reply with just the number.\n", env=attach)
        out["unsloth_chat_rc"] = rc
        out["unsloth_chat_tail"] = chat_text[-800:]
        rc, list_text = U.run([shutil.which("unsloth"), "chat"], name="chat-list-attached", timeout=300, input="x\n", env=attach)
        out["unsloth_chat_lists_npu"] = "qwen3-0.6b-FLM" in list_text
        return out

    step("c_unsloth_run_npu", lambda: {k: v for k, v in run_cli(["run", "--model", "qwen3-0.6b-FLM", "-p", "8899"], "run-npu",
                                                                  ready=ready, timeout=1500, then=serve_and_chat).items() if k != "proc"}, shoot=False)
    step("c_unsloth_chat_no_studio", lambda: dict(zip(("rc", "tail"), (lambda r: (r[0], r[1][-1200:]))(
        U.run([shutil.which("unsloth"), "chat", "qwen3-0.6b-FLM"], name="chat-npu-no-studio", timeout=300, input="hi\n")))), shoot=False)
    step("c_unsloth_start_npu", lambda: dict(zip(("rc", "tail"), (lambda r: (r[0], r[1][-1200:]))(
        U.run([shutil.which("unsloth"), "start", "claude", "--model", "qwen3-0.6b-FLM", "--no-launch"], name="start-npu", timeout=300)))), shoot=False)


if __name__ == "__main__":
    main()
