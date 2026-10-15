"""Context-full parity: the same reasoning-heavy chat on the NPU model and on the same model as a GGUF
(runner iGPU), both loaded at a 2048-token context. Records what the chat shows and whether it
continues on its own."""
import json, os, sys, time

sys.path.insert(0, os.path.dirname(__file__))
import uxlib as U
from uxlib import step, shot, api
import ux_ui as X
import ux_ab as AB
from playwright.sync_api import sync_playwright

BARS = ("filled the model's context window", "hit the Max Tokens limit", "empty response",
        "Start a new chat", "Context Length")
GGUF = {"model_path": "unsloth/Qwen3-0.6B-GGUF", "gguf_variant": "Q4_K_M"}


def settle(page, quiet=45, limit=1500):
    """Wait until no generation has run for `quiet` seconds, so auto-continue rounds are counted."""
    t0, last_busy = time.monotonic(), time.monotonic()
    while time.monotonic() - t0 < limit:
        if page.locator(".aui-composer-cancel").count():
            last_busy = time.monotonic()
        elif time.monotonic() - last_busy > quiet:
            return
        time.sleep(1)


def chat_case(page, label, load_body):
    def run():
        api("POST", "/api/inference/load", json={**load_body, "max_seq_length": 2048}, timeout=3600)
        page.goto(U.BASE + "/chat", wait_until="networkidle")
        time.sleep(4)
        posts = []
        listener = lambda req: posts.append(req.url.replace(U.BASE, "")) if req.method == "POST" else None
        page.on("request", listener)
        r = X.send_chat(AB.LOOP_PROMPT, timeout=900, new_chat=True)
        settle(page)
        page.remove_listener("request", listener)
        msgs = page.locator(".aui-assistant-message-root")
        reply = msgs.last.inner_text() if msgs.count() else ""
        shot(f"{label}-context-full", full=False)
        gen = [p for p in posts if "completions" in p or "generat" in p or "/chat" in p]
        return {"seconds_first_run": r["seconds"], "generation_posts": gen,
                "bars": [b for b in BARS if b in reply],
                "continue_button": msgs.last.get_by_role("button", name="Continue").count() if msgs.count() else None,
                "reply_tail": reply[-1200:], "status": X.inference_status().get("loaded")}
    return step(f"{label}_context_full_in_chat", run, shoot=False)


def api_case(label, model):
    def run():
        r = U.session.post(U.BASE + "/v1/chat/completions", timeout=900, stream=True,
                           json={"model": model, "stream": True, "max_tokens": 4096,
                                 "messages": [{"role": "user", "content": AB.LOOP_PROMPT}]})
        return {"status": r.status_code, **AB.last_events([l for l in r.iter_lines(decode_unicode=True) if l])}
    return step(f"{label}_context_full_api", run, shoot=False)


def main():
    step("start_studio", lambda: X.start_studio("parity"), shoot=False)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=["--no-sandbox"])
        ctx = browser.new_context(locale="en-US", viewport={"width": 1440, "height": 1000})
        if not U.WIN:
            ctx.add_init_script("document.addEventListener('DOMContentLoaded',()=>{const s=document.createElement('style');s.textContent='*{font-family:\"DejaVu Sans\",sans-serif!important}';document.head.appendChild(s)})")
        page = ctx.new_page()
        page.set_default_timeout(30000)
        U.page = page
        try:
            page.goto(U.BASE + "/", wait_until="networkidle")
            time.sleep(2)
            step("login", X.ensure_login)
            step("enable", lambda: (api("POST", "/api/npu/enable", timeout=900), X.npu_status()["ready"])[1], shoot=False)
            small = step("download_small", lambda: AB.dl(AB.SMALL), shoot=False)
            if small:
                chat_case(page, "npu", {"model_path": small["model_path"]})
                api_case("npu", small["model_path"])
            chat_case(page, "gguf", GGUF)
            api_case("gguf", GGUF["model_path"])
        finally:
            browser.close()
    X.stop_studio()
    print("NOTES:", json.dumps(U.NOTES, indent=1))


if __name__ == "__main__":
    main()
