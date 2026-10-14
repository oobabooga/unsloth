"""Focused NPU probes: intermittent empty answers, a draft lost after a model switch, and what the chat
shows when reasoning fills the context."""
import json, os, re, sys, time

sys.path.insert(0, os.path.dirname(__file__))
import uxlib as U
from uxlib import step, note, shot, api, poll
import ux_ui as X
from playwright.sync_api import sync_playwright

RUNS = []


def run_events(run_id):
    """Replay a finished chat run's events and summarise what the model actually produced."""
    r = U.session.post(U.BASE + f"/api/inference/chat-runs/{run_id}/events?after=0", timeout=120)
    if r.status_code == 405:
        r = U.session.get(U.BASE + f"/api/inference/chat-runs/{run_id}/events?after=0", timeout=120)
    content, reasoning, finish, usage = "", "", None, None
    for line in r.text.splitlines():
        if not line.startswith("data:"):
            continue
        try:
            ev = json.loads(line[5:])
        except ValueError:
            continue
        payload = ev.get("payload") or {}
        for ch in payload.get("choices") or []:
            d = ch.get("delta") or {}
            content += d.get("content") or ""
            reasoning += d.get("reasoning_content") or ""
            finish = ch.get("finish_reason") or finish
        usage = payload.get("usage") or usage
        if ev.get("type") == "run.completed":
            finish = f"{finish} / run:{payload.get('finishReason')}"
    return {"http": r.status_code, "content": content[-300:], "content_len": len(content), "reasoning_len": len(reasoning),
            "reasoning_tail": reasoning[-300:], "finish": finish,
            "usage": {k: (usage or {}).get(k) for k in ("completion_tokens", "total_tokens", "max_kv_token_capacity")}}


def main():
    step("start_studio", lambda: X.start_studio("probe"), shoot=False)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=["--no-sandbox"])
        ctx = browser.new_context(locale="en-US", viewport={"width": 1440, "height": 1000})
        if not U.WIN:
            ctx.add_init_script("document.addEventListener('DOMContentLoaded',()=>{const s=document.createElement('style');s.textContent='*{font-family:\"DejaVu Sans\",sans-serif!important}';document.head.appendChild(s)})")
        page = ctx.new_page()
        page.set_default_timeout(30000)
        U.page = page

        def on_request(req):
            if req.method == "POST" and req.url.rstrip("/").endswith("/api/inference/chat-runs"):
                try:
                    RUNS.append(json.loads(req.post_data or "{}").get("runId"))
                except ValueError:
                    pass
        page.on("request", on_request)
        try:
            body(page)
        finally:
            browser.close()
            X.stop_studio()
    print("NOTES:", json.dumps(U.NOTES, indent=1))


def body(page):
    page.goto(U.BASE + "/", wait_until="networkidle")
    step("login", X.ensure_login)

    def setup():
        api("POST", "/api/npu/enable", timeout=900)
        poll(lambda: X.npu_status()["ready"], 900, 2)
        out = {}
        for ck in ("qwen3:0.6b", "qwen3.5:2b"):
            m = X.model_for(ck)
            if not m["downloaded"]:
                list(U.session.post(U.BASE + f"/api/npu/models/{m['id']}/download", stream=True, timeout=3600).iter_lines())
            out[ck] = X.model_for(ck)
        return out
    models = step("setup_enable_and_download", setup, shoot=False)
    small = models["qwen3:0.6b"]
    big = models["qwen3.5:2b"]

    def load_small_ui():
        page.reload(wait_until="networkidle")
        time.sleep(3)
        X.open_picker("On Device")
        X.picker_row(small["id"]).click()
        poll(lambda: X.is_loaded(small), 600, 1)
        X.close_picker()
        return X.inference_status().get("loaded")
    step("load_small_from_picker", load_small_ui)

    def composer_probe(label):
        info = page.evaluate("""() => [...document.querySelectorAll("textarea[aria-label='Message input']")].map(t => {
            const r = t.getBoundingClientRect();
            return {visible: !!(t.offsetParent), x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), readOnly: t.readOnly, disabled: t.disabled, value: t.value};
        })""")
        box = page.locator("textarea[aria-label='Message input']").filter(visible=True).first
        box.click()
        page.keyboard.type("hello there", delay=40)
        time.sleep(0.5)
        typed = box.input_value()
        active = page.evaluate("() => { const a = document.activeElement; return a ? (a.tagName + ' ' + (a.getAttribute('aria-label') || '')) : null }")
        shot(f"typed-{label}")
        box.press("Control+a"); box.press("Delete")
        return {"textareas": info, "typed_value": typed, "active": active}

    def s1_deep():
        out = {}
        out["baseline_small"] = composer_probe("baseline-small")
        out["baseline_chat"] = X.send_chat("Reply with the single word: ok", timeout=300)
        for m, label in ((big, "after-switch-to-2b"), (small, "after-switch-back-to-0.6b")):
            X.open_picker("On Device")
            X.picker_row(m["id"]).click()
            poll(lambda m=m: X.is_loaded(m), 600, 1)
            X.close_picker()
            time.sleep(3)
            out[label] = composer_probe(label)
        if not out["after-switch-back-to-0.6b"]["typed_value"]:
            note("After switching qwen3.5-2b-FLM -> qwen3-0.6b-FLM in the picker, keystrokes in the composer are lost (value stays empty).")
        page.get_by_role("button", name="New chat").first.click()
        time.sleep(2)
        out["after_new_chat"] = composer_probe("after-new-chat")
        page.reload(wait_until="networkidle")
        time.sleep(4)
        out["after_reload"] = composer_probe("after-reload")
        return out
    step("s1_deep", s1_deep, shoot=False)


if __name__ == "__main__":
    main()
