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

    # E1: how often does a trivial question come back with no visible answer?
    def repeat_3x4():
        rows = []
        for i in range(6):
            r = X.send_chat("What is 3 times 4? Reply with just the number.", new_chat=True, timeout=400)
            ev = run_events(RUNS[-1]) if RUNS else None
            visible = [l for l in r["reply"].splitlines() if l.strip() and not l.startswith("Worked for")][:1]
            rows.append({"i": i, "visible_first_line": visible[0] if visible else None, "seconds": r["seconds"], "events": ev,
                         "expanded": r.get("expanded_thinking", "")[-300:]})
            shot(f"3x4-{i}", full=False)
        empty = [x for x in rows if x["visible_first_line"] in (None, "Copy")]
        if empty:
            note(f"{len(empty)}/6 fresh chats of 'What is 3 times 4?' on qwen3-0.6b-FLM showed no answer.")
        return rows
    step("e1_repeat_3x4", repeat_3x4, shoot=False)

    # S1: does a draft typed right after a model switch survive?
    def switch_and_type():
        out = {}
        for m in (big, small):
            X.open_picker("On Device")
            X.picker_row(m["id"]).click()
            t = time.monotonic()
            poll(lambda m=m: X.is_loaded(m), 600, 1)
            loaded_after = round(time.monotonic() - t, 1)
            X.close_picker()
            box = page.locator("textarea[aria-label='Message input']").first
            box.click()
            box.fill("Reply with the single word: ok")
            reads = [box.input_value()]
            for _ in range(5):
                time.sleep(1)
                reads.append(box.input_value())
            shot(f"draft-after-switch-{m['id']}")
            out[m["id"]] = {"backend_loaded_s": loaded_after, "draft_readbacks": reads,
                            "toasts": [t.inner_text() for t in page.locator("[data-sonner-toast]").all()][:3]}
            if reads[-1] != reads[0]:
                note(f"Text typed right after switching to {m['id']} was cleared from the composer ({reads}).")
            try:
                out[m["id"]]["chat"] = X.send_chat("Reply with the single word: ok", timeout=300)
            except Exception as exc:  # noqa: BLE001
                out[m["id"]]["chat_error"] = str(exc)[:600]
        return out
    step("s1_switch_then_type", switch_and_type, shoot=False)

    # R1: reasoning fills a small context; what does the chat say?
    def context_full():
        X.open_picker("On Device")
        page.get_by_role("button", name=f"Inference settings for {small['id']}").first.click()
        time.sleep(1.5)
        box = page.get_by_label("Context Length", exact=True).first
        box.click(force=True)
        box.press("Control+a")
        box.type("2048", delay=60)
        box.press("Tab")
        page.get_by_role("button", name=re.compile(r"^(Reload model|Load model)$")).filter(visible=True).first.click(force=True)
        poll(lambda: (s := X.npu_status()) and s.get("context_length") == 2048, 300, 1)
        X.close_picker()
        r = X.send_chat("Think very carefully, step by step, and list every prime number below 3000 with a one-line justification for each.", timeout=600, new_chat=True)
        r["events"] = run_events(RUNS[-1]) if RUNS else None
        r["banners"] = [b for b in ("hit the Max Tokens limit", "filled the model's context window", "empty response", "Continue", "Resume") if b in r["reply"]]
        if not r["banners"]:
            note("When reasoning filled the 2048-token NPU context the chat showed no explanation.")
        return r
    step("r1_context_full_message", context_full)


if __name__ == "__main__":
    main()
