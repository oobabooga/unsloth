"""Auto-switch safeguards on hardware: a GGUF reply streaming on the iGPU while an API request
names the NPU model, and a tool-result image naming a text-only NPU model. Then the earlier
refusal checks (ux_ab2)."""
import base64, json, os, sys, threading, time

sys.path.insert(0, os.path.dirname(__file__))
import uxlib as U
from uxlib import step, api
import ux_ui as X
import ux_ab as AB
import ux_ab2 as AB2

GGUF = {"model_path": "unsloth/Qwen3-0.6B-GGUF", "gguf_variant": "Q4_K_M"}
LONG = ("Write a detailed essay of at least 1500 words on the history of ocean navigation, "
        "from Polynesian wayfinding to satellite positioning. Use many paragraphs.")


def loaded():
    return X.inference_status().get("loaded") or []


def main():
    step("start_studio", lambda: X.start_studio("switch"), shoot=False)
    try:
        body()
    finally:
        X.stop_studio()
    AB2.main()
    print("NOTES:", json.dumps(U.NOTES, indent=1))


def body():
    from playwright.sync_api import sync_playwright
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
            step("login", X.ensure_login, shoot=False)
            checks()
        finally:
            U.page = None
            browser.close()


def checks():
    step("enable", lambda: (api("POST", "/api/npu/enable", timeout=900), X.npu_status()["ready"])[1], shoot=False)
    small = step("download_small", lambda: AB.dl(AB.SMALL), shoot=False)
    key = small["model_path"]
    gguf_id = None

    def load_gguf():
        api("POST", "/api/inference/load", json=GGUF, timeout=3600)
        return loaded()

    def gguf_reply_survives_npu_switch():
        nonlocal gguf_id
        api("POST", "/api/inference/load", json=GGUF, timeout=3600)
        gguf_id = X.inference_status().get("active_model")
        api("PUT", "/api/settings/openai-auto-switch", json={"enabled": True})
        out = {"gguf": {}, "npu": {}}

        started = threading.Event()

        def stream_gguf():
            t = time.monotonic()
            lines, err = [], None
            try:
                r = U.session.post(U.BASE + "/v1/chat/completions", timeout=900, stream=True, json={
                    "model": gguf_id, "stream": True, "max_tokens": 1200,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "messages": [{"role": "user", "content": LONG}]})
                out["gguf"]["status"] = r.status_code
                for line in r.iter_lines(decode_unicode=True):
                    if line:
                        lines.append(line)
                        if len(lines) >= 40:
                            started.set()
            except Exception as exc:  # noqa: BLE001
                err = str(exc)[:300]
            text = ""
            for line in lines:
                if line.startswith("data:") and "[DONE]" not in line:
                    try:
                        ev = json.loads(line[5:])
                    except ValueError:
                        continue
                    for ch in ev.get("choices") or []:
                        text += (ch.get("delta") or {}).get("content") or ""
            out["gguf"].update(AB.last_events(lines))
            out["gguf"].update({"ended_at": round(time.monotonic() - t0, 1), "seconds": round(time.monotonic() - t, 1),
                                "done_marker": any("[DONE]" in l for l in lines), "error": err,
                                "chars": len(text), "tail": text[-120:],
                                "error_events": [l[:300] for l in lines if '"error"' in l][:3]})

        t0 = time.monotonic()
        th = threading.Thread(target=stream_gguf)
        th.start()
        started.wait(300)
        out["npu_sent_at"] = round(time.monotonic() - t0, 1)
        r = U.session.post(U.BASE + "/v1/chat/completions", timeout=900, json={
            "model": key, "max_tokens": 30, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": "Reply with the single word: ok"}]})
        out["npu"] = {"status": r.status_code, "ended_at": round(time.monotonic() - t0, 1), "body": r.text[:300]}
        th.join(900)
        out["loaded_after"] = loaded()
        out["gguf_finished_before_npu"] = out["gguf"].get("ended_at", 1e9) <= out["npu"]["ended_at"]
        return out
    step("gguf_reply_survives_npu_switch", gguf_reply_survives_npu_switch, shoot=False)

    def tool_image_with_gguf_resident():
        api("POST", "/api/inference/load", json=GGUF, timeout=3600)
        before = loaded()
        r = U.session.post(U.BASE + "/v1/chat/completions", timeout=600, json={
            "model": key, "max_tokens": 30, "messages": [
                {"role": "user", "content": "Take a screenshot and describe it."},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "screenshot", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + AB2.RED}}]}],
            "tools": [{"type": "function", "function": {"name": "screenshot", "parameters": {"type": "object", "properties": {}}}}]})
        return {"status": r.status_code, "body": r.text[:400], "loaded_before": before, "loaded_after": loaded()}
    step("tool_image_naming_text_npu_model", tool_image_with_gguf_resident, shoot=False)

    def ui_style_load_npu():
        # The picker's load path (on_reload_confirmed set): idle, so it must load.
        t = time.monotonic()
        r = api("POST", "/api/inference/load", json={"model_path": key}, timeout=900, ok=None)
        return {"status": r["status"], "seconds": round(time.monotonic() - t, 1), "loaded_after": loaded()}
    step("load_route_npu_when_idle", ui_style_load_npu, shoot=False)
    api("PUT", "/api/settings/openai-auto-switch", json={"enabled": False}, ok=None)


if __name__ == "__main__":
    main()
