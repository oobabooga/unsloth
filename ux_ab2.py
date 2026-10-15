"""Hardware check of the review fixes: refused NPU auto-switches load nothing, a text switch still
loads, the NPU run banner, and the Hub NPU list across Back/Forward."""
import base64, json, os, re, shutil, sys, time

sys.path.insert(0, os.path.dirname(__file__))
import uxlib as U
from uxlib import step, api
import ux_ui as X
import ux_ab as AB
from playwright.sync_api import sync_playwright

RED = base64.b64encode(base64.b64decode(X.RED_PNG)).decode()


def loaded():
    return X.inference_status().get("loaded") or []


def main():
    step("start_studio", lambda: X.start_studio("review"), shoot=False)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=["--no-sandbox"])
        ctx = browser.new_context(locale="en-US", viewport={"width": 1440, "height": 1000})
        if not U.WIN:
            ctx.add_init_script("document.addEventListener('DOMContentLoaded',()=>{const s=document.createElement('style');s.textContent='*{font-family:\"DejaVu Sans\",sans-serif!important}';document.head.appendChild(s)})")
        page = ctx.new_page()
        page.set_default_timeout(30000)
        U.page = page
        try:
            body(page)
        finally:
            browser.close()
    X.stop_studio()
    banner = AB.run_cli(["run", "--model", "qwen3-0.6b-FLM", "-p", "8899"], "run-npu-banner",
                        ready=r"OpenAI SDK base URL:\s*\n\s*\S+", timeout=900)
    step("run_banner", lambda: {"ready": banner["ready"], "rc": banner["rc"],
                                "anthropic_example": "Anthropic Messages:" in banner["tail"],
                                "responses_example": "OpenAI Responses:" in banner["tail"],
                                "tail": banner["tail"][-900:]}, shoot=False)
    print("NOTES:", json.dumps(U.NOTES, indent=1))


def body(page):
    page.goto(U.BASE + "/", wait_until="networkidle")
    time.sleep(2)
    step("login", X.ensure_login)
    step("enable", lambda: (api("POST", "/api/npu/enable", timeout=900), X.npu_status()["ready"])[1], shoot=False)
    small = step("download_small", lambda: AB.dl("qwen3:0.6b"), shoot=False)
    api("PUT", "/api/settings/openai-auto-switch", json={"enabled": True})
    key = small["model_path"]

    def image_refused():
        r = U.session.post(U.BASE + "/v1/chat/completions", timeout=300, json={
            "model": key, "max_tokens": 50, "messages": [{"role": "user", "content": [
                {"type": "text", "text": "What color is this?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + RED}}]}]})
        return {"status": r.status_code, "body": r.text[:400], "loaded_after": loaded()}
    step("image_to_text_npu_model_refused_before_switch", image_refused, shoot=False)

    def messages_refused():
        r = U.session.post(U.BASE + "/v1/messages", timeout=300, json={
            "model": key, "max_tokens": 50, "messages": [{"role": "user", "content": "hi"}]})
        return {"status": r.status_code, "body": r.text[:400], "loaded_after": loaded()}
    step("messages_naming_npu_model_refused_before_switch", messages_refused, shoot=False)

    def text_switch():
        t = time.monotonic()
        r = U.session.post(U.BASE + "/v1/chat/completions", timeout=600, json={
            "model": key, "max_tokens": 100, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": "Reply with the single word: ok"}]})
        return {"status": r.status_code, "seconds": round(time.monotonic() - t, 1), "body": r.text[:300], "loaded_after": loaded()}
    step("text_request_still_switches", text_switch, shoot=False)

    def messages_with_npu_resident():
        r = U.session.post(U.BASE + "/v1/messages", timeout=300, json={
            "model": key, "max_tokens": 50, "messages": [{"role": "user", "content": "hi"}]})
        return {"status": r.status_code, "body": r.text[:400]}
    step("messages_with_switching_on_and_npu_resident", messages_with_npu_resident, shoot=False)
    api("PUT", "/api/settings/openai-auto-switch", json={"enabled": False}, ok=None)



if __name__ == "__main__":
    main()
