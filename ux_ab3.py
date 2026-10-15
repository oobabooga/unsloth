"""Hardware recheck after trimming the scope: folded picker NPU group that opens from its label,
Hub NPU format filter, one-click load and chat, `unsloth chat` model list with no NPU probe,
`unsloth run` with an NPU id."""
import json, os, re, shutil, sys, time

sys.path.insert(0, os.path.dirname(__file__))
import uxlib as U
from uxlib import step, shot, api
import ux_ui as X
import ux_ab as AB
from playwright.sync_api import sync_playwright


def main():
    step("start_studio", lambda: X.start_studio("trim"), shoot=False)
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

    def npu_list_requests():
        return "".join(p.read_text(encoding="utf-8", errors="replace") for p in U.OUT.glob("studio-*-trim.log")).count('"/api/npu/models"')

    def chat_list():
        before = npu_list_requests()
        t = time.monotonic()
        rc, text = U.run([shutil.which("unsloth"), "chat"], name="chat-list", timeout=300, input="x\n",
                         env=dict(os.environ, UNSLOTH_STUDIO_URL=U.BASE))
        time.sleep(2)
        return {"seconds": round(time.monotonic() - t, 1), "lists_npu": "qwen3-0.6b-FLM" in text,
                "npu_model_list_requests_during_unsloth_chat": npu_list_requests() - before, "tail": text[-400:]}
    step("unsloth_chat_list_without_npu_probe", chat_list, shoot=False)
    X.stop_studio()
    run = AB.run_cli(["run", "--model", "qwen3-0.6b-FLM", "-p", "8899"], "run-npu", ready=r"OpenAI SDK base URL:\s*\n\s*\S+", timeout=900)
    step("unsloth_run_npu", lambda: {"ready": bool(run["ready"]), "rc": run["rc"], "tail": run["tail"][-500:]}, shoot=False)
    print("NOTES:", json.dumps(U.NOTES, indent=1))


def body(page):
    page.goto(U.BASE + "/", wait_until="networkidle")
    time.sleep(2)
    step("login", X.ensure_login)

    def picker():
        X.open_picker("Recommended")
        menu = page.locator(".unsloth-model-selector-menu").first
        time.sleep(2)
        enable = menu.get_by_role("button", name="Enable NPU")
        out = {"folded_on_first_look": enable.count() == 0}
        shot("picker-first-look", full=False)
        menu.get_by_text("NPU", exact=True).first.click()
        time.sleep(1.5)
        out["label_click_shows_enable_npu"] = enable.count() > 0
        shot("picker-after-label-click", full=False)
        enable.first.click()
        st = U.poll(lambda: (s := X.npu_status()) and s["state"] in ("ready", "failed") and s, 900, 1)
        time.sleep(3)
        out["ready"] = st["ready"]
        out["model_row_after_enable"] = menu.get_by_text("qwen3-0.6b-FLM", exact=True).count() > 0
        shot("picker-after-enable", full=False)
        return out
    step("picker_folded_label_opens_enable", picker, shoot=False)

    def one_click():
        X.picker_row("qwen3-0.6b-FLM").click()
        small = U.poll(lambda: (m := X.model_for("qwen3:0.6b")) and m["downloaded"] and m, 1800, 2)
        U.poll(lambda: X.is_loaded(small), 600, 1)
        X.close_picker()
        return X.send_chat("Reply with the single word: ok", timeout=300)
    step("picker_click_downloads_loads_and_chats", one_click)

    def hub():
        X.goto_hub()
        time.sleep(3)
        out = {"prompt_row": page.get_by_test_id("hub-npu-prompt").count() > 0}
        page.get_by_label("Format filter", exact=True).first.click()
        page.get_by_role("option", name="NPU", exact=True).click()
        page.get_by_test_id("hub-npu-catalog").wait_for(timeout=60000)
        out["npu_rows"] = page.get_by_test_id("hub-npu-row").count()
        out["url"] = page.url.replace(U.BASE, "")
        return out
    step("hub_npu_via_format_filter", hub)


if __name__ == "__main__":
    main()
