"""After Studio was killed mid-download and restarted: what the picker offers, and the download it starts.

env: as ab_ui.py (BASE, TOKENS, OUT, TAG, MODEL). Screenshots go to OUT/<TAG>_resume_*.png.
"""

import json
import re
import time

from playwright.sync_api import sync_playwright

import ab_ui as ui


def main():
    results = {"tag": ui.TAG, "model": ui.MODEL, "steps": []}

    def record(step, **data):
        row = {"t": round(time.monotonic() - ui.T0, 1), "step": step, **data}
        results["steps"].append(row)
        print(json.dumps(row), flush = True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport = {"width": 1440, "height": 900}, locale = "en-US")
        ctx.add_init_script(ui.INIT)
        page = ctx.new_page()
        page.goto(ui.BASE + "/chat")
        ui.setup_account(page)
        page.locator(".unsloth-model-selector-trigger").first.wait_for(timeout = 120000)
        page.wait_for_timeout(2000)
        row = ui.open_picker(page)
        record("after_restart", row = ui.row_text(page), shot = ui.shot(page, "resume_0_after_restart"))
        row.click()
        page.wait_for_function(
            """(sel) => /Downloading \\d+%/.test(document.querySelector(sel)?.innerText || "")""",
            arg = ui.POP, timeout = 120000,
        )
        first = re.search(r"Downloading (\d+)%", ui.row_text(page))
        record(
            "first_percent_shown",
            percent = int(first.group(1)) if first else None,
            row = ui.row_text(page),
            shot = ui.shot(page, "resume_1_downloading"),
        )
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            models = ui.api(page, "/api/npu/models")["models"]
            if next(m for m in models if m["id"] == ui.MODEL)["downloaded"]:
                break
            time.sleep(2)
        page.wait_for_timeout(4000)
        # A finished download picks the model, which closes the picker.
        page.screenshot(path = ui.OUT / f"{ui.TAG}_resume_2_finished.png")
        record("finished", url = page.url)
        ctx.close()
        browser.close()
    (ui.OUT / f"{ui.TAG}_resume.json").write_text(json.dumps(results, indent = 1))


if __name__ == "__main__":
    main()
