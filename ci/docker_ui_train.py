"""Studio Train page, the way a user drives it: model picker, dataset picker, max steps,
Start training, then watch the Current Run tab until it finishes. Screenshots to OUT.

Env: STUDIO_URL, STUDIO_PW (current), STUDIO_NEW_PW, OUT, TRAIN_MODEL_Q, TRAIN_MODEL_PICK,
     TRAIN_DS_Q, TRAIN_DS_PICK, TRAIN_STEPS, TRAIN_TIMEOUT_S
"""
import json
import os
import re
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import docker_ui_probe as P  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

MODEL_Q = os.environ.get("TRAIN_MODEL_Q", "Qwen3-0.6B")
MODEL_PICK = os.environ.get("TRAIN_MODEL_PICK", "unsloth/Qwen3-0.6B-unsloth-bnb-4bit")
DS_Q = os.environ.get("TRAIN_DS_Q", "alpaca-cleaned")
DS_PICK = os.environ.get("TRAIN_DS_PICK", "yahma/alpaca-cleaned")
STEPS = os.environ.get("TRAIN_STEPS", "20")
TIMEOUT_S = float(os.environ.get("TRAIN_TIMEOUT_S", "1800"))


def pick(page, tour, noun, query, want):
    root = page.locator(f'[data-tour="{tour}"]').first
    btn = root.get_by_role("button", name = re.compile(r"Select (model|dataset)")).first
    if not btn.count():
        btn = root
    btn.click()
    page.wait_for_timeout(800)
    page.get_by_role("textbox", name = f"Search {noun}").first.fill(query)
    opt = page.locator('[data-picker-option="true"]').filter(has_text = want).first
    opt.wait_for(state = "visible", timeout = 90_000)
    P.shoot(page, f"train-{noun}-options")
    opt.click()
    page.wait_for_timeout(1500)


def main():
    with sync_playwright() as p:
        b = p.chromium.launch()
        ctx = b.new_context(viewport = {"width": 1440, "height": 1000}, locale = "en-US")
        page = ctx.new_page()
        page.set_default_timeout(60_000)
        try:
            P.studio_sign_in(page)
            page.get_by_role("button", name = "Train", exact = True).first.click()
            page.wait_for_url(re.compile(r"/studio"), timeout = 60_000)
            page.wait_for_timeout(2500)
            P.shoot(page, "train-page")
            pick(page, "studio-model-picker", "models", MODEL_Q, MODEL_PICK)
            pick(page, "studio-dataset", "datasets", DS_Q, DS_PICK)
            P.shoot(page, "train-picked")
            # Max Steps input sits next to the "Max Steps" label in the Simple params card
            params = page.locator('[data-tour="studio-params"]').first
            lbl = params.get_by_text("Max Steps", exact = True).first
            target = lbl.locator("xpath=ancestor::*[.//input][1]").locator("input").first
            P.log(f"max steps input value before: {target.input_value()!r}")
            target.click()
            target.fill(STEPS)
            target.press("Tab")
            page.wait_for_timeout(800)
            P.shoot(page, "train-steps-set")
            start = page.locator('[data-tour="studio-start"]').first
            t_wait = time.time()
            while time.time() - t_wait < 300 and not re.search(r"start", start.inner_text(), re.I):
                page.wait_for_timeout(2000)
            P.log(f"start button text: {start.inner_text()!r} (waited {time.time() - t_wait:.0f}s)")
            start.scroll_into_view_if_needed()
            start.click()
            page.wait_for_timeout(3000)
            P.shoot(page, "train-started")
            t0 = time.time()
            last_shot = 0
            state = ""
            while time.time() - t0 < TIMEOUT_S:
                st = P.api(page, "/api/train/status")
                body = st.get("body") or {}
                state = json.dumps(body)[:600]
                phase = str(body.get("phase") or "")
                if time.time() - last_shot > 60:
                    P.shoot(page, "train-progress")
                    P.log(f"status: {state}")
                    last_shot = time.time()
                if phase == "completed":
                    break
                if phase in ("error", "stopped"):
                    raise AssertionError(f"training failed: {state}")
                page.wait_for_timeout(5000)
            else:
                raise AssertionError(f"training did not finish in {TIMEOUT_S}s: {state}")
            page.wait_for_timeout(3000)
            P.shoot(page, "train-finished")
            runs = P.api(page, "/api/train/runs")
            P.record("studio_train_via_ui", True, f"{time.time() - t0:.0f}s final={state[:400]}")
            P.record("studio_train_runs_listed", runs.get("status") == 200, json.dumps(runs.get("body"))[:600])
        except Exception as e:
            P.shoot(page, "train-error")
            P.record("studio_train_via_ui", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1200:]}")
        finally:
            b.close()
    (P.OUT / "train_results.json").write_text(json.dumps(P.RESULTS, indent = 2))
    bad = [k for k, v in P.RESULTS.items() if not v["ok"]]
    P.log(f"SUMMARY: {len(P.RESULTS) - len(bad)}/{len(P.RESULTS)} passed; failed={bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
