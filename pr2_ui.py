"""Load an oversized GGUF from the chat picker like a user, and photograph what Studio says.

usage: pr2_ui.py BASE HOME PWFILE FOLDER OUTDIR LABEL

Fresh browser context, English locale. Registers FOLDER as a model folder, opens the chat model
picker's On Device tab, picks the Q5 quant of Qwen3.8-Flash-Next, and waits up to 25 minutes for
the load's outcome toast (error, or the "loaded with a warning" toast). Screenshots are taken the
moment that toast shows, since Studio dismisses it after 8 to 12 seconds.
"""

import asyncio
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

from playwright.async_api import async_playwright

BASE, HOME, PWFILE, FOLDER, OUT, LABEL = sys.argv[1:7]
os.makedirs(OUT, exist_ok=True)
facts = {"label": LABEL}


def call(method, path, body=None, token=None, timeout=60):
    req = urllib.request.Request(BASE + path, method=method,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {token}"} if token else {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", "replace")
            return r.status, (json.loads(raw) if raw.strip()[:1] in "{[" else raw)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, raw


def login():
    boot = os.path.join(HOME, "auth", ".bootstrap_password")
    cands = [open(PWFILE).read().strip() if os.path.exists(PWFILE) else None,
             open(boot).read().strip() if os.path.exists(boot) else None]
    for cand in cands:
        if not cand:
            continue
        st, r = call("POST", "/api/auth/login", {"username": "unsloth", "password": cand})
        if st == 200 and isinstance(r, dict) and r.get("access_token"):
            if r.get("must_change_password"):
                new = "DevlabPass-12345!"
                st2, r2 = call("POST", "/api/auth/change-password",
                               {"current_password": cand, "new_password": new}, r["access_token"])
                open(PWFILE, "w").write(new)
                if isinstance(r2, dict) and r2.get("access_token"):
                    return r2
                st, r = call("POST", "/api/auth/login", {"username": "unsloth", "password": new})
            return r
    raise SystemExit("login failed")


async def main():
    auth = login()
    tok = auth["access_token"]
    st, r = call("POST", "/api/hub/scan-folders", {"path": FOLDER, "recursive": True}, tok)
    facts["scan_folder"] = [st, r if isinstance(r, dict) else str(r)[:200]]
    seed = {"unsloth_auth_token": tok, "unsloth_refresh_token": auth.get("refresh_token", "")}
    init = ("(() => { const s = %s; for (const k of Object.keys(s)) { try { localStorage.setItem(k, s[k]); } "
            "catch (e) {} } })();" % json.dumps(seed))
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(locale="en-US", viewport={"width": 1280, "height": 860},
                                        device_scale_factor=2)
        await ctx.add_init_script(init)
        # Text paints only with a forced family on these runners.
        await ctx.add_init_script("""(() => { const add = () => { const s = document.createElement('style');
            s.textContent = '* { font-family: "DejaVu Sans", sans-serif !important; }';
            document.documentElement.appendChild(s); };
            if (document.documentElement) add(); else document.addEventListener('DOMContentLoaded', add); })();""")
        page = await ctx.new_page()
        page.set_default_timeout(60_000)
        await page.goto(BASE + "/chat", wait_until="domcontentloaded")
        trig = page.locator(".unsloth-model-selector-trigger, [data-tour='chat-model-selector']").first
        await trig.wait_for(state="visible", timeout=180_000)
        await page.wait_for_timeout(3_000)
        await trig.click()
        menu = page.locator(".unsloth-model-selector-menu").first
        await menu.wait_for(state="visible")
        await menu.get_by_text("On Device", exact=True).first.click()
        await page.wait_for_timeout(5_000)
        await page.screenshot(path=f"{OUT}/{LABEL}_0_on_device.png")
        facts["menu_text"] = (await menu.inner_text())[:800]
        row = menu.get_by_text(re.compile(os.environ.get("PR2_MODEL_RE", r"Qwen3\.8-Flash-Next"))).first
        await row.wait_for(state="visible", timeout=120_000)
        await row.click()
        await page.wait_for_timeout(2_000)
        await page.screenshot(path=f"{OUT}/{LABEL}_1_expanded.png")
        variant = menu.get_by_text(re.compile(os.environ.get("PR2_VARIANT_RE", r"Q5_K_XL"))).first
        if await variant.count():
            facts["variant_text"] = (await variant.inner_text())[:80]
            await variant.click()
        t0 = time.time()
        await page.wait_for_timeout(3_000)
        await page.screenshot(path=f"{OUT}/{LABEL}_2_loading.png")
        outcome = None
        while time.time() - t0 < 1500:
            toasts = await page.locator("[data-sonner-toast]").evaluate_all(
                "els => els.map(e => [e.getAttribute('data-type'), e.innerText])")
            kinds = {k for k, _ in toasts}
            if "error" in kinds or "warning" in kinds:
                await page.wait_for_timeout(700)
                await page.screenshot(path=f"{OUT}/{LABEL}_3_outcome.png")
                outcome = toasts
                break
            if "success" in kinds and time.time() - t0 > 5:
                # A plain success: give a warning toast a moment to join it.
                await page.wait_for_timeout(1_500)
                await page.screenshot(path=f"{OUT}/{LABEL}_3_outcome.png")
                outcome = await page.locator("[data-sonner-toast]").evaluate_all(
                    "els => els.map(e => [e.getAttribute('data-type'), e.innerText])")
                break
            await page.wait_for_timeout(1_000)
        facts["outcome_seconds"] = round(time.time() - t0, 1)
        facts["toasts"] = outcome
        boxes = await page.locator("[data-sonner-toast]").evaluate_all(
            "els => els.map(e => { const r = e.getBoundingClientRect(); "
            "return [e.getAttribute('data-type'), r.x, r.y, r.width, r.height]; })")
        facts["toast_boxes"] = boxes
        header = await page.locator(".unsloth-model-selector-trigger, [data-tour='chat-model-selector']").first.bounding_box()
        facts["trigger_box"] = header
        facts["trigger_text"] = (await page.locator(
            ".unsloth-model-selector-trigger, [data-tour='chat-model-selector']").first.inner_text())[:120]
        await page.wait_for_timeout(15_000)
        await page.screenshot(path=f"{OUT}/{LABEL}_4_settled.png")
        st, status = call("GET", "/api/inference/status", None, tok)
        facts["status_active_model"] = status.get("active_model") if isinstance(status, dict) else None
        await browser.close()
    json.dump(facts, open(f"{OUT}/{LABEL}_facts.json", "w"), indent=1, default=str)
    print(json.dumps({k: facts.get(k) for k in ("label", "outcome_seconds", "toasts", "status_active_model")},
                     default=str)[:2000])


asyncio.run(main())
