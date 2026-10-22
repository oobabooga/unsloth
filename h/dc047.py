"""DC-047 A/B: can a Chat only (GGUF) Desktop user find and run the upgrade to the full install?

Usage: python dc047.py <phase>   phase in: probe, upgrade
Env: H_APP (exe), H_ART (artifact dir), H_ROOT (studio root), H_LABEL (A or B)

probe:   relaunch on the existing Chat only install and record every route a user has to training:
         the Train row (state, badge, hover text, what a click does), Settings > General, Settings
         search for the words such a user would type, and the Repair installation confirm text.
upgrade: relaunch, click Train, accept the Add training dialog, watch the progress screen, then
         record the verdict, the sidebar and the Train page once the backend is back.
"""
import json
import os
import re
import sys
import time
import traceback

PHASE = sys.argv[1]
LABEL = os.environ.get("H_LABEL", "X")
sys.argv = [sys.argv[0], f"{LABEL}-{PHASE}"]
import harness as H  # noqa: E402  (reads sys.argv[1] as its phase name for file names)

log, rec, shot, body_text, visible = H.log, H.rec, H.shot, H.body_text, H.visible


def train_row(page):
    return page.locator('[data-testid="nav-row-train"]').first


def train_state(page):
    row = train_row(page)
    if row.count() == 0:
        return {"present": False}
    return {
        "present": True,
        "text": re.sub(r"\s+", " ", row.inner_text()).strip(),
        "disabled": row.evaluate("b => b.disabled"),
        "opacity": row.evaluate("b => getComputedStyle(b).opacity"),
    }


def hover_train(page):
    row = train_row(page)
    row.scroll_into_view_if_needed(timeout=5000)
    # A disabled button sits inside a focusable span that owns the tooltip; hover the box either way.
    box = row.bounding_box()
    page.mouse.move(5, 5)
    time.sleep(0.4)
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, steps=6)
    time.sleep(2)
    tips = page.locator('[data-slot="tooltip-content"], [role="tooltip"]')
    texts = []
    for i in range(tips.count()):
        try:
            t = tips.nth(i).inner_text().strip()
            if t and t not in texts:
                texts.append(t)
        except Exception:
            pass
    shot(page, "train-hover")
    page.mouse.move(5, 5)
    time.sleep(0.5)
    return texts


def click_train(page):
    """Click Train like a user and report what changed."""
    url_before = page.url
    row = train_row(page)
    box = row.bounding_box()
    # A real pointer click at the row's centre, so a disabled button gets the same click a user gives it.
    page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    time.sleep(2.5)
    dialogs = page.get_by_role("dialog")
    dialog_text = None
    if dialogs.count():
        dialog_text = re.sub(r"\s+", " ", dialogs.last.inner_text()).strip()
    shot(page, "train-click")
    return {"url_before": url_before, "url_after": page.url, "dialog": dialog_text}


def close_dialogs(page):
    for _ in range(3):
        page.keyboard.press("Escape")
        time.sleep(0.4)


def settings_dialog(page):
    if not H.open_settings(page):
        return None
    return page.get_by_role("dialog").last


def settings_general_top(page):
    dlg = settings_dialog(page)
    if dlg is None:
        return {"opened": False}
    tab = dlg.locator('[data-testid="settings-tab-general"]').first
    if tab.count():
        tab.click()
        time.sleep(1.5)
    # The panel scrolls inside the dialog; start from its top, where the version rows are.
    page.evaluate("() => document.querySelectorAll('[role=dialog] *').forEach(e => { if (e.scrollHeight > e.clientHeight + 4) e.scrollTop = 0; })")
    time.sleep(0.8)
    shot(page, "settings-general-top")
    text = re.sub(r"\s+", " ", dlg.inner_text())
    add_row = dlg.locator('[data-settings-label="Install type"]')
    res = {
        "opened": True,
        "add_training_row": add_row.count() > 0,
        "add_training_row_text": re.sub(r"\s+", " ", add_row.first.inner_text()).strip() if add_row.count() else None,
        "general_head": text[:900],
    }
    close_dialogs(page)
    return res


def settings_search(page, queries):
    out = {}
    for q in queries:
        dlg = settings_dialog(page)
        if dlg is None:
            out[q] = "settings did not open"
            continue
        box = dlg.locator('input[aria-label^="Search settings"]').first
        box.click()
        box.fill(q)
        time.sleep(1.5)
        shot(page, f"settings-search-{q.replace(' ', '-')}")
        # The results list replaces the tab rail while a query is typed.
        out[q] = re.sub(r"\s+", " ", dlg.inner_text())[:400]
        close_dialogs(page)
    return out


def repair_confirm(page):
    dlg = settings_dialog(page)
    if dlg is None:
        return None
    tab = dlg.locator('[data-testid="settings-tab-general"]').first
    if tab.count():
        tab.click()
        time.sleep(1.5)
    row = dlg.locator('[data-settings-label="Repair installation"]').first
    row.scroll_into_view_if_needed()
    time.sleep(0.8)
    shot(page, "settings-repair-row")
    btn = row.get_by_role("button", name=re.compile(r"^Repair install$", re.I)).first
    btn.scroll_into_view_if_needed()
    btn.click()
    time.sleep(1.5)
    confirm = page.get_by_role("dialog").last
    text = re.sub(r"\s+", " ", confirm.inner_text()).strip()
    shot(page, "repair-confirm")
    cancel = confirm.get_by_role("button", name=re.compile(r"^Cancel$", re.I))
    if cancel.count():
        cancel.first.click()
    close_dialogs(page)
    return text


def step(name, fn):
    try:
        rec(name, fn())
    except Exception as e:
        log(f"{name} failed: {traceback.format_exc()[-1200:]}")
        rec(name, f"err {str(e)[:300]}")
        try:
            close_dialogs(H_PAGE[0])
        except Exception:
            pass


H_PAGE = [None]


def phase_probe(pw, root):
    proc, page, port = H.phase_relaunch(pw, root, f"{LABEL}-probe")
    H_PAGE[0] = page
    try:
        if page is None:
            return
        step("train_state", lambda: train_state(page))
        step("train_hover", lambda: hover_train(page))
        step("train_click", lambda: click_train(page))
        close_dialogs(page)
        step("settings_general", lambda: settings_general_top(page))
        step("settings_search", lambda: settings_search(page, ["train", "pytorch", "chat only", "add training"]))
        step("repair_confirm", lambda: repair_confirm(page))
    finally:
        if proc:
            H.kill_app(proc)


def phase_upgrade(pw, root):
    proc, page, port = H.phase_relaunch(pw, root, f"{LABEL}-pre-upgrade")
    H_PAGE[0] = page
    try:
        if page is None:
            rec("upgrade_started", False)
            return
        rec("train_state_before", train_state(page))
        click = click_train(page)
        rec("train_click", click)
        dlg = page.get_by_role("dialog").last
        btn = dlg.get_by_role("button", name=re.compile(r"^Upgrade$", re.I))
        if not btn.count():
            rec("upgrade_started", False)
            return
        btn.first.click()
        rec("upgrade_started", True)
        t = time.time()
        seen = []
        end = t + 60 * 45
        n = 0
        while time.time() < end:
            txt = body_text(page, 500)
            if not seen or seen[-1] != txt:
                seen.append(txt)
                log(f"progress screen: {txt}")
            if n in (1, 4, 12):
                shot(page, f"upgrade-progress-{n:03d}")
            n += 1
            if n > 3 and visible(page.locator('[data-testid="nav-row-train"]')):
                break
            time.sleep(5)
        rec("progress_screens", seen[:12])
        rec("upgrade_seconds", round(time.time() - t))
        time.sleep(15)
        port, _ = H.backend_port()
        if port:
            H.health_summary(port)
        shot(page, "after-upgrade")
        rec("train_state_after", train_state(page))
        rec("train_hover_after", hover_train(page))
        row = train_row(page)
        row.click()
        time.sleep(5)
        rec("train_page_after", {"url": page.url, "text": body_text(page, 600)})
        shot(page, "train-page-after-upgrade")
        step("settings_general_after", lambda: settings_general_top(page))
        step("settings_search_after", lambda: settings_search(page, ["train"]))
    finally:
        if proc:
            H.kill_app(proc)


def box(loc):
    try:
        if loc.count() == 0:
            return None
        b = loc.first.bounding_box()
        return {k: round(v) for k, v in b.items()} if b else None
    except Exception:
        return None


def snap(page, name, facts, **boxes):
    """A full-window shot plus the page-space boxes the annotator draws on."""
    page.screenshot(path=str(H.SHOTS / f"{name}.png"), timeout=20000)
    facts[name] = {k: box(v) if not isinstance(v, (dict, type(None))) else v for k, v in boxes.items()}
    rec("boxes", facts)


def phase_shots(pw, root):
    """Same views on both sides, same viewport, for the BEFORE | AFTER composites."""
    proc, page, port = H.phase_relaunch(pw, root, f"{LABEL}-shots")
    H_PAGE[0] = page
    facts = {}
    try:
        if page is None:
            return
        rec("locale", page.evaluate("() => ({nav: navigator.language, langs: navigator.languages, html: document.documentElement.lang})"))
        page.mouse.move(700, 600)
        time.sleep(1)
        row = train_row(page)
        # 01: the Train row as a user first meets it, hovered.
        b = row.bounding_box()
        page.mouse.move(b["x"] + b["width"] / 2, b["y"] + b["height"] / 2, steps=6)
        time.sleep(2)
        tip = page.locator('[data-slot="tooltip-content"]')
        snap(page, "01-train-row", facts, row=row, tooltip=tip)
        # 02: what a click on it does.
        page.mouse.click(b["x"] + b["width"] / 2, b["y"] + b["height"] / 2)
        time.sleep(2.5)
        page.mouse.move(700, 850)
        time.sleep(1)
        dlg = page.get_by_role("dialog")
        snap(page, "02-train-click", facts, row=row, dialog=dlg if dlg.count() else None)
        close_dialogs(page)
        # 03: Settings > General, top.
        sd = settings_dialog(page)
        if sd is not None:
            tab = sd.locator('[data-testid="settings-tab-general"]').first
            if tab.count():
                tab.click()
                time.sleep(1.5)
            page.evaluate("() => document.querySelectorAll('[role=dialog] *').forEach(e => { if (e.scrollHeight > e.clientHeight + 4) e.scrollTop = 0; })")
            time.sleep(0.8)
            page.mouse.move(5, 895)
            snap(page, "03-settings-general", facts, dialog=sd, training_row=sd.locator('[data-settings-label="Install type"]'))
            # 04: search for what such a user would type.
            search = sd.locator('input[aria-label^="Search settings"]').first
            search.click()
            search.fill("pytorch")
            time.sleep(1.5)
            snap(page, "04-settings-search", facts, dialog=sd, search=search)
            close_dialogs(page)
        # 05: the Repair installation confirm dialog.
        sd = settings_dialog(page)
        if sd is not None:
            tab = sd.locator('[data-testid="settings-tab-general"]').first
            if tab.count():
                tab.click()
                time.sleep(1.5)
            rrow = sd.locator('[data-settings-label="Repair installation"]').first
            rrow.scroll_into_view_if_needed()
            time.sleep(0.8)
            rrow.get_by_role("button", name=re.compile(r"^Repair install$", re.I)).first.click()
            time.sleep(1.5)
            confirm = page.get_by_role("dialog").last
            page.mouse.move(5, 895)
            snap(page, "05-repair-confirm", facts, dialog=confirm)
            close_dialogs(page)
        # 08: the Export page, whose Chat only hint names where to add PyTorch.
        H.nav_click(page, "More")
        time.sleep(1.2)
        item = page.get_by_role("menuitem", name=re.compile(r"^\s*Export", re.I))
        if item.count() == 0:
            item = page.locator("[data-radix-popper-content-wrapper]").get_by_text(re.compile(r"^\s*Export", re.I))
        item.first.click(timeout=5000)
        time.sleep(6)
        hint = page.get_by_text(re.compile(r"Export was left out of this install"))
        page.mouse.move(5, 895)
        time.sleep(1)
        rec("export_hint", hint.first.inner_text() if hint.count() else body_text(page, 600))
        snap(page, "08-export", facts, hint=hint)
    except Exception:
        log("shots failed: " + traceback.format_exc()[-1500:])
    finally:
        if proc:
            H.kill_app(proc)


def phase_upgrade_shots(pw, root):
    """Run the upgrade the way each side offers it and shoot the progress screen and the result."""
    proc, page, port = H.phase_relaunch(pw, root, f"{LABEL}-upgrade-shots")
    H_PAGE[0] = page
    facts = {}
    try:
        if page is None:
            return
        if LABEL == "A":
            # main: the only in-app route is Settings > General > Danger zone > Repair install.
            sd = settings_dialog(page)
            tab = sd.locator('[data-testid="settings-tab-general"]').first
            if tab.count():
                tab.click()
                time.sleep(1.5)
            rrow = sd.locator('[data-settings-label="Repair installation"]').first
            rrow.scroll_into_view_if_needed()
            rrow.get_by_role("button", name=re.compile(r"^Repair install$", re.I)).first.click()
            time.sleep(1.5)
            page.get_by_role("button", name=re.compile(r"^Repair now$", re.I)).first.click()
        else:
            # This PR: Settings > General > Install type > Upgrade to full install, then confirm.
            sd = settings_dialog(page)
            tab = sd.locator('[data-testid="settings-tab-general"]').first
            if tab.count():
                tab.click()
                time.sleep(1.5)
            irow = sd.locator('[data-settings-label="Install type"]').first
            irow.scroll_into_view_if_needed()
            irow.get_by_role("button", name=re.compile(r"^Upgrade to full install$", re.I)).first.click()
            time.sleep(1.5)
            page.get_by_role("dialog").last.get_by_role("button", name=re.compile(r"^Upgrade$", re.I)).first.click()
        t = time.time()
        shot_taken = False
        seen = []
        while time.time() < t + 60 * 45:
            txt = body_text(page, 400)
            if not seen or seen[-1] != txt:
                seen.append(txt)
                log(f"progress screen: {txt}")
            if not shot_taken and time.time() - t > 12 and "unsloth" in txt.lower() and not visible(page.locator('[data-testid="nav-row-train"]')):
                page.mouse.move(5, 5)
                snap(page, "06-progress", facts)
                shot_taken = True
            if time.time() - t > 20 and visible(page.locator('[data-testid="nav-row-train"]')):
                break
            time.sleep(3)
        rec("progress_screens", seen[:12])
        rec("upgrade_seconds", round(time.time() - t))
        # What the Train row says once the shell is back: main kept the pre-repair verdict.
        timeline = []
        t_back = time.time()
        while time.time() - t_back < 60:
            st = train_state(page)
            st["spinner"] = page.locator('[data-testid="nav-row-train"][data-spinner="true"]').count() > 0
            timeline.append({"s": round(time.time() - t_back, 1), **st})
            if not st.get("disabled") and "Add" not in st.get("text", "") and not st["spinner"]:
                break
            time.sleep(2)
        rec("train_after_restart", timeline)
        port, _ = H.backend_port()
        if port:
            H.health_summary(port)
        # 07: the Train row once the shell is back, hovered the same way on both sides. main's
        # repair started from Settings, and the dialog reopens with the shell: close it, so both
        # sides show the bare sidebar.
        close_dialogs(page)
        time.sleep(1)
        row = train_row(page)
        b = row.bounding_box()
        page.mouse.move(5, 5)
        time.sleep(0.5)
        page.mouse.move(b["x"] + b["width"] / 2, b["y"] + b["height"] / 2, steps=6)
        time.sleep(2)
        snap(page, "07-after-upgrade", facts, row=row, tooltip=page.locator('[data-slot="tooltip-content"]'))
        page.mouse.move(700, 850)
        if row.evaluate("b => b.disabled"):
            rec("train_still_disabled_after_60s", True)
            return
        row.click()
        time.sleep(5)
        page.mouse.move(700, 850)
        snap(page, "07b-train-page", facts, row=row)
        rec("train_page_after", {"url": page.url, "text": body_text(page, 300)})
    except Exception:
        log("upgrade shots failed: " + traceback.format_exc()[-1500:])
    finally:
        if proc:
            H.kill_app(proc)


def main():
    from playwright.sync_api import sync_playwright

    root = os.environ.get("H_ROOT", "")
    H.prepare_desktop()
    try:
        with sync_playwright() as pw:
            if PHASE == "probe":
                phase_probe(pw, root)
            elif PHASE == "upgrade":
                phase_upgrade(pw, root)
            elif PHASE == "shots":
                phase_shots(pw, root)
            elif PHASE == "upgrade_shots":
                phase_upgrade_shots(pw, root)
    except Exception:
        log("EXCEPTION " + traceback.format_exc())
        H.desktop_shot("exception")
        sys.exit(1)


if __name__ == "__main__":
    main()
