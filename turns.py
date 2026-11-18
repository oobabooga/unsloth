"""One long session, one turn per workflow step: restore the liked take, then follow-ups steered from the Mac.

argv: turn K. Turn 1 restores the conversation and sends the gap fix on its own; every later turn waits for a
command in pick.txt on this branch ("<seq> send <text>" | "<seq> fresh" | "<seq> reshoot <dx>" | "<seq> final").
State lives in $STATE between steps. Every turn writes tK-* frames, tK-reply.md and tK-summary.json.
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from shoot import *  # noqa: F401,F403
import shoot

STATE = Path(os.environ["STATE"])
REPO, BRANCH = os.environ["REPO"], os.environ["BRANCH"]
ORIGINAL_PROMPT = Path(os.environ["ORIGINAL_PROMPT"]).read_text()
RESTORED_HTML = Path(os.environ["RESTORED_HTML"]).read_text()
TITLE = (
    "using html with embedded JavaScript with 3js make a voxel art model of a medieval kingdom, please dont just "
    "make it out…"
)
GAP_FIX = (
    "The cubes have gaps between them. Make every voxel fill its grid cell so the blocks touch and the kingdom "
    "looks solid."
)


def state():
    return json.loads(STATE.read_text()) if STATE.exists() else {"seq": 0}


def save(st):
    STATE.write_text(json.dumps(st, indent = 1))


def wait_command(after_seq, minutes = 150):
    """Poll the branch head (git, no API quota) and read pick.txt at that exact commit."""
    url = f"https://github.com/{REPO}.git"
    seen = None
    end = time.time() + minutes * 60
    while time.time() < end:
        try:
            out = subprocess.run(["git", "ls-remote", url, f"refs/heads/{BRANCH}"], capture_output = True,
                                 text = True, timeout = 60).stdout.split()
            sha = out[0] if out else None
            if sha and sha != seen:
                seen = sha
                raw = urllib.request.urlopen(f"https://raw.githubusercontent.com/{REPO}/{sha}/pick.txt",
                                             timeout = 30).read().decode().strip()
                seq, _, cmd = raw.partition(" ")
                if seq.isdigit() and int(seq) > after_seq:
                    log("COMMAND", seq, cmd[:300])
                    return int(seq), cmd
        except Exception as exc:
            log("poll", type(exc).__name__, str(exc)[:120])
        time.sleep(45)
    return None, None


def messages(tid):
    return call("GET", f"/api/chat/threads/{tid}/messages", token = TOKEN)["messages"]


def text_of(msg):
    parts = msg.get("content") or []
    if isinstance(parts, str):
        return parts
    return "\n".join(p.get("text", "") for p in parts if isinstance(p, dict) and p.get("type") == "text")


def badges(page):
    return page.get_by_text("tok/s").count()


def send(page, text, tag):
    before = badges(page)
    box = page.get_by_placeholder("Ask anything")
    box.click()
    page.keyboard.type(text, delay = 5)
    page.wait_for_timeout(400)
    page.keyboard.press("Enter")
    log(tag, "sent:", text[:160])
    t0 = time.time()
    while time.time() - t0 < 2400:
        page.wait_for_timeout(5000)
        if badges(page) > before:
            break
    else:
        raise RuntimeError("no reply in 40 minutes")
    log(tag, f"reply in {time.time() - t0:.0f}s, badge", page.get_by_text("tok/s").last.inner_text())
    page.wait_for_timeout(1500)


def open_thread(page, tid):
    page.goto(f"{BASE}/chat?thread={tid}")
    page.wait_for_timeout(4000)
    dismiss(page)
    collapse_sidebar(page)
    scroll_chat_end(page)


def ensure_model(page):
    st = call("GET", "/api/npu/status", token = TOKEN)
    if st.get("loaded_model") != f"lemonade:{MODEL}":
        log("model not loaded, loading")
        load_model(page)


def restore(page, st):
    """A real chat made through the UI, then its two messages rewritten to the liked take's turn."""
    page.goto(f"{BASE}/chat")
    page.wait_for_timeout(3500)
    dismiss(page)
    before = {t["id"] for t in threads()}
    send(page, "hi", "restore")
    new = [t for t in threads() if t["id"] not in before]
    tid = new[0]["id"]
    msgs = messages(tid)
    (OUT / "t01-seed-messages.json").write_text(json.dumps(msgs, indent = 1)[:200000])
    user = next(m for m in msgs if m["role"] == "user")
    bot = next(m for m in msgs if m["role"] == "assistant")
    user = dict(user, content = [{"type": "text", "text": ORIGINAL_PROMPT}])
    meta = {k: v for k, v in (bot.get("metadata") or {}).items()
            if not any(w in k.lower() for w in ("time", "token", "usage", "speed", "stat", "perf", "custom"))}
    bot = dict(bot, content = [{"type": "text", "text": "```html\n" + RESTORED_HTML.rstrip() + "\n```"}],
               metadata = meta or None)
    call("PUT", f"/api/chat/threads/{tid}/messages", {"messages": [user, bot], "pruneMissing": True}, TOKEN)
    thread = call("GET", f"/api/chat/threads/{tid}", token = TOKEN)
    thread["title"] = TITLE
    call("POST", "/api/chat/threads", thread, TOKEN)
    log("restored into", tid, "user", user["id"], "assistant", bot["id"], "kept metadata keys", list(meta))
    st["tid"] = tid
    save(st)
    open_thread(page, tid)
    shot(page, "t01-restored-chat")
    return tid


def judge_and_shoot(page, tag, st):
    """Open the newest preview, check it, frame it, write the frame set and the reply."""
    tid = st["tid"]
    msgs = messages(tid)
    last = [m for m in msgs if m["role"] == "assistant"][-1]
    (OUT / f"{tag}-reply.md").write_text(text_of(last))
    scroll_chat_end(page)
    r = present(page, tag)
    summary = {"tag": tag, "thread": tid, "result": r}
    if r and r.get("ok"):
        view, m = pick_view(page, tag)
        st["view"] = view
        scroll_chat_end(page)
        park_mouse(page)
        shot(page, f"{tag}-1-chat-and-kingdom")
        page.locator("iframe").last.screenshot(path = str(OUT / f"{tag}-2-closeup.png"), timeout = 180000)
        drag(page, 45, 0)
        frame_scene(page, f"{tag}-b")
        scroll_chat_end(page)
        park_mouse(page)
        shot(page, f"{tag}-1b-alt")
        page.locator("iframe").last.screenshot(path = str(OUT / f"{tag}-2b-closeup-alt.png"), timeout = 180000)
        summary["view"] = view
    else:
        shot(page, f"{tag}-rejected")
        try:
            page.locator("iframe").last.screenshot(path = str(OUT / f"{tag}-rejected-canvas.png"), timeout = 60000)
        except Exception:
            pass
    (OUT / f"{tag}-summary.json").write_text(json.dumps(summary, indent = 1))
    st.setdefault("turns", []).append(summary)
    save(st)
    log(tag, "SUMMARY", json.dumps(summary)[:400])


def final(page, st):
    tid = st["tid"]
    delete_threads([t["id"] for t in threads() if t["id"] != tid])
    open_thread(page, tid)
    r = present(page, "final")
    log("final present", json.dumps(r)[:300])
    drag(page, st.get("view", 0), 0)
    frame_scene(page, "final")
    scroll_chat_end(page)
    park_mouse(page)
    shot(page, "1-final-chat-and-kingdom")
    page.locator("iframe").last.screenshot(path = str(OUT / "2-canvas-closeup.png"), timeout = 180000)
    drag(page, 45, 0)
    frame_scene(page, "final-b")
    scroll_chat_end(page)
    park_mouse(page)
    shot(page, "1b-final-alt-angle")
    page.locator("iframe").last.screenshot(path = str(OUT / "2b-canvas-closeup-alt-angle.png"), timeout = 180000)
    if open_picker(page):
        name, _ = find_row(page)
        log("PICKER_LABEL_FINAL", name)
        page.wait_for_timeout(800)
        shot(page, "3-picker-open-new-name")
        page.keyboard.press("Escape")
        page.wait_for_timeout(600)
    expand_sidebar(page)
    scroll_chat_end(page)
    park_mouse(page)
    shot(page, "4-sidebar-recents-one-chat")
    log("RECENTS", json.dumps([[t["id"], t.get("title")] for t in threads()]))


def run(k):
    from playwright.sync_api import sync_playwright

    st = state()
    tag = f"t{k:02d}"
    if k == 1:
        seq, cmd = 0, "restore"
    else:
        if st.get("done"):
            log("already done")
            return
        seq, cmd = wait_command(st["seq"])
        if seq is None:
            log("no command in time; ending")
            st["done"] = True
            save(st)
            return
        st["seq"] = seq
        save(st)
    with sync_playwright() as p:
        browser, kw, software = pick_browser(p)
        ctx = browser.new_context(viewport = {"width": W, "height": H}, device_scale_factor = 1 if software else 2,
                                  color_scheme = "light")
        ctx.add_init_script(seed())
        page = ctx.new_page()
        page.on("pageerror", lambda e: (shoot.ERRORS.append(str(e)), log("pageerror", str(e)[:300])))
        try:
            if k == 1:
                try:
                    network_setting(page)
                except Exception as exc:
                    log("settings step failed", str(exc)[:200])
                delete_threads([t["id"] for t in threads()])
                load_model(page)
                restore(page, st)
                # a new context, so nothing cached from the seed "hi" turn can show
                ctx.close()
                ctx = browser.new_context(viewport = {"width": W, "height": H},
                                          device_scale_factor = 1 if software else 2, color_scheme = "light")
                ctx.add_init_script(seed())
                page = ctx.new_page()
                page.on("pageerror", lambda e: (shoot.ERRORS.append(str(e)), log("pageerror", str(e)[:300])))
                open_thread(page, st["tid"])
                shot(page, "t01-restored-chat-reloaded")
                judge_and_shoot(page, "t01-restored", st)
                open_thread(page, st["tid"])
                send(page, GAP_FIX, tag)
                judge_and_shoot(page, tag, st)
            elif cmd.startswith("send "):
                ensure_model(page)
                open_thread(page, st["tid"])
                send(page, cmd[5:], tag)
                judge_and_shoot(page, tag, st)
            elif cmd.startswith("fresh"):
                ensure_model(page)
                page.goto(f"{BASE}/chat")
                page.wait_for_timeout(3500)
                dismiss(page)
                collapse_sidebar(page)
                before = {t["id"] for t in threads()}
                send(page, cmd[6:].strip() or ORIGINAL_PROMPT, tag)
                st["tid"] = [t for t in threads() if t["id"] not in before][0]["id"]
                save(st)
                judge_and_shoot(page, tag, st)
            elif cmd.startswith("reshoot"):
                open_thread(page, st["tid"])
                judge_and_shoot(page, tag, st)
            elif cmd.startswith("thread "):
                st["tid"] = cmd[7:].strip()
                save(st)
                open_thread(page, st["tid"])
                judge_and_shoot(page, tag, st)
            elif cmd.startswith("final"):
                final(page, st)
                st["done"] = True
                save(st)
            else:
                log("unknown command", cmd)
        except Exception as exc:
            log(tag, "FAILED", type(exc).__name__, str(exc)[:500])
            shot(page, f"{tag}-failure")
        finally:
            ctx.close()
            browser.close()


if __name__ == "__main__":
    run(int(sys.argv[2]))
