"""Shared helpers for the FastFlow/Lemonade user-perspective runs on DevLab."""
import json, os, pathlib, subprocess, sys, time, traceback

import requests

ROOT = pathlib.Path(os.environ["UX_ROOT"])
OUT = ROOT / "evidence"
OUT.mkdir(parents=True, exist_ok=True)
SHOTS = OUT / "shots"
SHOTS.mkdir(exist_ok=True)
WIN = sys.platform == "win32"
PORT = int(os.environ.get("UX_PORT", "8888"))
BASE = f"http://127.0.0.1:{PORT}"
RESULTS = []
NOTES = []
session = requests.Session()
page = None  # set by the UI driver
_shot_n = [0]


def note(text):
    """A UX observation worth reading even when the step passed."""
    row = {"t": round(time.time()), "note": text}
    NOTES.append(row)
    print("NOTE " + text, flush=True)
    (OUT / "notes.json").write_text(json.dumps(NOTES, indent=2), encoding="utf-8")


def shot(name, full=True):
    if page is None:
        return None
    _shot_n[0] += 1
    path = SHOTS / f"{_shot_n[0]:03d}-{name}.png"
    try:
        page.screenshot(path=str(path), full_page=full, timeout=20000)
        (SHOTS / f"{_shot_n[0]:03d}-{name}.txt").write_text(
            page.locator("body").inner_text(timeout=10000), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        print(f"screenshot {name} failed: {exc}", flush=True)
    return str(path)


def step(name, fn, *, shoot=True):
    t = time.monotonic()
    try:
        value = fn()
        row = {"test": name, "ok": True, "seconds": round(time.monotonic() - t, 2), "value": value}
    except Exception as exc:  # noqa: BLE001
        row = {"test": name, "ok": False, "seconds": round(time.monotonic() - t, 2),
               "error": str(exc)[:4000], "trace": traceback.format_exc()[-4000:]}
    print("RESULT " + json.dumps(row, default=str)[:3000], flush=True)
    RESULTS.append(row)
    (OUT / "results.json").write_text(json.dumps(RESULTS, indent=2, default=str), encoding="utf-8")
    if shoot:
        shot(name + ("" if row["ok"] else "-FAIL"))
    return row.get("value") if row["ok"] else None


def api(method, path, *, ok=(200,), timeout=600, **kw):
    r = session.request(method, BASE + path, timeout=timeout, **kw)
    body = None
    try:
        body = r.json()
    except ValueError:
        body = r.text[:4000]
    if ok and r.status_code not in ok:
        raise RuntimeError(f"{method} {path} -> {r.status_code}: {json.dumps(body)[:3000] if not isinstance(body, str) else body}")
    return {"status": r.status_code, "body": body}


def poll(fn, timeout=180, every=2):
    until = time.monotonic() + timeout
    last = None
    while time.monotonic() < until:
        try:
            last = fn()
            if last:
                return last
        except (requests.RequestException, ValueError) as exc:
            last = exc
        time.sleep(every)
    raise TimeoutError(f"timed out; last={last!r}"[:2000])


def run(cmd, *, timeout=600, env=None, name=None, input=None):
    """Run a user command, log it whole, return (rc, text)."""
    t = time.monotonic()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=timeout, env=env, input=input, shell=isinstance(cmd, str))
        rc, text = p.returncode, (p.stdout or "") + ("\n[stderr]\n" + p.stderr if p.stderr else "")
    except subprocess.TimeoutExpired as exc:
        rc = "timeout"
        text = ((exc.stdout or b"").decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")) + \
            "\n[stderr]\n" + ((exc.stderr or b"").decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or ""))
    name = name or "cmd"
    (OUT / "cmd").mkdir(exist_ok=True)
    (OUT / "cmd" / f"{name}.log").write_text(f"$ {cmd}\nrc={rc} seconds={time.monotonic()-t:.1f}\n\n{text}", encoding="utf-8")
    return rc, text
