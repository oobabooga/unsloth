"""Drive the real desktop app through WebView2's DevTools port: click Get Started, wait for the
setup screen's error, print it and save a screenshot."""

import base64
import json
import sys
import time
import urllib.request

import websocket

PORT = 9222
OUT = sys.argv[1]


def targets():
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json", timeout=5) as r:
        return json.load(r)


deadline = time.time() + 180
page = None
while time.time() < deadline:
    try:
        pages = [t for t in targets() if t.get("type") == "page"]
        loaded = [t for t in pages if t.get("url", "").startswith("http")]
        if loaded or (pages and time.time() > deadline - 120):
            page = (loaded or pages)[0]
            break
    except Exception as e:
        last = e
    time.sleep(2)
if not page:
    sys.exit("no WebView2 page on the DevTools port")
print("page:", page.get("url"))

ws = websocket.create_connection(page["webSocketDebuggerUrl"], timeout=30, suppress_origin=True)
seq = 0


def call(method, **params):
    global seq
    seq += 1
    ws.send(json.dumps({"id": seq, "method": method, "params": params}))
    while True:
        msg = json.loads(ws.recv())
        if msg.get("id") == seq:
            return msg.get("result", {})


def js(expr):
    res = call("Runtime.evaluate", expression=expr, returnByValue=True)
    return res.get("result", {}).get("value")


def body():
    return js("document.body ? document.body.innerText : ''") or ""


def wait_for(text, seconds):
    end = time.time() + seconds
    while time.time() < end:
        if text in body():
            return True
        time.sleep(1)
    return False


def shot(name):
    data = call("Page.captureScreenshot", format="png").get("data")
    if data:
        with open(f"{OUT}/{name}.png", "wb") as f:
            f.write(base64.b64decode(data))


if not wait_for("Get Started", 180):
    print(body())
    shot("no-get-started")
    sys.exit("the not-installed screen never appeared")
shot("1-not-installed")
clicked = js(
    "(() => { const b = [...document.querySelectorAll('button')]"
    ".find(b => b.innerText.trim() === 'Get Started'); if (!b) return false; b.click(); return true; })()"
)
print("clicked Get Started:", clicked)
if not wait_for("Setup ran into a problem", 300):
    print(body())
    shot("no-error")
    sys.exit("the setup error screen never appeared")
time.sleep(1)
shot("2-setup-error")
error = js(
    "(() => { const h = [...document.querySelectorAll('p')]"
    ".find(p => p.innerText.trim() === 'Setup ran into a problem');"
    " return h && h.nextElementSibling ? h.nextElementSibling.innerText : null; })()"
)
print("SETUP ERROR SHOWN TO THE USER:")
print(error)
with open(f"{OUT}/setup-error.txt", "w", encoding="utf-8") as f:
    f.write(error or "")
