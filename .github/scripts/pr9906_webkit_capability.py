"""WebKit capability check for unslothai/unsloth#9906.

The PR's fix rests on three browser behaviours:
  1. a read-only <input> can be focused with focus({preventScroll: true});
  2. select() on it selects the entire value;
  3. its text is selectable by hand, which is what the old <button><code> wrapper
     prevented and is the whole point of the change.

Chromium and Firefox were verified against a live Studio on Linux. WebKit could not
be launched there (the bundled GTK libraries need root to install), so it is checked
here on macOS, where WebKit is the engine Safari and the Tauri WKWebView both use.

This exercises the engine, not the whole app: the markup and the mount effect are
reproduced verbatim from key-reveal-card.tsx.
"""

import sys

from playwright.sync_api import sync_playwright

TOKEN = "sk-unsloth-0123456789abcdef0123456789abcdef"

# Same shape the component renders: a read-only input carrying the token, focused and
# selected on mount, re-selecting on focus.
PAGE = """
<!doctype html><html><body>
<div id="wrap" style="display:flex;gap:8px;padding:8px;width:560px">
  <input id="tok" readonly value="%s" aria-label="New access token created"
         data-reload-snapshot-sensitive
         style="flex:1;font-family:monospace;font-size:14px;border:0;background:transparent">
  <button id="copy" aria-label="Copy access token">c</button>
</div>
<script>
  var input = document.getElementById('tok');
  input.addEventListener('focus', function () { input.select(); });
  input.focus({ preventScroll: true });
  input.select();
</script>
</body></html>
""" % TOKEN

FAIL = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ((" -- " + detail) if detail else ""))
    if not ok:
        FAIL.append(name)


with sync_playwright() as p:
    browser = p.webkit.launch()
    page = browser.new_context(viewport={"width": 1280, "height": 720}).new_page()
    page.set_content(PAGE)
    page.wait_for_timeout(500)

    st = page.evaluate(
        """() => {
          const f = document.getElementById('tok');
          return {active: f === document.activeElement, s: f.selectionStart,
                  e: f.selectionEnd, len: f.value.length,
                  userSelect: getComputedStyle(f).webkitUserSelect || getComputedStyle(f).userSelect};
        }"""
    )
    print("  engine: webkit", page.evaluate("() => navigator.userAgent"))
    check("focus({preventScroll}) focuses the read-only input", st["active"] is True)
    check("select() selects the whole token",
          st["s"] == 0 and st["e"] == st["len"] == len(TOKEN),
          "sel=%s..%s len=%s" % (st["s"], st["e"], st["len"]))
    check("input is not user-select:none", st["userSelect"] != "none", str(st["userSelect"]))

    # Manual drag selection: 0 characters is what the old button-wrapped <code> gave.
    box = page.evaluate(
        """() => { const r = document.getElementById('tok').getBoundingClientRect();
                   return {x: r.x, y: r.y, w: r.width, h: r.height}; }"""
    )
    page.evaluate("() => { const f = document.getElementById('tok'); f.blur(); }")
    page.mouse.move(box["x"] + 4, box["y"] + box["h"] / 2)
    page.mouse.down()
    page.mouse.move(box["x"] + box["w"] - 4, box["y"] + box["h"] / 2, steps=15)
    page.mouse.up()
    page.wait_for_timeout(250)
    drag = page.evaluate(
        """() => { const f = document.getElementById('tok');
                   return f.value.slice(f.selectionStart, f.selectionEnd); }"""
    )
    check("the token is selectable by hand in WebKit", len(drag) > 0,
          "%d chars selected" % len(drag))

    # The control the PR removes: text inside a <button> is what could not be selected.
    page.set_content(
        "<button id=b style='width:560px'><code id=c>%s</code></button>" % TOKEN
    )
    cbox = page.evaluate(
        """() => { const r = document.getElementById('c').getBoundingClientRect();
                   return {x: r.x, y: r.y, w: r.width, h: r.height}; }"""
    )
    page.mouse.move(cbox["x"] + 2, cbox["y"] + cbox["h"] / 2)
    page.mouse.down()
    page.mouse.move(cbox["x"] + cbox["w"] - 2, cbox["y"] + cbox["h"] / 2, steps=15)
    page.mouse.up()
    page.wait_for_timeout(250)
    old = page.evaluate("() => String(window.getSelection())")
    print("  control: chars selectable inside the old button-wrapped code: %d" % len(old))

    browser.close()

print("\n%s" % ("WEBKIT CAPABILITY OK" if not FAIL else "FAILURES: %s" % FAIL))
sys.exit(1 if FAIL else 0)
