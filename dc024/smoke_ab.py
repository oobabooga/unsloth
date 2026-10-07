"""A/B the dark dropdown glow probe on the real heavy-thread smoke page.

Arm A runs main's lib/dropdown-surround.ts, arm B the branch's, both transpiled to plain
modules and served at /dc024/<arm>.js by route interception. Each arm opens the composer's
real menus (Radix, modal and non-modal) in dark mode over a seeded heavy thread, and records
frame gaps after the click, the longest rAF callback, and the glow it applied.

usage: smoke_ab.py <frontend_dir> <old.js> <new.js> <out.json> <engine[,engine]> [sizes] [reps]
"""
import json, os, statistics, sys
from pathlib import Path

FRONTEND = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(FRONTEND.parent.parent / "tests" / "studio"))
from playwright.sync_api import sync_playwright  # noqa: E402
from _playwright_robust import chromium_launch_args, start_vite, stop_process, wait_for_smoke_page  # noqa: E402

MODULES = {"A": Path(sys.argv[2]).read_text(), "B": Path(sys.argv[3]).read_text()}
OUT = Path(sys.argv[4])
ENGINES = sys.argv[5].split(",")
SIZES = [int(s) for s in (sys.argv[6] if len(sys.argv) > 6 else "100000,300000").split(",")]
REPS = int(sys.argv[7]) if len(sys.argv) > 7 else 7
PORT = int(os.environ.get("PW_PORT", "5473"))
BASE = f"http://127.0.0.1:{PORT}"

INIT = """
(() => {
  try { localStorage.setItem('theme', 'dark'); } catch (e) {}
  const raf = window.requestAnimationFrame.bind(window);
  window.__frames = [];
  const loop = () => { window.__frames.push(performance.now()); raf(loop); };
  raf(loop);
  window.__rafCost = [];
  window.requestAnimationFrame = (cb) => raf((t) => {
    const s = performance.now();
    try { cb(t); } finally {
      const d = performance.now() - s;
      if (d > 0.5) window.__rafCost.push([s, d]);
    }
  });
  window.__beats = [];
  let lastBeat = performance.now();
  setInterval(() => {
    const now = performance.now();
    if (now - lastBeat > 50) window.__beats.push([lastBeat, now - lastBeat]);
    lastBeat = now;
  }, 10);
  window.__t0 = 0;
  addEventListener('pointerdown', () => { window.__t0 = performance.now(); }, true);
})();
"""
MENUS = [
    ("composer + (modal)", '[aria-label="Tools and attachments"]'),
    ("composer permission (modal)", '[aria-label="Permission level for tool calls"]'),
    ("message More (non-modal)", '[data-slot="tooltip-trigger"][aria-haspopup="menu"]'),
]
COLLECT = """(t0) => {
  const frames = window.__frames.filter((f) => f >= t0 && f <= t0 + 2500);
  const gaps = []; let prev = t0;
  for (const f of frames) { gaps.push(f - prev); prev = f; }
  gaps.push(Math.max(0, t0 + 2500 - prev));
  const raf = window.__rafCost.filter(([s]) => s >= t0 && s <= t0 + 2500).map(([, d]) => d);
  const open = [...document.querySelectorAll('[data-state="open"]')].filter((el) => el.matches('[data-slot="dropdown-menu-content"],[data-slot="popover-content"],[data-slot="select-content"]'));
  const menu = open[open.length - 1];
  return {
    frozen: gaps.reduce((a, g) => a + Math.max(0, g - 16.7), 0),
    long: gaps.filter((g) => g > 50).length,
    maxGap: Math.max(0, ...gaps),
    frames: frames.length,
    blocked: window.__beats.filter(([s]) => s >= t0 - 50 && s <= t0 + 2500)
      .reduce((a, [, d]) => a + d - 10, 0),
    rafMax: Math.max(0, ...raf),
    modal: document.body.style.pointerEvents === 'none',
    surround: menu ? menu.style.getPropertyValue('--dropdown-surround-bg') : null,
    blend: menu ? menu.hasAttribute('data-surround-blend') : null,
    elements: document.getElementsByTagName('*').length,
  };
}"""


HEADLESS = os.environ.get("PW_HEADED") != "1"


def launch(p, engine):
    if engine == "chromium":
        return p.chromium.launch(headless=HEADLESS, args=chromium_launch_args())
    if engine == "msedge":
        return p.chromium.launch(headless=HEADLESS, channel="msedge", args=chromium_launch_args())
    return getattr(p, engine).launch(headless=HEADLESS)


def open_arm(browser, arm, size):
    ctx = browser.new_context(viewport={"width": 1440, "height": 900})
    ctx.add_init_script(INIT)
    ctx.route("**/dc024/*.js", lambda route: route.fulfill(
        status=200, content_type="application/javascript", body=MODULES[route.request.url.rsplit("/", 1)[1][0]]))
    page = ctx.new_page()
    page.goto(f"{BASE}/smoke-heavy-thread.html")
    page.wait_for_function("() => window.__heavyThread", timeout=180000)
    page.evaluate("(n) => window.__heavyThread.seed(n)", size)
    page.wait_for_timeout(8000)
    page.evaluate("""async (arm) => {
      document.documentElement.classList.add('dark');
      const m = await import(`/dc024/${arm}.js`);
      m.watchDropdownSurround(window);
    }""", arm)
    page.wait_for_timeout(1500)
    return page


def measure(page, sel):
    page.bring_to_front()
    loc = page.locator(sel).first
    loc.wait_for(state="visible", timeout=15000)
    page.wait_for_timeout(400)
    loc.click(force=True)
    page.wait_for_timeout(2700)
    row = page.evaluate(COLLECT, page.evaluate("() => window.__t0"))
    page.keyboard.press("Escape")
    page.wait_for_timeout(600)
    return row


results = {}
proc = start_vite(PORT)
try:
    wait_for_smoke_page(f"{BASE}/smoke-heavy-thread.html", "/smoke-heavy-thread-main.tsx", proc=proc)
    with sync_playwright() as p:
        for engine in ENGINES:
            for k, size in enumerate(SIZES):
                rows = {name: {"A": [], "B": []} for name, _ in MENUS}
                # One browser per arm, so neither page is a throttled background tab.
                for arm in (("A", "B") if k % 2 == 0 else ("B", "A")):
                    browser = launch(p, engine)
                    page = open_arm(browser, arm, size)
                    for name, sel in MENUS:
                        try:
                            for rep in range(REPS + 1):
                                r = measure(page, sel)
                                if rep:
                                    rows[name][arm].append(r)
                        except Exception as exc:
                            print(f"{engine} {size} {name} {arm}: skipped ({str(exc).splitlines()[0]})", flush=True)
                    browser.close()
                for name, _ in MENUS:
                    if not (rows[name]["A"] and rows[name]["B"]):
                        continue
                    cell = {}
                    for arm, rs in rows[name].items():
                        cell[arm] = {
                            "frozen_med": round(statistics.median(r["frozen"] for r in rs), 1),
                            "long_med": statistics.median(r["long"] for r in rs),
                            "maxGap_med": round(statistics.median(r["maxGap"] for r in rs), 1),
                            "rafMax_med": round(statistics.median(r["rafMax"] for r in rs), 1),
                            "frames_med": statistics.median(r["frames"] for r in rs),
                            "blocked_med": round(statistics.median(r["blocked"] for r in rs), 1),
                            "modal": rs[-1]["modal"],
                            "surround": sorted({str(r["surround"]) for r in rs}),
                            "blend": sorted({str(r["blend"]) for r in rs}),
                            "elements": rs[-1]["elements"],
                        }
                    results[f"{engine}|{size}|{name}"] = cell
                    print(engine, size, name, json.dumps(cell), flush=True)
finally:
    stop_process(proc)
OUT.write_text(json.dumps(results, indent=1))
