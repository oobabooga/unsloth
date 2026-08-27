// Offline smoke test for r9695_scene.js against a hand-rolled DOM stub.
//
// A bug in this scene is not cheap: it costs an exclusive slot on a shared GPU runner, and the
// failure is only visible once the job has already completed. Nothing here measures anything
// real. The point is that the scene RUNS TO COMPLETION, posts a payload with the shape
// criteria/studio_r9695_rebase.py reads, and that its checks can return BOTH answers rather than
// always the convenient one.
//
// REAL TIMERS, NOT A VIRTUAL CLOCK. `performance.now` is the real one and rAF is driven off a
// real `setInterval` at 8 ms -- the cadence rAF actually has under the headless X server this
// scene runs on, where it is not vsync locked -- because the one thing this scene is FOR is
// proving that its frame channel can report a blocked main thread. The hog spins synchronously inside the same event
// loop, so it really does starve the rAF driver and `liveness.drop_fraction` is a measurement
// here rather than an arithmetic identity. Under a virtual clock the jam scenario would pass on a
// scene that had lost its positive control.
//
// Six scenarios, and the ones this whole run turns on are 2, 3 and 5:
//
//   1. HAPPY PATH at a big rung: three phases in order, every window scored, all four actions
//      ran, the clamp calibrated, mount attributed to the seeded marker
//   2. THE JAM RESOLVES: effective fps over wall time falls under the scene's own positive
//      control, and the hog is turned off again afterwards
//   3. THE BLIND CHANNEL IS SHOWN TO BE BLIND: 1000/p50 barely moves on the SAME rAF series, so
//      the report can show what would have been concluded from it
//   4. THE SCORED GESTURE IS THE ALL-PANES ONE: every reasoning pane opens and closes, and the
//      census taken while they are OPEN differs from the ones taken around it
//   5. THE FIDELITY CENSUS RETURNS BOTH ANSWERS: a `main`-like page and a `head`-like page must
//      be distinguishable on the reasoning-scoped span count and the deferred shells, because on
//      a real runner that census is the only in-page evidence of which build was measured
//   6. A FAILURE IS REPORTED AS ONE: ok false, an error naming the timeout, and phase/marks so
//      it is attributable, with NO phases so nothing from it can be scored by accident
//
// Run: node selftest_r9695_scene.mjs   (about 40 s, almost all of it real waiting)

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SRC = fs.readFileSync(path.join(HERE, "r9695_scene.js"), "utf8");
const T0 = Date.now();

let failures = 0;
const check = (label, cond, detail = "") => {
  if (cond) console.log(`  ok   ${label}`);
  else { console.log(`  FAIL ${label}${detail ? " -- " + detail : ""}`); failures++; }
};
const num = (v) => typeof v === "number" && Number.isFinite(v);

// The seeded thread's tail. The scene will not call a thread mounted until this string is in
// document.body.innerText, so it has to be real text inside the last message.
const MARKER = "SEEDED_TAIL_MARKER_0042";
const ANCHOR_PROBE = "anchor-name: --unsloth-probe";

// ── the stub ─────────────────────────────────────────────────────────────────────────────────

class El {
  constructor(tag, cls = "", attrs = {}, text = "") {
    this.tagName = tag.toUpperCase();
    this.nodeType = 1;
    this.className = cls;
    this.attrs = attrs;
    this.children = [];
    this.parentElement = null;
    this._text = text;
    // Real elements have one, and the scene writes an inline scroll-behavior onto the scroller.
    // Without it the assignment throws, the failure is swallowed into a note, and the guard would
    // report success for the wrong reason.
    this.style = {};
    this._listeners = {};
    this._onclick = null;
  }
  get textContent() { return this._text + this.children.map((c) => c.textContent).join(""); }
  set textContent(v) { this._text = String(v); this.children = []; }
  // `document.body.innerText` is how the scene decides the thread is mounted, so it has to be
  // derived from the tree rather than pinned to a constant: an empty rung must NOT contain the
  // marker of a seeded one.
  get innerText() { return this.textContent; }
  appendChild(c) { c.parentElement = this; this.children.push(c); return c; }
  remove() {
    if (!this.parentElement) return;
    const i = this.parentElement.children.indexOf(this);
    if (i >= 0) this.parentElement.children.splice(i, 1);
    this.parentElement = null;
  }
  hasAttribute(k) { return k in this.attrs; }
  getAttribute(k) { return (k in this.attrs) ? this.attrs[k] : null; }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  removeAttribute(k) { delete this.attrs[k]; }
  querySelector(s) { return queryAll(this, s)[0] || null; }
  querySelectorAll(s) { return queryAll(this, s); }
  getElementsByTagName(t) {
    const d = descendants(this);
    return t === "*" ? d : d.filter((e) => e.tagName === t.toUpperCase());
  }
  getBoundingClientRect() {
    const r = this._rect || { left: 0, top: 0, width: 0, height: 0 };
    return { left: r.left, top: r.top, width: r.width, height: r.height,
             right: r.left + r.width, bottom: r.top + r.height };
  }
  addEventListener(t, fn) { (this._listeners[t] = this._listeners[t] || []).push(fn); }
  dispatchEvent(ev) {
    for (const fn of this._listeners[ev.type] || []) fn(ev);
    return true;
  }
  focus() { this._focused = true; }
  click() { if (this._onclick) this._onclick(); }
}

// The composer is reached through `Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype,
// "value").set`, which is how the scene writes into a React-controlled textarea. So the accessor
// has to live on a PROTOTYPE and has to really store, or `fill` spins for ten seconds and the
// send never happens.
class HTMLTextAreaElement extends El {
  constructor(cls, attrs) { super("textarea", cls, attrs); this._value = ""; }
  get value() { return this._value; }
  set value(v) { this._value = String(v); }
}

const classesOf = (el) => String(el.className || "").split(" ").filter(Boolean);
const descendants = (root, out = []) => {
  for (const c of root.children) { out.push(c); descendants(c, out); }
  return out;
};

// Compound simple selectors, because the scene asks for `textarea[aria-label="Message input"]`
// and `[data-slot="reasoning-root"][data-state="open"]`: a matcher that only understood one token
// would silently report zero open reasoning panes in every census.
const SIMPLE = /^(?:\*|[a-zA-Z][\w-]*|\.[\w-]+|\[[^\]]+\])/;
const matchCompound = (el, sel) => {
  let rest = sel.trim();
  if (!rest) return false;
  while (rest) {
    const m = SIMPLE.exec(rest);
    if (!m) return false;
    const tok = m[0];
    rest = rest.slice(tok.length);
    if (tok === "*") continue;
    if (tok[0] === ".") {
      if (!classesOf(el).includes(tok.slice(1))) return false;
    } else if (tok[0] === "[") {
      const inner = tok.slice(1, -1);
      const eq = inner.indexOf("=");
      if (eq < 0) {
        if (!(inner in el.attrs)) return false;
      } else {
        const k = inner.slice(0, eq);
        const v = inner.slice(eq + 1).replace(/^["']|["']$/g, "");
        if (el.attrs[k] !== v) return false;
      }
    } else if (el.tagName !== tok.toUpperCase()) return false;
  }
  return true;
};

// A descendant chain is split on whitespace OUTSIDE brackets and quotes. Every selector this
// scene aims at the composer and the send button carries a space inside its attribute value
// (`textarea[aria-label="Message input"]`), so a naive split on /\s+/ finds neither, the scene
// waits out its mount timeout, and the self-test reports a scene bug that is really a stub bug.
const splitChain = (part) => {
  const out = [];
  let cur = "", depth = 0, quote = null;
  for (const ch of part) {
    if (quote) { cur += ch; if (ch === quote) quote = null; continue; }
    if (ch === '"' || ch === "'") { quote = ch; cur += ch; continue; }
    if (ch === "[") { depth += 1; cur += ch; continue; }
    if (ch === "]") { depth -= 1; cur += ch; continue; }
    if (/\s/.test(ch) && depth === 0) { if (cur) out.push(cur); cur = ""; continue; }
    cur += ch;
  }
  if (cur) out.push(cur);
  return out;
};

// Comma lists and descendant chains: `pre span` is the count the fence half of the readback is
// read from, so the chain has to be real rather than approximated by the last compound.
const queryAll = (root, sel) => {
  const out = [];
  const seen = new Set();
  for (const part of String(sel).split(",").map((s) => s.trim()).filter(Boolean)) {
    const chain = splitChain(part);
    let cur = descendants(root).filter((e) => matchCompound(e, chain[0]));
    for (let i = 1; i < chain.length; i++) {
      const next = [];
      for (const c of cur) for (const d of descendants(c)) if (matchCompound(d, chain[i])) next.push(d);
      cur = next;
    }
    for (const e of cur) if (!seen.has(e)) { seen.add(e); out.push(e); }
  }
  return out;
};

// Every timer the SCENE creates goes through here. Two reasons: scenario 8 needs to know whether
// the hog's interval was cleared, and every scenario needs the scene's two endless loops (the rAF
// re-registration and the 1 ms tick chain) stopped afterwards, or six abandoned scenes would be
// spinning in the background while the next one is being timed.
function makeTimers() {
  const timeouts = new Set();
  const intervals = new Set();
  const stats = { intervals_created: 0, intervals_cleared: 0 };
  return {
    stats,
    liveIntervals: intervals,
    setTimeout(fn, ms, ...args) {
      const id = setTimeout((...a) => { timeouts.delete(id); fn(...a); }, ms, ...args);
      timeouts.add(id);
      return id;
    },
    clearTimeout(id) { timeouts.delete(id); clearTimeout(id); },
    setInterval(fn, ms, ...args) {
      stats.intervals_created += 1;
      const id = setInterval(fn, ms, ...args);
      intervals.add(id);
      return id;
    },
    clearInterval(id) {
      if (intervals.delete(id)) stats.intervals_cleared += 1;
      clearInterval(id);
    },
    teardown() {
      for (const id of timeouts) clearTimeout(id);
      for (const id of intervals) clearInterval(id);
      timeouts.clear();
      intervals.clear();
    },
  };
}

// ── the page ─────────────────────────────────────────────────────────────────────────────────
//
// One builder, two arms, and the difference between them is EXACTLY the difference PR 9695 makes
// to the DOM: on `head` a fence inside a reasoning pane renders the same `DeferredFenceShell` an
// unreached fence already shows, so it carries `data-unsloth-fence-deferred="true"` and contains
// no highlight spans; on `main` the same fence is tokenised into spans.
//
// Fences are placed BOTH inside and outside the reasoning panes on purpose. PR 9695 touches only
// the ones inside, so a thread-wide span count cannot tell "the mechanism did not engage" from
// "it engaged and the thread has other code in it". If the scene ever stopped scoping its census
// to reasoning roots, scenario 5 would go red.
function buildPage(opts) {
  const o = Object.assign({ arm: "head", empty: false, composer: true, marker: MARKER,
                            messages: 6, reasoningFences: 3, bodyFences: 2,
                            spansPerFence: 12 }, opts || {});
  const head = new El("head");
  head.appendChild(new El("script", "", { src: "/assets/index-4f21ab.js" }));
  head.appendChild(new El("link", "", { rel: "stylesheet", href: "/assets/index-9c02de.css" }));

  const html = new El("html");
  const body = new El("body");
  html.appendChild(head);
  html.appendChild(body);

  const scroller = new El("div", "aui-thread-viewport");
  scroller._top = 0;
  scroller.clientHeight = 800;
  scroller.scrollHeight = o.empty ? 800 : 240000;
  Object.defineProperty(scroller, "scrollTop", {
    configurable: true,
    get() { return this._top; },
    set(v) {
      this._top = Math.max(0, Math.min(Math.max(0, this.scrollHeight - this.clientHeight),
                                       Math.round(v)));
    },
  });
  body.appendChild(scroller);

  const roots = [];
  const mkFence = (deferred) => {
    // On `head` the shell is deferred and carries no spans. On `main` it is highlighted. The
    // `pre` element exists on both arms either way, which is what makes `plain_fences` a reading
    // rather than an absence.
    const pre = new El("pre", "", deferred ? { "data-unsloth-fence-deferred": "true" } : {},
                       "def f():\n    return 1\n");
    if (!deferred) {
      for (let s = 0; s < o.spansPerFence; s++) pre.appendChild(new El("span", "hljs-keyword", {}, "def"));
    }
    return pre;
  };

  if (!o.empty) {
    for (let m = 0; m < o.messages; m++) {
      scroller.appendChild(new El("div", "aui-user-message", { "data-role": "user" },
                                  `question ${m} `));
      const last = m === o.messages - 1;
      const msg = new El("div", "aui-assistant-message", { "data-role": "assistant" },
                         `answer ${m} ` + (last ? o.marker + " " : ""));
      scroller.appendChild(msg);

      const reasoning = new El("div", "", { "data-slot": "reasoning-root",
                                            "data-state": "closed" });
      // `data-slot="reasoning-trigger"` is what the ALL-PANES gesture selects on. The one-pane
      // gesture takes `root.querySelector("button")` instead, so the two selectors have to be
      // satisfied by the same element or scenario 4 would be testing a different button from the
      // one the scored window clicks.
      const trigger = new El("button", "", { "data-slot": "reasoning-trigger",
                                             "aria-label": "Toggle reasoning" }, "Thinking");
      trigger._onclick = () => {
        reasoning.attrs["data-state"] =
          reasoning.attrs["data-state"] === "open" ? "closed" : "open";
      };
      reasoning.appendChild(trigger);
      const content = new El("div", "", { "data-slot": "reasoning-content" },
                             `chain of thought for message ${m}, `.repeat(4));
      for (let k = 0; k < o.reasoningFences; k++) {
        content.appendChild(mkFence(o.arm === "head"));
      }
      reasoning.appendChild(content);
      msg.appendChild(reasoning);
      roots.push(reasoning);

      // Fences in the BODY of the message, which PR 9695 does not touch, so they are highlighted
      // on both arms.
      for (let k = 0; k < o.bodyFences; k++) msg.appendChild(mkFence(false));
    }
  }

  const world = { html, body, head, scroller, roots, streaming: false, timers: null };
  if (o.composer) {
    const composer = new HTMLTextAreaElement("", { "aria-label": "Message input" });
    composer._rect = { left: 100, top: 820, width: 640, height: 44 };
    body.appendChild(composer);
    world.composer = composer;
    // The scene never sends, but `W.dom.sendButtons()` is still read on the ERROR path, so one
    // has to exist or scenario 6 would fail inside the failure reporter.
    const send = new El("button", "", { "aria-label": "Send message" }, "Send");
    body.appendChild(send);
    world.sendButton = send;
  }

  const doc = {
    documentElement: html,
    body,
    head,
    scrollingElement: scroller,
    querySelector: (s) => queryAll(html, s)[0] || null,
    querySelectorAll: (s) => queryAll(html, s),
    getElementsByTagName: (t) => (t === "*" ? descendants(html)
      : descendants(html).filter((e) => e.tagName === t.toUpperCase())),
    elementFromPoint: () => scroller,
    createElement: (t) => new El(t),
    createRange: () => ({ node: null, selectNodeContents(n) { this.node = n; } }),
    execCommand: (cmd) => cmd === "copy",
  };
  world.doc = doc;
  world.computed = (el) => ({
    overflowY: el === scroller ? "auto" : "visible",
    scrollBehavior: el.style.scrollBehavior || "smooth",
    contentVisibility: "visible",
  });
  world.supports = () => true;
  return world;
}

// ── driving the scene ────────────────────────────────────────────────────────────────────────

async function runScene(pageOpts, runOpts, hold) {
  const world = buildPage(pageOpts);
  const timers = makeTimers();
  world.timers = timers;
  const posted = [];
  const postErrors = [];

  // rAF off a REAL interval, created with the GLOBAL setInterval on purpose so the interval
  // accounting sees the scene's hog and nothing else. All callbacks due in one turn get one
  // timestamp, the way a vsync callback list does.
  //
  // 8 ms, NOT 16, AND THE NUMBER IS LOAD-BEARING. Under the headless X server this scene really
  // runs on, rAF is not vsync locked and reports 8-9 ms gaps on an idle page. That cadence is
  // the whole reason `1000/p50` is blind: within each unblocked stretch the page gets a dozen
  // cheap frames and the jam contributes ONE long gap per period, so the median never leaves the
  // cheap bucket while the wall-time rate collapses. At a 16 ms cadence the stub is not this
  // venue -- there are only two cheap frames per stretch, the median moves too, and scenario 3
  // would report the p50 channel as sighted, which is the opposite of what was measured on the
  // real box (62.0 -> 16.6 on the wall-time channel against 62.5 -> 62.5 on p50).
  const rafQ = [];
  const rafDriver = setInterval(() => {
    if (!rafQ.length) return;
    const due = rafQ.splice(0, rafQ.length);
    const t = performance.now();
    for (const fn of due) { try { fn(t); } catch (e) { postErrors.push(String(e)); } }
  }, 8);

  const selection = {
    _range: null,
    removeAllRanges() { this._range = null; },
    addRange(r) { this._range = r; },
    toString() { return this._range && this._range.node ? this._range.node.textContent : ""; },
  };
  const mkEvent = (name) => {
    const C = { [name]: class { constructor(type, init) { Object.assign(this, init || {}); this.type = type; } } };
    return C[name];
  };

  const sandbox = {
    document: world.doc,
    getComputedStyle: world.computed,
    CSS: { supports: world.supports },
    performance,
    Date,
    // Both substrings matter: engine_probe.is_webkit_gtk_ua is an AND over /AppleWebKit/ and
    // /X11/, and a criteria gate fails any session that was not WebKitGTK.
    navigator: {
      userAgent: "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/605.1.15 (KHTML, like Gecko) "
                 + "Version/17.0 Safari/605.1.15",
      vendor: "Apple Computer, Inc.",
      hardwareConcurrency: 16,
    },
    location: { href: "http://127.0.0.1:5601/chat?thread=t-selftest" },
    innerWidth: 1440,
    innerHeight: 900,
    devicePixelRatio: 1,
    setTimeout: timers.setTimeout,
    clearTimeout: timers.clearTimeout,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
    requestAnimationFrame: (fn) => rafQ.push(fn),
    HTMLTextAreaElement,
    Event: mkEvent("Event"),
    MouseEvent: mkEvent("MouseEvent"),
    PointerEvent: mkEvent("PointerEvent"),
    KeyboardEvent: mkEvent("KeyboardEvent"),
    getSelection: () => selection,
    webkit: { messageHandlers: { bench: { postMessage: (s) => {
      // The scene swallows anything this throws, so a parse failure here would look exactly like
      // a scene that never posted. Record it instead of losing it.
      try { posted.push(JSON.parse(s)); } catch (e) { postErrors.push(String(e)); }
    } } } },
  };
  sandbox.window = sandbox;

  const factory = new Function(
    "window", "document", "performance", "navigator", "location", "getComputedStyle", "CSS",
    "setTimeout", "clearTimeout", "setInterval", "clearInterval", "Date",
    SRC + "\n; return window.__av;");
  const W = factory(sandbox, sandbox.document, performance, sandbox.navigator, sandbox.location,
                    sandbox.getComputedStyle, sandbox.CSS, sandbox.setTimeout,
                    sandbox.clearTimeout, sandbox.setInterval, sandbox.clearInterval, Date);

  const stop = () => { clearInterval(rafDriver); timers.teardown(); };
  if (hold) {
    const promise = W.run(runOpts);
    return { W, world, timers, posted, postErrors, promise, stop };
  }
  await W.run(runOpts);
  stop();
  return { W, world, timers, posted, postErrors,
           payload: posted.filter((p) => p && p.__done).pop() };
}

// idleMs is short but NOT zero: the clamp needs at least 40 idle ticks to calibrate at all, and a
// window with no elapsed time reports busy_pct null WITH A REASON, which would make every numeric
// assertion below vacuous for a reason that has nothing to do with the scene.
const IDLE_MS = 700;
const phaseNames = (p) => ((p && p.phases) || []).map((x) => x.phase);
const markNames = (p) => ((p && p.marks) || []).map((x) => x.name);
const actionOf = (p, name) => ((p && p.actions) || []).find((a) => a.name === name);
const ACTIONS = ["reasoning_toggle", "reasoning_toggle_all", "reasoning_fidelity_settled",
                 "select_all_copy"];
const runOpts = (extra) => Object.assign(
  { idleMs: IDLE_MS, rung: "100K", arm: "head", lastMarker: MARKER, mountTimeoutMs: 8000,
    // A 50% duty cycle rather than the runner's 80%, for the same reason the rAF cadence is 8 ms:
    // the point of the stub is to reproduce the SHAPE the real venue has (many cheap frames
    // between one long gap per period), not the exact duty. At 80% duty against an 8 ms cadence
    // there are too few cheap frames per period for the median to stay in the cheap bucket, and
    // scenario 3 would be testing the stub rather than the scene.
    composerTimeoutMs: 8000, skipSend: true, hogMs: 100, hogPeriodMs: 200 }, extra || {});

let headPayload = null;
let mainPayload = null;

console.log("scenario 1: the happy path at a big rung");
{
  const r = await runScene({ arm: "head", messages: 6 }, runOpts({ arm: "head" }));
  const p = r.payload;
  headPayload = p;
  // NAMED, NOT THROWN. If the scene died before posting, every assertion below would read a
  // property of undefined and this file would report a TypeError with a line number instead of
  // saying what went wrong.
  check("the scene completed and posted ok:true", Boolean(p && p.ok === true),
        JSON.stringify((p && (p.error || p.phase)) || r.postErrors || "nothing was posted"));
  if (!p || p.ok !== true) {
    check("scenario 1 cannot continue without a posted payload", false, "the run did not complete");
  } else {
    check("the three scored phases are present, in the order the criteria module indexes them",
          JSON.stringify(phaseNames(p)) === JSON.stringify(["idle", "idle_jammed", "scroll"]),
          JSON.stringify(phaseNames(p)));
    for (const ph of p.phases || []) {
      check(`${ph.phase}: eff_fps, elapsed_ms and raf.n are real readings`,
            num(ph.eff_fps) && ph.eff_fps > 0 && num(ph.elapsed_ms) && ph.elapsed_ms > 0
            && num(ph.raf && ph.raf.n) && ph.raf.n > 0,
            JSON.stringify({ eff_fps: ph.eff_fps, elapsed_ms: ph.elapsed_ms,
                             n: ph.raf && ph.raf.n }));
      // The criteria module recomputes the headline from raf.n and elapsed_ms and GATES on the
      // two agreeing. If they ever disagreed, a headline would be read off a field whose meaning
      // had moved, so it is checked here too rather than only on the runner.
      check(`${ph.phase}: eff_fps really is frames over wall time`,
            Math.abs(ph.eff_fps - 1000 * ph.raf.n / ph.elapsed_ms) < 0.15,
            `${ph.eff_fps} vs ${1000 * ph.raf.n / ph.elapsed_ms}`);
    }
    check("the setTimeout clamp calibrated, so busy_pct is a number rather than null with a reason",
          num(p.clamp && p.clamp.clamp_ms) && (p.clamp.reason === null),
          JSON.stringify(p.clamp));
    for (const name of ACTIONS) {
      const a = actionOf(p, name);
      check(`action ${name} ran and is scored, not skipped`,
            Boolean(a && a.ok === true && !a.not_applicable && num(a.eff_fps)),
            JSON.stringify(a && { ok: a.ok, na: a.not_applicable, fps: a.eff_fps, d: a.detail }));
    }
    check("mount is attributed to the seeded marker, not to the composer",
          p.mount && p.mount.by === "last_seeded_marker", JSON.stringify(p.mount && p.mount.by));
    check("the marks name every phase boundary, in the order the film runs",
          markNames(p).join(",") === ["mount", "idle:calibrate", "idle", "idle_jammed", "scroll",
                                      "action:reasoning_toggle", "action:reasoning_toggle_all",
                                      "action:reasoning_fidelity_settled",
                                      "action:select_all_copy"].join(","),
          markNames(p).join(","));
    check("the engine probe reports WebKitGTK, which a criteria gate requires",
          p.engine_probe && p.engine_probe.is_webkit_gtk_ua === true
          && p.engine_probe.has_chrome === false
          && p.engine_probe.has_webkit_message_handlers === true,
          JSON.stringify(p.engine_probe));
    check("the payload says the send was skipped rather than leaving the missing phases "
          + "unexplained", p.skipped_send === true, JSON.stringify(p.skipped_send));
    check("no send, stream or recover mark was ever laid down",
          !markNames(p).some((m) => ["send", "stream", "recover"].includes(m)),
          markNames(p).join(","));
    check("the scroll gesture was forced out of smooth scrolling and really travelled",
          (p.phases || []).some((x) => x.phase === "scroll" && x.guard
                                  && x.guard.behavior_after === "auto"
                                  && x.travel && x.travel.travelled_px > 0),
          JSON.stringify((p.phases || []).find((x) => x.phase === "scroll") || {}).slice(0, 200));
  }
  check("the scene created exactly one interval (the hog) and cleared it again",
        r.timers.stats.intervals_created === 1 && r.timers.stats.intervals_cleared === 1
        && r.timers.liveIntervals.size === 0,
        JSON.stringify(r.timers.stats));
}

console.log("scenario 2: the JAMMED positive control resolves");
{
  const p = headPayload;
  const idle = (p && p.phases || []).find((x) => x.phase === "idle");
  const jam = (p && p.phases || []).find((x) => x.phase === "idle_jammed");
  if (!idle || !jam) {
    check("scenario 2 needs both idle windows from scenario 1", false, JSON.stringify(phaseNames(p)));
  } else {
    check("the jammed window reports a LOWER effective frame rate than the clean one",
          jam.eff_fps < idle.eff_fps, `${idle.eff_fps} -> ${jam.eff_fps}`);
    check("and the drop clears the 0.25 the criteria module discards a repetition under",
          p.liveness && p.liveness.drop_fraction >= 0.25,
          JSON.stringify(p.liveness));
    check("the independent timer-lag channel agrees the thread was blocked",
          num(jam.busy && jam.busy.busy_pct) && jam.busy.busy_pct > (idle.busy.busy_pct || 0),
          `${idle.busy && idle.busy.busy_pct}% -> ${jam.busy && jam.busy.busy_pct}%`);
    check("the jammed phase is labelled, with the hog it was priced at",
          jam.jammed === true && num(jam.hog_ms) && num(jam.hog_period_ms),
          JSON.stringify({ jammed: jam.jammed, ms: jam.hog_ms, period: jam.hog_period_ms }));
    check("and the windows after the jam recover, so the hog did not contaminate them",
          (p.phases.find((x) => x.phase === "scroll") || {}).eff_fps >= idle.eff_fps * 0.5,
          `scroll ${(p.phases.find((x) => x.phase === "scroll") || {}).eff_fps} vs idle ${idle.eff_fps}`);
  }
}

console.log("scenario 3: the blind channel is shown to be blind, on the SAME series");
{
  const p = headPayload;
  const l = (p && p.liveness) || {};
  check("1000/p50 is reported for both legs, so the disagreement stays visible",
        num(l.clean_fps_p50) && num(l.jammed_fps_p50), JSON.stringify(l));
  if (num(l.clean_fps_p50) && num(l.jammed_fps_p50) && num(l.drop_fraction)) {
    const p50Drop = 1 - l.jammed_fps_p50 / l.clean_fps_p50;
    check("the p50 channel moves far less than the wall-time channel on the same rAF series",
          p50Drop < l.drop_fraction / 2,
          `p50 drop ${p50Drop.toFixed(3)} vs wall-time drop ${l.drop_fraction}`);
    check("and it stays under the 0.25 bar, so a run scored on it would have been discarded as "
          + "having no working control",
          p50Drop < 0.25, `p50 drop ${p50Drop.toFixed(3)}`);
  }
}

console.log("scenario 4: the scored gesture is the ALL-PANES one");
{
  const p = headPayload;
  const one = actionOf(p, "reasoning_toggle");
  const all = actionOf(p, "reasoning_toggle_all");
  check("the one-pane gesture is recorded, so the report can print it beside the scored one",
        Boolean(one && one.ok), JSON.stringify(one && one.detail));
  check("the all-panes gesture opened EVERY pane and closed them again",
        Boolean(all && all.ok && /open 0 -> 6 of 6, then closed/.test(all.detail || "")),
        JSON.stringify(all && all.detail));
  check("and it reports the synchronous cost of the click handlers separately from the window",
        num(all && all.app_sync_ms) && all.app_sync_ms >= 0 && all.elapsed_ms > all.app_sync_ms,
        JSON.stringify(all && { sync: all.app_sync_ms, window: all.elapsed_ms }));
  check("the census taken while the panes were OPEN is carried, because the ones around it are "
        + "both taken with the panes closed",
        Boolean(all && all.census_open && all.census_open.reasoning_open === 6
                && all.census_before.reasoning_open === 0
                && all.census_after.reasoning_open === 0),
        JSON.stringify(all && { open: all.census_open && all.census_open.reasoning_open,
                                before: all.census_before && all.census_before.reasoning_open,
                                after: all.census_after && all.census_after.reasoning_open }));
  check("every pane really ended closed, so the next window is not measuring an open thread",
        Boolean(p && p.final && p.final.reasoning_open === 0),
        JSON.stringify(p && p.final && p.final.reasoning_open));
}

console.log("scenario 5: the fidelity census returns BOTH answers");
{
  const r = await runScene({ arm: "main", messages: 6 }, runOpts({ arm: "main" }));
  mainPayload = r.payload;
  check("the main-like session completed and posted ok:true",
        Boolean(mainPayload && mainPayload.ok === true),
        JSON.stringify((mainPayload && mainPayload.error) || r.postErrors));
  const H = actionOf(headPayload, "reasoning_fidelity_settled");
  const M = actionOf(mainPayload, "reasoning_fidelity_settled");
  if (!H || !M) {
    check("scenario 5 cannot continue without both arms' settled census", false,
          JSON.stringify({ head: Boolean(H), main: Boolean(M) }));
  } else {
    check("the settled census really settled on both arms, so neither is a snapshot of a page "
          + "still working", H.settled === true && M.settled === true,
          JSON.stringify({ head: H.settled, main: M.settled }));
    const hf = (H.fence_census || {}).reasoning || {};
    const mf = (M.fence_census || {}).reasoning || {};
    check("both arms have the SAME number of reasoning fences: the mechanism changes how they "
          + "render, not how many there are",
          hf.fences === mf.fences && hf.fences === 18,
          JSON.stringify({ head: hf.fences, main: mf.fences }));
    check("reasoning SPANS distinguish the two arms: none on head, many on main",
          hf.spans === 0 && mf.spans > 0, JSON.stringify({ head: hf.spans, main: mf.spans }));
    check("deferred shells inside reasoning distinguish them the other way round",
          (H.fence_census || {}).reasoning_deferred_shells === 18
          && (M.fence_census || {}).reasoning_deferred_shells === 0,
          JSON.stringify({ head: (H.fence_census || {}).reasoning_deferred_shells,
                           main: (M.fence_census || {}).reasoning_deferred_shells }));
    // THE SCOPING CHECK. Fences in the message body are highlighted on BOTH arms, so a
    // thread-wide count is not zero on head. If the census ever stopped being scoped to reasoning
    // roots, this goes red and the arms would look far less different than they are.
    const ht = (H.fence_census || {}).thread || {};
    check("the census is SCOPED: head still has highlighted fences thread-wide, in the message "
          + "bodies PR 9695 does not touch",
          ht.spans > 0 && ht.fences > hf.fences,
          JSON.stringify({ thread_spans: ht.spans, thread_fences: ht.fences,
                           reasoning_fences: hf.fences }));
    check("and plain_fences is a reading rather than an absence: the `pre` elements exist on both "
          + "arms", hf.plain_fences === 18 && mf.plain_fences === 0,
          JSON.stringify({ head: hf.plain_fences, main: mf.plain_fences }));
  }
}

console.log("scenario 6: a failure is reported as a failure");
{
  // No composer at all, so the very first wait cannot be satisfied. A scene that swallowed this
  // would post ok:true with empty phases, and a criteria module counting completed sessions would
  // score a page that never rendered.
  const r = await runScene({ arm: "head", composer: false },
                           runOpts({ composerTimeoutMs: 1200, mountTimeoutMs: 1200 }));
  const p = r.payload;
  check("the scene posted a payload rather than dying silently", Boolean(p),
        JSON.stringify(r.postErrors));
  if (p) {
    check("ok is FALSE, so the completion gate counts the session as failed instead of scoring it",
          p.ok === false, JSON.stringify(p.ok));
    check("the error names the timeout and what it was waiting for",
          /timeout waiting for/.test(String(p.error)), String(p.error));
    check("the failure is attributable: the phase it died in is recorded",
          typeof p.phase === "string" && p.phase.length > 0, String(p.phase));
    check("and the marks it reached are recorded", Array.isArray(p.marks) && p.marks.length > 0,
          JSON.stringify(markNames(p)));
    check("a failed session carries NO phases, so no number from it can be scored by accident",
          p.phases === undefined, JSON.stringify(phaseNames(p)));
    check("and the DOM state at the moment of death is recorded",
          p.dom && typeof p.dom.elements === "number", JSON.stringify(p.dom && p.dom.elements));
  }
}

console.log();
if (failures) {
  console.log(`${failures} FAILED`);
  process.exit(1);
}
console.log(`all checks passed in ${((Date.now() - T0) / 1000).toFixed(0)}s`);
