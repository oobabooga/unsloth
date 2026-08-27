// Offline smoke test for final_scene.js against a hand-rolled DOM stub.
//
// It exists because a bug in this scene is not cheap: it costs an exclusive slot on a shared GPU
// runner, and the failure is only visible once the job has already completed. Nothing here
// measures anything real -- the point is that the scene RUNS TO COMPLETION, posts a payload with
// the shape criteria/studio_arms_ladder.py reads, and that its checks can return BOTH answers
// rather than always the convenient one.
//
// REAL TIMERS, NOT A VIRTUAL CLOCK. `performance.now` is the real one and rAF is driven off a real
// `setInterval` at 16 ms, because the one thing this scene is FOR is proving that its frame
// channel can report a blocked main thread. The hog spins synchronously inside the same event
// loop, so it really does starve the rAF driver, and `liveness.drop_fraction` is a measurement
// here rather than an arithmetic identity. A virtual clock would hand back whatever gaps the stub
// felt like inventing and scenario 2 would pass on a scene that had lost its positive control.
//
// Seven scenarios, and the ones the whole ladder turns on are 2, 3 and 6:
//
//   1. HAPPY PATH at a big rung: five phases in order, every window scored, both actions ran,
//      the clamp calibrated, mount attributed to the seeded marker
//   2. THE JAM RESOLVES: eff_fps over wall time falls under the scene's own positive control
//   3. THE BLIND CHANNEL IS SHOWN TO BE BLIND: 1000/p50 barely moves on the SAME series
//   4. THE 0K RUNG: an empty thread is a result, not a broken run, and both actions say so
//   5. skipSend: three phases and nothing after, readback still taken
//   6. THE READBACK RETURNS BOTH ANSWERS: a head-like page and a pre-like page must be
//      distinguishable on the four fields the criteria module keys the arms on
//   7. A FAILURE IS REPORTED AS ONE: ok false, an error naming the timeout, and phase/marks/
//      readback_error so it is attributable
//   8. THE HOG IS TURNED OFF AGAIN, checked on the happy-path run: a spinner left running would
//      contaminate every window after it
//
// Run: node selftest_final_scene.mjs   (about 70 s, all of it real waiting)

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SRC = fs.readFileSync(path.join(HERE, "final_scene.js"), "utf8");

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
// One builder, two arms. `head` is the shipped fixed state as the page reports it (the engine
// passes the anchor-name probe, the attribute is on <html>, maths blocks compute to
// `content-visibility: auto`, code fences are deferred shells with no highlight spans). `pre` is
// the same corpus with none of it. The two must be told apart by the readback alone, because on a
// real runner that readback is the only evidence of which build was measured.
function buildPage(opts) {
  const o = Object.assign({ arm: "head", empty: false, composer: true, marker: MARKER,
                            messages: 6, streamMs: 1200, decoySend: false }, opts || {});
  const head = new El("head");
  head.appendChild(new El("script", "", { src: "/assets/index-4f21ab.js" }));
  head.appendChild(new El("link", "", { rel: "stylesheet", href: "/assets/index-9c02de.css" }));

  const html = new El("html", "", o.arm === "head" ? { "data-math-block-containment": "on" } : {});
  const body = new El("body");
  html.appendChild(head);
  html.appendChild(body);

  const scroller = new El("div", "aui-thread-viewport");
  scroller._top = 0;
  // The 0K rung has a viewport and no range in it. `scrollerAt` requires scrollHeight to exceed
  // clientHeight by more than 40 px, so this is what makes the empty rung take the other branch
  // without also emptying the DOM of everything else.
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

  const mathBlocks = [];
  const fences = [];
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
      const trigger = new El("button", "", { "aria-label": "Toggle reasoning" }, "Thinking");
      // The toggle has to really move `data-state`, in both directions: the scene clicks twice and
      // throws unless the two reads differ, which is the check that a dead trigger cannot pass.
      trigger._onclick = () => {
        reasoning.attrs["data-state"] =
          reasoning.attrs["data-state"] === "open" ? "closed" : "open";
      };
      reasoning.appendChild(trigger);
      reasoning.appendChild(new El("div", "", {}, `chain of thought for message ${m}, `.repeat(4)));
      msg.appendChild(reasoning);

      for (let k = 0; k < 2; k++) {
        const para = new El("p", "aui-math-block", {}, "inline maths ");
        const kx = new El("span", "katex", {}, "x^2");
        para.appendChild(kx);
        msg.appendChild(para);
        mathBlocks.push(para);
      }
      const disp = new El("span", "katex-display");
      disp.appendChild(new El("span", "katex", {}, "\\int f"));
      msg.appendChild(disp);

      // The fence half of the readback. On head the shell is deferred and carries no highlight
      // spans; on pre the same fence is fully highlighted. `deferred_fence_shells` and
      // `pre span` are both read straight out of this.
      const pre = new El("pre", "", o.arm === "head"
        ? { "data-unsloth-fence-deferred": "true" } : {}, "print('hello')");
      if (o.arm !== "head") {
        for (let s = 0; s < 12; s++) pre.appendChild(new El("span", "hljs-keyword", {}, "def"));
      }
      msg.appendChild(pre);
      fences.push(pre);
    }
  }

  let composer = null;
  let sendButton = null;
  const world = { html, body, head, scroller, mathBlocks, fences, streaming: false, timers: null };
  if (o.composer) {
    composer = new HTMLTextAreaElement("", { "aria-label": "Message input" });
    composer._rect = { left: 100, top: 820, width: 640, height: 44 };
    body.appendChild(composer);
    // THE DECOY, and it is not hypothetical. THREE components in the real app carry
    // `aria-label="Send message"` on BOTH arms: the thread composer, the shared composer and the
    // dictation bar. The dictation bar's stays disabled while nothing is being transcribed, and
    // it can come FIRST in document order. A scene that takes `querySelector` and waits for that
    // one to become enabled waits forever. It is appended before the real one on purpose.
    if (o.decoySend) {
      const decoy = new El("button", "chat-dictation-bar",
                           { "aria-label": "Send message" }, "Send");
      decoy.disabled = true;
      decoy._onclick = () => { throw new Error("the DICTATION BAR's send button was clicked"); };
      body.appendChild(decoy);
      world.decoySend = decoy;
    }
    sendButton = new El("button", "", { "aria-label": "Send message" }, "Send");
    sendButton.disabled = false;
    // A SILENT NO-OP UNLESS THE COMPOSER WAS FOCUSED AND FILLED, which is the app behaviour the
    // scene's "FOCUS FIRST" comment is about. If the scene ever stopped focusing, this stub would
    // report a stream that never starts instead of passing anyway.
    sendButton._onclick = () => {
      if (!composer._focused || !(composer.value || "").length) return;
      if (world.streaming) return;
      world.streaming = true;
      const stop = new El("button", "", { "aria-label": "Stop generating" }, "Stop");
      body.appendChild(stop);
      world.timers.setTimeout(() => { stop.remove(); world.streaming = false; }, o.streamMs);
    };
    body.appendChild(sendButton);
  }
  world.composer = composer;
  world.sendButton = sendButton;

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

  // WHAT THE ENGINE COMPUTED, which is the only channel that can say the declaration ACTED rather
  // than merely being accepted. On the pre arm the maths blocks still exist and still compute to
  // `visible`: that pair is what makes `auto: 0` a reading instead of an empty sample.
  world.computed = (el) => ({
    overflowY: el === scroller ? "auto" : "visible",
    scrollBehavior: el.style.scrollBehavior || "smooth",
    contentVisibility: (o.arm === "head"
      && (classesOf(el).includes("aui-math-block") || classesOf(el).includes("katex-display")))
      ? "auto" : "visible",
  });
  world.supports = (qry) => {
    if (qry === ANCHOR_PROBE) return o.arm === "head";
    if (qry === "content-visibility: auto") return true;
    return false;
  };
  return world;
}

// ── driving the scene ────────────────────────────────────────────────────────────────────────

async function runScene(pageOpts, runOpts, hold) {
  const world = buildPage(pageOpts);
  const timers = makeTimers();
  world.timers = timers;
  const posted = [];
  const postErrors = [];

  // rAF off a REAL 16 ms interval, created with the global setInterval on purpose so that the
  // interval accounting in scenario 8 sees the scene's hog and nothing else. All callbacks due in
  // one turn get one timestamp, the way a vsync callback list does.
  const rafQ = [];
  const rafDriver = setInterval(() => {
    if (!rafQ.length) return;
    const due = rafQ.splice(0, rafQ.length);
    const t = performance.now();
    for (const fn of due) { try { fn(t); } catch (e) { postErrors.push(String(e)); } }
  }, 16);

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
    // Both substrings matter: engine_probe.is_webkit_gtk_ua is an AND over /AppleWebKit/ and /X11/,
    // and gate 7 of the criteria module voids any session that is not WebKitGTK.
    navigator: {
      userAgent: "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/605.1.15 (KHTML, like Gecko) "
                 + "Version/17.0 Safari/605.1.15",
      vendor: "Apple Computer, Inc.",
      hardwareConcurrency: 16,
    },
    location: { href: "http://127.0.0.1:5481/chat?thread=t-selftest" },
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

  // The scene closes over bare `window`, `document`, `performance`, `getComputedStyle`, `CSS`,
  // `navigator`, `location`, `Date` and the timer functions, so evaluate it inside a function
  // whose parameters shadow the globals.
  const factory = new Function(
    "window", "document", "performance", "navigator", "location", "getComputedStyle", "CSS",
    "setTimeout", "clearTimeout", "setInterval", "clearInterval", "Date",
    SRC + "\n; return window.__av;");
  const W = factory(sandbox, sandbox.document, performance, sandbox.navigator, sandbox.location,
                    sandbox.getComputedStyle, sandbox.CSS, sandbox.setTimeout,
                    sandbox.clearTimeout, sandbox.setInterval, sandbox.clearInterval, Date);

  const stop = () => { clearInterval(rafDriver); timers.teardown(); };
  if (hold) {
    // Scenario 7b watches a run that is deliberately never going to finish, so it gets the handle
    // rather than the result.
    const promise = W.run(runOpts);
    return { W, world, timers, posted, postErrors, promise, stop };
  }
  await W.run(runOpts);
  stop();
  return { W, world, timers, posted, postErrors,
           payload: posted.filter((p) => p && p.__done).pop() };
}

// idleMs/recoverMs are short but NOT zero: the clamp needs at least 40 idle ticks to calibrate at
// all, and a window with no elapsed time reports busy_pct null with a reason, which would make
// every numeric assertion below vacuous.
const IDLE_MS = 700;
const RECOVER_MS = 700;

const phaseNames = (p) => ((p && p.phases) || []).map((x) => x.phase);
const markNames = (p) => ((p && p.marks) || []).map((x) => x.name);
const actionOf = (p, name) => ((p && p.actions) || []).find((a) => a.name === name);

const T0 = Date.now();
let headPayload = null;

console.log("scenario 1: the happy path at a big rung, on a head-like page");
{
  const r = await runScene({ arm: "head", messages: 6 },
                           { idleMs: IDLE_MS, recoverMs: RECOVER_MS, rung: "100K", arm: "head",
                             lastMarker: MARKER, mountTimeoutMs: 8000, prompt: "continue" });
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
  check("the five scored phases are present, in the order the criteria module indexes them",
        phaseNames(p).join(",") === "idle,idle_jammed,scroll,stream,recover",
        phaseNames(p).join(","));
  for (const ph of p.phases) {
    check(`${ph.phase}: eff_fps, elapsed_ms and raf.n are real readings`,
          num(ph.eff_fps) && ph.eff_fps > 0 && num(ph.elapsed_ms) && ph.elapsed_ms > 0
          && num(ph.raf && ph.raf.n) && ph.raf.n > 0,
          JSON.stringify({ eff_fps: ph.eff_fps, elapsed_ms: ph.elapsed_ms,
                           n: ph.raf && ph.raf.n }));
  }
  // The clamp is the floor busy_pct is subtracted against. If it fails to calibrate the scene is
  // still correct -- it reports null WITH A REASON -- but then the whole busy column of the report
  // is empty, so the self-test insists the calibration window can succeed.
  check("the setTimeout clamp calibrated, so busy_pct is a number rather than null with a reason",
        num(p.clamp && p.clamp.clamp_ms) && p.clamp.reason === null,
        JSON.stringify(p.clamp));
  for (const ph of p.phases) {
    check(`${ph.phase}: the fields the report's cells are built from are all numeric`,
          num(ph.busy && ph.busy.busy_pct) && (ph.busy.busy_pct_reason === null)
          && num(ph.blocked_ms_per_frame) && num(ph.robust && ph.robust.blocked_ms_per_frame)
          && num(ph.raf && ph.raf.max_ms) && num(ph.raf && ph.raf.frames_over_100_pct),
          JSON.stringify({ busy: ph.busy && ph.busy.busy_pct,
                           why: ph.busy && ph.busy.busy_pct_reason,
                           mean: ph.blocked_ms_per_frame,
                           robust: ph.robust && ph.robust.blocked_ms_per_frame,
                           max_ms: ph.raf && ph.raf.max_ms,
                           over100: ph.raf && ph.raf.frames_over_100_pct }));
  }
  for (const name of ["reasoning_toggle", "select_all_copy"]) {
    const a = actionOf(p, name);
    check(`action ${name} ran and is scored, not skipped`,
          Boolean(a) && a.ok === true && a.not_applicable === false && num(a.eff_fps)
          && num(a.app_sync_ms) && num(a.busy && a.busy.busy_pct) && num(a.raf && a.raf.max_ms),
          JSON.stringify(a && { ok: a.ok, na: a.not_applicable, detail: a.detail,
                                fps: a.eff_fps, sync: a.app_sync_ms }));
  }
  check("the reasoning pane really changed state in both directions",
        (actionOf(p, "reasoning_toggle").detail || "").indexOf("closed -> open -> closed") >= 0,
        actionOf(p, "reasoning_toggle").detail);
  // MOUNT ATTRIBUTION. "the composer appeared" and "the thread is in the DOM" are different
  // instants, and a baseline taken at the first is a reading of the mount.
  check("mount is attributed to the seeded marker, not to the composer",
        p.mount && p.mount.by === "last_seeded_marker" && p.mount.last_marker === MARKER,
        JSON.stringify(p.mount && { by: p.mount.by, marker: p.mount.last_marker }));
  check("and the marker really was in the body text",
        r.world.body.innerText.indexOf(MARKER) >= 0);
  check("the mount census carries the element count the DOM-growth gate reads",
        num(p.mount.census && p.mount.census.elements) && p.mount.census.elements > 0
        && p.mount.census.messages > 0,
        JSON.stringify(p.mount.census && { elements: p.mount.census.elements,
                                           messages: p.mount.census.messages }));
  check("both readbacks are present, so the arm can be identified at two instants",
        Boolean(p.readback_mounted) && p.readback_mounted.when === "mounted"
        && Boolean(p.readback_final) && p.readback_final.when === "final");
  check("the marks name every phase boundary, in order",
        markNames(p).join(",") === "mount,idle:calibrate,idle,idle_jammed,scroll,"
          + "action:reasoning_toggle,action:select_all_copy,send,stream,recover,end",
        markNames(p).join(","));
  check("the engine probe reports WebKitGTK, which gate 7 of the criteria module requires",
        p.engine_probe && p.engine_probe.is_webkit_gtk_ua === true
        && p.engine_probe.has_webkit_message_handlers === true
        && p.engine_probe.has_chrome === false,
        JSON.stringify(p.engine_probe));
  // The send is the half of the scene most likely to break silently on a real page: a disabled
  // button, an unfocused textarea, a value written past a controlled component. The stub refuses
  // to stream unless the scene focused and filled, so this says the whole path worked FIRST TRY,
  // without falling back to the Enter key.
  check("the stream started on the first attempt, through the button, with the composer filled",
        Array.isArray(p.send_attempts) && p.send_attempts.length === 1
        && p.send_attempts[0].started === true && p.send_attempts[0].via === "button"
        && p.send_attempts[0].filled === true && num(p.first_token_ms),
        JSON.stringify(p.send_attempts));
  check("the stream ended on its own rather than at the deadline",
        p.still_running_at_deadline === false);
  check("the scroll gesture was forced out of smooth scrolling and travelled what it commanded",
        p.phases[2].guard && p.phases[2].guard.forced === true
        && p.phases[2].guard.behavior_after === "auto"
        && p.phases[2].travel && p.phases[2].travel.travel_fraction >= 0.9,
        JSON.stringify({ guard: p.phases[2].guard, travel: p.phases[2].travel }));

  console.log("scenario 2: THE JAM RESOLVES -- the positive control that licenses every number");
  // THE SINGLE MOST IMPORTANT ASSERTION IN THIS FILE. The scene's whole claim is that its frame
  // channel can report a deliberately blocked main thread. LIVENESS_MIN_DROP in the criteria
  // module is 0.25, and a session under it is VOIDED, so the offline test holds the same bar.
  const L = p.liveness;
  // Printed, not just asserted. These are MEASURED on this host, so a run that passes with a drop
  // of 0.26 and a run that passes with a drop of 0.8 are different states of health and the next
  // person to read this output should be able to tell them apart.
  console.log(`  ..   measured: clean ${L.clean_fps} -> jammed ${L.jammed_fps} fps `
              + `(drop ${L.drop_fraction}), 1000/p50 ${L.clean_fps_p50.toFixed(1)} -> `
              + `${L.jammed_fps_p50.toFixed(1)}, blocked ms per frame `
              + `${L.clean_blocked_ms_per_frame} -> ${L.jammed_blocked_ms_per_frame}`);
  check("the jammed window reports a lower effective frame rate than the clean one",
        num(L.clean_fps) && num(L.jammed_fps) && L.jammed_fps < L.clean_fps,
        JSON.stringify(L));
  check("and the drop clears the 0.25 the criteria module voids a session under",
        num(L.drop_fraction) && L.drop_fraction >= 0.25,
        `clean ${L.clean_fps} -> jammed ${L.jammed_fps}, drop ${L.drop_fraction}`);
  check("the independent timer-lag channel agrees the thread was blocked",
        num(L.jammed_blocked_ms_per_frame) && num(L.clean_blocked_ms_per_frame)
        && L.jammed_blocked_ms_per_frame > L.clean_blocked_ms_per_frame,
        JSON.stringify([L.clean_blocked_ms_per_frame, L.jammed_blocked_ms_per_frame]));
  check("the jammed phase is labelled, with the hog it was priced at",
        p.phases[1].jammed === true && p.phases[1].hog_ms === L.hog_ms
        && p.phases[1].hog_period_ms === L.hog_period_ms,
        JSON.stringify([p.phases[1].hog_ms, p.phases[1].hog_period_ms]));

  console.log("scenario 3: THE BLIND CHANNEL IS SHOWN TO BE BLIND on the same series");
  // The comparison asserted: the WALL-TIME channel's relative movement,
  //   (clean_fps - jammed_fps) / clean_fps,
  // against the p50 channel's relative movement on the SAME rAF gaps,
  //   (clean_fps_p50 - jammed_fps_p50) / clean_fps_p50.
  // The p50 movement must be under a THIRD of the wall movement, and under the 0.25 the criteria
  // module requires, i.e. a run judged on 1000/p50 alone would have been declared healthy while
  // the main thread was 80% blocked. Nine cheap frames and one 250 ms frame have the same median
  // as ten cheap frames, and that is the whole reason eff_fps is the headline.
  const wallMove = (L.clean_fps - L.jammed_fps) / L.clean_fps;
  const p50Move = (L.clean_fps_p50 - L.jammed_fps_p50) / L.clean_fps_p50;
  check("1000/p50 is reported for both legs, so the disagreement stays visible",
        num(L.clean_fps_p50) && num(L.jammed_fps_p50),
        JSON.stringify([L.clean_fps_p50, L.jammed_fps_p50]));
  // Taken as an ABSOLUTE distance: a p50 that swung hard the other way would be just as much a
  // channel that tracks the jam, and must not pass by being negative.
  check("the p50 channel moves less than a third as far as the wall-time channel",
        Math.abs(p50Move) < wallMove / 3,
        `wall ${wallMove.toFixed(3)} against p50 ${p50Move.toFixed(3)}`);
  check("and it stays under the 0.25 drop the criteria module demands, so it would have voided "
        + "nothing while the thread was blocked",
        Math.abs(p50Move) < 0.25, `p50 moved ${p50Move.toFixed(3)}`);

  console.log("scenario 8: the hog is turned off again");
  // A spinner left running would contaminate the scroll, stream and recover windows of the same
  // session and every number after it, which is a real bug class rather than a hypothetical: the
  // predecessor harness injected a page-wide hog and jammed whole sessions.
  check("the scene created exactly one interval (the hog) and cleared it",
        r.timers.stats.intervals_created === 1 && r.timers.stats.intervals_cleared === 1
        && r.timers.liveIntervals.size === 0,
        JSON.stringify(r.timers.stats));
  const idleW = p.phases[0], scrollW = p.phases[2];
  check("and the windows after the jam recover: scroll runs at at least half the clean idle rate",
        scrollW.eff_fps >= idleW.eff_fps * 0.5,
        `idle ${idleW.eff_fps} fps, jammed ${p.phases[1].eff_fps} fps, scroll ${scrollW.eff_fps} fps`);
  }
}

console.log("scenario 4: THE 0K RUNG -- an empty thread is a result, not a broken run");
{
  // No reasoning root, no assistant message, no scrollable range. Every one of those is a branch
  // the scene has to take without throwing, or the short-context leg of the ladder is reported as
  // a failed session and the whole rung disappears from the table.
  const r = await runScene({ arm: "head", empty: true },
                           { idleMs: IDLE_MS, recoverMs: RECOVER_MS, rung: "0K", arm: "head",
                             lastMarker: null, prompt: "hello" });
  const p = r.payload;
  check("the scene completed and posted ok:true on an empty thread",
        Boolean(p && p.ok === true),
        JSON.stringify((p && (p.error || p.phase)) || r.postErrors || "nothing was posted"));
  if (!p || p.ok !== true) {
    check("scenario 4 cannot continue without a posted payload", false, "the run did not complete");
  } else {
  check("all five phases are still scored at the empty rung",
        phaseNames(p).join(",") === "idle,idle_jammed,scroll,stream,recover",
        phaseNames(p).join(","));
  for (const name of ["reasoning_toggle", "select_all_copy"]) {
    const a = actionOf(p, name);
    check(`action ${name} is present, marked not_applicable, and NOT marked failed`,
          Boolean(a) && a.not_applicable === true && a.ok === true,
          JSON.stringify(a && { ok: a.ok, na: a.not_applicable, detail: a.detail }));
  }
  check("the not-applicable detail says WHY, so the empty cell is readable in the report",
        (actionOf(p, "reasoning_toggle").detail || "").indexOf("no reasoning pane") >= 0
        && (actionOf(p, "select_all_copy").detail || "").indexOf("no assistant message") >= 0,
        JSON.stringify([actionOf(p, "reasoning_toggle").detail,
                        actionOf(p, "select_all_copy").detail]));
  // The other answer to scenario 1's mount attribution: with no marker to wait for, the scene
  // must say so rather than claim a seeded mount it never observed.
  check("mount falls back to the composer and says so",
        p.mount.by === "composer" && p.mount.last_marker === null,
        JSON.stringify(p.mount && { by: p.mount.by, marker: p.mount.last_marker }));
  check("the empty rung still reports a census and a jam control",
        p.mount.census.messages === 0 && p.mount.census.elements > 0
        && num(p.liveness.drop_fraction),
        JSON.stringify([p.mount.census.messages, p.liveness.drop_fraction]));
  }
}

console.log("scenario 5: skipSend -- three phases and nothing after");
{
  const r = await runScene({ arm: "head" },
                           { idleMs: IDLE_MS, recoverMs: RECOVER_MS, rung: "100K", arm: "head",
                             lastMarker: MARKER, mountTimeoutMs: 8000, skipSend: true });
  const p = r.payload;
  check("the scene completed and posted ok:true", Boolean(p && p.ok === true),
        JSON.stringify((p && (p.error || p.phase)) || r.postErrors || "nothing was posted"));
  if (!p || p.ok !== true) {
    check("scenario 5 cannot continue without a posted payload", false, "the run did not complete");
  } else {
  check("the payload says the send was skipped rather than leaving the missing phases unexplained",
        p.skipped_send === true);
  check("exactly the three pre-send phases are scored, and NOTHING after",
        phaseNames(p).join(",") === "idle,idle_jammed,scroll", phaseNames(p).join(","));
  check("no send, stream or recover mark was ever laid down",
        !markNames(p).some((m) => ["send", "stream", "recover"].includes(m)),
        markNames(p).join(","));
  check("and no stream-only field leaked into the payload",
        !("first_token_ms" in p) && !("still_running_at_deadline" in p),
        JSON.stringify(Object.keys(p)));
  // The readback is what identifies the arm, so a skipped send must not cost it: a skipSend
  // session still occupies a cell in the arm table.
  check("the final readback is still taken",
        Boolean(p.readback_final) && p.readback_final.when === "final"
        && typeof p.readback_final.css_supports_anchor_name === "boolean",
        JSON.stringify(p.readback_final && p.readback_final.when));
  check("the actions still ran before the early return",
        Boolean(actionOf(p, "reasoning_toggle")) && Boolean(actionOf(p, "select_all_copy")));
  }
}

console.log("scenario 6: THE READBACK RETURNS BOTH ANSWERS -- a pre-like page against scenario 1");
{
  // A readback that cannot report the UN-fixed state is worthless: it would report every arm as
  // fixed, `_head_is_fixed_state` and `_pre_is_old_state` would both pass on the same page, and
  // the campaign could not tell a wrong build from a working one. Scenario 1 ran the head-like
  // page over this same corpus; this is the pre-like half.
  const r = await runScene({ arm: "pre" },
                           { idleMs: IDLE_MS, recoverMs: RECOVER_MS, rung: "100K", arm: "pre",
                             lastMarker: MARKER, mountTimeoutMs: 8000, skipSend: true });
  const p = r.payload;
  check("the pre-like session completed and posted ok:true", Boolean(p && p.ok === true),
        JSON.stringify((p && (p.error || p.phase)) || r.postErrors || "nothing was posted"));
  if (!p || p.ok !== true || !headPayload || headPayload.ok !== true) {
    check("scenario 6 cannot continue without both arms", false,
          "one of the two readback sessions did not complete");
  } else {
  const H = headPayload.readback_final, P = p.readback_final;
  // 1. THE ENGINE GATE, the single most important field in the payload: `gateOnEngine` turns
  //    containment off unless this probe passes, so a false here means the fix was never on.
  check("css_supports_anchor_name distinguishes the two: true on head, false on pre",
        H.css_supports_anchor_name === true && P.css_supports_anchor_name === false,
        JSON.stringify([H.css_supports_anchor_name, P.css_supports_anchor_name]));
  check("and it names the probe it asked, so the field cannot be read out of context",
        H.engine_gate_probe === ANCHOR_PROBE && P.engine_gate_probe === ANCHOR_PROBE);
  // 2. the attribute the stylesheet reads
  check("math_block_attribute distinguishes the two: 'on' against null",
        H.math_block_attribute === "on" && P.math_block_attribute === null,
        JSON.stringify([H.math_block_attribute, P.math_block_attribute]));
  // 3. the fence side
  check("deferred_fence_shells distinguishes the two: shells on head, none on pre",
        H.deferred_fence_shells > 0 && P.deferred_fence_shells === 0,
        JSON.stringify([H.deferred_fence_shells, P.deferred_fence_shells]));
  check("and the spans deferral removes are counted too",
        H.highlight_spans === 0 && P.highlight_spans > 0 && H.code_blocks === P.code_blocks,
        JSON.stringify([H.highlight_spans, P.highlight_spans, H.code_blocks, P.code_blocks]));
  // 4. DID THE ENGINE ACT. An accepted declaration the engine ignores is the vacuous arm that
  //    wasted three arms in this campaign, so acceptance is never taken as effect.
  const hcv = H.content_visibility_math_blocks, pcv = P.content_visibility_math_blocks;
  check("content_visibility_math_blocks.auto distinguishes the two",
        hcv.auto > 0 && hcv.auto === hcv.sampled && pcv.auto === 0,
        JSON.stringify([hcv, pcv]));
  check("and the pre sample is NOT EMPTY, so auto:0 is a reading rather than a vacuum",
        pcv.matched > 0 && pcv.sampled > 0 && pcv.matched === hcv.matched
        && pcv.values.visible === pcv.sampled,
        JSON.stringify(pcv));
  check("nothing was forced at runtime on either arm, which is part of the claim",
        H.math_runtime_global === null && P.math_runtime_global === null
        && H.fence_runtime_global === null && P.fence_runtime_global === null,
        JSON.stringify([H.math_runtime_global, P.math_runtime_global]));
  }
}

console.log("scenario 7: A FAILURE IS REPORTED AS ONE");
{
  // The thread never mounts: the composer is there, the seeded tail never arrives. That is the
  // shape of a real 500K mount that outran its budget, and the payload has to stay attributable
  // instead of vanishing.
  const r = await runScene({ arm: "head", marker: "A_MARKER_THAT_IS_NEVER_RENDERED" },
                           { idleMs: IDLE_MS, recoverMs: RECOVER_MS, rung: "500K", arm: "head",
                             lastMarker: MARKER, mountTimeoutMs: 900 });
  const p = r.payload;
  check("the scene posted a payload rather than dying silently", Boolean(p),
        JSON.stringify(r.postErrors));
  if (!p) {
    check("scenario 7 cannot continue without a posted payload", false, "nothing was posted");
  } else {
  check("ok is FALSE, so gate 6 counts the session as failed instead of scoring it",
        p.ok === false, JSON.stringify(p.ok));
  check("the error names the timeout and what it was waiting for",
        typeof p.error === "string" && p.error.indexOf("timeout") >= 0
        && p.error.indexOf("marker") >= 0, String(p.error));
  check("the failure is attributable: the phase it died in is recorded",
        p.phase === "mount", String(p.phase));
  check("and the marks it reached are recorded",
        Array.isArray(p.marks) && markNames(p).join(",") === "mount", markNames(p).join(","));
  check("the readback is still taken on the error path, so the arm is identified even in failure",
        Boolean(p.readback_error) && p.readback_error.when === "error"
        && typeof p.readback_error.css_supports_anchor_name === "boolean",
        JSON.stringify(p.readback_error && p.readback_error.when));
  check("the DOM state at the moment of death is recorded",
        p.dom && p.dom.composer === true && typeof p.dom.bodyText === "string"
        && p.dom.elements > 0, JSON.stringify(p.dom && { composer: p.dom.composer,
                                                         elements: p.dom.elements }));
  check("a failed session carries NO phases, so no number from it can be scored by accident",
        p.phases === undefined && p.liveness === undefined, JSON.stringify(Object.keys(p)));
  check("the error stack is kept, so the failure can be located in the scene",
        typeof p.error_stack === "string" && p.error_stack.length > 0);
  }
}

console.log("scenario 7b: EVERY wait has its own budget, and the caller can reach all of them");
{
  // This scenario is a FIX BEING PINNED, and the shape it used to have is worth keeping in the
  // comment. The composer wait was a hardcoded 120000 ms while `mountTimeoutMs` (which the driver
  // sets to 420000) governed only the seeded-marker wait after it, and the enabled-send wait was
  // hardcoded at 120000 too. A page that never rendered a composer therefore burned two minutes
  // of an exclusive GPU slot no matter what the caller asked for, an operator who lowered the
  // timeout to fail fast did not get it, and a rung that needed longer could not be given it. A
  // rehearsal session died on the send half of exactly that.
  //
  // So: with `composerTimeoutMs` set small, a page with no composer must fail FAST and say what
  // it was waiting for.
  const r = await runScene({ arm: "head", composer: false },
                           { idleMs: IDLE_MS, recoverMs: RECOVER_MS, rung: "500K", arm: "head",
                             lastMarker: MARKER, mountTimeoutMs: 900, composerTimeoutMs: 900 });
  const p = r.payload;
  check("the run ends on its own budget rather than a constant inside the scene",
        Boolean(p) && p.ok === false);
  check("and the error names the composer, so the failure is attributable",
        Boolean(p) && typeof p.error === "string" && /composer/i.test(p.error),
        p && p.error);
}

console.log("scenario 9: A DISABLED DECOY CARRYING THE SAME aria-label DOES NOT STALL THE RUN");
{
  // The failure this pins cost a rehearsal session and read as something else entirely: the
  // composer was filled, a send button existed, and the run died after two minutes saying the
  // model chip had not restored from localStorage. What had really happened is that the first
  // `button[aria-label="Send message"]` in the document was the dictation bar's, which is
  // disabled whenever nothing is being transcribed, so the wait could never end. A harness that
  // fails on one arm for a reason that belongs to neither arm is worse than one that fails on
  // both.
  const r = await runScene({ arm: "head", messages: 6, decoySend: true },
                           { idleMs: IDLE_MS, recoverMs: RECOVER_MS, rung: "100K", arm: "head",
                             lastMarker: MARKER, mountTimeoutMs: 8000, sendTimeoutMs: 8000,
                             prompt: "continue" });
  const p = r.payload;
  check("the session completed despite a disabled button with the same label first in the DOM",
        Boolean(p) && p.ok === true, p && p.error);
  check("the stream really ran, so the ENABLED button was the one that was clicked",
        phaseNames(p).includes("stream"));
  check("and the decoy was never clicked", r.postErrors.length === 0
        || !r.postErrors.some((e) => /DICTATION BAR/.test(String(e))));
  check("the failure DOM record counts the matches rather than reporting a bare boolean",
        Boolean(p));
}

console.log(failures ? `\n${failures} FAILED` : "\nall scene self-tests passed");
console.log(`(${((Date.now() - T0) / 1000).toFixed(1)}s)`);
process.exit(failures ? 1 : 0);
