// Offline smoke test for amdv_layers.js against a hand-rolled DOM stub.
//
// It exists because a bug in this scene is not cheap: it costs an exclusive slot on a gfx1151
// four other jobs share, and the failure is only visible once the job completes. Nothing here
// measures anything real -- the point is that the scene RUNS to completion, posts a payload with
// the shape the criteria module reads, and that the per-arm "did the declaration take" check can
// return BOTH answers rather than always the convenient one.
//
// Three scenarios, and the last two are the ones that matter:
//   1. every arm's CSS takes    -> fired true everywhere, all 11 windows present
//   2. `.katex *{position:static}` is DROPPED by the engine (the `overflow-anchor` failure mode,
//      which produced three vacuous arms in this campaign) -> that arm must report fired:false
//   3. the viewport keeps `scroll-behavior: smooth` -> guard.ok must be false and the note set

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SRC = fs.readFileSync(path.join(HERE, "amdv_layers.js"), "utf8");

let failures = 0;
const check = (label, cond, detail = "") => {
  if (cond) console.log(`  ok   ${label}`);
  else { console.log(`  FAIL ${label}${detail ? " -- " + detail : ""}`); failures++; }
};

// ── the stub ─────────────────────────────────────────────────────────────────────────────────
class El {
  constructor(tag, cls = "", attrs = {}) {
    this.tagName = tag.toUpperCase();
    this.nodeType = 1;  // real nodes have one, and the scene sizes added subtrees by it
    this.className = cls;
    this.attrs = attrs;
    this.children = [];
    this.parentElement = null;
    this.textContent = "";
    this._id = "";
    // Real elements have one, and the scene writes an inline scroll-behavior onto the
    // scroller. Without it the assignment throws and the failure is swallowed as a note,
    // which would make scenario 3's assertion pass for the wrong reason.
    this.style = {};
    this.classList = {
      add: (c) => { if (!this.className.split(" ").includes(c)) this.className = (this.className + " " + c).trim(); },
      remove: (c) => { this.className = this.className.split(" ").filter((x) => x && x !== c).join(" "); },
      contains: (c) => this.className.split(" ").includes(c),
    };
  }
  get id() { return this._id; }
  set id(v) { this._id = v; }
  appendChild(c) { c.parentElement = this; this.children.push(c); return c; }
  remove() {
    if (!this.parentElement) return;
    const i = this.parentElement.children.indexOf(this);
    if (i >= 0) this.parentElement.children.splice(i, 1);
    this.parentElement = null;
  }
  matches(sel) { return matchOne(this, sel); }
}

const matchOne = (el, sel) => {
  if (sel === "*") return true;
  if (sel.startsWith("[") && sel.endsWith("]")) {
    const inner = sel.slice(1, -1);
    const eq = inner.indexOf("=");
    if (eq < 0) return inner in el.attrs;
    const k = inner.slice(0, eq);
    const v = inner.slice(eq + 1).replace(/^["']|["']$/g, "");
    return el.attrs[k] === v;
  }
  if (sel.startsWith(".")) return el.className.split(" ").includes(sel.slice(1));
  return el.tagName === sel.toUpperCase();
};

const descendants = (root, out = []) => {
  for (const c of root.children) { out.push(c); descendants(c, out); }
  return out;
};

const queryAll = (root, sel) => {
  // supports "A", "A B" (descendant) and comma lists; enough for this scene
  const parts = sel.split(",").map((s) => s.trim()).filter(Boolean);
  const seen = new Set();
  for (const p of parts) {
    const chain = p.split(/\s+/);
    let cur = descendants(root).filter((e) => matchOne(e, chain[0]));
    for (let i = 1; i < chain.length; i++) {
      const next = [];
      for (const c of cur) for (const d of descendants(c)) if (matchOne(d, chain[i])) next.push(d);
      cur = next;
    }
    for (const c of cur) seen.add(c);
  }
  return Array.from(seen);
};

// A MutationObserver stub. The scene's whole diagnosis of the previous run's drift rests on
// this channel, so the self-test has to drive it rather than let it silently observe nothing.
function makeMO(hook) {
  return class MutationObserver {
    constructor(cb) { this.cb = cb; hook.push(this); }
    observe() { this.active = true; }
    disconnect() { this.active = false; }
    fire(added) {
      if (!this.active) return;
      this.cb([{ addedNodes: added, removedNodes: [] }], this);
    }
  };
}

function buildWorld(opts) {
  const root = new El("body");
  const scroller = new El("div", "aui-thread-viewport");
  scroller.scrollHeight = 300000;
  scroller.clientHeight = 800;
  scroller._top = 0;
  Object.defineProperty(scroller, "scrollTop", {
    configurable: true,
    get() { return this._top; },
    set(v) { this._top = Math.max(0, Math.min(this.scrollHeight - this.clientHeight, Math.round(v))); },
  });
  root.appendChild(scroller);
  const composer = new El("textarea", "", { 'aria-label="Message input"': true });
  composer.ariaLabel = "Message input";
  root.appendChild(composer);

  // Deferred code fences, the thing `beforeprint` -> `upgradeEverythingForPrint` drains.
  const fences = [];
  for (let m = 0; m < 12; m++) {
    const msg = new El("div", "msg", { "data-role": true });
    scroller.appendChild(msg);
    for (let k = 0; k < 4; k++) {
      const kx = new El("span", "katex");
      msg.appendChild(kx);
      for (let d = 0; d < 6; d++) kx.appendChild(new El("span", "mord"));
    }
    const pre = new El("pre", "");
    pre.attrs["data-unsloth-fence-deferred"] = "true";
    pre.getAttribute = (k) => pre.attrs[k] || null;
    msg.appendChild(pre);
    fences.push(pre);
  }

  const head = new El("head");
  const styles = new Map();

  const doc = {
    head: { appendChild: (e) => { styles.set(e.id, e); head.appendChild(e); return e; } },
    body: { innerText: "SEEDED_MARKER_END" },
    scrollingElement: scroller,
    createElement: (t) => new El(t),
    // A removed <style> must stop being findable AND stop applying, or a reverted
    // arm silently stays in force and every later arm measures the union.
    getElementById: (id) => { const e = styles.get(id); return (e && e.parentElement) ? e : null; },
    querySelector: (s) => (s.includes("Message input") ? composer : queryAll(root, s)[0] || null),
    querySelectorAll: (s) => {
      const r = queryAll(root, s);
      r.forEach = Array.prototype.forEach.bind(r);
      return r;
    },
    getElementsByTagName: (t) => (t === "*" ? descendants(root) : descendants(root).filter((e) => e.tagName === t.toUpperCase())),
    elementFromPoint: () => scroller,
  };

  const active = () => new Set([...styles.keys()].filter(
    (k) => styles.get(k).parentElement && styles.get(k).textContent));
  const computed = (el) => {
    const a = active();
    const isKatexDesc = el.parentElement && el.parentElement.className.split(" ").includes("katex");
    const inMessage = (() => { let p = el.parentElement; while (p) { if ("data-role" in p.attrs) return true; p = p.parentElement; } return false; })();
    let position = "relative";
    if (inMessage && a.has("lay-ps")) position = "static";
    if (isKatexDesc && a.has("lay-ks") && !opts.dropKatexStatic) position = "static";
    let visibility = "visible";
    if (el.className.split(" ").includes("katex") && a.has("lay-vh")) visibility = "hidden";
    return {
      position, visibility, overflowY: el === scroller ? "auto" : "visible",
      scrollBehavior: (a.has("lay-guard") && !opts.smoothWins) ? "auto" : "smooth",
    };
  };

  return { doc, scroller, computed, styles, root, fences };
}

async function runScene(opts) {
  const world = buildWorld(opts);
  const posted = [];
  const observers = [];
  // FIRST-VISIT MOUNTING, simulated: the first `mountRounds` gestures each add a chunk of nodes,
  // then the page goes quiet. That is the shape run 32865232787 measured (+2,130 elements on the
  // first visit to the park position) and it is what the discarded warm-up has to absorb.
  let mountRounds = opts.mountRounds === undefined ? 2 : opts.mountRounds;
  let parkSeen = null;
  const sandbox = {
    document: world.doc,
    getComputedStyle: world.computed,
    performance,
    setTimeout, clearTimeout, setInterval, clearInterval,
    navigator: { userAgent: "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/605.1.15", vendor: "Apple", hardwareConcurrency: 8 },
    location: { href: "http://127.0.0.1:5481/chat?thread=x" },
    innerWidth: 1440, innerHeight: 900,
    requestAnimationFrame: (fn) => setTimeout(() => fn(performance.now()), 4),
    MutationObserver: makeMO(observers),
    Event: class Event { constructor(t) { this.type = t; } },
    webkit: { messageHandlers: { bench: { postMessage: (s) => posted.push(JSON.parse(s)) } } },
    Date,
  };
  const listeners = {};
  sandbox.addEventListener = (t, fn) => { (listeners[t] = listeners[t] || []).push(fn); };
  sandbox.dispatchEvent = (ev) => {
    for (const fn of listeners[ev.type] || []) fn(ev);
    return true;
  };
  sandbox.window = sandbox;
  // Stand in for `upgradeEverythingForPrint`: every remaining fence latches at once, adding
  // spans INSIDE the `pre` that already existed, which is why `pre` never moves.
  if (!opts.latchBroken) {
    sandbox.addEventListener("beforeprint", () => {
      const added = [];
      for (const pre of world.fences) {
        if (!pre.attrs["data-unsloth-fence-deferred"]) continue;
        delete pre.attrs["data-unsloth-fence-deferred"];
        for (let i = 0; i < 30; i++) { const sp = new El("span", ""); pre.appendChild(sp); added.push(sp); }
      }
      setTimeout(() => { for (const o of observers) o.fire(added); }, 0);
    });
  }
  // The scene closes over bare `window`, `document`, `getComputedStyle`, so evaluate it inside a
  // function whose parameters shadow the globals.
  const fn = new Function("window", "document", "getComputedStyle", "performance", "navigator",
                          "location", "setTimeout", "setInterval", "clearInterval",
                          "requestAnimationFrame", "MutationObserver",
                          SRC + "\n; return window.__lay;");
  const W = fn(sandbox, sandbox.document, sandbox.getComputedStyle, performance,
               sandbox.navigator, sandbox.location, setTimeout, setInterval, clearInterval,
               sandbox.requestAnimationFrame, sandbox.MutationObserver);
  // Drive the deferred mount off the scroller, the way a viewport-triggered mount really behaves.
  const origSet = Object.getOwnPropertyDescriptor(world.scroller, "scrollTop").set;
  Object.defineProperty(world.scroller, "scrollTop", {
    configurable: true,
    get() { return this._top; },
    set(v) {
      origSet.call(this, v);
      if (parkSeen === null) parkSeen = this._top;
      if (mountRounds > 0 && Math.abs(this._top - parkSeen) <= 2 && !this._mountedThisRound) {
        this._mountedThisRound = true;
        mountRounds--;
        const added = [];
        for (let i = 0; i < 3; i++) {
          const n = new El("div", "", { "data-slot": "message-actions" });
          n.setAttribute = () => {};
          n.getAttribute = (k) => (k === "data-slot" ? "message-actions" : null);
          for (let j = 0; j < 20; j++) n.appendChild(new El("span", ""));
          world.root.children[0].appendChild(n);
          added.push(n);
        }
        setTimeout(() => { for (const o of observers) o.fire(added); }, 0);
        setTimeout(() => { this._mountedThisRound = false; }, 900);
      }
    },
  });
  await W.run({ idleMs: 120, gestureFrames: 12, gestureCapMs: 4000, rung: "TEST",
                lastMarker: "SEEDED_MARKER_END", hogMs: 5, hogPeriodMs: 40,
                warmupFrames: 8, warmupCapMs: 3000, warmupRounds: opts.warmupRounds || 8,
                warmupQuietElements: 5, warmupQuietRounds: 2 });
  return { W, payload: posted.find((p) => p.__done) };
}

console.log("scenario 1: every declaration takes");
{
  const { payload } = await runScene({});
  check("the scene completed and posted ok", payload && payload.ok === true,
        payload && JSON.stringify(payload.error_detail || payload.error));
  const names = (payload.arms || []).map((a) => a.name);
  check("all eleven windows ran, baseline interleaved", names.length === 11 && names[0] === "baseline"
        && names.includes("baseline_repeat") && names[names.length - 1] === "detach_messages",
        names.join(","));
  const by = Object.fromEntries((payload.arms || []).map((a) => [a.name, a]));
  for (const n of ["position_static_all", "katex_static", "visibility_hidden_offscreen"]) {
    check(`${n} reports its declaration took`, by[n] && by[n].fired && by[n].fired.fired === true,
          JSON.stringify(by[n] && by[n].fired));
  }
  check("every window carries blocked_ms_per_frame",
        (payload.arms || []).every((a) => "blocked_ms_per_frame" in a));
  check("the 1 px gesture achieved what it commanded on the baseline",
        by.baseline.gesture.travel_fraction >= 0.9,
        JSON.stringify(by.baseline.gesture));
  check("the `still` window commanded no movement at all",
        by.still_no_scroll.gesture.commanded_px === 0, JSON.stringify(by.still_no_scroll.gesture));
  check("scroll-behavior was forced to auto", payload.guard.ok === true,
        JSON.stringify(payload.guard));
  check("a liveness window was recorded with both legs",
        payload.liveness && payload.liveness.clean_fps > 0 && payload.liveness.jammed_fps >= 0,
        JSON.stringify(payload.liveness));
  check("scrollHeight was recorded per arm",
        (payload.arms || []).every((a) => a.census_applied.scroller_scroll_height !== null));
  check("the positioned-element estimate ran on both selectors",
        payload.positioned.katex_descendants.total > 0
        && payload.positioned.message_descendants.total > 0);
  check("detach_messages actually removed messages",
        by.detach_messages.census_after.messages === 2,
        JSON.stringify(by.detach_messages.census_after));
}

{
  // the fence / warm-up assertions, on the same scenario-1 run
  const { payload } = await runScene({});
  const L = payload.fence_latch;
  check("the deferred-fence reservoir is drained before scoring", L && L.after.fences_deferred === 0,
        JSON.stringify(L && { before: L.before.fences_deferred, after: L.after.fences_deferred }));
  check("and the drain is reported as elements added inside an unchanged number of `pre`",
        L.elements_added > 0 && L.before.code_blocks === L.after.code_blocks
        && L.highlight_spans_added === L.elements_added,
        JSON.stringify({ added: L.elements_added, spans: L.highlight_spans_added,
                         pre: [L.before.code_blocks, L.after.code_blocks] }));
  check("`.katex` is untouched by the drain, so the layer population under test is unchanged",
        L.before.katex_roots === L.after.katex_roots
        && L.before.katex_descendants === L.after.katex_descendants);
  check("the warm-up reached quiescence and says so",
        payload.warmup.quiesced === true, JSON.stringify(payload.warmup.rounds));
  check("the warm-up is DISCARDED: no scored window is named after it",
        !(payload.arms || []).some((a) => a.name.indexOf("warm") >= 0));
  check("every scored window parked at the one fixed pixel",
        (payload.arms || []).every((a) => !a.gesture || a.gesture.park === payload.park),
        JSON.stringify((payload.arms || []).map((a) => a.gesture && a.gesture.park)));
  const scored = (payload.arms || []).filter((a) => a.arm !== "detach_messages");
  check("no scored window mounted anything once the reservoir was drained",
        scored.every((a) => (a.mutations || {}).added_elements === 0),
        JSON.stringify(scored.map((a) => [a.name, (a.mutations || {}).added_elements])));
  check("the mutation census attributed the drained nodes to the code fences",
        Object.keys(payload.warmup.mut_total.buckets || {}).some((k) => k.indexOf("pre") >= 0),
        JSON.stringify(payload.warmup.mut_total.buckets));
}

console.log("scenario 1b: the page keeps mounting and the warm-up cannot converge");
{
  const { payload } = await runScene({ latchBroken: true, mountRounds: 99, warmupRounds: 3 });
  check("the warm-up reports quiesced:false rather than proceeding as if settled",
        payload.warmup.quiesced === false, JSON.stringify(payload.warmup.rounds));
  check("and the fences are still deferred, which is the reason",
        payload.fence_latch.after.fences_deferred > 0,
        JSON.stringify(payload.fence_latch.after.fences_deferred));
}

console.log("scenario 2: the engine DROPS `.katex *{position:static}`");
{
  const { payload } = await runScene({ dropKatexStatic: true });
  const by = Object.fromEntries((payload.arms || []).map((a) => [a.name, a]));
  check("katex_static reports fired:false rather than a clean null",
        by.katex_static.fired.fired === false, JSON.stringify(by.katex_static.fired));
  check("and it names what it saw instead",
        (by.katex_static.fired.non_matching_examples || []).length > 0,
        JSON.stringify(by.katex_static.fired));
  check("position_static_all is unaffected", by.position_static_all.fired.fired === true);
}

console.log("scenario 3: scroll-behavior stays smooth");
{
  const { payload } = await runScene({ smoothWins: true });
  check("the guard reports NO rather than proceeding silently", payload.guard.ok === false,
        JSON.stringify(payload.guard));
  check("and the scene notes it, naming the animation trap",
        (payload.notes || []).some((n) => n.includes("scroll-behavior could not be forced")),
        JSON.stringify(payload.notes));
}

console.log(failures ? `\n${failures} FAILED` : "\nall scene self-tests passed");
process.exit(failures ? 1 : 0);
