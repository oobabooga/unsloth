// Offline smoke test for cvk_layers.js against a hand-rolled DOM stub.
//
// It exists because a bug in this scene is not cheap: it costs an exclusive slot on a gfx1151
// four other jobs share, and the failure is only visible once the job completes. Nothing here
// measures anything real -- the point is that the scene RUNS to completion, posts a payload with
// the shape the criteria module reads, and that the per-arm checks can return BOTH answers rather
// than always the convenient one.
//
// This scene asks something the layers scene could not. `content-visibility: auto` is ACCEPTED on
// any box and ACTS only on a box that can take size containment, so "the engine took the
// declaration" (`fired`) and "the engine did something with it" (`took_effect`) come apart, and
// the arm that reads auto while skipping nothing is not a bug in the instrument -- it is the
// measurement. Six scenarios, and the second is the one the whole run turns on:
//
//   1. every declaration takes AND acts     -> 13 windows in order, fired true on all three
//                                              content-visibility arms, heights change under each
//   2. THE VACUOUS ARM: `content_visibility_katex_all` is accepted on the inline roots and skips
//      nothing there, because size containment does not apply to a non-atomic inline-level box.
//      The payload must carry fired:true WITH changed:0 -- if the scene cannot record that pair,
//      the campaign reads an inert arm as a null result and ships a rule on a guess.
//   3. a declaration the engine DROPS entirely (the `overflow-anchor` failure mode, three vacuous
//      arms in this campaign) -> that arm must report fired:false
//   4. the viewport keeps `scroll-behavior: smooth` -> guard.ok false and the note set
//   5. NO SCROLLABLE RANGE (the 0K rung)    -> no_scroll_range, an empty arm table, ok still true,
//                                              and the idle windows still reported
//   6. markMathBlocks found the block ancestors, classed them, and measured their heights

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SRC = fs.readFileSync(path.join(HERE, "cvk_layers.js"), "utf8");

let failures = 0;
const check = (label, cond, detail = "") => {
  if (cond) console.log(`  ok   ${label}`);
  else { console.log(`  FAIL ${label}${detail ? " -- " + detail : ""}`); failures++; }
};

// The window the scene measures against, and the geometry the fake layout hands back.
const VIEWPORT_H = 900;      // window.innerHeight, and the band `content-visibility: auto` keeps
const CONTAINED_H = 12;      // used height of a SKIPPED subtree: the placeholder, not the content
const MSG_STRIDE = 2000;     // document pixels between messages, so most of the thread is off-screen
const P_H = 40, INLINE_H = 20, DISPLAY_H = 54, MORD_H = 12;

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
    // Set by place(). An element the scene never measures keeps a zero rect, which the height
    // probe reads as off-screen with height 0 -- so everything the probes can reach IS placed.
    this._layout = null;
    // Real elements have one, and the scene writes an inline scroll-behavior onto the
    // scroller. Without it the assignment throws and the failure is swallowed as a note,
    // which would make scenario 4's assertion pass for the wrong reason.
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
  hasAttribute(k) { return k in this.attrs; }
  getAttribute(k) { return (k in this.attrs) ? this.attrs[k] : null; }
  // `closest` is how the scene splits inline maths from display maths
  // (`.katex` that is not inside a `.katex-display`), which is the split the fix's ceiling is
  // computed from, so it has to include self and walk the whole way up.
  closest(sel) {
    let e = this;
    while (e) { if (matchOne(e, sel)) return e; e = e.parentElement; }
    return null;
  }
  getElementsByTagName(t) {
    const d = descendants(this);
    return t === "*" ? d : d.filter((e) => e.tagName === t.toUpperCase());
  }
  // THE BEHAVIOURAL SIGNAL. `content-visibility` is not readable from the computed value alone:
  // an inline root reports `auto` and is skipped by nothing. What separates the two is that a
  // SKIPPED box is size-contained, so its used height becomes the placeholder. That is exactly
  // what the scene's heightProbe reads, so this is where the stub has to be able to say both.
  getBoundingClientRect() {
    const L = this._layout;
    if (!L) return { top: 0, bottom: 0, left: 0, right: 0, width: 0, height: 0 };
    const h = L.world.contained(this) ? CONTAINED_H : L.height;
    const top = L.top - L.world.scroller.scrollTop;
    return { top, bottom: top + h, left: 0, right: L.width, width: L.width, height: h };
  }
}

const place = (world, el, top, height, width = 700) => {
  el._layout = { world, top, height, width };
  return el;
};

const classesOf = (el) => String(el.className || "").split(" ").filter(Boolean);

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
  if (sel.startsWith(".")) return classesOf(el).includes(sel.slice(1));
  return el.tagName === sel.toUpperCase();
};

const descendants = (root, out = []) => {
  for (const c of root.children) { out.push(c); descendants(c, out); }
  return out;
};

const queryAll = (root, sel) => {
  // supports "A", "A B" (descendant), comma lists and a trailing ":not(...)"; enough for this scene
  const parts = sel.split(",").map((s) => s.trim()).filter(Boolean);
  const seen = new Set();
  for (const p of parts) {
    // `:not(...)` is peeled off first and its argument is evaluated as its own query over the
    // whole tree, then subtracted. That is what makes `.katex:not(.katex-display *)` -- the
    // INLINE maths population, the one the vacuous arm is about -- expressible at all. The
    // subtraction is applied to the finished chain rather than to the compound it was written
    // on, which is the same answer for every selector this scene uses.
    const excludes = [];
    const base = p.replace(/:not\(([^)]*)\)/g, (_, inner) => { excludes.push(inner.trim()); return ""; }).trim();
    const chain = base.split(/\s+/).filter(Boolean);
    let cur = descendants(root).filter((e) => matchOne(e, chain[0]));
    for (let i = 1; i < chain.length; i++) {
      const next = [];
      for (const c of cur) for (const d of descendants(c)) if (matchOne(d, chain[i])) next.push(d);
      cur = next;
    }
    for (const ex of excludes) {
      const out = new Set(queryAll(root, ex));
      cur = cur.filter((e) => !out.has(e));
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

// Shipped katex.css gives `.katex-display` `display: block` and gives `.katex` no `display` at
// all, so inline maths is an inline box. That one line is the entire reason one of these two arms
// can act and the other cannot, so the stub models display rather than hard-coding the answer.
const BLOCK_TAGS = new Set(["BODY", "DIV", "P", "PRE", "SECTION", "ARTICLE"]);
const displayOf = (el) => {
  if (classesOf(el).includes("katex-display")) return "block";
  if (el.tagName === "LI") return "list-item";
  return BLOCK_TAGS.has(el.tagName) ? "block" : "inline";
};
const isBlockish = (d) => d === "block" || d === "flow-root" || d === "list-item";

function buildWorld(opts) {
  const world = {};
  const dropped = new Set(opts.dropRules || []);
  const root = new El("body");
  world.root = root;
  const scroller = new El("div", "aui-thread-viewport");
  // Scenario 5 shortens this to under the 200 px the scene requires. Everything else about the
  // page is left alone, so the check isolates the range gate instead of also emptying the DOM.
  scroller.scrollHeight = opts.scrollHeight === undefined ? 300000 : opts.scrollHeight;
  scroller.clientHeight = 800;
  scroller._top = 0;
  Object.defineProperty(scroller, "scrollTop", {
    configurable: true,
    get() { return this._top; },
    set(v) { this._top = Math.max(0, Math.min(this.scrollHeight - this.clientHeight, Math.round(v))); },
  });
  root.appendChild(scroller);
  world.scroller = scroller;
  const composer = new El("textarea", "", { "aria-label": "Message input" });
  root.appendChild(composer);

  // THE CORPUS, in the shape the fix's ceiling is computed from: every message carries a
  // paragraph of INLINE maths (`<p>` -> `.katex`, an inline box inside a block ancestor) and two
  // DISPLAY blocks (`.katex-display` -> `.katex`, the block box the shippable rule can reach).
  // Both populations exist in every message so the two selectors can never be told apart by
  // where in the thread they live.
  const fences = [];
  let y = 0;
  for (let m = 0; m < 12; m++) {
    const msg = place(world, new El("div", "msg", { "data-role": "assistant" }), y, MSG_STRIDE - 40);
    scroller.appendChild(msg);

    const para = place(world, new El("p"), y + 20, P_H);
    msg.appendChild(para);
    for (let k = 0; k < 2; k++) {
      const kx = place(world, new El("span", "katex"), y + 30, INLINE_H, 120);
      para.appendChild(kx);
      for (let d = 0; d < 6; d++) place(world, kx.appendChild(new El("span", "mord")), y + 30, MORD_H, 12);
    }

    for (let k = 0; k < 2; k++) {
      const top = y + 200 + k * 400;
      const disp = place(world, new El("span", "katex-display"), top, DISPLAY_H);
      msg.appendChild(disp);
      const kx = place(world, new El("span", "katex"), top + 2, DISPLAY_H - 4, 400);
      disp.appendChild(kx);
      for (let d = 0; d < 6; d++) place(world, kx.appendChild(new El("span", "mord")), top + 2, MORD_H, 12);
    }

    const pre = place(world, new El("pre", ""), y + 1000, 300);
    pre.attrs["data-unsloth-fence-deferred"] = "true";
    msg.appendChild(pre);
    fences.push(pre);
    y += MSG_STRIDE;
  }

  const head = new El("head");
  const styles = new Map();

  const doc = {
    documentElement: root,
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
  const live = (id, a) => a.has(id) && !dropped.has(id);

  // What `getComputedStyle(el).contentVisibility` returns: the SPECIFIED value of whichever arm
  // stylesheet is installed, whether or not the engine goes on to skip anything. A rule named in
  // `dropRules` is one the engine threw away at parse time, and then the computed value never
  // moves off `visible` -- that is the `overflow-anchor` failure mode, scenario 3.
  const contentVisibilityOf = (el, a) => {
    const cls = classesOf(el);
    if (live("cvk-d", a) && cls.includes("katex-display")) return "auto";
    if (live("cvk-a", a) && (cls.includes("katex") || cls.includes("katex-display"))) return "auto";
    if (live("cvk-m", a) && cls.includes("cvk-mathblock")) return "auto";
    return "visible";
  };

  // Off-screen by the UNSKIPPED height on purpose: if this asked the rect, the rect would ask
  // back and the model would chase its own tail.
  const offscreen = (el) => {
    const L = el._layout;
    if (!L) return true;
    const top = L.top - scroller.scrollTop;
    return !(top + L.height > 0 && top < VIEWPORT_H);
  };

  // DOES THE DECLARATION ACT? css-contain-2 #containment-size: size containment does not apply to
  // a non-atomic inline-level box, and WebKit checks exactly that in
  // `Style::ContainmentChecker::shouldApplySizeContainment`. So an inline `.katex` accepts
  // `content-visibility: auto` and is skipped by nothing. `inlineContainmentActs` is the
  // counterfactual engine, used only to prove the instrument can also return the other answer.
  world.contained = (el) => {
    const a = active();
    if (contentVisibilityOf(el, a) !== "auto") return false;
    const acts = isBlockish(displayOf(el)) ? true : Boolean(opts.inlineContainmentActs);
    return acts && offscreen(el);
  };

  const inMessage = (el) => {
    let p = el.parentElement;
    while (p) { if ("data-role" in p.attrs) return true; p = p.parentElement; }
    return false;
  };

  const computed = (el) => {
    const a = active();
    const cv = contentVisibilityOf(el, a);
    return {
      display: displayOf(el),
      // `RenderBoxModelObject::requiresLayer()` is true for `isPositioned()`, so the positioned
      // census needs something other than static to count inside the thread.
      position: inMessage(el) ? "relative" : "static",
      visibility: (classesOf(el).includes("katex") && a.has("lay-vh")) ? "hidden" : "visible",
      contentVisibility: cv,
      containIntrinsicSize: cv === "auto" ? "auto 3.5rem" : "none",
      overflowY: el === scroller ? "auto" : "visible",
      scrollBehavior: (a.has("lay-guard") && !opts.smoothWins) ? "auto" : "smooth",
    };
  };

  world.doc = doc;
  world.computed = computed;
  world.styles = styles;
  world.fences = fences;
  return world;
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
    innerWidth: 1440, innerHeight: VIEWPORT_H,
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
                          SRC + "\n; return window.__cvk;");
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
                warmupQuietElements: 5, warmupQuietRounds: 2, only: opts.only || null });
  return { W, world, payload: posted.find((p) => p.__done) };
}

// The sequence the criteria module indexes by name. `position_static_all` and `katex_static` are
// gone: both were disqualified on scrollHeight in run 32869180652 and neither could ever ship.
const SEQUENCE = ["baseline", "noop_touch", "baseline_2", "content_visibility_katex_display",
                  "baseline_3", "content_visibility_katex_all", "baseline_4",
                  "content_visibility_math_blocks", "baseline_5", "visibility_hidden_offscreen",
                  "baseline_repeat", "still_no_scroll", "detach_messages"];

console.log("scenario 1: every declaration takes AND acts");
{
  // The counterfactual engine, the one that also skips inline boxes. Nothing real behaves this
  // way; the run exists so that "changed: 0" in scenario 2 is a reading rather than a stub that
  // cannot produce anything else.
  const { payload } = await runScene({ inlineContainmentActs: true });
  check("the scene completed and posted ok", payload && payload.ok === true,
        payload && JSON.stringify(payload.error_detail || payload.error));
  const names = (payload.arms || []).map((a) => a.name);
  check("all thirteen windows ran, in order, baseline interleaved",
        names.join(",") === SEQUENCE.join(","), names.join(","));
  const by = Object.fromEntries((payload.arms || []).map((a) => [a.name, a]));
  for (const n of ["content_visibility_katex_display", "content_visibility_katex_all",
                   "content_visibility_math_blocks", "visibility_hidden_offscreen"]) {
    check(`${n} reports its declaration took`, by[n] && by[n].fired && by[n].fired.fired === true,
          JSON.stringify(by[n] && by[n].fired));
  }
  for (const n of ["content_visibility_katex_display", "content_visibility_katex_all",
                   "content_visibility_math_blocks"]) {
    const t = by[n] && by[n].took_effect;
    check(`${n} reports the engine ACTED on it, not merely accepted it`,
          t && t.compared > 0 && t.fraction_changed >= 0.9 && t.shrank === t.changed
          && t.mean_height_after < t.mean_height_before, JSON.stringify(t));
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
  check("the scroller had a range to gesture over, so the arms were scored",
        payload.no_scroll_range === false && payload.scrollable_px > 200,
        JSON.stringify([payload.no_scroll_range, payload.scrollable_px]));
  // THE SPLIT THE FIX'S CEILING IS COMPUTED FROM. A total over both populations would hide the
  // only number that decides how much `content-visibility` can be worth.
  const P = payload.positioned;
  check("the positioned census keeps display and inline maths apart, and neither is empty",
        P.katex_display_roots.total > 0 && P.katex_inline_roots.total > 0
        && P.katex_display_descendants.total > 0 && P.katex_inline_descendants.total > 0
        && P.katex_descendants.total > 0 && P.message_descendants.total > 0,
        JSON.stringify(P));
  check("the display and inline root counts partition `.katex`",
        P.katex_display_roots.total + P.katex_inline_roots.total
          === payload.baseline_census.katex_roots,
        JSON.stringify([P.katex_display_roots.total, P.katex_inline_roots.total,
                        payload.baseline_census.katex_roots]));
  check("the census carries the display-descendant and math-block counts the report reads",
        payload.baseline_census.katex_display_descendants > 0
        && payload.baseline_census.math_blocks > 0,
        JSON.stringify(payload.baseline_census));
  check("detach_messages actually removed messages",
        by.detach_messages.census_after.messages === 2,
        JSON.stringify(by.detach_messages.census_after));

  // the fence / warm-up assertions, on this same run: the drain is the reason a warm-up can
  // converge at all, so it is scored where it happened rather than in a second session
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
        && L.before.katex_descendants === L.after.katex_descendants
        && L.before.katex_display === L.after.katex_display);
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

  // scenario 6 rides on the same run: the class is added ONCE, before the warm-up, so every
  // window including every baseline sees the identical DOM and the arm only toggles a stylesheet
  console.log("scenario 6: the maths-bearing blocks were found, classed and measured");
  const M = payload.math_blocks;
  check("markMathBlocks found block ancestors for the maths roots",
        M && M.count > 0 && M.roots === payload.baseline_census.katex_roots,
        JSON.stringify(M));
  check("and the class actually landed on exactly those blocks",
        payload.baseline_census.math_blocks === M.count,
        JSON.stringify([payload.baseline_census.math_blocks, M.count]));
  check("the placeholder height comes from the measured median of those blocks",
        typeof M.median_height === "number" && M.median_height > 0
        && M.heights.p50 !== null && M.heights.min <= M.median_height
        && M.median_height <= M.heights.max, JSON.stringify(M.heights));
  check("the exploratory arm names that measured height in its own applied detail",
        by.content_visibility_math_blocks.apply_detail.indexOf(`${M.median_height}px`) >= 0,
        by.content_visibility_math_blocks.apply_detail);
}

console.log("scenario 1b: the page keeps mounting and the warm-up cannot converge");
{
  const { payload } = await runScene({ latchBroken: true, mountRounds: 99, warmupRounds: 3,
                                       only: ["baseline"] });
  check("the warm-up reports quiesced:false rather than proceeding as if settled",
        payload.warmup.quiesced === false, JSON.stringify(payload.warmup.rounds));
  check("and the fences are still deferred, which is the reason",
        payload.fence_latch.after.fences_deferred > 0,
        JSON.stringify(payload.fence_latch.after.fences_deferred));
}

console.log("scenario 2: THE VACUOUS ARM -- accepted on inline maths, acts on nothing");
{
  // The real engine. `content_visibility_katex_all` adds `.katex` to the selector, the inline
  // roots take the declaration, and size containment does not apply to them, so nothing is
  // skipped. If the scene cannot report that as fired:true WITH changed:0 then this arm reads
  // as a null result and the campaign cannot tell "inline maths is out of reach" from "the arm
  // never applied".
  const { payload } = await runScene({ only: ["content_visibility_katex_display",
                                              "content_visibility_katex_all"] });
  const by = Object.fromEntries((payload.arms || []).map((a) => [a.name, a]));
  const all = by.content_visibility_katex_all;
  check("the engine ACCEPTED the declaration on the inline roots",
        all.fired.fired === true && all.fired.total > 0 && all.fired.fraction === 1,
        JSON.stringify(all.fired));
  check("the sample it acted on is not empty, so `changed: 0` is a reading and not a vacuum",
        all.took_effect && all.took_effect.compared > 0
        && all.took_effect.offscreen_before === all.took_effect.offscreen_after,
        JSON.stringify(all.took_effect));
  check("and it SKIPPED NOTHING: no inline root changed height",
        all.took_effect.changed === 0 && all.took_effect.fraction_changed === 0
        && all.took_effect.mean_height_after === all.took_effect.mean_height_before,
        JSON.stringify(all.took_effect));
  check("the probe was aimed at the inline population specifically",
        all.took_effect.selector === ".katex:not(.katex-display *)", all.took_effect.selector);
  // The contrast is what makes it readable: the very same session, the very same instrument,
  // moves on the display roots. Without this leg "changed: 0" could be a broken probe.
  const dsp = by.content_visibility_katex_display;
  check("while in the SAME session the display roots did shrink",
        dsp.fired.fired === true && dsp.took_effect.fraction_changed >= 0.9
        && dsp.took_effect.mean_height_after < dsp.took_effect.mean_height_before,
        JSON.stringify(dsp.took_effect));
}

console.log("scenario 3: the engine DROPS the content-visibility declaration");
{
  const { payload } = await runScene({ dropRules: ["cvk-m"],
                                       only: ["content_visibility_katex_display",
                                              "content_visibility_math_blocks"] });
  const by = Object.fromEntries((payload.arms || []).map((a) => [a.name, a]));
  const mb = by.content_visibility_math_blocks;
  check("content_visibility_math_blocks reports fired:false rather than a clean null",
        mb.fired.fired === false, JSON.stringify(mb.fired));
  check("and it names what it saw instead",
        (mb.fired.non_matching_examples || []).includes("visible"), JSON.stringify(mb.fired));
  check("a dropped declaration also acts on nothing, and both facts are recorded",
        mb.took_effect && mb.took_effect.compared > 0 && mb.took_effect.changed === 0,
        JSON.stringify(mb.took_effect));
  check("content_visibility_katex_display is unaffected",
        by.content_visibility_katex_display.fired.fired === true
        && by.content_visibility_katex_display.took_effect.fraction_changed >= 0.9,
        JSON.stringify(by.content_visibility_katex_display.fired));
}

console.log("scenario 4: scroll-behavior stays smooth");
{
  const { payload } = await runScene({ smoothWins: true, only: ["baseline"] });
  check("the guard reports NO rather than proceeding silently", payload.guard.ok === false,
        JSON.stringify(payload.guard));
  check("and the scene notes it, naming the animation trap",
        (payload.notes || []).some((n) => n.includes("scroll-behavior could not be forced")),
        JSON.stringify(payload.notes));
}

console.log("scenario 5: no scrollable range, which is a result and not an error");
{
  // The 0K rung. There is nothing to scroll, so there is nothing for any of these arms to
  // remove. Throwing here would report the short-context leg as a broken run instead of as the
  // flat leg the short-context gate is asking for.
  const { payload } = await runScene({ scrollHeight: 900 });
  check("the scene still completed and posted ok", payload && payload.ok === true,
        payload && JSON.stringify(payload.error_detail || payload.error));
  check("it says so in the payload rather than leaving the empty table unexplained",
        payload.no_scroll_range === true && payload.scrollable_px === 100,
        JSON.stringify([payload.no_scroll_range, payload.scrollable_px]));
  check("no arm window was scored", (payload.arms || []).length === 0,
        JSON.stringify((payload.arms || []).map((a) => a.name)));
  check("and a note explains why",
        (payload.notes || []).some((n) => n.includes("scrollable range")
                                          && n.includes("no gesture window is scored")),
        JSON.stringify(payload.notes));
  // The short rungs are still expected to deliver the controls the flat leg is read against.
  check("the idle and idle_jammed windows are still reported",
        payload.idle && payload.idle.name === "idle" && payload.idle.jammed === false
        && payload.idle_jammed && payload.idle_jammed.name === "idle_jammed"
        && payload.idle_jammed.jammed === true,
        JSON.stringify([payload.idle && payload.idle.name,
                        payload.idle_jammed && payload.idle_jammed.name]));
  check("and so is the census, so the rung is still describable",
        payload.final && payload.final.elements > 0 && payload.baseline_census.katex_roots > 0,
        JSON.stringify(payload.baseline_census && payload.baseline_census.elements));
}

console.log(failures ? `\n${failures} FAILED` : "\nall scene self-tests passed");
process.exit(failures ? 1 : 0);
