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
// measurement. Seven scenarios, and two of them are the ones the whole run turns on:
//
//   1. every declaration takes AND acts     -> 17 windows in order, fired true on all three
//                                              content-visibility arms and on the product arm,
//                                              heights change under each
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
//   7. THE PRODUCT ARM, which applies no stylesheet at all: it sets `data-math-block-containment`
//      on the document element and the declaration comes from the build under test. So the run
//      has to prove the census FOUND the product rule (verbatim selectorText and cssText), that
//      the attribute moved computed style, that the arm reverted the attribute, and that it
//      injected no `<style>` of its own. 7b is the dangerous half: a bundle built WITHOUT the
//      rule, where the class is still on the page, the arm still runs, and the window is a clean
//      null -- the stub has to be able to say `present: false` and `fired: false` there, or the
//      campaign cannot tell a wrong build from a product implementation that does not work.
//      7c puts the rule inside a grouping rule, because a census that does not recurse reports a
//      correct build as a missing one.

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
  // The product arm is a PURE ATTRIBUTE TOGGLE on the document element, so these two are the
  // entire mechanism under test in scenario 7 and the stub has to implement them for real. A
  // `removeAttribute` that left the key behind would make the revert look like it worked while
  // every later window still carried the feature.
  setAttribute(k, v) { this.attrs[k] = String(v); }
  removeAttribute(k) { delete this.attrs[k]; }
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
// ── the product branch's frozen names, and the rule it ships ─────────────────────────────────
//
// The renderer adds `.aui-math-block` to the nearest block-level ancestor of every INLINE maths
// root at render time, ALWAYS, whether the feature is on or off, so the DOM is identical in every
// measured window. Display maths needs no marker: `span.katex-display` is already block-level and
// the rule names it directly. The only thing the flag moves is the attribute on `<html>`.
const PRODUCT_ATTR = "data-math-block-containment";
const PRODUCT_SELECTOR_TEXT = 'html[data-math-block-containment="on"] .aui-thread-root '
                            + ":is(.aui-math-block, .katex-display)";
const PRODUCT_CSS_TEXT = PRODUCT_SELECTOR_TEXT
                       + " { content-visibility: auto; contain-intrinsic-size: auto 7.5rem; }";
// A rule that mentions the CLASS but not the ATTRIBUTE. The census keys on the attribute, and this
// is here so that "found the product rule" cannot pass by matching the class instead.
const DECOY_CSS_TEXT = ".aui-math-block { margin-block: 0.5rem; }";

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
  // THE SHIP DEFAULT, MODELLED. On the head of PR 9731 `SHIP_DEFAULT` is `"contain"`, so
  // `applyMathBlockContainment()` sets this attribute before the first render and the page the
  // scene is injected into is ALREADY the fixed state. Every baseline would then be the fix, and
  // the harness's one rule makes that a VOID and not a pass, so the scene has to clear it and the
  // stub has to be able to hand it a page that needs clearing.
  if (opts.bootContained) root.setAttribute(PRODUCT_ATTR, "on");
  // ...and the other answer. A document element whose `removeAttribute` does nothing for this key
  // stands in for any build where the clear does not take. Without this the readback assertions in
  // scenario 7d would pass on a scene that never cleared anything.
  if (opts.attributeStuck) {
    root.removeAttribute = (k) => { if (k !== PRODUCT_ATTR) delete root.attrs[k]; };
  }
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

    // `.aui-math-block` is emitted by the PRODUCT RENDERER, so it is here in every scenario,
    // including the ones where the stylesheet rule is missing. That is the shape the product ships
    // (defect #48: the DOM must be identical in every measured window whether the feature is on or
    // off), and it is also what makes "the class is there but the rule is not" expressible at all.
    const para = place(world, new El("p", "aui-math-block"), y + 20, P_H);
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

  // ── the CSSOM the product-rule census walks ────────────────────────────────────────────────
  //
  // `findProductRule` reads `document.styleSheets` and every sheet's `cssRules`, which is the only
  // way to answer "does this BUILD contain the fix" without measuring anything. Three things have
  // to be modelled or the census cannot be trusted:
  //   * a cross-origin sheet THROWS on `.cssRules` (CSSOM security, enforced by WebKit), and the
  //     census must count it rather than fold it into "not found";
  //   * grouping rules (`@media`, `@supports`, `@layer`) nest their own `cssRules`, and a bundler
  //     is free to put the product rule inside one;
  //   * the arm stylesheets this harness injects are sheets too, and none of them mention the
  //     attribute, so they must not be mistaken for the product's rule.
  const rule = (selectorText, cssText) => ({ selectorText, cssText });
  const group = (cssText, rules) => ({ selectorText: undefined, cssText, cssRules: rules });
  const sheetOf = (href, cssRules) => ({ href, cssRules });
  const crossOriginSheet = (href) => ({
    href,
    get cssRules() {
      throw new Error("SecurityError: Not allowed to access cross-origin stylesheet");
    },
  });
  const bundleSheet = () => {
    const rules = [rule(".katex", ".katex { text-rendering: auto; }"),
                   rule(".aui-math-block", DECOY_CSS_TEXT)];
    if (!opts.productRuleMissing) {
      const product = rule(PRODUCT_SELECTOR_TEXT, PRODUCT_CSS_TEXT);
      // The same rule, reached only through a grouping rule's own `cssRules`. A census that does
      // not recurse reports a build that HAS the fix as one that does not, which is the exact
      // wrong answer this whole precondition exists to prevent.
      rules.push(opts.productRuleNested
        ? group("@media screen { " + PRODUCT_CSS_TEXT + " }", [product])
        : product);
    }
    return sheetOf("/assets/index-4f21ab.css", rules);
  };

  const doc = {
    documentElement: root,
    get styleSheets() {
      const out = [bundleSheet()];
      for (let i = 0; i < (opts.crossOriginSheets === undefined ? 1 : opts.crossOriginSheets); i++) {
        out.push(crossOriginSheet("https://fonts.googleapis.com/css2?family=Inter"));
      }
      for (const [id, el] of styles) {
        if (!el.parentElement || !el.textContent) continue;
        const txt = String(el.textContent);
        out.push(sheetOf("(inline <style>)#" + id,
                         [rule(txt.split("{")[0].trim(), txt)]));
      }
      return out;
    },
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
    // THE PRODUCT PATH. Two independent conditions, and separating them is the point: the RULE has
    // to be in the bundle (`productRuleMissing` is a build that never had the fix) AND the
    // ATTRIBUTE has to be set on the document element (the feature flag, which defaults to OFF).
    // A build without the rule therefore reports `visible` no matter what the arm does, which is
    // what makes the arm's toggle inert and its window a clean null.
    if (!opts.productRuleMissing && root.getAttribute(PRODUCT_ATTR) === "on"
        && (cls.includes("aui-math-block") || cls.includes("katex-display"))) return "auto";
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
  // `installOverrideWatcher` from the product, modelled: assigning the global REAPPLIES, which is
  // the whole reason that accessor exists (a devtools flip that changed nothing left the session
  // measuring the arm it was already in). `overrideWatcher: false` is a build that predates it, and
  // that is the case the scene's direct `removeAttribute` has to cover on its own.
  world.overrideWrites = [];
  if (opts.overrideWatcher !== false) {
    let held;
    Object.defineProperty(sandbox, "__UNSLOTH_MATH_BLOCK_CONTAINMENT__", {
      configurable: true,
      enumerable: true,
      get: () => held,
      set: (next) => {
        held = next;
        world.overrideWrites.push(next);
        if (next === true || next === "1" || next === "contain") {
          world.root.setAttribute(PRODUCT_ATTR, "on");
        } else {
          world.root.removeAttribute(PRODUCT_ATTR);
        }
      },
    });
  }
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
const SEQUENCE = ["baseline", "noop_touch", "baseline_2", "katex_root_visibility_hidden",
                  "baseline_3", "content_visibility_katex_display", "baseline_4",
                  "content_visibility_katex_all", "baseline_5",
                  "content_visibility_math_blocks", "baseline_6",
                  "product_math_block_containment", "baseline_7",
                  "katex_root_visibility_hidden_late", "baseline_repeat", "still_no_scroll",
                  "detach_messages"];

console.log("scenario 1: every declaration takes AND acts");
{
  // The counterfactual engine, the one that also skips inline boxes. Nothing real behaves this
  // way; the run exists so that "changed: 0" in scenario 2 is a reading rather than a stub that
  // cannot produce anything else.
  const { payload } = await runScene({ inlineContainmentActs: true });
  check("the scene completed and posted ok", payload && payload.ok === true,
        payload && JSON.stringify(payload.error_detail || payload.error));
  const names = (payload.arms || []).map((a) => a.name);
  check("all seventeen windows ran, in order, baseline interleaved, the upper bound twice",
        names.join(",") === SEQUENCE.join(","), names.join(","));
  const by = Object.fromEntries((payload.arms || []).map((a) => [a.name, a]));
  for (const n of ["content_visibility_katex_display", "content_visibility_katex_all",
                   "content_visibility_math_blocks", "product_math_block_containment",
                   "katex_root_visibility_hidden"]) {
    check(`${n} reports its declaration took`, by[n] && by[n].fired && by[n].fired.fired === true,
          JSON.stringify(by[n] && by[n].fired));
  }
  for (const n of ["content_visibility_katex_display", "content_visibility_katex_all",
                   "content_visibility_math_blocks", "product_math_block_containment"]) {
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
  // THE MEAN, NOT THE MEDIAN. The scrollHeight error a placeholder introduces is the SUM of
  // (placeholder - actual) over the blocks that were never rendered, so the statistic that
  // zeroes it is the arithmetic mean. Run 32876363634 used the median and moved scrollHeight by
  // +3.3% at r500K, which disqualified the arm, while the same arm moved it by -0.01% at r100K
  // where the two statistics happened to coincide. Both are still recorded, because the gap
  // between them is what says the distribution is skewed.
  check("the placeholder height comes from the measured MEAN of those blocks",
        typeof M.mean_height === "number" && M.mean_height > 0
        && M.heights.min <= M.mean_height && M.mean_height <= M.heights.max,
        JSON.stringify({ mean: M.mean_height, median: M.median_height, heights: M.heights }));
  check("the median is recorded too, so a skewed distribution is visible",
        typeof M.median_height === "number" && M.median_height > 0,
        JSON.stringify([M.mean_height, M.median_height]));
  check("what those blocks ARE is counted, so the follow-up knows whether a CSS selector exists",
        M.kinds && typeof M.kinds === "object" && Object.keys(M.kinds).length > 0,
        JSON.stringify(M.kinds));
  check("the exploratory arm names that measured height in its own applied detail",
        by.content_visibility_math_blocks.apply_detail.indexOf(`${M.mean_height}px`) >= 0,
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

console.log("scenario 7: THE PRODUCT ARM -- a pure attribute toggle over the build's own rule");
{
  // The product arm applies NO STYLESHEET. Everything it depends on -- the class on the block
  // ancestor and the declaration gated on the attribute -- comes from the build under test, so the
  // only things this arm can be checked on are: did the rule turn out to be there, did setting the
  // attribute move computed style, and did the attribute come back off afterwards.
  const { payload, world } = await runScene({ only: ["content_visibility_math_blocks",
                                                     "product_math_block_containment"] });
  const by = Object.fromEntries((payload.arms || []).map((a) => [a.name, a]));
  const arm = by.product_math_block_containment;
  const P = payload.product_rule;
  // NAMED, NOT THROWN. If the scene died before posting, every assertion below would read a
  // property of undefined and this file would report a TypeError with a line number instead of
  // saying what went wrong. A test that fails by crashing is a test whose failure has to be
  // debugged before it can be read.
  check("the scene completed and posted its product-rule precondition",
        Boolean(payload && payload.ok === true && P && arm),
        JSON.stringify((payload && (payload.error_detail || payload.error))
                       || Object.keys(payload || {})));
  if (!P || !arm) {
    check("scenario 7 cannot continue without a posted precondition", false,
          "the scene did not reach the end of its run");
  } else {
  check("it found the product rule in the loaded stylesheets", P.present === true,
        JSON.stringify(P.matches));
  check("and recorded the matched selectorText VERBATIM", P.selector_text === PRODUCT_SELECTOR_TEXT,
        String(P.selector_text));
  check("and the matched cssText VERBATIM", P.css_text === PRODUCT_CSS_TEXT, String(P.css_text));
  check("it keyed on the ATTRIBUTE, not on the class: the decoy rule did not match",
        P.matches.length === 1 && P.matches[0].css_text.indexOf("content-visibility") >= 0,
        JSON.stringify(P.matches.map((m) => m.selector_text)));
  check("a cross-origin sheet that throws on .cssRules is COUNTED, not folded into 'not found'",
        P.sheets_unreadable === 1 && P.sheets_readable >= 1
        && (P.unreadable[0].why || "").indexOf("SecurityError") >= 0,
        JSON.stringify([P.sheets_readable, P.sheets_unreadable, P.unreadable]));
  check("the census counted the populations the rule covers",
        P.blocks > 0 && P.katex_display > 0 && P.blocks === payload.baseline_census.product_math_blocks,
        JSON.stringify([P.blocks, P.katex_display, payload.baseline_census.product_math_blocks]));
  const T = P.toggle_check;
  check("the BEHAVIOURAL half ran: computed style sampled with the attribute off, then on",
        T.sampled > 0 && T.off_values.length > 0 && T.on_values.length === T.off_values.length,
        JSON.stringify(T));
  check("off it reads visible, on it reads auto, and the whole sample moved",
        T.off_values.every((v) => v === "visible") && T.on_values.every((v) => v === "auto")
        && T.moved === T.sampled && T.moved_fraction === 1, JSON.stringify(T));
  check("and the attribute was put back exactly as it was found, so no other window saw it",
        T.attribute_before === null && T.attribute_after === null, JSON.stringify(T));
  check("the arm's fired check is the same one the other arms use, over `.aui-math-block`",
        arm.fired.fired === true && arm.fired.selector === ".aui-math-block"
        && arm.fired.prop === "contentVisibility" && arm.fired.want === "auto",
        JSON.stringify(arm.fired));
  check("its height probe is the same one the other arms use, and the engine ACTED",
        arm.took_effect.selector === ".aui-math-block" && arm.took_effect.compared > 0
        && arm.took_effect.fraction_changed >= 0.9
        && arm.took_effect.mean_height_after < arm.took_effect.mean_height_before,
        JSON.stringify(arm.took_effect));
  // DEFECT #50: the arm's identity is the selector it applied, and this arm applied someone
  // else's, so it has to quote the one it found rather than one this file believes in.
  check("the apply detail quotes the PRODUCT stylesheet's own selector, read back from the page",
        arm.apply_detail.indexOf(PRODUCT_SELECTOR_TEXT) === 0, arm.apply_detail);
  check("and names the attribute it set and the populations it covers",
        arm.apply_detail.indexOf(`set [${PRODUCT_ATTR}="on"] on <html>`) > 0
        && arm.apply_detail.indexOf(`${P.blocks} .aui-math-block`) > 0
        && arm.apply_detail.indexOf(`${P.katex_display} .katex-display`) > 0, arm.apply_detail);
  check("REVERT: the attribute is gone from <html> after the run, not merely set to something else",
        !(PRODUCT_ATTR in world.root.attrs), JSON.stringify(world.root.attrs));
  check("the arm injected NO stylesheet of its own: the only ids ever created are the guard's "
        + "and the exploratory arm's",
        [...world.styles.keys()].sort().join(",") === "cvk-m,lay-guard",
        JSON.stringify([...world.styles.keys()]));
  check("every window saw the identical DOM, because the class is emitted whether the flag is on "
        + "or off",
        arm.census_before.product_math_blocks === arm.census_applied.product_math_blocks
        && arm.census_before.product_math_blocks
           === by.content_visibility_math_blocks.census_before.product_math_blocks
        && arm.census_before.product_math_blocks > 0,
        JSON.stringify([arm.census_before.product_math_blocks,
                        arm.census_applied.product_math_blocks]));
  }
}

console.log("scenario 7b: THE BUNDLE WAS BUILT WITHOUT THE RULE -- the toggle is inert");
{
  // The dangerous case. Nothing about the DOM changes: the renderer still emits the class, the
  // arm still sets the attribute, the window still produces a number. Only the rule is missing,
  // and the number is a clean null that looks exactly like a product implementation that does not
  // work. Both of the stub's answers are exercised here: `present` false AND `fired` false.
  const { payload, world } = await runScene({ productRuleMissing: true,
                                              only: ["product_math_block_containment"] });
  const by = Object.fromEntries((payload.arms || []).map((a) => [a.name, a]));
  const arm = by.product_math_block_containment;
  const P = payload.product_rule;
  check("the scene completed and posted its product-rule precondition even on the wrong build",
        Boolean(payload && payload.ok === true && P && arm),
        JSON.stringify((payload && (payload.error_detail || payload.error))
                       || Object.keys(payload || {})));
  if (!P || !arm) {
    check("scenario 7b cannot continue without a posted precondition", false,
          "the scene did not reach the end of its run, so the wrong-build case proved nothing");
  } else {
  check("the census says the rule is NOT present, and says it with an empty match list",
        P.present === false && P.matches.length === 0 && P.selector_text === null
        && P.css_text === null, JSON.stringify(P.matches));
  check("it still read the sheets rather than failing to look",
        P.sheets_readable >= 1 && P.rules_scanned > 0,
        JSON.stringify([P.sheets_readable, P.rules_scanned]));
  check("the class is STILL on the page, so the DOM is identical to the working build",
        P.blocks > 0 && payload.baseline_census.product_math_blocks === P.blocks,
        JSON.stringify([P.blocks, payload.baseline_census.product_math_blocks]));
  check("the toggle check moved NOTHING, which is the other answer the stub can give",
        P.toggle_check.sampled > 0 && P.toggle_check.moved === 0
        && P.toggle_check.moved_fraction === 0
        && P.toggle_check.on_values.every((v) => v === "visible"),
        JSON.stringify(P.toggle_check));
  check("the arm reports fired:false rather than a clean null",
        arm.fired.fired === false && (arm.fired.non_matching_examples || []).includes("visible"),
        JSON.stringify(arm.fired));
  check("and its height probe moved nothing either, so both channels agree",
        arm.took_effect.compared > 0 && arm.took_effect.changed === 0,
        JSON.stringify(arm.took_effect));
  check("the apply detail SAYS the rule was not found instead of quoting a selector it never saw",
        arm.apply_detail.indexOf("NO RULE MENTIONING [" + PRODUCT_ATTR + "]") === 0
        && arm.apply_detail.indexOf("applies NOTHING") > 0, arm.apply_detail);
  check("and the scene notes it once, naming the build rather than the arm",
        (payload.notes || []).some((n) => n.indexOf("built without the product fix") >= 0),
        JSON.stringify(payload.notes));
  check("the attribute is still reverted, even though it bought nothing",
        !(PRODUCT_ATTR in world.root.attrs), JSON.stringify(world.root.attrs));
  }
}

console.log("scenario 7c: the product rule inside a grouping rule is still found");
{
  // A bundler is free to emit the rule inside `@media`, `@supports` or `@layer`. A census that
  // does not recurse would report a build that HAS the fix as one that does not, and the arm would
  // be voided for a fault that is in the census.
  const { payload } = await runScene({ productRuleNested: true, crossOriginSheets: 0,
                                       only: ["product_math_block_containment"] });
  const P = payload.product_rule;
  const arm = (payload.arms || [])[0];
  check("the scene completed and posted its product-rule precondition",
        Boolean(payload && payload.ok === true && P),
        JSON.stringify((payload && (payload.error_detail || payload.error))
                       || Object.keys(payload || {})));
  if (!P) {
    check("scenario 7c cannot continue without a posted precondition", false,
          "the scene did not reach the end of its run");
  } else {
  check("the nested rule was found, with its selectorText verbatim",
        P.present === true && P.selector_text === PRODUCT_SELECTOR_TEXT, JSON.stringify(P.matches));
  check("and the toggle still moves computed style",
        P.toggle_check.moved === P.toggle_check.sampled && P.toggle_check.sampled > 0,
        JSON.stringify(P.toggle_check));
  check("and with no cross-origin sheet the unreadable count is zero, not a constant",
        P.sheets_unreadable === 0, JSON.stringify([P.sheets_readable, P.sheets_unreadable]));
  check("the arm fired", arm && arm.fired && arm.fired.fired === true,
        JSON.stringify(arm && arm.fired));
  }
}

console.log("scenario 7d: THE PAGE BOOTS FIXED -- the ship default is now `contain`");
{
  // The case this whole revision exists for. `SHIP_DEFAULT` on the head of PR 9731 is `"contain"`,
  // so the page arrives with the attribute already set and every baseline would be the FIXED
  // state. A session run unchanged would report every arm flat and be read as "the fix does
  // nothing", which is the harness's one rule (base does not exhibit the defect => VOID) arriving
  // silently. The scene has to clear it before anything is measured, and say so in the payload.
  const { payload, world } = await runScene({ bootContained: true, crossOriginSheets: 0,
                                              only: ["baseline", "product_math_block_containment"] });
  const B = payload && payload.math_containment_boot;
  check("the scene completed and posted its boot readback",
        Boolean(payload && payload.ok === true && B),
        JSON.stringify((payload && (payload.error_detail || payload.error))
                       || Object.keys(payload || {})));
  if (!B) {
    check("scenario 7d cannot continue without a posted boot readback", false,
          "the scene did not reach the end of its run");
  } else {
  check("it SAW the page boot in the fixed state, rather than assuming a flag it could not read",
        B.attribute_at_boot === "on" && B.ship_default_was_on === true, JSON.stringify(B));
  check("it drove the product's OWN runtime override, and the override took",
        B.override_assigned === true && world.overrideWrites.length === 1
        && world.overrideWrites[0] === false, JSON.stringify(world.overrideWrites));
  check("the attribute is gone immediately after, read back off the document element",
        B.attribute_after === null, JSON.stringify(B));
  check("and gone again once the thread has really mounted, with real blocks to sample",
        B.post_mount && B.post_mount.attribute === null && B.post_mount.blocks > 0
        && B.post_mount.any_auto === false
        && B.post_mount.cv.every((v) => v === "visible"), JSON.stringify(B.post_mount));
  const base = (payload.arms || []).filter((a) => a.arm === "baseline");
  check("so the BASELINE windows are the flag-off arm, which is the premise of the whole run",
        base.length > 0, JSON.stringify((payload.arms || []).map((a) => a.arm)));
  const arm = (payload.arms || []).find((a) => a.arm === "product_math_block_containment");
  check("and the product arm still turns it back on and fires",
        Boolean(arm && arm.fired && arm.fired.fired === true), JSON.stringify(arm && arm.fired));
  check("the toggle check saw `visible` off and `auto` on, not `auto` on both sides",
        payload.product_rule.toggle_check.off_values.every((v) => v === "visible")
        && payload.product_rule.toggle_check.moved
           === payload.product_rule.toggle_check.sampled,
        JSON.stringify(payload.product_rule.toggle_check));
  check("and the attribute is off again at the end, so nothing leaks into a later window",
        !(PRODUCT_ATTR in world.root.attrs), JSON.stringify(world.root.attrs));
  }
}

console.log("scenario 7e: THE CLEAR DOES NOT TAKE -- the readback must say so, not stay quiet");
{
  // Anti-vacuity for 7d. Every assertion above would pass on a scene that cleared nothing if the
  // stub could not boot contained; this is the other direction, where the clear is defeated. The
  // readback has to REPORT the fixed state rather than reporting what it asked for. The criteria
  // module's `flag_off_premise` gate is what turns this into a VOID; here the only claim is that
  // the scene's own record can carry the bad news.
  const { payload } = await runScene({ bootContained: true, attributeStuck: true,
                                       overrideWatcher: false, crossOriginSheets: 0,
                                       only: ["baseline", "product_math_block_containment"] });
  const B = payload && payload.math_containment_boot;
  check("the scene still completed and still posted a boot readback",
        Boolean(payload && payload.ok === true && B),
        JSON.stringify((payload && (payload.error_detail || payload.error))
                       || Object.keys(payload || {})));
  if (!B) {
    check("scenario 7e cannot continue without a posted boot readback", false,
          "the scene did not reach the end of its run");
  } else {
  check("the attribute is REPORTED as still set, which is the answer 7d could not produce",
        B.attribute_at_boot === "on" && B.attribute_after === "on", JSON.stringify(B));
  check("and the post-mount sample reports real blocks computing to `auto`",
        B.post_mount && B.post_mount.attribute === "on" && B.post_mount.blocks > 0
        && B.post_mount.any_auto === true, JSON.stringify(B.post_mount));
  }
}

console.log(failures ? `\n${failures} FAILED` : "\nall scene self-tests passed");
process.exit(failures ? 1 : 0);
