// ONE-PIXEL scroll ablation: what does `content-visibility: auto` on KaTeX actually buy?
//
// This is `amdv_layers.js`, the scene that produced run 32869180652 (baseline 287.6 ms blocked
// per scroll event = 3.2 fps, `.katex{visibility:hidden}` 10.0 ms = 61.0 fps, +97%), with arms
// added and NOTHING ELSE CHANGED about the instrument. Same one-pixel gesture, same fixed park,
// same drained fence reservoir, same discarded warm-up, same interleaved baselines, same
// thresholds.
//
// SECOND ATTEMPT, after run 32876363634. That run passed every instrument gate at every rung and
// failed exactly one: the reference upper bound. `.katex{visibility:hidden}` removed 90% at
// r100K, as published, but only 55% at r500K against the published 97%. Two things changed here
// and nothing else: the upper bound now runs TWICE, early and late, so order-dependence is
// measured rather than argued; and the exploratory arm's placeholder height is now the
// arithmetic MEAN of the blocks it covers rather than the median, because the scrollHeight error
// a placeholder introduces is the SUM of (placeholder - actual) and the mean is the value that
// zeroes it.
//
// WHAT IS BEING ASKED. `.katex{visibility:hidden}` is a MECHANISM PROBE, not a shippable fix: it
// works by making the maths invisible. Its 97% is an UPPER BOUND on what a real fix could
// recover. `content-visibility: auto` is the shippable form of the same idea -- off-screen maths
// generates no boxes and no layers, on-screen maths renders exactly as it does today -- and the
// three new arms measure it, bound it, and say honestly where it cannot reach:
//
//   * `content_visibility_katex_display`  THE SHIPPABLE FIX, exactly the rule the PR adds.
//   * `content_visibility_katex_all`      the same rule with `.katex` added to the selector.
//                                         `content-visibility` needs SIZE CONTAINMENT to take
//                                         effect and size containment does not apply to a
//                                         non-atomic inline-level box (css-contain-2
//                                         #containment-size; WebKit checks it in
//                                         `Style::ContainmentChecker::shouldApplySizeContainment`).
//                                         Shipped `katex.css` gives `.katex` no `display`, so
//                                         inline maths is `display: inline` and the declaration
//                                         is silently inert on it. 910 of the 1,027 roots in this
//                                         corpus are inline. If that reading is right this arm
//                                         reads the SAME as the one above, and this is the arm
//                                         that proves it rather than asserting it.
//   * `content_visibility_math_blocks`    EXPLORATORY, NOT SHIPPABLE AS WRITTEN. Hoists the
//                                         declaration to the nearest block-level ancestor of
//                                         every maths root, which is what a renderer-side change
//                                         would have to do to reach inline maths. It bounds what
//                                         that work would be worth before anyone does it.
//   * `product_math_block_containment`    THE PRODUCT IMPLEMENTATION of that same hoist, measured
//                                         instead of imitated. The branch adds the class
//                                         `.aui-math-block` to the nearest block-level ancestor of
//                                         every INLINE maths root at render time, ALWAYS, whether
//                                         the feature is on or off, and ships the declaration in
//                                         the product bundle gated on
//                                         `html[data-math-block-containment="on"]`. So this arm
//                                         applies NO STYLESHEET OF ITS OWN: it sets one attribute
//                                         on the document element and takes it off again.
//
// THE DANGEROUS FAILURE OF THE PRODUCT ARM, and it is dangerous precisely because it is quiet.
// An arm that applies nothing of its own is INERT when the bundle under test does not contain the
// rule it is toggling, and an inert arm reads as a clean null -- which looks exactly like "the
// product implementation does not reproduce the harness arm" when what actually happened is that
// the wrong build was measured. That is defect #35/#13 (a vacuous arm) arriving through defect #50
// (an arm's identity is the selector it applied, never its label). So before any window is
// measured this scene walks `document.styleSheets` and records, VERBATIM, the selectorText and
// cssText of the product rule it found, how many sheets it could not read, and a BEHAVIOURAL
// toggle check: the computed `content-visibility` of a sample of `.aui-math-block` elements with
// the attribute off, then with it on, then off again. Those two facts together say "the rule
// shipped" without depending on measuring anything, and the criteria module VOIDS the arm rather
// than reporting its null when they do not hold.
//
// The class those blocks carry is added BEFORE the warm-up, so every window including every
// baseline sees the same DOM and the arm only ever toggles a stylesheet. An arm that mutates the
// DOM inside its own measured window is billing itself for someone else's work (defect #48).
//
// THE QUESTION. On the local llvmpipe WebKitGTK rig the 500K scroll cost is attributed to
// `RenderLayer::recursiveUpdateLayerPositionsAfterScroll` walking ~22,000 descendant
// RenderLayers once per scroll EVENT, and `.katex *{position:static}` removes 79% of it. On a
// real GPU `usesCompositedScrolling()` is true, which adds `setNeedsCompositingGeometryUpdate()`,
// `setDescendantsNeedUpdateBackingAndHierarchyTraversal()` and `updateCompositingLayersAfterScroll()`
// -- a SECOND full descendant walk per scroll that llvmpipe never takes. If that is right the
// attribution must transfer HERE and be LARGER here.
//
// WHY ONE PIXEL, and not the 900 px gesture the earlier ablation used. Locally, assigning the
// SAME scrollTop costs 1.8 ms/frame while changing it by ONE PIXEL costs 30.4 ms/frame. So a
// 1 px change buys the entire per-scroll-EVENT descendant walk and almost no per-pixel paint or
// compositing. A 120 px or 900 px gesture confounds the walk with repaint of newly exposed
// content, and cannot separate "the walk is expensive" from "painting is expensive".
//
// ONE SCROLL EVENT PER PAINTED FRAME. The gesture is paced by requestAnimationFrame, so
// `blocked_ms_per_frame` IS the cost of one scroll event. That is the primary metric on purpose:
// `busy%` is a fraction of WALL time, and on this campaign it hid a 62% work reduction as a 19%
// change, because when frames get cheaper more of them fit into the same wall clock.
//
// THE CONTROLS, none of which are optional:
//   * LIVENESS (idle_jammed): the same idle window with the main thread deliberately blocked.
//     Priced at IDLE, not during the gesture, because the gesture window is already saturated
//     and a control priced where the page is saturated cannot resolve. It MUST fall hard here or
//     nothing below this line is readable.
//   * NEGATIVE (noop_touch): touch every message node, add a class that styles nothing. Must NOT
//     recover. If it does, the arms measure the act of mutating the DOM.
//   * POSITIVE (detach_messages): keep two messages. Must recover. Destructive, so it runs last.
//   * FLOOR (still_no_scroll): assign the SAME scrollTop every frame. No position change, so WebKit's
//     scrollTo early-outs and no descendant walk happens at all. This is the floor the arms are
//     trying to reach, measured on the very same page rather than assumed.
//   * DRIFT (baseline between every arm, plus baseline_repeat): a baseline runs before and after
//     every arm, so each arm is scored against its OWN two neighbours. The local run of the
//     earlier ablation failed exactly this check (17.5 vs 30.1 fps) and that is what invalidated
//     it. A drift-corrected number is the only honest one.
//   * WARM-UP (discarded): run 32865232787 failed its own drift gate, reading a first baseline of
//     261.4 ms blocked per scroll event against a 162.6 / 193.9 / 197.5 / 184.5 cluster, while
//     the DOM grew 11,205 elements across the session with `.katex` (1,027), `.katex *`
//     (101,306) and `pre` (330) all pinned. The cause is `code-fence-defer.tsx`, which ships
//     `SHIP_DEFAULT = "defer"` and latches a deferred fence into a highlighted block whose spans
//     land INSIDE the `pre` that already existed. It latches on an IntersectionObserver with a
//     100% root margin, which re-delivers on LAYOUT CHANGE with no scroll at all, so any arm
//     that changes the thread's height mounts markup and permanently contaminates every window
//     after it. This scene therefore drains the reservoir through the app's own
//     `beforeprint` -> `upgradeEverythingForPrint` path, then runs the identical gesture at the
//     identical park position until the DOM stops changing, and throws that reading away. The
//     threshold on the drift gate is UNCHANGED: the artefact is removed, not accommodated.
//   * FIRED (per arm): a sample of the nodes the CSS claims to target is re-read through
//     getComputedStyle after the arm is applied. An arm whose declaration was DROPPED reads as a
//     clean null otherwise -- three `overflow-anchor` arms in this campaign were vacuous that way,
//     because CSSScrollAnchoringEnabled is false in every WebKitGTK this campaign has touched.
//   * SCROLLHEIGHT (per arm): recorded before, after apply and after the gesture. An arm that
//     collapses scrollHeight pins the gesture, and one such arm already reported 111 fps at 184%
//     busy and meant nothing.
(() => {
  if (window.__cvk) return;
  const W = { gaps: [], ticks: [], marks: [], arms: [], notes: [] };
  window.__cvk = W;

  const nativeRaf = window.requestAnimationFrame.bind(window);
  let last = performance.now();
  const frame = (now) => { W.gaps.push(now - last); last = now; nativeRaf(frame); };
  nativeRaf(frame);

  let lastTick = performance.now();
  const tick = () => {
    const now = performance.now();
    W.ticks.push(now - lastTick); lastTick = now;
    setTimeout(tick, 1);
  };
  setTimeout(tick, 1);

  // ── WHAT IS MUTATING, and where ───────────────────────────────────────────────────────────
  //
  // A count alone would not have settled the last run's drift: the census showed 11,205 elements
  // appearing and could not say what they were. This attributes every added node to the nearest
  // ancestor that identifies it, so "the first-visit work" is a named thing with a number rather
  // than a hypothesis. It is installed before anything else so it sees the whole session.
  const MUT = { added_nodes: 0, added_elements: 0, removed_nodes: 0, records: 0, buckets: {} };
  const bucketOf = (n) => {
    let e = (n && n.nodeType === 1) ? n : (n && n.parentElement);
    let hops = 0;
    while (e && hops++ < 40) {
      const cls = String(e.className || "");
      if (e.tagName === "PRE" || e.tagName === "CODE") return "inside_pre_or_code";
      if (cls.split && cls.split(" ").indexOf("katex") >= 0) return "inside_katex";
      if (e.getAttribute) {
        if (e.getAttribute("data-streamdown")) return "streamdown:" + e.getAttribute("data-streamdown");
        const slot = e.getAttribute("data-slot");
        if (slot) return "data-slot:" + slot;
        if (e.getAttribute("data-role")) return "message_direct";
      }
      e = e.parentElement;
    }
    return "other";
  };
  try {
    new MutationObserver((list) => {
      for (const m of list) {
        MUT.records++;
        MUT.removed_nodes += m.removedNodes.length;
        for (const n of m.addedNodes) {
          MUT.added_nodes++;
          let size = 1;
          if (n.nodeType === 1 && n.getElementsByTagName) {
            size = 1 + n.getElementsByTagName("*").length;
          } else if (n.nodeType !== 1) {
            size = 0;
          }
          MUT.added_elements += size;
          const b = bucketOf(n);
          MUT.buckets[b] = (MUT.buckets[b] || 0) + size;
        }
      }
    }).observe(document.documentElement,
               { childList: true, subtree: true, characterData: false });
  } catch (e) { /* recorded as a note once W exists */ }
  const mutSnapshot = () => ({ added_nodes: MUT.added_nodes, added_elements: MUT.added_elements,
                               removed_nodes: MUT.removed_nodes, records: MUT.records,
                               buckets: Object.assign({}, MUT.buckets) });
  const mutDelta = (a, b) => {
    const d = { added_nodes: b.added_nodes - a.added_nodes,
                added_elements: b.added_elements - a.added_elements,
                removed_nodes: b.removed_nodes - a.removed_nodes,
                records: b.records - a.records, buckets: {} };
    for (const k of Object.keys(b.buckets)) {
      const v = b.buckets[k] - (a.buckets[k] || 0);
      if (v) d.buckets[k] = v;
    }
    return d;
  };

  const q = (s) => document.querySelector(s);
  const post = (o) => {
    if (o && o.__done) W.result = o;
    try { window.webkit.messageHandlers.bench.postMessage(JSON.stringify(o)); } catch (e) {}
  };
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  // JavaScriptCore's Error.prototype.stack does NOT carry the message, so a scene that logs only
  // the stack reports a nameless failure. Both, always.
  const err = (e) => ({ message: String((e && e.message) || e),
                        name: (e && e.name) || null,
                        stack: String((e && e.stack) || "").slice(0, 1200) });
  const waitFor = async (fn, ms, label) => {
    const dl = performance.now() + ms;
    let lastErr = null;
    while (performance.now() < dl) {
      let v; try { v = fn(); } catch (e) { v = null; lastErr = err(e); }
      if (v) return v;
      await sleep(100);
    }
    const e = new Error("timeout waiting for " + label
                        + (lastErr ? " (last error: " + lastErr.message + ")" : ""));
    throw e;
  };

  const pc = (s, p) => s.length
    ? s[Math.min(s.length - 1, Math.max(0, Math.round(p * (s.length - 1))))] : null;
  const summarise = (arr) => {
    const s = arr.slice().sort((a, b) => a - b);
    if (!s.length) return { n: 0 };
    return { n: s.length, p50_ms: pc(s, 0.5), p95_ms: pc(s, 0.95), max_ms: s[s.length - 1],
             over_100: s.filter((g) => g > 100).length };
  };

  const MAX_CLAMP_MS = 10.0;
  let clamp = { clamp_ms: null, reason: "not calibrated" };
  const calibrate = (t) => {
    const s = t.slice().sort((a, b) => a - b);
    if (s.length < 40) return { clamp_ms: null, reason: `only ${s.length} idle ticks` };
    const med = pc(s, 0.5);
    if (!(med > 0) || med > MAX_CLAMP_MS) {
      return { clamp_ms: null, samples: s.length, p50: med,
               reason: `idle tick median ${med && med.toFixed(1)} ms is not a floor` };
    }
    return { clamp_ms: med, samples: s.length, p50: med };
  };
  // THE PRIMARY METRIC IS A MEAN, AND A MEAN OVER ONE OUTLIER IS A REPORT OF WHERE THE OUTLIER
  // LANDED. This is not hypothetical: on this venue at r500K the app blocks the main thread for
  // about 8.6 SECONDS every 30 SECONDS. Re-scored from the untouched rAF series of runs
  // 32869180652 and 32876363634, every window longer than about 3 s catches exactly one of
  // those, and every window shorter than that catches none. `.katex{visibility:hidden}` runs in
  // 1.8 s when it works, so in the published run it missed the stall and read 10.0 ms per frame,
  // and in the next session it caught one and read 112.9 -- an 11x difference produced entirely
  // by one frame, with both sessions reporting an identical p50 of 17 ms and p95 of 18 ms.
  //
  // So every window now also reports what it costs with its single worst frame removed, and how
  // many frames over one second it contained. Neither replaces the mean; a window whose two
  // numbers disagree is a window that caught a stall, and that is a fact about the window rather
  // than about the arm.
  const robustOver = (t, g, el) => {
    if (clamp.clamp_ms === null || !(el > 0) || !t.length || !g.length) {
      return { blocked_ms: null, frames: null, blocked_ms_per_frame: null,
               stall_frames_over_1s: null, worst_tick_ms: null, worst_gap_ms: null };
    }
    const ts = t.slice().sort((a, b) => b - a);
    const gs = g.slice().sort((a, b) => b - a);
    let b = 0;
    for (let i = 1; i < ts.length; i++) { const x = ts[i] - clamp.clamp_ms; if (x > 0) b += x; }
    const frames = Math.max(1, g.length - 1);
    return { blocked_ms: Math.round(b), frames,
             blocked_ms_per_frame: Math.round((b / frames) * 10) / 10,
             stall_frames_over_1s: g.filter((x) => x > 1000).length,
             worst_tick_ms: Math.round(ts[0]), worst_gap_ms: Math.round(gs[0]) };
  };

  const busyOver = (t, el) => {
    if (clamp.clamp_ms === null || !(el > 0)) {
      return { busy_pct: null, blocked_ms: null,
               busy_pct_reason: clamp.reason || "no elapsed time" };
    }
    let b = 0;
    for (const g of t) { const x = g - clamp.clamp_ms; if (x > 0) b += x; }
    return { busy_pct: Math.round((b / el) * 1000) / 10, blocked_ms: Math.round(b) };
  };

  // ── the scroller, and the guard that stops a real gesture reading as inert ────────────────
  //
  // `.aui-thread-viewport` carries `scroll-smooth`. Under `scroll-behavior: smooth` an assignment
  // starts an ANIMATION, so a same-turn read-back returns the position it started from and the
  // gesture reads as if nothing moved. Force it to auto and PROVE the computed value took, rather
  // than assuming it.
  const scrollerAt = (x, y) => {
    let el = document.elementFromPoint(x, y);
    while (el && el !== document.body) {
      if (el.scrollHeight > el.clientHeight + 40) {
        const ov = getComputedStyle(el).overflowY;
        if (ov === "auto" || ov === "scroll") return el;
      }
      el = el.parentElement;
    }
    return document.scrollingElement;
  };

  const style = (id, css) => {
    let s = document.getElementById(id);
    if (!s) { s = document.createElement("style"); s.id = id; document.head.appendChild(s); }
    s.textContent = css;
  };
  const unstyle = (id) => { const s = document.getElementById(id); if (s) s.remove(); };

  const census = (el) => ({
    elements: document.getElementsByTagName("*").length,
    messages: document.querySelectorAll("[data-role]").length,
    message_descendants: document.querySelectorAll("[data-role] *").length,
    katex_roots: document.querySelectorAll(".katex").length,
    katex_descendants: document.querySelectorAll(".katex *").length,
    katex_display: document.querySelectorAll(".katex-display").length,
    // The two maths populations, kept apart everywhere. `content-visibility` can reach the first
    // and provably cannot reach the second, so a total over both would hide the only number that
    // decides how much this fix can be worth.
    katex_display_descendants: document.querySelectorAll(".katex-display *").length,
    math_blocks: document.querySelectorAll(".cvk-mathblock").length,
    // The PRODUCT's own marker class, counted separately from the harness's. The branch emits it
    // at render time whether the feature is on or off, so this count must be the SAME in every
    // window of the session including every baseline; if it moves, the DOM is not identical
    // across windows and defect #48 is back.
    product_math_blocks: document.querySelectorAll(".aui-math-block").length,
    code_blocks: document.querySelectorAll("pre").length,
    // Run 32865232787 grew the DOM by 11,205 elements across one session while `.katex`,
    // `.katex *` and `pre` all stayed pinned, so the growth was none of those. These buckets
    // exist to name it rather than guess at it.
    highlight_spans: document.querySelectorAll("pre span").length,
    // The deferred code fences that have NOT yet upgraded. `code-fence-defer.tsx` ships with
    // `SHIP_DEFAULT: FenceMode = "defer"`, so a fence renders a plain shell containing a real
    // `<pre><code>` and swaps in the highlighted block later. That is why the last run saw
    // 11,205 elements appear while `pre` never moved: the spans land INSIDE a `pre` that already
    // existed.
    fences_deferred: document.querySelectorAll('[data-unsloth-fence-deferred="true"]').length,
    code_block_bodies: document.querySelectorAll('[data-streamdown="code-block-body"]').length,
    all_spans: document.getElementsByTagName("span").length,
    buttons: document.getElementsByTagName("button").length,
    svgs: document.getElementsByTagName("svg").length,
    data_slots: document.querySelectorAll("[data-slot]").length,
    doc_scroll_height: (document.scrollingElement || document.body).scrollHeight,
    scroller_scroll_height: el ? el.scrollHeight : null,
    scroller_client_height: el ? el.clientHeight : null,
  });

  // A sampled estimate of how many elements are POSITIONED, because
  // `RenderBoxModelObject::requiresLayer()` returns true for `isPositioned()` and `isPositioned()`
  // is `position != static`. A full getComputedStyle scan over ~112,000 elements is itself a
  // multi-second layout-flushing cost, so this samples and reports the sample size.
  const positionedEstimate = (sel, cap) => {
    const all = document.querySelectorAll(sel);
    const n = all.length;
    if (!n) return { selector: sel, total: 0, sampled: 0, non_static: 0, estimate: 0 };
    const stride = Math.max(1, Math.floor(n / cap));
    let sampled = 0, nonStatic = 0;
    for (let i = 0; i < n; i += stride) {
      const cs = getComputedStyle(all[i]);
      sampled++;
      if (cs.position !== "static") nonStatic++;
    }
    return { selector: sel, total: n, sampled, non_static: nonStatic,
             estimate: Math.round(n * (nonStatic / sampled)) };
  };

  // DID THE ARM FIRE? Re-read the property the arm claims to set, on a sample of the very nodes
  // the selector names. Returns the share that now carry the intended value.
  const firedCheck = (sel, prop, want, cap) => {
    const all = document.querySelectorAll(sel);
    const n = all.length;
    if (!n) return { selector: sel, prop, want, total: 0, sampled: 0, matching: 0,
                     fraction: null, fired: false,
                     note: "selector matched nothing, so this arm removed nothing" };
    const stride = Math.max(1, Math.floor(n / cap));
    let sampled = 0, ok = 0;
    const examples = [];
    for (let i = 0; i < n; i += stride) {
      const cs = getComputedStyle(all[i]);
      sampled++;
      const v = cs[prop];
      if (v === want) ok++;
      else if (examples.length < 5) examples.push(v);
    }
    return { selector: sel, prop, want, total: n, sampled, matching: ok,
             fraction: Math.round((ok / sampled) * 1000) / 1000,
             fired: ok === sampled, non_matching_examples: examples };
  };

  // DID THE DECLARATION TAKE EFFECT, as opposed to merely being accepted?
  //
  // `getComputedStyle(el).contentVisibility` returns the SPECIFIED value whether or not the
  // engine is skipping anything, so for `content-visibility` the `firedCheck` above answers the
  // wrong question: it would report `auto` on an inline `.katex` that the engine is ignoring
  // completely. The behavioural signal is size containment. A skipped root is size-contained, so
  // its used height becomes the `contain-intrinsic-size` instead of the height its contents would
  // produce, and its descendants stop being laid out.
  //
  // So this snapshots the heights of a fixed, index-strided sample of OFF-SCREEN elements, and
  // the same sample is taken again after the arm is applied. Same indices, same elements, so the
  // two are comparable element by element. An arm that changed nothing returns zero changed.
  const heightProbe = (sel, cap) => {
    const all = document.querySelectorAll(sel);
    const n = all.length;
    if (!n) return { selector: sel, total: 0, sampled: 0, offscreen: 0, heights: [] };
    const stride = Math.max(1, Math.floor(n / cap));
    const vh = window.innerHeight;
    const heights = [], idx = [];
    let offscreen = 0;
    for (let i = 0; i < n; i += stride) {
      const r = all[i].getBoundingClientRect();
      const visible = r.bottom > 0 && r.top < vh;
      if (visible) continue;
      offscreen++;
      idx.push(i);
      heights.push(Math.round(r.height * 100) / 100);
    }
    return { selector: sel, total: n, sampled: idx.length, offscreen, indices: idx, heights };
  };
  const heightDelta = (before, after) => {
    if (!before || !after || !before.heights || !after.heights) return null;
    const n = Math.min(before.heights.length, after.heights.length);
    let changed = 0, shrank = 0, sumBefore = 0, sumAfter = 0;
    for (let i = 0; i < n; i++) {
      const a = before.heights[i], b = after.heights[i];
      sumBefore += a; sumAfter += b;
      if (Math.abs(a - b) > 0.5) { changed++; if (b < a) shrank++; }
    }
    return { compared: n, changed, shrank,
             fraction_changed: n ? Math.round((changed / n) * 1000) / 1000 : null,
             mean_height_before: n ? Math.round((sumBefore / n) * 100) / 100 : null,
             mean_height_after: n ? Math.round((sumAfter / n) * 100) / 100 : null,
             selector: before.selector, total: before.total,
             offscreen_before: before.offscreen, offscreen_after: after.offscreen };
  };

  // ── THE PRODUCT IMPLEMENTATION'S OWN NAMES, frozen, and never re-derived from a label ──────
  //
  // These three strings are the contract with the branch under test. The class is emitted at
  // render time on the nearest block-level ancestor of every INLINE maths root, ALWAYS, so the DOM
  // is identical whether the feature is on or off (defect #48). Display maths needs no marker: its
  // own `span.katex-display` root is already block-level, and the product rule names it directly.
  // The attribute is the only thing the feature flag moves, and it lives on the document element.
  const PRODUCT_ATTR = "data-math-block-containment";
  const PRODUCT_BLOCK = "aui-math-block";
  const PRODUCT_SEL = "." + PRODUCT_BLOCK;

  // DOES THE RULE THE ARM TOGGLES ACTUALLY EXIST IN THIS BUILD?
  //
  // This is the precondition the product arm cannot be read without. The arm applies no stylesheet
  // of its own, so if the bundle was built without the fix the attribute toggle is INERT and the
  // window reads as a clean null -- indistinguishable, from the numbers alone, from a product
  // implementation that does not work. Walking the loaded stylesheets and quoting the matched rule
  // VERBATIM is the only way to tell those two apart, and it does not depend on measuring
  // anything.
  //
  // Cross-origin stylesheets throw on `.cssRules` (CSSOM, and WebKit enforces it), so every sheet
  // is read inside its own try and the number that could not be read is REPORTED rather than
  // silently folded into "not found": "the rule is absent" and "the rule may be in a sheet I could
  // not open" are different answers and must not be printed as the same one.
  const findProductRule = () => {
    const sheets = document.styleSheets || [];
    const out = { attribute: PRODUCT_ATTR, block_class: PRODUCT_BLOCK, block_selector: PRODUCT_SEL,
                  sheets_total: sheets.length, sheets_readable: 0, sheets_unreadable: 0,
                  unreadable: [], rules_scanned: 0, matches: [] };
    const visit = (rules, href, depth) => {
      if (!rules || depth > 6) return;
      for (let i = 0; i < rules.length; i++) {
        const r = rules[i];
        out.rules_scanned++;
        let sel = null;
        try { sel = r.selectorText; } catch (e) { sel = null; }
        if (typeof sel === "string" && sel.indexOf(PRODUCT_ATTR) >= 0) {
          let css = "";
          try { css = String(r.cssText || ""); }
          catch (e) { css = "(cssText threw: " + err(e).message + ")"; }
          out.matches.push({ selector_text: String(sel), css_text: css.slice(0, 800), sheet: href });
        }
        // Grouping rules (`@media`, `@supports`, `@layer`) carry their own `cssRules`, and a
        // bundler is free to put the product rule inside one.
        let kids = null;
        try { kids = r.cssRules; } catch (e) { kids = null; }
        if (kids && kids.length) visit(kids, href, depth + 1);
      }
    };
    for (let i = 0; i < sheets.length; i++) {
      const s = sheets[i];
      const href = String((s && s.href) || "(inline <style>)");
      let rules = null;
      try { rules = s.cssRules; }
      catch (e) {
        out.sheets_unreadable++;
        if (out.unreadable.length < 8) out.unreadable.push({ sheet: href, why: err(e).message });
        continue;
      }
      if (rules === null || rules === undefined) {
        out.sheets_unreadable++;
        if (out.unreadable.length < 8) out.unreadable.push({ sheet: href, why: "cssRules was null" });
        continue;
      }
      out.sheets_readable++;
      visit(rules, href, 0);
    }
    out.present = out.matches.length > 0;
    // VERBATIM, both of them, and taken from the MATCH LIST rather than from the `present` flag,
    // so the quoted rule and the claim that there is one cannot disagree. A census that answers
    // YES and quotes nothing is exactly the shape this record exists to prevent, and deriving the
    // quotes from the flag would also let a wrong flag throw in here and lose the whole payload.
    out.selector_text = out.matches.length ? out.matches[0].selector_text : null;
    out.css_text = out.matches.length ? out.matches[0].css_text : null;
    return out;
  };

  // THE BEHAVIOURAL HALF OF THE SAME PRECONDITION. Finding the rule's text says the bundle carries
  // it; this says the engine ACTS on it. Sample the computed `content-visibility` of the product's
  // own blocks with the attribute OFF (expected `visible`, or `normal` on an engine that reports
  // the initial value that way), set the attribute, sample the SAME elements again (expected
  // `auto`), then put the attribute back exactly as it was found. Nothing else in the session may
  // observe a difference, so the restore is unconditional and is itself recorded.
  const productToggleCheck = (cap) => {
    const all = document.querySelectorAll(PRODUCT_SEL);
    const n = all.length;
    const was = document.documentElement.getAttribute(PRODUCT_ATTR);
    const res = { selector: PRODUCT_SEL, total: n, sampled: 0, attribute_before: was,
                  off_values: [], on_values: [], moved: 0, moved_fraction: null,
                  auto_when_on: 0, note: null };
    if (!n) {
      res.note = "no `" + PRODUCT_SEL + "` element exists at this rung, so there is nothing to "
               + "toggle. At an empty thread that is the CORRECT answer and not a failure, but "
               + "the RULE must still be present in the bundle here";
      res.attribute_after = document.documentElement.getAttribute(PRODUCT_ATTR);
      return res;
    }
    const stride = Math.max(1, Math.floor(n / cap));
    const sample = [];
    for (let i = 0; i < n; i += stride) sample.push(all[i]);
    for (const e of sample) res.off_values.push(getComputedStyle(e).contentVisibility);
    document.documentElement.setAttribute(PRODUCT_ATTR, "on");
    for (const e of sample) res.on_values.push(getComputedStyle(e).contentVisibility);
    if (was === null || was === undefined) document.documentElement.removeAttribute(PRODUCT_ATTR);
    else document.documentElement.setAttribute(PRODUCT_ATTR, was);
    res.attribute_after = document.documentElement.getAttribute(PRODUCT_ATTR);
    res.sampled = sample.length;
    for (let i = 0; i < sample.length; i++) {
      if (res.on_values[i] === "auto") res.auto_when_on++;
      if (res.on_values[i] === "auto" && res.off_values[i] !== "auto") res.moved++;
    }
    res.moved_fraction = Math.round((res.moved / res.sampled) * 1000) / 1000;
    return res;
  };

  const cut = { g: 0, t: 0 };
  const slice = () => {
    const g = W.gaps.slice(cut.g), t = W.ticks.slice(cut.t);
    cut.g = W.gaps.length; cut.t = W.ticks.length;
    return { g, t };
  };

  // ── THE GESTURE: one pixel, once per painted frame ────────────────────────────────────────
  //
  // A FIXED NUMBER OF FRAMES, capped by a wall clock. Fixed frames is what the local rig uses
  // (80), and it is the right primary bound here: every arm then performs exactly the same number
  // of scroll EVENTS, so blocked-ms-per-frame compares equal-sized samples of the same work. The
  // wall cap exists only so that an arm which collapses to a fraction of a frame per second
  // cannot eat the whole job's budget; which bound ended the window is recorded.
  //
  // TRAVEL IS MEASURED ON THE NEXT FRAME, never by reading back on the same turn. Under
  // `scroll-behavior: smooth` an assignment starts an ANIMATION, so a same-turn read-back reports
  // a real gesture as inert -- that is the trap that made a 54,000 px commanded gesture travel
  // 6,610 px while the instrument saw nothing wrong. The guard that forces `auto` is kept as
  // well, but a measurement must not DEPEND on a guard having worked.
  //
  // THE PARK POSITION IS A FIXED ABSOLUTE PIXEL, chosen once and reused by every window. It used
  // to be recomputed as `scrollHeight * 0.5` per window, and that is what broke run
  // 32865232787: `[data-role] *{position:static}` makes the thread 7.3% taller, so the mid point
  // moved ~11,000 px, the window landed on CONTENT THAT HAD NEVER BEEN VISITED, and mounting it
  // was billed to whichever arm happened to move the scroller there. The drift gate caught it as
  // a first baseline of 261.4 ms against a 162-198 ms cluster. A fixed pixel means every window
  // looks at the same content.
  const gesture = async (el, frames, capMs, mode, park) => {
    el.scrollTop = park;
    await sleep(400);
    const behaviour = getComputedStyle(el).scrollBehavior;
    const start = performance.now();
    let commanded = 0, travelled = 0, steps = 0, snapback = 0, dir = 1;
    let want = null;
    const first = el.scrollTop;
    let stoppedBy = "frames";
    await new Promise((resolve) => {
      const step = () => {
        const now = performance.now();
        const cur = el.scrollTop;
        if (want !== null) {
          // What the scroller ACHIEVED by the next frame, not what it was asked for.
          travelled += Math.abs(cur - lastFrom);
          if (Math.abs(cur - want) > 4) snapback++;
        }
        if (steps >= frames) { stoppedBy = "frames"; resolve(); return; }
        if (now - start >= capMs) { stoppedBy = "wall_cap"; resolve(); return; }
        // ONE PIXEL, alternating, so the scroller never drifts anywhere and no new content is
        // ever exposed. The only thing that changes is that the scroll position CHANGED.
        const next = (mode === "still") ? cur : (cur + dir);
        dir = -dir;
        lastFrom = cur;
        el.scrollTop = next;
        want = next;
        commanded += Math.abs(next - cur);
        steps++;
        nativeRaf(step);
      };
      let lastFrom = first;
      nativeRaf(step);
    });
    return { mode, steps, stopped_by: stoppedBy, park,
             elapsed_ms: Math.round(performance.now() - start),
             commanded_px: commanded, travelled_px: travelled,
             travel_fraction: commanded > 0 ? Math.round((travelled / commanded) * 1000) / 1000
                                            : null,
             snapback_frames: snapback, scroll_behavior: behaviour,
             start_top: first, end_top: el.scrollTop,
             scroll_height: el.scrollHeight, span: Math.max(0, el.scrollHeight - el.clientHeight) };
  };

  // ── a deliberate main-thread jam, used ONLY to prove the channel can report one ────────────
  let hogTimer = null;
  const hogOn = (busyMs, periodMs) => {
    if (hogTimer !== null) return;
    hogTimer = setInterval(() => {
      const t = performance.now();
      while (performance.now() - t < busyMs) { /* spin */ }
    }, periodMs);
  };
  const hogOff = () => { if (hogTimer !== null) { clearInterval(hogTimer); hogTimer = null; } };

  // ── the arms ──────────────────────────────────────────────────────────────────────────────
  const ARMS = {
    baseline: {
      why: "the page as it is. Runs before and after every arm, so each arm has its own two "
         + "neighbouring baselines and page drift cannot be read as an arm effect.",
      apply: async () => "none", revert: async () => {}, fired: null },

    noop_touch: {
      why: "NEGATIVE CONTROL. Touch every message node and add a class that styles nothing. Same "
         + "traversal, same mutation count, nothing removed. Must NOT recover.",
      apply: async () => {
        const n = document.querySelectorAll("[data-role]");
        style("lay-noop", ".lay-noop-marker{}");
        n.forEach((e) => e.classList.add("lay-noop-marker"));
        return `touched ${n.length} messages`;
      },
      revert: async () => {
        document.querySelectorAll(".lay-noop-marker").forEach((e) =>
          e.classList.remove("lay-noop-marker"));
        unstyle("lay-noop");
      },
      fired: null },

    position_static_all: {
      why: "`RenderBoxModelObject::requiresLayer()` returns true for `isPositioned()`, and "
         + "`isPositioned()` is `position != static`, so `position: relative` alone buys a "
         + "RenderLayer. Forcing every descendant of every message to static should delete most "
         + "of the ~22,000 layers `recursiveUpdateLayerPositionsAfterScroll` walks per scroll "
         + "event. Locally this removed 78% of the cost.",
      apply: async () => {
        style("lay-ps", "[data-role] *{position:static !important}");
        return "[data-role] *{position:static}";
      },
      revert: async () => unstyle("lay-ps"),
      fired: () => firedCheck("[data-role] *", "position", "static", 400) },

    katex_static: {
      why: "The same removal aimed only at KaTeX, which generates 82% of the elements in this "
         + "corpus. If this reproduces most of position_static_all then KaTeX is the source of "
         + "the layers rather than merely correlated with them. Locally 79%.",
      apply: async () => {
        style("lay-ks", ".katex *{position:static !important}");
        return ".katex *{position:static}";
      },
      revert: async () => unstyle("lay-ks"),
      fired: () => firedCheck(".katex *", "position", "static", 400) },

    // NAMED AFTER ITS SELECTOR, deliberately. The old name for this arm was
    // `visibility_hidden_offscreen`, and `wq_final.js` has a DIFFERENT arm of that exact name
    // which hides off-screen MESSAGES (`.f-vh` on `[data-role]`) rather than KaTeX roots. Two
    // arms, adjacent names, different selectors: that is instrument defect #40, and it cost a
    // round of investigation when this arm read 55% here against a published 97%. It was not the
    // collision -- both sessions recorded `.katex{visibility:hidden}` and both fired 514/514 --
    // but the only reason that could be established is that the payload records the SELECTOR
    // independently of the label. The name now carries the selector so the question cannot be
    // asked again.
    katex_root_visibility_hidden: {
      why: "`recursiveUpdateLayerPositionsAfterScroll` has exactly ONE early-out: "
         + "`!m_hasVisibleDescendant && !m_hasVisibleContent`. `visibility: hidden` on the KaTeX "
         + "roots is the only cheap way to reach it, and it does NOT change layout, so "
         + "scrollHeight must be unchanged. A recovery here indicts the walk specifically rather "
         + "than the layers' existence. Locally 86%, busy 90 -> 46.7%.",
      apply: async () => {
        style("lay-vh", ".katex{visibility:hidden !important}");
        return ".katex{visibility:hidden}";
      },
      revert: async () => unstyle("lay-vh"),
      fired: () => firedCheck(".katex", "visibility", "hidden", 400) },

    // ── the shippable fix, verbatim ──────────────────────────────────────────────────────────
    //
    // The declaration is byte for byte what `studio/frontend/src/index.css` adds, modulo the
    // `.aui-thread-root` scope, which is a no-op inside the thread and is dropped here only so
    // the arm cannot be confused with a scope bug. The placeholder height is chosen from the
    // corpus: rendered `.katex-display` heights over the 500K thread run min 22px, p25 41.7,
    // p50 53.9, p75 56.3, p90 57, max 57, so 3.5rem sits in the middle of the distribution and
    // the `auto` keyword makes the engine remember each block's real size once it has been seen.
    content_visibility_katex_display: {
      why: "THE SHIPPABLE FIX. `content-visibility: auto` reaches the same early-out that "
         + "`.katex{visibility:hidden}` reaches (`computeHasVisibleContent()` returns false for "
         + "`isSkippedContent()`), but only for content the user cannot see, so on-screen maths "
         + "renders exactly as before. `.katex-display` is `display: block` by shipped "
         + "katex.css, so size containment applies and the declaration can take effect.",
      apply: async () => {
        style("cvk-d", ".katex-display{content-visibility:auto;"
                     + "contain-intrinsic-size:auto 3.5rem}");
        return ".katex-display{content-visibility:auto;contain-intrinsic-size:auto 3.5rem}";
      },
      revert: async () => unstyle("cvk-d"),
      fired: () => firedCheck(".katex-display", "contentVisibility", "auto", 400),
      probe: () => heightProbe(".katex-display", 400) },

    content_visibility_katex_all: {
      why: "The same rule with `.katex` added to the selector. Size containment does not apply "
         + "to a non-atomic inline-level box, and 910 of this corpus's 1,027 `.katex` roots are "
         + "`display: inline`, so this arm should read the SAME as the display-only arm. If it "
         + "does, inline maths is out of reach of `content-visibility` as a measured fact rather "
         + "than as a reading of the spec. If it reads BETTER, the reading is wrong and the "
         + "shipped rule is leaving something on the table.",
      apply: async () => {
        style("cvk-a", ".katex,.katex-display{content-visibility:auto;"
                     + "contain-intrinsic-size:auto 3.5rem}");
        return ".katex,.katex-display{content-visibility:auto;contain-intrinsic-size:auto 3.5rem}";
      },
      revert: async () => unstyle("cvk-a"),
      fired: () => firedCheck(".katex", "contentVisibility", "auto", 400),
      // Probed on the INLINE roots specifically, because those are the ones the claim is about.
      probe: () => heightProbe(".katex:not(.katex-display *)", 400) },

    content_visibility_math_blocks: {
      why: "EXPLORATORY, NOT SHIPPABLE AS WRITTEN. Hoists the declaration to the nearest "
         + "block-level ancestor of every maths root, which is what a renderer-side change would "
         + "have to do to reach inline maths, since the inline box itself cannot be contained. "
         + "The class is applied before the warm-up so no window sees a different DOM from any "
         + "other, and the placeholder height is the measured median of those very blocks. It "
         + "bounds what that work would be worth before anyone does it.",
      apply: async () => {
        const h = W.mathBlocks && W.mathBlocks.mean_height ? W.mathBlocks.mean_height : 24;
        style("cvk-m", `.cvk-mathblock{content-visibility:auto;contain-intrinsic-size:auto ${h}px}`);
        return `.cvk-mathblock{content-visibility:auto;contain-intrinsic-size:auto ${h}px} `
             + `over ${(W.mathBlocks || {}).count} blocks`;
      },
      revert: async () => unstyle("cvk-m"),
      fired: () => firedCheck(".cvk-mathblock", "contentVisibility", "auto", 400),
      probe: () => heightProbe(".cvk-mathblock", 400) },

    // ── the PRODUCT implementation of the arm above, measured rather than imitated ────────────
    //
    // A PURE ATTRIBUTE TOGGLE. It injects no stylesheet: the declaration is already in the product
    // bundle, gated on `html[data-math-block-containment="on"]`, and the class the rule selects is
    // emitted at render time whether the feature is on or off. So this arm changes exactly one
    // attribute on the document element, which is the smallest possible difference between the
    // measured window and its neighbouring baselines.
    //
    // BECAUSE IT APPLIES NOTHING OF ITS OWN, IT CANNOT BE READ WITHOUT ITS PRECONDITION. On a
    // bundle built without the fix this toggle is inert and the window is a clean null. The
    // `product_rule` record posted with every payload -- the matched selectorText and cssText
    // verbatim, the count of unreadable sheets, and the off/on computed-style toggle check -- is
    // what separates "the product implementation did not reproduce the harness arm" from "the
    // wrong build was measured", and the criteria module VOIDS this arm rather than reporting its
    // number when that record does not hold up.
    //
    // The apply detail quotes the SELECTOR THE PRODUCT STYLESHEET ITSELF CARRIES, read back out of
    // the page, not a selector this file believes the product uses. Defect #50: an arm's identity
    // is the selector it applied and the fired check that proves it applied, never its label.
    product_math_block_containment: {
      why: "THE PRODUCT IMPLEMENTATION. The chat markdown renderer adds `.aui-math-block` to the "
         + "nearest block-level ancestor of every INLINE maths root, always, and the product "
         + "stylesheet carries `content-visibility: auto` for that class and `.katex-display` "
         + "gated on `html[data-math-block-containment=\"on\"]`. This arm therefore injects NO CSS "
         + "of its own and only sets the attribute, which is what makes it a measurement of the "
         + "shipped code rather than of a rule this harness wrote. It is scored against the same "
         + "neighbouring baselines and the same reference upper bound as every other candidate, "
         + "and it sits next to `content_visibility_math_blocks` so that the product-versus-"
         + "harness comparison is a single table.",
      apply: async () => {
        const pr = W.productRule || {};
        document.documentElement.setAttribute(PRODUCT_ATTR, "on");
        const blocks = document.querySelectorAll(PRODUCT_SEL).length;
        const disp = document.querySelectorAll(".katex-display").length;
        const sel = (pr.present && pr.selector_text)
          ? pr.selector_text
          : `NO RULE MENTIONING [${PRODUCT_ATTR}] WAS FOUND IN ANY READABLE STYLESHEET `
            + `(${pr.sheets_readable} readable, ${pr.sheets_unreadable} unreadable), so this `
            + `attribute toggle applies NOTHING`;
        return `${sel} <- set [${PRODUCT_ATTR}="on"] on <html>, covering ${blocks} `
             + `${PRODUCT_SEL} and ${disp} .katex-display`;
      },
      revert: async () => { document.documentElement.removeAttribute(PRODUCT_ATTR); },
      fired: () => firedCheck(PRODUCT_SEL, "contentVisibility", "auto", 400),
      probe: () => heightProbe(PRODUCT_SEL, 400) },

    still_no_scroll: {
      why: "FLOOR. Assign the SAME scrollTop every frame. The position never changes, so "
         + "`RenderLayerScrollableArea::scrollTo` early-outs and no descendant walk happens. "
         + "This is what a perfect fix would read, measured rather than assumed.",
      apply: async () => "gesture mode: still", revert: async () => {}, fired: null,
      mode: "still" },

    detach_messages: {
      why: "POSITIVE CONTROL. Remove all but the last two messages. Must recover to near the "
         + "floor. If it does not, this harness cannot detect a win and no arm above means "
         + "anything. Destructive, so it is last.",
      destructive: true,
      apply: async () => {
        const n = Array.from(document.querySelectorAll("[data-role]"));
        const keep = new Set(n.slice(-2));
        let removed = 0;
        n.forEach((e) => { if (!keep.has(e)) { e.remove(); removed++; } });
        return `removed ${removed} of ${n.length} messages`;
      },
      revert: null, fired: null },
  };

  // Baseline BETWEEN every arm. `baseline_repeat` is the same arm as the first one and is the
  // named drift check.
  // `position_static_all` and `katex_static` are dropped from this sequence. Both were already
  // DISQUALIFIED on scrollHeight in run 32869180652 (+7.27% and +3.83%), both are ablations that
  // could never ship, and every window they occupy is a window the shippable arms do not get.
  // `visibility_hidden_offscreen` is kept because it is the UPPER BOUND the shippable arms are
  // being read against, and re-measuring it in the same session is the only way to know that
  // this session is the same experiment as the one that produced the 97%.
  // THE REFERENCE UPPER BOUND RUNS TWICE, EARLY AND LATE, and that is the point of this
  // sequence. In run 32876363634 `.katex{visibility:hidden}` removed 90% at r100K but only 55%
  // at r500K, against the 97% run 32869180652 published for the same arm on the same corpus and
  // the same venue. Every other control in that session behaved -- the floor was 6.0 ms, the
  // positive control 2.0 ms, both reps agreed to within a few percent, and one arm in the same
  // wave removed 92% -- so the session was not blind. What differed from the published run is
  // the arm's POSITION: it ran tenth, after three arms that skip and unskip large subtrees,
  // rather than eighth after two that force `position: static`. Running it in both places in one
  // session is the only way to settle whether that history is what moved it, and if the two
  // readings agree then the position hypothesis is dead and the difference is between sessions.
  const SEQUENCE = [
    ["baseline", "baseline"],
    ["noop_touch", "noop_touch"],
    ["baseline", "baseline_2"],
    ["katex_root_visibility_hidden", "katex_root_visibility_hidden"],
    ["baseline", "baseline_3"],
    ["content_visibility_katex_display", "content_visibility_katex_display"],
    ["baseline", "baseline_4"],
    ["content_visibility_katex_all", "content_visibility_katex_all"],
    ["baseline", "baseline_5"],
    ["content_visibility_math_blocks", "content_visibility_math_blocks"],
    ["baseline", "baseline_6"],
    // IMMEDIATELY AFTER the harness's own hoist, so the product implementation and the arm that
    // bounds it are separated by one baseline and nothing else. The two are read against each
    // other in one table, and putting eight windows between them would make that comparison a
    // comparison between two different parts of the session.
    ["product_math_block_containment", "product_math_block_containment"],
    ["baseline", "baseline_7"],
    ["katex_root_visibility_hidden", "katex_root_visibility_hidden_late"],
    ["baseline", "baseline_repeat"],
    ["still_no_scroll", "still_no_scroll"],
    ["detach_messages", "detach_messages"],
  ];

  W.run = async (opts) => {
    const o = Object.assign({ idleMs: 8000, gestureFrames: 80, gestureCapMs: 45000,
                              warmupFrames: 60, warmupCapMs: 30000, warmupRounds: 10,
                              warmupQuietElements: 50, warmupQuietRounds: 2,
                              lastMarker: null,
                              mountTimeoutMs: 420000, rung: "?", only: null,
                              hogMs: 200, hogPeriodMs: 250 }, opts || {});
    let el = null;
    try {
      W.marks.push({ name: "mount", wall_ms: Date.now() });
      const t0 = performance.now();
      await waitFor(() => q('textarea[aria-label="Message input"]'), 120000, "composer");
      if (o.lastMarker) {
        await waitFor(() => (document.body.innerText || "").indexOf(o.lastMarker) >= 0,
                      o.mountTimeoutMs, "the last seeded marker");
      }
      await new Promise((r) => nativeRaf(() => nativeRaf(r)));
      const mountMs = performance.now() - t0;

      el = scrollerAt(window.innerWidth / 2, window.innerHeight * 0.5);
      if (!el) throw new Error("no scroller was found under the middle of the viewport");
      const behaviourBefore = getComputedStyle(el).scrollBehavior;
      style("lay-guard", "*,.aui-thread-viewport{scroll-behavior:auto !important}");
      // Belt and braces, exactly as the local rig does it: the stylesheet covers descendants and
      // anything the app mounts later, the inline style is the one declaration an author rule
      // cannot outrank.
      try { el.style.scrollBehavior = "auto"; } catch (e) { W.notes.push(err(e).message); }
      await sleep(200);
      const behaviourAfter = getComputedStyle(el).scrollBehavior;
      const guard = { scroll_behavior_before: behaviourBefore,
                      scroll_behavior_after: behaviourAfter,
                      ok: behaviourAfter === "auto",
                      scroller_tag: el.tagName,
                      scroller_class: String(el.className || "").slice(0, 200) };
      if (!guard.ok) {
        W.notes.push("scroll-behavior could not be forced to auto: a 1 px assignment will start "
                   + "an animation and read back as inert");
      }
      const scrollable = el.scrollHeight - el.clientHeight;
      // AT THE SHORT RUNGS THERE IS NOTHING TO SCROLL, and that is a result rather than an
      // error. The 0K rung is an empty thread: no maths, no scrollable range, and therefore
      // nothing for any of these arms to remove. Throwing here would report the short-context
      // leg as a broken run instead of as the flat leg the short-context gate is asking for. The
      // idle windows and the census still run, and `no_scroll_range` says why the arm table is
      // empty.
      const noScrollRange = scrollable < 200;
      if (noScrollRange) {
        W.notes.push(`the scroller has only ${scrollable} px of scrollable range at rung `
                   + `${o.rung}, so no gesture window is scored here`);
      }
      slice();

      // Calibrate on a dedicated idle window, never on a window that is also reported.
      W.marks.push({ name: "idle:calibrate", wall_ms: Date.now() });
      await sleep(o.idleMs);
      clamp = calibrate(slice().t);

      const idleWindow = async (label, jam) => {
        if (jam) hogOn(o.hogMs, o.hogPeriodMs);
        await sleep(600);
        slice();
        W.marks.push({ name: label, wall_ms: Date.now() });
        const s0 = performance.now();
        await sleep(o.idleMs);
        const s = slice(), elapsed = performance.now() - s0;
        if (jam) hogOff();
        const b = busyOver(s.t, elapsed);
        return { name: label, jammed: Boolean(jam), elapsed_ms: Math.round(elapsed),
                 frames: s.g.length,
                 eff_fps: Math.round((1000 * s.g.length / elapsed) * 10) / 10,
                 blocked_ms_per_frame: (b.blocked_ms !== null && s.g.length)
                   ? Math.round((b.blocked_ms / s.g.length) * 10) / 10 : null,
                 raf: summarise(s.g), busy: b };
      };

      // LIVENESS, priced at idle. Idle is 61 fps at every rung on this host, so a jam has room to
      // show here; during the gesture the page is already saturated and a control priced there
      // cannot resolve (that mistake nearly failed a whole run for lack of a control that had
      // ample power in a different phase).
      const idleClean = await idleWindow("idle", false);
      await sleep(1500);
      const idleJammed = await idleWindow("idle_jammed", true);
      await sleep(1500);
      hogOff();

      const liveness = {
        clean_fps: idleClean.eff_fps, jammed_fps: idleJammed.eff_fps,
        drop_fraction: (idleClean.eff_fps > 0)
          ? Math.round((1 - idleJammed.eff_fps / idleClean.eff_fps) * 1000) / 1000 : null,
        clean_blocked_ms_per_frame: idleClean.blocked_ms_per_frame,
        jammed_blocked_ms_per_frame: idleJammed.blocked_ms_per_frame,
      };
      W.liveness = liveness;

      // ── THE PARK POSITION, fixed once, before anything is measured ──────────────────────
      const park = Math.round(Math.max(0, el.scrollHeight - el.clientHeight) * 0.5);

      // ── MARK THE MATHS-BEARING BLOCKS, ONCE, BEFORE ANY WINDOW ─────────────────────────
      //
      // The exploratory arm needs a selector for "the nearest block-level ancestor of a maths
      // root". There is no such CSS selector that is not `:has()`, and `:has()` is the thing
      // that owns the 500K scroll cost on Chromium (0f), so a class is added instead. It is
      // added HERE, before the warm-up and therefore before every scored window including every
      // baseline, so no window ever sees a different DOM from any other and the arm only toggles
      // a stylesheet. Adding it inside the arm's own window would bill the arm for its own DOM
      // mutation, which is defect #48.
      //
      // The placeholder height for that arm is the MEASURED median of these very blocks, taken
      // now, rather than a number chosen in advance. A wrong placeholder moves scrollHeight,
      // which is gated, so getting it from the page is the only way an exploratory arm can be
      // scored at all.
      const markMathBlocks = () => {
        const roots = document.querySelectorAll(".katex");
        const blocks = new Set();
        for (const r of roots) {
          let e = r.parentElement, hops = 0;
          while (e && hops++ < 12) {
            if (e.hasAttribute && e.hasAttribute("data-role")) break;
            const d = getComputedStyle(e).display;
            if (d === "block" || d === "flow-root" || d === "list-item") { blocks.add(e); break; }
            e = e.parentElement;
          }
        }
        const hs = [];
        const kinds = {};
        for (const b of blocks) {
          b.classList.add("cvk-mathblock");
          hs.push(b.getBoundingClientRect().height);
          // What these blocks ARE, because if they turn out to be one identifiable kind of
          // element then the follow-up is a CSS rule and not a renderer change, and if they are
          // a grab-bag then it is a renderer change. Guessing either way would be guessing.
          const cls = String(b.className || "");
          const k = b.tagName + (cls.indexOf("katex-display") >= 0 ? ".katex-display" : "")
                  + (b.getAttribute && b.getAttribute("data-streamdown")
                     ? "[data-streamdown=" + b.getAttribute("data-streamdown") + "]" : "");
          kinds[k] = (kinds[k] || 0) + 1;
        }
        const sum = hs.reduce((a, b) => a + b, 0);
        hs.sort((a, b) => a - b);
        const at = (p) => hs.length
          ? hs[Math.min(hs.length - 1, Math.max(0, Math.round(p * (hs.length - 1))))] : null;
        // THE MEAN, NOT THE MEDIAN, and this is a correction rather than a preference. The
        // scrollHeight error a placeholder introduces is the SUM of (placeholder - actual) over
        // the blocks that have not been rendered, so the value that makes that sum zero is the
        // arithmetic mean. Run 32876363634 used the median, 138 px against a mean of about
        // 133, and moved scrollHeight by +3.3% at r500K -- which disqualified the arm under the
        // 2% rule even though the same arm moved scrollHeight by -0.01% at r100K, where the two
        // statistics happened to coincide.
        const mean = hs.length ? Math.round(sum / hs.length) : null;
        return { count: blocks.size, roots: roots.length,
                 mean_height: mean,
                 median_height: at(0.5) === null ? null : Math.round(at(0.5)),
                 kinds,
                 heights: { min: at(0), p25: at(0.25), p50: at(0.5), p75: at(0.75),
                            p90: at(0.9), max: at(1) } };
      };
      W.mathBlocks = markMathBlocks();

      // ── THE PRODUCT ARM'S PRECONDITION, ASSERTED ONCE, BEFORE ANY WINDOW ────────────────
      //
      // Alongside `markMathBlocks()` and for the same reason: it is a fact about the page that
      // every window shares, so it is established once rather than inside a measured window. Two
      // independent halves, both recorded whatever they say:
      //
      //   * the RULE EXISTS: `document.styleSheets` is walked and the matched selectorText and
      //     cssText are quoted verbatim. This arm applies no CSS of its own, so if the bundle
      //     lacks the product rule the attribute toggle is inert and the window reads as a clean
      //     null. Without this record that null is indistinguishable from a product
      //     implementation that does not work, and the wrong conclusion is the plausible one.
      //   * the RULE ACTS: computed `content-visibility` on a sample of `.aui-math-block` with
      //     the attribute off, then on, then off again.
      //
      // At the 0K rung there is no maths, so ZERO BLOCKS IS THE CORRECT ANSWER and the toggle
      // check says so rather than failing; the rule must still be found in the bundle there,
      // because it is the same bundle at every rung.
      const productRule = findProductRule();
      productRule.blocks = document.querySelectorAll(PRODUCT_SEL).length;
      productRule.katex_display = document.querySelectorAll(".katex-display").length;
      productRule.toggle_check = productToggleCheck(24);
      W.productRule = productRule;
      if (!productRule.present) {
        W.notes.push(`no rule mentioning \`${PRODUCT_ATTR}\` was found in any of the `
                   + `${productRule.sheets_readable} readable stylesheets `
                   + `(${productRule.sheets_unreadable} unreadable), so this bundle was built `
                   + `without the product fix and \`product_math_block_containment\` toggles `
                   + `nothing`);
      }
      await sleep(1000);

      // ── A DISCARDED WARM-UP, run to QUIESCENCE ─────────────────────────────────────────
      //
      // Run 32865232787 failed its own drift gate because the FIRST scored window paid a
      // first-visit cost: 261.4 ms blocked per scroll event against a 162.6 / 193.9 / 197.5 /
      // 184.5 cluster, while the DOM grew by 11,205 elements across the session. That growth was
      // NOT KaTeX and NOT new code fences -- `.katex` stayed at 1,027, `.katex *` at 101,306 and
      // `pre` at 330 throughout -- so it is some other viewport-triggered mount, and the
      // mutation census above is what names it.
      //
      // This window runs the identical 1 px gesture at the identical fixed park position and
      // THROWS THE READING AWAY, repeating until the DOM stops changing. It is bounded by rounds
      // and by a wall clock, and whether it actually reached quiescence is recorded rather than
      // assumed: a warm-up that did not converge is a reportable condition, not a silent one.
      // Widening the drift threshold instead would have turned off the one gate that was telling
      // the truth.
      W.marks.push({ name: "warmup", wall_ms: Date.now() });
      const warm = { rounds: [], quiesced: false, park,
                     mut_before: mutSnapshot(), census_before: census(el) };
      const warmT0 = performance.now();

      // DRAIN THE DEFERRED-FENCE RESERVOIR FIRST, or no warm-up can converge.
      //
      // `useFenceReached` latches on three triggers, and two of them are not scrolling at all: an
      // IntersectionObserver with a 100% root margin, and a capturing scroll listener that treats
      // any move larger than one root height as a jump. An IntersectionObserver re-delivers on
      // LAYOUT CHANGE with no scroll whatsoever, so `[data-role] *{position:static}` -- which
      // makes this thread 7.3% taller -- slides fences into the band and latches them. That is
      // not the arm's effect on scrolling; it is the arm paying for markup that any arm could
      // have triggered, and it permanently contaminates every window after it (the last run's
      // baseline_3 and baseline_4 mounted 1,898 and 2,322 elements during their SETTLE, having
      // moved nothing).
      //
      // The reservoir is emptied on purpose, using the app's own path: `code-fence-defer.tsx`
      // registers `beforeprint` -> `upgradeEverythingForPrint`, which latches every remaining
      // fence at once. After that there is nothing left to mount, so a height change cannot
      // trigger one.
      //
      // THE COST OF DOING THIS IS STATED RATHER THAN HIDDEN: it moves the page off the state the
      // local llvmpipe rig measured (111,995 elements at this rung, which this venue reproduced
      // to within one element at 111,994). Both counts are recorded below so the comparison with
      // the local 78% / 79% / 86% carries its caveat as a number.
      const latch = { before: census(el), dispatched: false, error: null };
      try {
        window.dispatchEvent(new Event("beforeprint"));
        latch.dispatched = true;
      } catch (e) { latch.error = err(e); }
      // flushSync lands the swap in one task; the grammars arrive on a requestIdleCallback with a
      // 2,000 ms timeout, and a fence whose grammar is late latches to a plain fallback first and
      // grows its spans only once it arrives.
      await sleep(6000);
      latch.after = census(el);
      latch.fences_latched = latch.before.fences_deferred - latch.after.fences_deferred;
      latch.elements_added = latch.after.elements - latch.before.elements;
      latch.highlight_spans_added = latch.after.highlight_spans - latch.before.highlight_spans;
      warm.fence_latch = latch;
      let quietRun = 0;
      for (let i = 0; i < o.warmupRounds; i++) {
        const m0 = mutSnapshot(), e0 = census(el);
        let g = null, gErr = null;
        try { g = await gesture(el, o.warmupFrames, o.warmupCapMs, "pixel", park); }
        catch (e) { gErr = err(e); }
        await sleep(800);
        const m1 = mutSnapshot(), e1 = census(el);
        const d = mutDelta(m0, m1);
        const round = { i: i + 1, mutations: d,
                        elements: e1.elements, element_delta: e1.elements - e0.elements,
                        highlight_spans: e1.highlight_spans,
                        scroll_height: e1.scroller_scroll_height,
                        frames: g ? g.steps : null, error: gErr };
        warm.rounds.push(round);
        if (Math.abs(round.element_delta) <= o.warmupQuietElements
            && d.added_elements <= o.warmupQuietElements) {
          quietRun++;
          if (quietRun >= o.warmupQuietRounds) { warm.quiesced = true; break; }
        } else {
          quietRun = 0;
        }
        if (performance.now() - warmT0 > o.warmupCapMs * o.warmupRounds) break;
      }
      warm.elapsed_ms = Math.round(performance.now() - warmT0);
      warm.mut_total = mutDelta(warm.mut_before, mutSnapshot());
      warm.census_after = census(el);
      W.warmup = warm;
      slice();

      const baselineCensus = census(el);
      // THE SPLIT THE WHOLE FIX TURNS ON. `content-visibility` can only reach the display-maths
      // population, so the honest ceiling on this fix is the share of positioned boxes that live
      // there. Counted rather than assumed, and reported whatever it says.
      const dspRoots = Array.from(document.querySelectorAll(".katex-display"));
      const inlRoots = Array.from(document.querySelectorAll(".katex"))
        .filter((e) => !e.closest(".katex-display"));
      const descOf = (roots) => {
        const out = [];
        for (const r of roots) {
          const kids = r.getElementsByTagName("*");
          for (let i = 0; i < kids.length; i++) out.push(kids[i]);
        }
        return out;
      };
      const positionedIn = (list, cap) => {
        const n = list.length;
        if (!n) return { total: 0, sampled: 0, non_static: 0, estimate: 0 };
        const stride = Math.max(1, Math.floor(n / cap));
        let sampled = 0, nonStatic = 0;
        for (let i = 0; i < n; i += stride) {
          sampled++;
          if (getComputedStyle(list[i]).position !== "static") nonStatic++;
        }
        return { total: n, sampled, non_static: nonStatic,
                 estimate: Math.round(n * (nonStatic / sampled)) };
      };
      const positioned = {
        message_descendants: positionedEstimate("[data-role] *", 1500),
        katex_descendants: positionedEstimate(".katex *", 1500),
        katex_display_roots: positionedIn(dspRoots, 1000),
        katex_inline_roots: positionedIn(inlRoots, 1000),
        katex_display_descendants: positionedIn(descOf(dspRoots), 1500),
        katex_inline_descendants: positionedIn(descOf(inlRoots), 1500),
      };

      for (const [armKey, windowName] of SEQUENCE) {
        if (o.only && o.only.indexOf(windowName) < 0) continue;
        if (noScrollRange) break;
        const arm = ARMS[armKey];
        const before = census(el);
        const mutAtStart = mutSnapshot();
        // The behavioural fired-check needs a BEFORE reading of the same sample, taken while the
        // arm is not applied. Same indices are re-read after apply, so the comparison is element
        // by element rather than distribution against distribution.
        let probeBefore = null;
        if (arm.probe) {
          try { probeBefore = arm.probe(); } catch (e) { probeBefore = { error: err(e) }; }
        }
        let detail = "", ok = true, applyError = null;
        try { detail = await arm.apply(); }
        catch (e) { ok = false; applyError = err(e); detail = applyError.message; }
        // Let the mutation settle so its own style/layout cost is not billed to the gesture.
        // Removing 28 messages is a far bigger job than adding a stylesheet and needs longer:
        // the positive control's window carried a 20,249 ms frame last run, which was the
        // removal itself leaking into the reading.
        await sleep(arm.destructive ? 5000 : 1500);
        const applied = census(el);
        let fired = null;
        if (arm.fired) {
          try { fired = arm.fired(); }
          catch (e) { fired = { error: err(e) }; }
        }
        let probeAfter = null, tookEffect = null;
        if (arm.probe) {
          try { probeAfter = arm.probe(); } catch (e) { probeAfter = { error: err(e) }; }
          tookEffect = heightDelta(probeBefore, probeAfter);
        }
        slice();

        W.marks.push({ name: "arm:" + windowName, wall_ms: Date.now() });
        const aT0 = performance.now();
        let gest = null, gestError = null;
        try { gest = await gesture(el, o.gestureFrames, o.gestureCapMs, arm.mode || "pixel",
                                   park); }
        catch (e) { ok = false; gestError = err(e); }
        const elapsed = performance.now() - aT0;
        const s = slice();
        const b = busyOver(s.t, elapsed);

        W.arms.push({
          name: windowName, arm: armKey, why: arm.why, ok,
          apply_detail: detail, apply_error: applyError,
          gesture: gest, gesture_error: gestError,
          // `fired` says the engine ACCEPTED the declaration. `took_effect` says it ACTED on it.
          // For `content-visibility` the two come apart: an inline box reports `auto` from
          // getComputedStyle and is skipped by nothing, which is precisely the vacuous-arm
          // failure this scene exists to make visible.
          fired,
          probe_before: probeBefore, probe_after: probeAfter, took_effect: tookEffect,
          destructive: Boolean(arm.destructive),
          elapsed_ms: Math.round(elapsed),
          frames: s.g.length,
          eff_fps: Math.round((1000 * s.g.length / elapsed) * 10) / 10,
          // THE PRIMARY METRIC. One scroll event per painted frame, so this is the cost of one
          // scroll event. busy% is reported too but it is a share of wall time and cannot
          // separate "less work per frame" from "more frames in the same second".
          blocked_ms_per_frame: (b.blocked_ms !== null && s.g.length)
            ? Math.round((b.blocked_ms / s.g.length) * 10) / 10 : null,
          // Reported ALONGSIDE the mean, never instead of it. See `robustOver`.
          robust: robustOver(s.t, s.g, elapsed),
          raf: summarise(s.g),
          raf_gaps_ms: s.g.map((x) => Math.round(x * 10) / 10),
          busy: b,
          // THE SELECTOR, on every window, so a report can quote the thing rather than the
          // label. Three times in this campaign a label has been trusted over what it names.
          selector: detail,
          census_before: before, census_applied: applied, census_after: census(el),
          // Per window, so a window that paid for someone else's deferred mounting is visible as
          // a number instead of being averaged into the arm it landed on.
          mutations: mutDelta(mutAtStart, mutSnapshot()),
          scroll_height_delta: (applied.scroller_scroll_height !== null
                                && before.scroller_scroll_height)
            ? Math.round(((applied.scroller_scroll_height / before.scroller_scroll_height) - 1)
                         * 10000) / 10000 : null,
          wall_start_ms: Date.now() - Math.round(elapsed), wall_end_ms: Date.now(),
        });

        if (arm.revert) {
          try { await arm.revert(); } catch (e) { W.notes.push("revert failed: " + err(e).message); }
          await sleep(1000);
          slice();
        }
      }
      W.marks.push({ name: "end", wall_ms: Date.now() });

      post({ __done: true, ok: true, rung: o.rung, ua: navigator.userAgent,
             engine_probe: {
               is_webkit_gtk_ua: /AppleWebKit/.test(navigator.userAgent)
                 && /X11/.test(navigator.userAgent),
               vendor: navigator.vendor, has_chrome: typeof window.chrome !== "undefined",
               hardwareConcurrency: navigator.hardwareConcurrency },
             url: location.href, mount: { ms: Math.round(mountMs), census: census(el) },
             clamp, guard, liveness, idle: idleClean, idle_jammed: idleJammed,
             warmup: warm, park, fence_latch: warm.fence_latch,
             no_scroll_range: noScrollRange, scrollable_px: scrollable,
             math_blocks: W.mathBlocks,
             // PER RUNG AND PER REP, because this payload is one repetition of one rung. The
             // product arm cannot be scored without it, and at 0K -- where there are no windows
             // at all -- it is the only thing this rung has to say about the product build.
             product_rule: W.productRule,
             mutations_total: mutDelta(warm.mut_before, mutSnapshot()),
             baseline_census: baselineCensus, positioned,
             gesture_frames: o.gestureFrames, gesture_cap_ms: o.gestureCapMs, idle_ms: o.idleMs,
             notes: W.notes, marks: W.marks, arms: W.arms, final: census(el) });
    } catch (e) {
      hogOff();
      const E = err(e);
      post({ __done: true, ok: false, rung: o.rung,
             error: E.message, error_detail: E, notes: W.notes,
             marks: W.marks, arms: W.arms, url: location.href,
             dom: { elements: document.getElementsByTagName("*").length,
                    messages: document.querySelectorAll("[data-role]").length,
                    katex_roots: document.querySelectorAll(".katex").length,
                    bodyText: (document.body.innerText || "").slice(0, 800) } });
    }
  };
})();
