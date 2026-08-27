// ONE-PIXEL scroll ablation: does the RenderLayer attribution transfer to a REAL GPU?
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
  if (window.__lay) return;
  const W = { gaps: [], ticks: [], marks: [], arms: [], notes: [] };
  window.__lay = W;

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

    visibility_hidden_offscreen: {
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
  const SEQUENCE = [
    ["baseline", "baseline"],
    ["noop_touch", "noop_touch"],
    ["baseline", "baseline_2"],
    ["position_static_all", "position_static_all"],
    ["baseline", "baseline_3"],
    ["katex_static", "katex_static"],
    ["baseline", "baseline_4"],
    ["visibility_hidden_offscreen", "visibility_hidden_offscreen"],
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
      if (scrollable < 200) {
        throw new Error(`the scroller has only ${scrollable} px of scrollable range, so the `
                      + `gesture cannot move`);
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
      const positioned = {
        message_descendants: positionedEstimate("[data-role] *", 1500),
        katex_descendants: positionedEstimate(".katex *", 1500),
      };

      for (const [armKey, windowName] of SEQUENCE) {
        if (o.only && o.only.indexOf(windowName) < 0) continue;
        const arm = ARMS[armKey];
        const before = census(el);
        const mutAtStart = mutSnapshot();
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
          fired,
          destructive: Boolean(arm.destructive),
          elapsed_ms: Math.round(elapsed),
          frames: s.g.length,
          eff_fps: Math.round((1000 * s.g.length / elapsed) * 10) / 10,
          // THE PRIMARY METRIC. One scroll event per painted frame, so this is the cost of one
          // scroll event. busy% is reported too but it is a share of wall time and cannot
          // separate "less work per frame" from "more frames in the same second".
          blocked_ms_per_frame: (b.blocked_ms !== null && s.g.length)
            ? Math.round((b.blocked_ms / s.g.length) * 10) / 10 : null,
          raf: summarise(s.g),
          raf_gaps_ms: s.g.map((x) => Math.round(x * 10) / 10),
          busy: b,
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
