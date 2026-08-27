// Injected at document-start into REAL WebKitGTK, before app code.
//
// THE QUESTION. A user reported "60 FPS downgrading to 5 FPS when the context length grows from
// 0K to 100K to 500K". Every number this campaign has published so far was taken on a BRANCH with
// a flag forced on. This scene exists to measure two SHIPPED builds against each other, as they
// ship, with nothing forced:
//
//   pre   the last commit before the campaign started. The "60 fps" side of the complaint.
//   head  today's main, whose ship defaults are already the fixed state.
//
// So this scene NEVER sets a flag, never injects a stylesheet and never mutates the DOM before a
// scored window. It only measures, and it READS THE ARM'S OWN STATE BACK OUT OF THE RUNNING PAGE
// so that "the head arm really was the fixed state" is a recorded fact rather than an assumption
// about which bytes were built.
//
// The film, per session (one arm, one rung, one repetition):
//
//   mount          navigation until the last SEEDED message is in the DOM
//   idle:calibrate a dedicated window whose only job is the setTimeout clamp
//   idle           nobody touches it
//   idle_jammed    THE POSITIVE CONTROL, at every rung, in every session
//   scroll         the gesture a user makes on a long thread
//   actions        reasoning_toggle and select_all_copy, the two windows that carried the report
//   stream         the ladder's own tail arrives through the pacer
//   recover        after the stream ends
//
// THE INSTRUMENT, and why two of the three plausible channels are banned.
//
//   * `GdkFrameClock::after-paint` under `begin_updating()` reads 60.0 fps with the main thread
//     80% blocked. It drives the clock independently of whether the app can do any work.
//   * `1000 / p50` of rAF gaps reads 16.0 ms jammed and unjammed alike: a median cannot see mass
//     that has moved into the tail. Nine cheap frames and one three-second frame have the same
//     p50 as ten cheap frames.
//
// The headline is therefore EFFECTIVE FRAME RATE OVER WALL TIME: how many frames the page
// actually produced, divided by how long the window really lasted. p50 is still reported, and it
// is labelled as a statistic that cannot see the tail. `idle_jammed` is what makes any of it
// mean anything: a channel that cannot report a deliberately blocked main thread cannot report a
// blocked one that happened by accident.
//
// busy_pct is calibrated, not assumed. A setTimeout(1) loop cannot resolve below the platform
// clamp, so the clamp is measured on its own window and blocked time is the excess over it. If
// the clamp cannot be established the field is null WITH A REASON, never 0.
(() => {
  if (window.__av) return;
  const W = { gaps: [], ticks: [], samples: [], actions: [], marks: [], notes: [], t0: null,
              phase: "boot", log: [] };
  window.__av = W;

  const nativeRaf = window.requestAnimationFrame.bind(window);
  let last = performance.now();
  const frame = (now) => { W.gaps.push(now - last); last = now; nativeRaf(frame); };
  nativeRaf(frame);

  // Timer-lag channel, independent of rAF. Every gap is kept, not just aggregates, because the
  // clamp has to be subtracted per gap and that cannot be done after the fact from a sum.
  let lastTick = performance.now();
  const tick = () => {
    const now = performance.now();
    W.ticks.push(now - lastTick);
    lastTick = now;
    setTimeout(tick, 1);
  };
  setTimeout(tick, 1);

  const q = (s) => document.querySelector(s);
  const qa = (s) => document.querySelectorAll(s);
  W.dom = {
    composer: () => q('textarea[aria-label="Message input"]'),
    isRunning: () => Boolean(q('button[aria-label="Stop generating"]') ||
                             q('button[aria-label="Queue message"]')),
    // THREE COMPONENTS CARRY `aria-label="Send message"` on BOTH arms: the thread composer, the
    // shared composer and the dictation bar. `querySelector` returns whichever is first in
    // document order, and the dictation bar's is permanently disabled when nothing is being
    // transcribed. Waiting for THAT one to become enabled is a wait that can never end, and it
    // cost a session in rehearsal: the composer was filled, a send button existed, and the run
    // timed out after two minutes reporting that the model chip had not restored. So take every
    // match and prefer one that is actually usable.
    sendButtons: () => Array.from(qa('button[aria-label="Send message"]')),
    elements: () => document.getElementsByTagName("*").length,
    messages: () => qa("[data-role]").length,
    assistantChars: () => Array.from(qa('[data-role="assistant"]'))
      .reduce((a, e) => a + (e.textContent || "").length, 0),
    reasoningRoots: () => qa('[data-slot="reasoning-root"]').length,
    reasoningOpen: () => qa('[data-slot="reasoning-root"][data-state="open"]').length,
    codeBlocks: () => qa("pre").length,
    highlightSpans: () => qa("pre span").length,
  };

  const post = (o) => {
    if (o && o.__done) W.result = o;
    try { window.webkit.messageHandlers.bench.postMessage(JSON.stringify(o)); } catch (e) {}
  };
  W.post = post;
  W.note = (m) => { W.notes.push(m); W.log.push([Date.now(), m]); post({ note: m }); };

  // A phase mark is a WALL CLOCK boundary, so a driver-side series can be cut at exactly these
  // instants. performance.now() is not comparable across the two processes.
  const mark = (name) => {
    W.marks.push({ name, wall_ms: Date.now(),
                   t_ms: W.t0 === null ? null : Math.round(performance.now() - W.t0) });
    W.phase = name;
  };
  W.mark = mark;

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const waitFor = async (fn, timeoutMs, label) => {
    const dl = performance.now() + timeoutMs;
    while (performance.now() < dl) {
      let v; try { v = fn(); } catch (e) { v = null; }
      if (v) return v;
      await sleep(100);
    }
    throw new Error("timeout waiting for " + label);
  };

  const census = () => ({
    elements: W.dom.elements(), messages: W.dom.messages(),
    assistant_chars: W.dom.assistantChars(),
    reasoning_roots: W.dom.reasoningRoots(), reasoning_open: W.dom.reasoningOpen(),
    code_blocks: W.dom.codeBlocks(), highlight_spans: W.dom.highlightSpans(),
    scroll_height: (document.scrollingElement || document.body).scrollHeight,
  });
  W.census = census;

  // ── WHAT THIS ARM ACTUALLY IS, read out of the running page ────────────────────────────────
  //
  // Neither arm is forced, so the only way to know which state was measured is to ask the page.
  // Three independent things are read, because they can disagree and the disagreement is the
  // finding:
  //
  //   1. the engine gate. `math-block-mode.ts::gateOnEngine` turns containment OFF unless
  //      `CSS.supports("anchor-name: --unsloth-probe")` is true. If this runner's WebKitGTK fails
  //      that probe then the head arm is running WITHOUT the scroll fix, and a small delta would
  //      otherwise be reported as "the fix is worth little" when it is really "the fix was never
  //      on". This is the single most important field in the payload.
  //   2. the attribute the stylesheet reads, on documentElement.
  //   3. whether the engine ACTED: a computed `content-visibility` sampled off real maths blocks.
  //      An accepted declaration that the engine ignores is the vacuous-arm failure that wasted
  //      three arms in this campaign, so acceptance is never taken as effect.
  //
  // The fence side is read the same way: `data-unsloth-fence-deferred` shells exist only on a
  // build where deferral is in force, and `pre span` counts the spans deferral removes.
  const ANCHOR_PROBE = "anchor-name: --unsloth-probe";
  const sampleContentVisibility = (sel, limit) => {
    const out = { selector: sel, matched: 0, sampled: 0, auto: 0, values: {} };
    const nodes = qa(sel);
    out.matched = nodes.length;
    const step = Math.max(1, Math.floor(nodes.length / limit));
    for (let i = 0; i < nodes.length; i += step) {
      let v = "";
      try { v = getComputedStyle(nodes[i]).contentVisibility || ""; } catch (e) { v = "?"; }
      out.sampled += 1;
      out.values[v] = (out.values[v] || 0) + 1;
      if (v === "auto") out.auto += 1;
      if (out.sampled >= limit) break;
    }
    return out;
  };
  const readback = (when) => {
    let supportsAnchor = null, supportsCV = null;
    try { supportsAnchor = CSS.supports(ANCHOR_PROBE); } catch (e) { supportsAnchor = "threw"; }
    try { supportsCV = CSS.supports("content-visibility: auto"); } catch (e) { supportsCV = "threw"; }
    const g = (k) => {
      try { return typeof window[k] === "undefined" ? null : window[k]; } catch (e) { return "threw"; }
    };
    return {
      when,
      ua: navigator.userAgent,
      // The engine gate, verified HERE rather than inherited from another box.
      css_supports_anchor_name: supportsAnchor,
      css_supports_content_visibility: supportsCV,
      engine_gate_probe: ANCHOR_PROBE,
      // The maths side.
      math_block_attribute: document.documentElement.getAttribute("data-math-block-containment"),
      math_runtime_global: g("__UNSLOTH_MATH_BLOCK_CONTAINMENT__"),
      aui_math_block: qa(".aui-math-block").length,
      aui_math_display: qa(".aui-math-display").length,
      katex_roots: qa(".katex").length,
      katex_display_roots: qa(".katex-display").length,
      content_visibility_math_blocks: sampleContentVisibility(".aui-math-block", 60),
      content_visibility_katex_display: sampleContentVisibility(".katex-display", 60),
      // The fence side.
      fence_runtime_global: g("__UNSLOTH_DEFER_FENCE_HIGHLIGHT__"),
      deferred_fence_shells: qa('[data-unsloth-fence-deferred="true"]').length,
      code_blocks: qa("pre").length,
      highlight_spans: qa("pre span").length,
    };
  };
  W.readback = readback;

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
  W.scrollerAt = scrollerAt;

  const pcOf = (sorted, p) => sorted.length
    ? sorted[Math.min(sorted.length - 1, Math.max(0, Math.round(p * (sorted.length - 1))))]
    : null;

  const summarise = (arr) => {
    const s = arr.slice().sort((a, b) => a - b);
    if (!s.length) return { n: 0 };
    const over = (t) => s.filter((g) => g > t).length;
    return {
      n: s.length, p50_ms: pcOf(s, 0.5), p90_ms: pcOf(s, 0.9), p95_ms: pcOf(s, 0.95),
      p99_ms: pcOf(s, 0.99), max_ms: s[s.length - 1],
      // REPORTED, NEVER THE HEADLINE. `fps_p50` is 1000/p50 and it reads the same jammed and
      // unjammed; it is here so the disagreement with eff_fps stays visible.
      fps_p50: 1000 / pcOf(s, 0.5), fps_p5: 1000 / pcOf(s, 0.95),
      fps_worst: 1000 / s[s.length - 1],
      frames_over_33: over(33), frames_over_33_pct: 100 * over(33) / s.length,
      frames_over_100: over(100), frames_over_100_pct: 100 * over(100) / s.length,
      frames_over_1000: over(1000),
      sum_ms: s.reduce((a, b) => a + b, 0),
    };
  };
  W.summarise = summarise;

  // ── the clamp, measured ────────────────────────────────────────────────────────────────────
  const MAX_CLAMP_MS = 10.0;
  let clamp = { clamp_ms: null, reason: "not calibrated" };
  W.calibrate = (tickSlice) => {
    const s = tickSlice.slice().sort((a, b) => a - b);
    if (s.length < 40) return { clamp_ms: null, reason: `only ${s.length} idle ticks, need 40` };
    const med = pcOf(s, 0.5);
    if (!(med > 0)) return { clamp_ms: null, reason: "idle tick median was not positive" };
    if (med > MAX_CLAMP_MS) {
      return { clamp_ms: null, samples: s.length, p50: med,
               reason: `idle tick median ${med.toFixed(1)} ms exceeds the ${MAX_CLAMP_MS} ms ` +
                       "ceiling, so the idle window was not idle and no clamp is trustworthy" };
    }
    return { clamp_ms: med, samples: s.length, p50: med, p95: pcOf(s, 0.95), reason: null };
  };

  const busyOver = (tickSlice, elapsedMs) => {
    if (clamp.clamp_ms === null) {
      return { busy_pct: null, blocked_ms: null, busy_pct_reason: clamp.reason,
               ticks: tickSlice.length };
    }
    if (!(elapsedMs > 0)) {
      return { busy_pct: null, blocked_ms: null, ticks: tickSlice.length,
               busy_pct_reason: "the window reported no elapsed time" };
    }
    let blocked = 0;
    for (const g of tickSlice) { const x = g - clamp.clamp_ms; if (x > 0) blocked += x; }
    return { busy_pct: Math.round((blocked / elapsedMs) * 1000) / 10,
             blocked_ms: Math.round(blocked * 10) / 10, ticks: tickSlice.length,
             busy_pct_reason: null };
  };

  // A MEAN OVER ONE OUTLIER IS A REPORT OF WHERE THE OUTLIER LANDED. At 500K this app blocks the
  // main thread for seconds at a time, so a window either catches one of those or does not, and
  // two windows that differ elevenfold can have an identical p50. Every window therefore also
  // reports what it cost with its single worst tick removed, and how many frames over a second it
  // contained. Neither replaces the mean: a window whose two numbers disagree is a window that
  // caught a stall, which is a fact about the window rather than about the arm.
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

  // THE HEADLINE. Frames the page really produced over the wall time the window really took.
  const effFps = (n, el) => (el > 0 ? Math.round((1000 * n / el) * 10) / 10 : null);

  // ── a deliberate main-thread jam, used ONLY to prove the channel can report one ─────────────
  let hogTimer = null;
  const hogOn = (busyMs, periodMs) => {
    if (hogTimer !== null) return;
    hogTimer = setInterval(() => {
      const t = performance.now();
      while (performance.now() - t < busyMs) { /* spin */ }
    }, periodMs);
  };
  const hogOff = () => { if (hogTimer !== null) { clearInterval(hogTimer); hogTimer = null; } };
  W.hogOn = hogOn; W.hogOff = hogOff;

  // ── the film ───────────────────────────────────────────────────────────────────────────────
  W.run = async (opts) => {
    const o = Object.assign({
      idleMs: 6000, recoverMs: 6000, maxMs: 900000, prompt: "continue",
      lastMarker: null, mountTimeoutMs: 300000, sendTimeoutMs: 240000,
      composerTimeoutMs: 180000, rung: "?", arm: "?",
      hogMs: 200, hogPeriodMs: 250, skipSend: false,
    }, opts || {});

    const cut = { gaps: 0, ticks: 0 };
    const slice = () => {
      const g = W.gaps.slice(cut.gaps), t = W.ticks.slice(cut.ticks);
      cut.gaps = W.gaps.length; cut.ticks = W.ticks.length;
      return { g, t };
    };

    // One shape for every scored window, so pre and head cannot be summarised differently.
    const windowOf = (name, s, elapsed, extra) => {
      const b = busyOver(s.t, elapsed);
      return Object.assign({
        phase: name,
        elapsed_ms: Math.round(elapsed),
        frames: s.g.length,
        // THE HEADLINE, and the only figure that both moves under a jam and means frames a user
        // would have seen.
        eff_fps: effFps(s.g.length, elapsed),
        blocked_ms_per_frame: (b.blocked_ms !== null && s.g.length)
          ? Math.round((b.blocked_ms / s.g.length) * 10) / 10 : null,
        robust: robustOver(s.t, s.g, elapsed),
        raf: summarise(s.g),
        raf_gaps_ms: s.g.map((x) => Math.round(x * 10) / 10),
        busy: b,
        census: census(),
      }, extra || {});
    };

    try {
      W.t0 = performance.now();
      mark("mount");
      const mountT0 = performance.now();
      // The composer appearing is NOT the thread being mounted. At 500K the app paints a shell
      // and then spends a long time putting messages in, and a baseline taken there is a reading
      // of the mount rather than of the mounted thread.
      // Its OWN budget, and an option rather than a constant. This used to be a hardcoded
      // 120000 while the caller passed `mountTimeoutMs: 420000`, so the number the operator
      // set governed only the second half of the mount and the first half could not be
      // extended for the rung that needs it most.
      const ta = await waitFor(W.dom.composer, o.composerTimeoutMs, "composer textarea");
      let mountedBy = "composer";
      if (o.lastMarker) {
        await waitFor(() => (document.body.innerText || "").indexOf(o.lastMarker) >= 0,
                      o.mountTimeoutMs, "last seeded marker in the DOM");
        mountedBy = "last_seeded_marker";
      }
      // Two nested rAFs: the marker being in the DOM is not the frame that shows it.
      await new Promise((r) => nativeRaf(() => nativeRaf(r)));
      const mountMs = performance.now() - mountT0;
      const censusMounted = census();
      const readbackMounted = readback("mounted");
      slice();

      // Two idle windows, not one. The clamp is the floor a busy_pct is subtracted against, so
      // deriving it from the same window it is then reported for is circular: half that window's
      // ticks are at or below their own median by construction.
      mark("idle:calibrate");
      const calT0 = performance.now();
      await sleep(o.idleMs);
      const calS = slice();
      clamp = W.calibrate(calS.t);
      clamp.window_ms = Math.round(performance.now() - calT0);

      mark("idle");
      const idleT0 = performance.now();
      await sleep(o.idleMs);
      const idleEl = performance.now() - idleT0;
      const idleW = windowOf("idle", slice(), idleEl);

      // ── THE POSITIVE CONTROL, AT EVERY RUNG, IN EVERY SESSION ──────────────────────────────
      //
      // Priced at idle deliberately. Idle is ~61 fps at every rung on this host, so a jam has
      // room to show; during the gesture at 500K the page is already saturated and a control
      // priced there cannot resolve, which nearly failed a whole run for want of a control that
      // had ample power one phase away.
      //
      // If this does not resolve, NOTHING in this session means anything and the criteria module
      // is required to VOID it. The documented pass is roughly 61 -> 17 fps.
      hogOn(o.hogMs, o.hogPeriodMs);
      await sleep(600);
      slice();
      mark("idle_jammed");
      const jamT0 = performance.now();
      await sleep(o.idleMs);
      const jamEl = performance.now() - jamT0;
      const jamS = slice();
      hogOff();
      const jamW = windowOf("idle_jammed", jamS, jamEl, { jammed: true,
                                                          hog_ms: o.hogMs,
                                                          hog_period_ms: o.hogPeriodMs });
      await sleep(1500);
      slice();

      const liveness = {
        clean_fps: idleW.eff_fps, jammed_fps: jamW.eff_fps,
        drop_fraction: (idleW.eff_fps > 0)
          ? Math.round((1 - jamW.eff_fps / idleW.eff_fps) * 1000) / 1000 : null,
        clean_blocked_ms_per_frame: idleW.blocked_ms_per_frame,
        jammed_blocked_ms_per_frame: jamW.blocked_ms_per_frame,
        // The blind channel, computed on the SAME series, so the report can show what would have
        // been concluded from it. It barely moves.
        clean_fps_p50: idleW.raf && idleW.raf.fps_p50,
        jammed_fps_p50: jamW.raf && jamW.raf.fps_p50,
        hog_ms: o.hogMs, hog_period_ms: o.hogPeriodMs,
      };
      W.liveness = liveness;

      mark("scroll");
      const scrollT0 = performance.now();
      const el = scrollerAt(window.innerWidth / 2, window.innerHeight * 0.5);
      let scrollDetail = "no scroller";
      let scrollGuard = { forced: false, behavior_before: null, behavior_after: null };
      let scrollTravel = null;
      if (el) {
        // `.aui-thread-viewport` carries `scroll-smooth`. Under `scroll-behavior: smooth` an
        // assignment starts an ANIMATION, so a same-turn read-back returns the position it
        // started from and a real gesture reads as if nothing moved. Force it to auto and PROVE
        // the computed value took, rather than assuming it.
        scrollGuard.behavior_before = getComputedStyle(el).scrollBehavior;
        try { el.style.scrollBehavior = "auto"; scrollGuard.forced = true; } catch (e) {}
        scrollGuard.behavior_after = getComputedStyle(el).scrollBehavior;
        if (scrollGuard.behavior_after !== "auto") {
          W.note("scroll-behavior could not be forced to auto, so an assignment starts an "
               + "animation and the gesture may read as inert");
        }
        const h0 = el.scrollHeight, top0 = el.scrollTop;
        let commanded = 0, travelled = 0;
        // Up through the thread, pause, back down. Three passes so one stall cannot dominate.
        for (let pass = 0; pass < 3; pass++) {
          for (let i = 0; i < 10; i++) {
            const before = el.scrollTop;
            el.scrollTop = Math.max(0, el.scrollTop - 900);
            commanded += Math.min(900, before);
            travelled += Math.abs(el.scrollTop - before);
            await sleep(80);
          }
          await sleep(300);
          for (let i = 0; i < 10; i++) {
            const before = el.scrollTop;
            el.scrollTop = el.scrollTop + 900;
            commanded += 900;
            travelled += Math.abs(el.scrollTop - before);
            await sleep(80);
          }
          await sleep(300);
        }
        el.scrollTop = el.scrollHeight;
        scrollTravel = { commanded_px: Math.round(commanded), travelled_px: Math.round(travelled),
                         travel_fraction: commanded > 0
                           ? Math.round((travelled / commanded) * 1000) / 1000 : null,
                         start_top: top0, end_top: el.scrollTop, scroll_height: h0,
                         span: Math.max(0, h0 - el.clientHeight) };
        scrollDetail = `${el.tagName}.${(el.className || "").toString().slice(0, 30)} ` +
                       `h=${h0} top ${top0}->${el.scrollTop}`;
      }
      const scrollEl = performance.now() - scrollT0;
      const scrollW = windowOf("scroll", slice(), scrollEl,
                               { detail: scrollDetail, guard: scrollGuard, travel: scrollTravel });

      // ── the action windows that carried the original complaint ─────────────────────────────
      //
      // app_sync_ms is recorded separately from the window: it is the time spent INSIDE the click
      // handler, which is app code and nothing else. The window around it includes this scene's
      // own waiting, and reporting only the window would bill the harness's cost to the app.
      const actions = [];
      const runAction = async (name, fn) => {
        const i0 = W.gaps.length, t0 = performance.now();
        const before = census();
        let detail = "", ok = true, appSync = null, na = false;
        try {
          const r = await fn();
          detail = (r && r.detail) || String(r || "");
          appSync = r && r.app_sync_ms;
          na = Boolean(r && r.not_applicable);
        } catch (e) { ok = false; detail = String((e && e.message) || e); }
        const dur = performance.now() - t0;
        const g = W.gaps.slice(i0), t = W.ticks.slice(cut.ticks);
        cut.gaps = W.gaps.length; cut.ticks = W.ticks.length;
        const b = busyOver(t, dur);
        actions.push({
          name, ok, not_applicable: na, detail, app_sync_ms: appSync,
          elapsed_ms: Math.round(dur), frames: g.length, eff_fps: effFps(g.length, dur),
          blocked_ms_per_frame: (b.blocked_ms !== null && g.length)
            ? Math.round((b.blocked_ms / g.length) * 10) / 10 : null,
          robust: robustOver(t, g, dur),
          raf: summarise(g), raf_gaps_ms: g.map((x) => Math.round(x * 10) / 10),
          busy: b, census_before: before, census_after: census(),
          wall_start_ms: Date.now() - Math.round(dur), wall_end_ms: Date.now() });
        return ok;
      };

      mark("action:reasoning_toggle");
      await runAction("reasoning_toggle", async () => {
        const root = q('[data-slot="reasoning-root"]');
        if (!root) {
          return { detail: "no reasoning pane at this rung: an empty thread has no assistant "
                           + "message until the first reply, so there is nothing to toggle",
                   not_applicable: true };
        }
        const trigger = root.querySelector("button");
        if (!trigger) throw new Error("no reasoning trigger button");
        const s0 = root.getAttribute("data-state");
        const c0 = performance.now();
        trigger.click();
        const sync1 = performance.now() - c0;
        await sleep(1500);
        const r2 = q('[data-slot="reasoning-root"]') || root;
        const s1 = r2.getAttribute("data-state");
        const t2 = r2.querySelector("button");
        let sync2 = null;
        if (t2) { const c1 = performance.now(); t2.click(); sync2 = performance.now() - c1; }
        await sleep(1500);
        const s2 = (q('[data-slot="reasoning-root"]') || r2).getAttribute("data-state");
        if (s0 === s1 && s1 === s2) throw new Error("the pane never changed state: " + s0);
        return { detail: `state ${s0} -> ${s1} -> ${s2}`,
                 app_sync_ms: Math.round(Math.max(sync1, sync2 || 0) * 10) / 10 };
      });

      mark("action:select_all_copy");
      await runAction("select_all_copy", async () => {
        const root = q('[data-slot="reasoning-root"]') || q('[data-role="assistant"]');
        if (!root) {
          return { detail: "no assistant message at this rung yet, so there is nothing to copy",
                   not_applicable: true };
        }
        const sel = window.getSelection();
        sel.removeAllRanges();
        const rng = document.createRange();
        rng.selectNodeContents(root);
        sel.addRange(rng);
        await sleep(150);
        const c0 = performance.now();
        let copied = false;
        try { copied = document.execCommand("copy"); } catch (e) { copied = false; }
        const sync = performance.now() - c0;
        const n = (sel.toString() || "").length;
        sel.removeAllRanges();
        if (!n) throw new Error("the selection was empty, so nothing rendered to copy");
        return { detail: `selected ${n} chars, execCommand(copy)=${copied}`,
                 app_sync_ms: Math.round(sync * 10) / 10 };
      });

      const engineProbe = () => ({
        is_webkit_gtk_ua: /AppleWebKit/.test(navigator.userAgent) && /X11/.test(navigator.userAgent),
        vendor: navigator.vendor,
        has_chrome: typeof window.chrome !== "undefined",
        has_webkit_message_handlers:
          typeof (window.webkit && window.webkit.messageHandlers) !== "undefined",
        hardwareConcurrency: navigator.hardwareConcurrency, dpr: window.devicePixelRatio,
      });
      const buildProbe = () => ({
        scripts: Array.from(qa("script[src]")).map((x) => x.getAttribute("src")),
        css: Array.from(qa('link[rel="stylesheet"]')).map((x) => x.getAttribute("href")),
      });

      if (o.skipSend) {
        post({
          __done: true, ok: true, ua: navigator.userAgent, rung: o.rung, arm: o.arm,
          skipped_send: true,
          engine_probe: engineProbe(), build: buildProbe(), url: location.href,
          mount: { ms: Math.round(mountMs), by: mountedBy, census: censusMounted,
                   last_marker: o.lastMarker },
          clamp, liveness, notes: W.notes,
          readback_mounted: readbackMounted, readback_final: readback("final"),
          scroll_detail: scrollDetail, marks: W.marks, actions,
          phases: [idleW, jamW, scrollW],
          final: census(),
        });
        return;
      }

      mark("send");
      const setter = Object.getOwnPropertyDescriptor(
        window.HTMLTextAreaElement.prototype, "value").set;
      // FOCUS FIRST. Setting .value on an unfocused textarea leaves the app's composer state
      // untouched: the button exists, the click lands, and no request is ever made.
      const focusComposer = async () => {
        try {
          const r = ta.getBoundingClientRect();
          const x = r.left + r.width / 2, y = r.top + r.height / 2;
          for (const type of ["pointerdown", "mousedown", "mouseup", "click"]) {
            const Ctor = type.startsWith("pointer") && window.PointerEvent
              ? window.PointerEvent : window.MouseEvent;
            ta.dispatchEvent(new Ctor(type, { bubbles: true, cancelable: true,
                                              clientX: x, clientY: y }));
          }
        } catch (e) {}
        ta.focus();
        await sleep(150);
      };
      const fill = async (text) => {
        await focusComposer();
        // BLANK FIRST, BUT ONLY WHEN A PLAIN SET WOULD BE SWALLOWED, and the "only when" is the
        // part that was wrong the first time.
        //
        // React tracks the last value it saw on the node, so setting the same string twice leaves
        // the tracker unchanged and the second `input` never becomes a change: a retry that
        // rewrites the same text is a no-op. Writing "" first makes it a real transition.
        //
        // Doing that UNCONDITIONALLY introduced a worse race than it fixed. The blank is itself a
        // state change to EMPTY, and `ComposerPrimitive.Send` is disabled while the composer is
        // empty, so a caller that checked the button immediately after could see it enabled from
        // the previous text and then watch the blank land: the wait resolved, and by the time the
        // click came the button was disabled again. Observed as six attempts in a row clicking a
        // disabled button with 8 characters sitting in the field.
        //
        // So the blank happens only when the field already holds the target, and the settle after
        // it is a real one rather than a token 30 ms.
        if ((ta.value || "") === text) {
          setter.call(ta, "");
          ta.dispatchEvent(new Event("input", { bubbles: true }));
          await sleep(150);
        }
        setter.call(ta, text);
        ta.dispatchEvent(new Event("input", { bubbles: true }));
        const dl = performance.now() + 10000;
        while (performance.now() < dl && (ta.value || "") !== text) await sleep(100);
        await sleep(250);
        return (ta.value || "") === text;
      };
      const pressEnter = async () => {
        for (const type of ["keydown", "keypress", "keyup"]) {
          ta.dispatchEvent(new KeyboardEvent(type, {
            key: "Enter", code: "Enter", keyCode: 13, which: 13,
            bubbles: true, cancelable: true }));
        }
        await sleep(200);
      };
      const composerT0 = performance.now();
      const filled0 = await fill(o.prompt);
      const composerMs = performance.now() - composerT0;
      // WAIT FOR THE BUTTON TO BE ENABLED, not merely present, AND RE-FILL WHILE WAITING.
      //
      // `ComposerPrimitive.Send` is disabled while the app's composer state is empty, and the
      // app's state is not the DOM's. Filling works by setting `.value` through the native
      // descriptor and dispatching `input`, which React turns into its own change event ONLY if
      // its listener is already attached. Dispatch during hydration and the text sits on the
      // element, visible and readable, while the app still believes the composer is empty. Send
      // then stays disabled for ever and no amount of waiting fixes it.
      //
      // Observed exactly that, twice, on an EMPTY thread and never on a seeded one, which is what
      // a hydration race looks like: an empty thread has nothing to render, so the app reaches
      // this point sooner. A rehearsal burned the full four-minute budget and reported that "the
      // model chip has not restored", which is a true statement about a disabled button and the
      // wrong diagnosis.
      //
      // So the wait RE-FILLS every couple of seconds instead of only looking. Both arms get the
      // identical treatment, it happens before any scored window of the stream phase, and how
      // many refills it took is recorded so a session that needed several is visible rather than
      // silently equal to one that needed none.
      const isOff = (b) => Boolean(b.disabled || b.getAttribute("aria-disabled") === "true");
      const enabledSend = () => W.dom.sendButtons().find((b) => !isOff(b)) || null;
      const refills = { count: 0, first_ok: filled0, ms: null };
      const refillT0 = performance.now();
      const sendOrRefill = async () => {
        const b = enabledSend();
        if (b) return b;
        refills.count += 1;
        await fill(o.prompt);
        return enabledSend();
      };
      const send = await waitFor(sendOrRefill, o.sendTimeoutMs,
                                 "an ENABLED send button. The composer was re-filled " +
                                 "repeatedly, so this is not the app missing an input event; " +
                                 "either the model chip never restored or Send is held by " +
                                 "something else");
      refills.ms = Math.round(performance.now() - refillT0);
      W.refills = refills;
      const sendT = performance.now();
      let started = false;
      const attempts = [];
      W.sendAttempts = attempts;
      for (let attempt = 0; attempt < 6 && !started; attempt++) {
        const filled = attempt === 0 ? filled0 : await fill(o.prompt);
        // RE-ESTABLISH THE ENABLED STATE IMMEDIATELY BEFORE CLICKING, rather than trusting the
        // one the wait above resolved on. Between the two there is a fill, and a fill can leave
        // the composer momentarily empty. Clicking a disabled button does nothing, silently, and
        // six silent no-ops read as "the stream never started".
        const btn = (await sendOrRefill()) || W.dom.sendButtons()[0] || send;
        const a = { attempt, filled, has_button: !!btn, via: null,
                    disabled: btn ? (btn.disabled || btn.getAttribute("aria-disabled")) : null,
                    composer_value_len: (ta.value || "").length };
        attempts.push(a);
        if (btn) { btn.click(); a.via = "button"; }
        let dl = performance.now() + 20000;
        while (performance.now() < dl && !started) {
          if (W.dom.isRunning()) started = true; else await sleep(50);
        }
        if (!started) {
          await pressEnter();
          a.via = a.via ? a.via + "+enter" : "enter";
          dl = performance.now() + 20000;
          while (performance.now() < dl && !started) {
            if (W.dom.isRunning()) started = true; else await sleep(50);
          }
        }
        a.started = started;
        if (!started) await sleep(1000 * (attempt + 1));
      }
      if (!started) {
        throw new Error("timeout waiting for stream to start (stop/queue button). attempts=" +
                        JSON.stringify(attempts));
      }
      const firstTokenMs = performance.now() - sendT;
      slice();

      mark("stream");
      const streamT0 = performance.now();
      const dl = performance.now() + o.maxMs;
      while (W.dom.isRunning() && performance.now() < dl) await sleep(250);
      const streamEl = performance.now() - streamT0;
      const stillRunning = W.dom.isRunning();
      const streamW = windowOf("stream", slice(), streamEl, { first_token_ms: firstTokenMs });

      mark("recover");
      const recT0 = performance.now();
      await sleep(o.recoverMs);
      const recEl = performance.now() - recT0;
      const recW = windowOf("recover", slice(), recEl);
      mark("end");

      post({
        __done: true, ok: true, ua: navigator.userAgent, rung: o.rung, arm: o.arm,
        engine_probe: engineProbe(), build: buildProbe(), url: location.href,
        mount: { ms: Math.round(mountMs), by: mountedBy, census: censusMounted,
                 last_marker: o.lastMarker },
        clamp, liveness, notes: W.notes,
        readback_mounted: readbackMounted, readback_final: readback("final"),
        first_token_ms: firstTokenMs, composer_fill_ms: Math.round(composerMs),
        send_attempts: attempts, composer_refills: refills,
        still_running_at_deadline: stillRunning,
        scroll_detail: scrollDetail,
        marks: W.marks, actions,
        phases: [idleW, jamW, scrollW, streamW, recW],
        final: census(),
      });
    } catch (e) {
      hogOff();
      post({ __done: true, ok: false, rung: o.rung, arm: o.arm, phase: W.phase,
             error: String((e && e.message) || e),
             error_stack: String((e && e.stack) || ""),
             send_attempts: W.sendAttempts || null,
             marks: W.marks, url: location.href, notes: W.notes,
             readback_error: (() => { try { return readback("error"); } catch (x) { return null; } })(),
             dom: { composer: !!W.dom.composer(), send: W.dom.sendButtons().length,
                    running: W.dom.isRunning(), elements: W.dom.elements(),
                    messages: W.dom.messages(),
                    bodyText: (document.body.innerText || "").slice(0, 1200) } });
    }
  };
})();
