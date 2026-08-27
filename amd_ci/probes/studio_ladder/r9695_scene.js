// Injected at document-start into REAL WebKitGTK, before app code.
//
// THE QUESTION. PR 9695 renders fenced code inside reasoning panes PLAIN, and was measured at
// 5.02x on `action:reasoning_toggle_all` at r100K (AMD runs 32833058576 / 32841817590, span
// census 74,250 -> 10,917). Since then main merged PR 9799 (the idle grammar pre-warm that was
// warming on an EMPTY STRING), PR 9731 and PR 9787, all on this same path. This scene exists to
// ask whether that 5.02x survives a rebase onto today's main, or whether the other work has
// already absorbed it.
//
//   main   origin/main at 0be140dbd
//   head   the same commit with PR 9695's rebased diff applied
//
// THE GESTURE IS `reasoning_toggle_all`, AND THE NAME MATTERS. This campaign has already been
// caught by two different gestures sharing one name: opening the FIRST pane and opening EVERY
// pane read 1.9% and 1.913x for the same mechanism, both correctly. `reasoning_toggle` below
// opens the first pane and is recorded but never scored; `reasoning_toggle_all` opens every pane
// and is the window the 5.02x was taken on.
//
// The film, per session (one arm, one rung, one repetition):
//
//   mount                        navigation until the last SEEDED message is in the DOM
//   idle:calibrate               a dedicated window whose only job is the setTimeout clamp
//   idle                         THE IDLE CONTROL. Nobody touches it.
//   idle_jammed                  THE POSITIVE CONTROL, at every rung, in every session
//   scroll                       the gesture a user makes on a long thread
//   reasoning_toggle             the ONE-pane gesture. Recorded, never scored.
//   reasoning_toggle_all         THE SCORED GESTURE
//   reasoning_fidelity_settled   not a performance window: the settled census that prices the
//                                fidelity this PR trades away
//   select_all_copy              the other window that carried the original report
//
// THE INSTRUMENT, and why two of the three plausible channels are banned.
//
//   * `GdkFrameClock::after-paint` under `begin_updating()` reads 60.0 fps with the main thread
//     80% blocked. It drives the clock independently of whether the app can do any work.
//   * `1000 / p50` of rAF gaps reads 16.0 ms jammed and unjammed alike: a median cannot see mass
//     that has moved into the tail. Measured on this venue: a jam took the correct channel from
//     62.0 to 16.6 fps and left `1000/p50` at 62.5 -> 62.5, to one decimal place.
//
// The headline is therefore EFFECTIVE FRAME RATE OVER WALL TIME. p50 is still reported and is
// labelled as a statistic that cannot see the tail.
//
// TWO CONTROLS, AND BOTH CAN VOID A READING.
//
//   `idle_jammed` is the POSITIVE control: a channel that cannot report a deliberately blocked
//   main thread cannot report one that happened by accident. The documented pass is ~61 -> ~17
//   fps.
//
//   `idle` is the IDLE control, and it is the one this particular question is most likely to get
//   wrong. At r500K a `plain` arm once read 4.16 fps against 2.77 on the arm it was being
//   compared with -- +50%, and it would have been published -- while its idle window was already
//   stalling in 3 repetitions of 5 (idle fps 18.6, 14.8 and 4.9 against 61-62 everywhere else,
//   with 7.9-9.3 s frames INSIDE the idle window). An arm whose idle window is already stalling
//   is not in a comparable state, and the criteria module discards those repetitions by name and
//   counts them in the report.
//
// busy_pct is calibrated, not assumed. A setTimeout(1) loop cannot resolve below the platform
// clamp, so the clamp is measured on its own window and blocked time is the excess over it. If
// the clamp cannot be established the field is null WITH A REASON, never 0, and a null must be
// carried as MISSING rather than averaged as zero.
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

  // ── THE FIDELITY SIDE, from amdv_scene_p123.js ────────────────────────────────────────────
  //
  // `highlight_spans` is a thread-wide total, so it cannot tell "no fence is highlighted any
  // more" from "the same fences are highlighted with fewer tokens". PR 9695 acts per fence and
  // only inside reasoning panes, so the honest census is per fence and scoped to those panes: a
  // thread-wide count mixes the container the gesture acts on with containers it does not touch.
  const reasoningRootsAll = () =>
    Array.from(qa('[data-slot="reasoning-root"]'));

  const codeFences = (root) => Array.from((root || document).querySelectorAll("pre")).map((p) => ({
    chars: (p.textContent || "").length,
    spans: p.querySelectorAll("span").length,
  }));

  //: The raw per-fence array would be thousands of entries at 500K, so the payload carries counts
  //: and buckets. The two thresholds are the ones the code actually uses:
  //: `isOversizedStreamingCode` at OVERSIZED_OPEN_CODE_CHARS = 4,096 and
  //: `shouldAutoHighlightStreamingCode` at MAX_AUTO_HIGHLIGHT_SOURCE_CODE_UNITS = 16,384.
  const P3_OPEN_CAP = 4 * 1024;
  const P3_UPGRADE_CAP = 16 * 1024;
  const fenceStats = (fences) => {
    const n = fences.length;
    const lens = fences.map((f) => f.chars).sort((a, b) => a - b);
    const hot = fences.filter((f) => f.spans > 0);
    const pc = (p) => (n ? lens[Math.min(n - 1, Math.max(0, Math.round(p * (n - 1))))] : null);
    return {
      fences: n,
      highlighted_fences: hot.length,
      plain_fences: n - hot.length,
      spans: fences.reduce((a, f) => a + f.spans, 0),
      chars: lens.reduce((a, b) => a + b, 0),
      p50_chars: pc(0.5), p90_chars: pc(0.9), p99_chars: pc(0.99),
      max_chars: n ? lens[n - 1] : null,
      over_p3_open: fences.filter((f) => f.chars >= P3_OPEN_CAP).length,
      over_p3_upgrade: fences.filter((f) => f.chars > P3_UPGRADE_CAP).length,
      deferred_shells: qa('[data-unsloth-fence-deferred="true"]').length,
    };
  };

  // DELIBERATELY NOT PART OF `census()`. The per-fence walk visits every `pre` and every span
  // beneath it, and the two arms differ by tens of thousands of spans, so folding it into the
  // census that runs INSIDE each measured window would charge the arm with MORE spans more
  // instrument cost than the arm with fewer: the measuring device co-varying with the subject, in
  // the one direction that would manufacture the result being looked for. It is called only from
  // the settled fidelity action, which runs after every measured window and is never scored.
  const fenceCensus = () => ({
    thread: fenceStats(codeFences(null)),
    reasoning: fenceStats(reasoningRootsAll().flatMap((r) => codeFences(r))),
    // The mechanism, read out of the DOM rather than out of the patch list: on `head` a fence
    // inside a reasoning pane renders the SAME `DeferredFenceShell` an unreached fence already
    // shows, so it carries `data-unsloth-fence-deferred="true"` and no spans.
    reasoning_deferred_shells: reasoningRootsAll()
      .reduce((a, r) => a + r.querySelectorAll('[data-unsloth-fence-deferred="true"]').length, 0),
    reasoning_chars: reasoningRootsAll()
      .reduce((a, e) => a + (e.textContent || "").length, 0),
  });
  W.fenceCensus = fenceCensus;

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
        let detail = "", ok = true, appSync = null, na = false, r0 = null;
        try {
          const r = await fn();
          r0 = r;
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
          // WHAT WAS MOUNTED WHILE THE PANES WERE OPEN. census_before and census_after are both
          // taken with the panes CLOSED, because the gesture opens and then closes them, so
          // neither can see what the open state put in the DOM.
          census_open: (r0 && r0.census_open) || null,
          // Carried so the criteria can refuse a settled census that never settled: a bounded
          // wait that timed out yields a snapshot of a page still doing work, which is the one
          // thing the settled action exists to avoid.
          settled: r0 ? r0.settled : undefined,
          settle_polls: r0 ? r0.settle_polls : undefined,
          fence_census: (r0 && r0.fence_census) || null,
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


      // EVERY PANE, WHICH IS A DIFFERENT GESTURE FROM THE ONE ABOVE, and it is the one PR 9695's
      // 5.02x was measured on. The one-pane gesture caps its own possible effect at a single
      // trace no matter how large the thread is; this is where the mechanism has room to matter.
      // From amdv_scene_p123.js, unchanged, so this run's window and the original run's window
      // are the same window.
      mark("action:reasoning_toggle_all");
      await runAction("reasoning_toggle_all", async () => {
        const triggers = Array.from(qa('[data-slot="reasoning-trigger"]'));
        if (triggers.length === 0) {
          return { detail: "no reasoning pane at this rung", not_applicable: true };
        }
        const open = () => qa('[data-slot="reasoning-root"][data-state="open"]').length;
        const before = open();
        const c0 = performance.now();
        for (const t of triggers) t.click();
        const sync1 = performance.now() - c0;
        await sleep(2500);
        const opened = open();
        const censusOpen = census();
        const c1 = performance.now();
        for (const t of Array.from(qa('[data-slot="reasoning-trigger"]'))) t.click();
        const sync2 = performance.now() - c1;
        await sleep(2500);
        if (opened === before) throw new Error("no pane opened: " + before + " -> " + opened);
        return { detail: `open ${before} -> ${opened} of ${triggers.length}, then closed`,
                 census_open: censusOpen,
                 app_sync_ms: Math.round(Math.max(sync1, sync2) * 10) / 10 };
      });

      // ── FIDELITY, SETTLED. Not a performance window; its frames are never scored. ──────────
      //
      // Every census above is a SNAPSHOT taken 2,500 ms after a click, and syntax highlighting on
      // this app is ASYNCHRONOUS. That makes `highlight_spans` in `census_open` a race: the same
      // arm read 11,530 spans on one repetition and 11,094 on the next, and a JAMMED page read
      // 7,259 -- not because a jam changes what the app renders, but because a blocked main
      // thread had not finished highlighting when the snapshot was taken. A number that moves
      // with how busy the page happens to be cannot price a FIDELITY change.
      //
      // So this opens every pane again, waits for the span count to stop changing, and censuses
      // that. It runs AFTER every measured window, so it cannot perturb one.
      mark("action:reasoning_fidelity_settled");
      await runAction("reasoning_fidelity_settled", async () => {
        const triggers = Array.from(qa('[data-slot="reasoning-trigger"]'));
        if (triggers.length === 0) {
          return { detail: "no reasoning pane at this rung", not_applicable: true };
        }
        for (const t of triggers) t.click();
        // Quiescence, not a fixed sleep: three consecutive identical readings of the quantity the
        // policy acts on. Bounded, because at 500K a page this loaded may never settle, and a
        // census that never happened must be REPORTED as such rather than waited for forever.
        const SETTLE_POLL_MS = 400, SETTLE_MAX_MS = 30000, SETTLE_STABLE_N = 3;
        const dl = performance.now() + SETTLE_MAX_MS;
        let lastN = -1, stable = 0, polls = 0, settled = false;
        while (performance.now() < dl) {
          await sleep(SETTLE_POLL_MS);
          polls += 1;
          const n = W.dom.highlightSpans();
          if (n === lastN) { stable += 1; } else { stable = 0; lastN = n; }
          if (stable >= SETTLE_STABLE_N) { settled = true; break; }
        }
        const censusOpen = census();
        const fences = fenceCensus();
        for (const t of Array.from(qa('[data-slot="reasoning-trigger"]'))) t.click();
        await sleep(500);
        return { detail: `settled=${settled} after ${polls} polls, spans=${lastN}`,
                 settled, settle_polls: polls, census_open: censusOpen,
                 fence_census: fences };
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


      // NO SEND, NO STREAM. This question is about a SETTLED thread whose panes are then opened,
      // and the composer/send path is the harness's largest single source of lost sessions (a
      // React value-tracking race that leaves text visible in the DOM and invisible to the app).
      // A window this run does not score is not worth a race that can cost the whole session.
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
    } catch (e) {
      hogOff();
      post({ __done: true, ok: false, rung: o.rung, arm: o.arm, phase: W.phase,
             error: String((e && e.message) || e),
             error_stack: String((e && e.stack) || ""),
             marks: W.marks, url: location.href, notes: W.notes,
             readback_error: (() => { try { return readback("error"); } catch (x) { return null; } })(),
             dom: { composer: !!W.dom.composer(), send: W.dom.sendButtons().length,
                    running: W.dom.isRunning(), elements: W.dom.elements(),
                    messages: W.dom.messages(),
                    bodyText: (document.body.innerText || "").slice(0, 1200) } });
    }
  };
})();
