// Injected at document-start into REAL WebKitGTK, before app code.
//
// scripts/wk_scene9477.js measures ONE streamed reply into a FRESH thread: it varies reply
// length and has no thread-size axis at all. This scene is the other axis, which is the one
// the user's report is actually about: "60 fps at 0K, 5 fps at 100K, 5 fps at 500K" is a
// claim about how big the THREAD is, not about how long the reply is.
//
// So the film here is: mount a thread that has already been seeded to the rung, wait until it
// is really mounted, then measure four phases with the rung as the only variable.
//
//   mount   navigation to the last seeded message being present in the DOM
//   idle    nobody touches it. If a big thread is slow at rest, this is where it shows.
//   scroll  the gesture a user makes on a long thread, at three depths
//   stream  the ladder's own 6,000-char tail arrives through the pacer
//   recover after the stream ends
//
// Two independent frame channels are recorded and NEITHER is trusted alone:
//   * rAF gaps, which in a headless X server measure MAIN THREAD AVAILABILITY (8-9 ms gaps on
//     an idle page, i.e. "120 Hz" on a server with no refresh rate);
//   * phase marks in wall-clock, so the driver's GdkFrameClock::after-paint series (one
//     emission per PRESENTED frame of the toplevel) can be sliced per phase in Python.
// The presented-frame channel is the headline. The disagreement between them is a finding.
//
// busy_pct is calibrated, not assumed. A setTimeout(1) loop cannot resolve below the platform
// clamp, so the clamp is measured during the idle phase and blocked time is the excess over
// it. If the clamp cannot be established the field is null WITH A REASON, never 0.
(() => {
  if (window.__av) return;
  const W = { gaps: [], ticks: [], samples: [], actions: [], marks: [], t0: null,
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
  W.dom = {
    composer: () => q('textarea[aria-label="Message input"]'),
    sendButton: () => q('button[aria-label="Send message"]'),
    isRunning: () => Boolean(q('button[aria-label="Stop generating"]') ||
                             q('button[aria-label="Queue message"]')),
    elements: () => document.getElementsByTagName("*").length,
    messages: () => document.querySelectorAll("[data-role]").length,
    assistantChars: () => Array.from(document.querySelectorAll('[data-role="assistant"]'))
      .reduce((a, e) => a + (e.textContent || "").length, 0),
    reasoningRoots: () => document.querySelectorAll('[data-slot="reasoning-root"]').length,
    reasoningOpen: () => document.querySelectorAll('[data-slot="reasoning-root"][data-state="open"]').length,
    codeBlocks: () => document.querySelectorAll("pre").length,
    highlightSpans: () => document.querySelectorAll("pre span").length,
    // ── the quantities PIECE 3 acts on, per fence rather than in aggregate ──────────────────
    //
    // `highlight_spans` is a thread-wide total, so it cannot tell "no fence is highlighted any
    // more" from "the same fences are highlighted with fewer tokens". Piece 3's cap is stated
    // per fence and against a SOURCE LENGTH, so the honest census is per fence and carries the
    // length: it is what makes the corpus-crossing distribution a measurement of the rendered
    // DOM instead of an inference from a constant.
    //
    // Scoped to reasoning panes as well as thread-wide, because the gesture under test opens
    // reasoning panes and `reasoning.tsx` renders them with `codeHighlighting="plain"`, so a
    // thread-wide count mixes the container the gesture acts on with containers it does not
    // touch.
    codeFences: (root) => Array.from((root || document).querySelectorAll("pre")).map((p) => ({
      chars: (p.textContent || "").length,
      spans: p.querySelectorAll("span").length,
    })),
  };

  const reasoningRootsAll = () =>
    Array.from(document.querySelectorAll('[data-slot="reasoning-root"]'));

  //: fences, summarised. The raw per-fence array would be thousands of entries at 500K, so the
  //: payload carries counts and buckets.
  //:
  //: THE BUCKETS ARE BOTH OF 9477's THRESHOLDS, because it has two and the campaign has been
  //: quoting only the larger one. `isOversizedStreamingCode` sends a fence down the plain-first
  //: path at OVERSIZED_OPEN_CODE_CHARS = 4,096, and `shouldAutoHighlightStreamingCode` then
  //: refuses the upgrade back to a highlighted subtree above
  //: MAX_AUTO_HIGHLIGHT_SOURCE_CODE_UNITS = 16,384. A census reported only against 16,384 would
  //: understate how many fences the policy touches by every fence between 4 KiB and 16 KiB.
  //:
  //: 20,000 is main's MAX_HIGHLIGHT_CHARS and is carried for comparison only. It is NOT a
  //: precedent for these: `markdownPluginNeeds` has two callers and neither is the assistant
  //: thread, so on main today every reply fence and every reasoning fence is highlighted at any
  //: size. The bucket is reported so that claim is checkable from the artifact.
  const P3_OPEN_CAP = 4 * 1024;
  const P3_UPGRADE_CAP = 16 * 1024;
  const MAIN_CAP = 20000;
  const fenceStats = (fences) => {
    const n = fences.length;
    const lens = fences.map((f) => f.chars).sort((a, b) => a - b);
    const hot = fences.filter((f) => f.spans > 0);
    const pc = (p) => (n ? lens[Math.min(n - 1, Math.max(0, Math.round(p * (n - 1))))] : null);
    const totalChars = lens.reduce((a, b) => a + b, 0);
    const overChars = (t) => fences.reduce((a, f) => a + (f.chars > t ? f.chars : 0), 0);
    return {
      fences: n,
      highlighted_fences: hot.length,
      plain_fences: n - hot.length,
      spans: fences.reduce((a, f) => a + f.spans, 0),
      chars: totalChars,
      p50_chars: pc(0.5), p90_chars: pc(0.9), p99_chars: pc(0.99),
      max_chars: n ? lens[n - 1] : null,
      over_p3_open: fences.filter((f) => f.chars >= P3_OPEN_CAP).length,
      over_p3_upgrade: fences.filter((f) => f.chars > P3_UPGRADE_CAP).length,
      over_main_cap: fences.filter((f) => f.chars > MAIN_CAP).length,
      chars_over_p3_open: overChars(P3_OPEN_CAP - 1),
      chars_over_p3_upgrade: overChars(P3_UPGRADE_CAP),
      chars_over_main_cap: overChars(MAIN_CAP),
    };
  };

  const post = (o) => {
    if (o && o.__done) W.result = o;
    try { window.webkit.messageHandlers.bench.postMessage(JSON.stringify(o)); } catch (e) {}
  };
  W.post = post;
  W.note = (m) => { W.log.push([Date.now(), m]); post({ note: m }); };

  // A phase mark is a WALL CLOCK boundary. The driver's after-paint series is wall clock too,
  // so Python can cut the presented-frame series at exactly these instants. performance.now()
  // would not be comparable across the two processes.
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

  // THE TEXT INSIDE THE REASONING PANES, on its own. `assistant_chars` is the whole thread, so a
  // pane that pagination cut from 12,149 characters to 7,721 moves it by 2% at r100K and by 0.4%
  // at r500K, and a gate scored on it cannot tell "the mechanism did not engage" from "the
  // mechanism engaged and is small relative to the document". Those are different findings.
  const reasoningChars = () => Array.from(
    document.querySelectorAll('[data-slot="reasoning-root"]'))
    .reduce((a, e) => a + (e.textContent || "").length, 0);

  const census = () => ({
    elements: W.dom.elements(), messages: W.dom.messages(),
    assistant_chars: W.dom.assistantChars(),
    reasoning_chars: reasoningChars(),
    show_more_buttons: document.querySelectorAll('[data-slot="reasoning-show-earlier"]').length,
    reasoning_roots: W.dom.reasoningRoots(), reasoning_open: W.dom.reasoningOpen(),
    code_blocks: W.dom.codeBlocks(), highlight_spans: W.dom.highlightSpans(),
    scroll_height: (document.scrollingElement || document.body).scrollHeight,
  });

  // DELIBERATELY NOT PART OF `census()`. The per-fence walk visits every `pre` and every span
  // beneath it, and the arms under test differ by up to 240,000 spans, so folding it into the
  // census that runs INSIDE each measured window would charge the arm with more spans more
  // instrument cost than the arm with fewer -- the measuring device co-varying with the subject,
  // in the one direction that would manufacture the result being looked for. It is called only
  // from the settled fidelity action, which runs after every measured window and is not scored.
  const fenceCensus = () => ({
    thread: fenceStats(W.dom.codeFences(null)),
    reasoning: fenceStats(reasoningRootsAll().flatMap((r) => W.dom.codeFences(r))),
  });
  W.fenceCensus = fenceCensus;
  W.census = census;

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
      fps_p50: 1000 / pcOf(s, 0.5), fps_p5: 1000 / pcOf(s, 0.95),
      fps_worst: 1000 / s[s.length - 1],
      frames_over_33: over(33), frames_over_33_pct: 100 * over(33) / s.length,
      frames_over_100: over(100), frames_over_100_pct: 100 * over(100) / s.length,
      sum_ms: s.reduce((a, b) => a + b, 0),
    };
  };
  W.summarise = summarise;

  // ── the clamp, measured ────────────────────────────────────────────────────────────────────
  // A setTimeout(1) never comes back in 1 ms. The floor is the platform's, and blocked time is
  // only the excess over it. Calibrated on the idle phase's own ticks, on this engine, on this
  // host, rather than assumed to be 4 ms or 1 ms.
  const MAX_CLAMP_MS = 10.0;
  W.calibrate = (tickSlice) => {
    const s = tickSlice.slice().sort((a, b) => a - b);
    if (s.length < 40) return { clamp_ms: null, reason: `only ${s.length} idle ticks, need 40` };
    const med = pcOf(s, 0.5);
    if (!(med > 0)) return { clamp_ms: null, reason: "idle tick median was not positive" };
    if (med > MAX_CLAMP_MS) {
      // An idle window whose own median is already this late was not idle. Reporting a busy_pct
      // computed against it would be measuring the calibration, not the page.
      return { clamp_ms: null, samples: s.length, p50: med,
               reason: `idle tick median ${med.toFixed(1)} ms exceeds the ${MAX_CLAMP_MS} ms ` +
                       "ceiling, so the idle window was not idle and no clamp is trustworthy" };
    }
    return { clamp_ms: med, samples: s.length, p50: med, p95: pcOf(s, 0.95), reason: null };
  };

  const busyOver = (tickSlice, elapsedMs, clamp) => {
    if (clamp.clamp_ms === null) {
      return { busy_pct: null, blocked_ms: null, busy_pct_reason: clamp.reason,
               ticks: tickSlice.length };
    }
    if (!(elapsedMs > 0)) {
      return { busy_pct: null, blocked_ms: null, ticks: tickSlice.length,
               busy_pct_reason: "the phase reported no elapsed time" };
    }
    let blocked = 0;
    for (const g of tickSlice) { const x = g - clamp.clamp_ms; if (x > 0) blocked += x; }
    return { busy_pct: Math.round((blocked / elapsedMs) * 1000) / 10,
             blocked_ms: Math.round(blocked * 10) / 10, ticks: tickSlice.length,
             busy_pct_reason: null };
  };

  // ── the film ───────────────────────────────────────────────────────────────────────────────
  W.run = async (opts) => {
    const o = Object.assign({
      idleMs: 6000, recoverMs: 6000, maxMs: 900000, prompt: "continue",
      lastMarker: null, mountTimeoutMs: 300000, rung: "?",
    }, opts || {});

    const cut = { gaps: 0, ticks: 0 };
    const slice = () => {
      const g = W.gaps.slice(cut.gaps), t = W.ticks.slice(cut.ticks);
      cut.gaps = W.gaps.length; cut.ticks = W.ticks.length;
      return { g, t };
    };

    try {
      W.t0 = performance.now();
      mark("mount");
      const mountT0 = performance.now();
      // The composer appearing is NOT the thread being mounted. At 500K the app paints a shell
      // and then spends a long time putting messages in, and an idle baseline taken there is a
      // reading of the mount, not of the mounted thread.
      const ta = await waitFor(W.dom.composer, 120000, "composer textarea");
      let mountedBy = "composer";
      if (o.lastMarker) {
        // The seeder's own marker string for the LAST seeded turn. If it is on the page, every
        // message before it exists too. seeder.turn_marker() must match this exactly.
        await waitFor(() => (document.body.innerText || "").indexOf(o.lastMarker) >= 0,
                      o.mountTimeoutMs, "last seeded marker in the DOM");
        mountedBy = "last_seeded_marker";
      }
      // Two nested rAFs: the marker being in the DOM is not the frame that shows it.
      await new Promise((r) => nativeRaf(() => nativeRaf(r)));
      const mountMs = performance.now() - mountT0;
      const censusMounted = census();
      slice();

      // Two idle windows, not one. The clamp is the floor a busy_pct is a subtraction against,
      // so deriving it from the same window it is then reported for is circular: half that
      // window's ticks are at or below their own median by construction. studiobench opens a
      // dedicated `idle:calibrate` window for this reason and this follows it.
      mark("idle:calibrate");
      const calT0 = performance.now();
      await sleep(o.idleMs);
      const calS = slice();
      const clamp = W.calibrate(calS.t);
      clamp.window_ms = Math.round(performance.now() - calT0);

      mark("idle");
      const idleT0 = performance.now();
      await sleep(o.idleMs);
      const idleEl = performance.now() - idleT0;
      const idleS = slice();

      mark("scroll");
      const scrollT0 = performance.now();
      const el = scrollerAt(window.innerWidth / 2, window.innerHeight * 0.5);
      let scrollDetail = "no scroller";
      if (el) {
        const h0 = el.scrollHeight, top0 = el.scrollTop;
        // Up through the thread, pause, back down. Three passes so one stall cannot dominate.
        for (let pass = 0; pass < 3; pass++) {
          for (let i = 0; i < 10; i++) { el.scrollTop = Math.max(0, el.scrollTop - 900); await sleep(80); }
          await sleep(300);
          for (let i = 0; i < 10; i++) { el.scrollTop = el.scrollTop + 900; await sleep(80); }
          await sleep(300);
        }
        el.scrollTop = el.scrollHeight;
        scrollDetail = `${el.tagName}.${(el.className || "").toString().slice(0, 30)} ` +
                       `h=${h0} top ${top0}->${el.scrollTop}`;
      }
      const scrollEl = performance.now() - scrollT0;
      const scrollS = slice();

      // ── functional proof + the 9477 window ─────────────────────────────────────────────────
      // reasoning_toggle is the window PR 9477 piece 2 has to be judged on, and it is also the
      // clearest proof that the app is really working: the pane cannot open and close unless
      // the thread rendered, the message mounted and React is committing.
      //
      // app_sync_ms is recorded separately from the window: it is the time spent INSIDE the
      // click handler, which is app code and nothing else. The window around it includes this
      // scene's own waiting. Reporting only the window would repeat the mistake of attributing
      // harness cost to the app, which turned a "153 ms" traced call into ~26 ms of real
      // listener elsewhere in this campaign.
      const actions = [];
      const runAction = async (name, fn) => {
        const i0 = W.gaps.length, t0 = performance.now();
        const before = census();
        let detail = "", ok = true, appSync = null;
        let na = false;
        let r0 = null;
        try {
          const r = await fn();
          r0 = r;
          detail = (r && r.detail) || String(r || "");
          appSync = r && r.app_sync_ms;
          na = Boolean(r && r.not_applicable);
        }
        catch (e) { ok = false; detail = String((e && e.message) || e); }
        const dur = performance.now() - t0;
        const g = W.gaps.slice(i0), t = W.ticks.slice(cut.ticks);
        cut.gaps = W.gaps.length; cut.ticks = W.ticks.length;
        actions.push({ name, ok, not_applicable: na, detail, app_sync_ms: appSync,
                       elapsed_ms: Math.round(dur), raf: summarise(g),
                       raf_gaps_ms: g.map((x) => Math.round(x * 10) / 10),
                       busy: busyOver(t, dur, clamp),
                       // WHAT WAS MOUNTED WHILE THE PANE WAS OPEN. census_before and
                       // census_after are both taken with the pane CLOSED, because the gesture
                       // opens and then closes it, so neither can show whether a windowed mount
                       // put less in the DOM. Without this a "pagination changed nothing" result
                       // is indistinguishable from "the flag never reached the DOM".
                       census_open: (r0 && r0.census_open) || null,
                       // Carried so the criteria can refuse a settled census that never settled.
                       // A bounded wait that timed out yields a snapshot of a page still doing
                       // work, which is the very thing this action exists to avoid.
                       settled: r0 ? r0.settled : undefined,
                       settle_polls: r0 ? r0.settle_polls : undefined,
                       fence_census: (r0 && r0.fence_census) || null,
                       census_before: before, census_after: census(),
                       wall_start_ms: Date.now() - Math.round(dur), wall_end_ms: Date.now() });
        return ok;
      };

      mark("action:reasoning_toggle");
      await runAction("reasoning_toggle", async () => {
        const root = document.querySelector('[data-slot="reasoning-root"]');
        if (!root) {
          return { detail: "no reasoning pane at this rung: an empty thread has no assistant "
                           + "message until the first reply, so there is nothing to toggle",
                   not_applicable: true };
        }
        const trigger = root.querySelector("button");
        if (!trigger) throw new Error("no reasoning trigger button");
        const s0 = root.getAttribute("data-state");
        // The synchronous cost of the click handler: app code, measured with nothing else in
        // the way. React commits inside this call.
        const c0 = performance.now();
        trigger.click();
        const sync1 = performance.now() - c0;
        await sleep(1500);
        const r2 = document.querySelector('[data-slot="reasoning-root"]') || root;
        const s1 = r2.getAttribute("data-state");
        // The one snapshot that can see a windowed mount: taken with the pane OPEN, before the
        // second click closes it again.
        const censusOpen = census();
        const t2 = r2.querySelector("button");
        let sync2 = null;
        if (t2) { const c1 = performance.now(); t2.click(); sync2 = performance.now() - c1; }
        await sleep(1500);
        const s2 = (document.querySelector('[data-slot="reasoning-root"]') || r2)
          .getAttribute("data-state");
        if (s0 === s1 && s1 === s2) throw new Error("the pane never changed state: " + s0);
        return { detail: `state ${s0} -> ${s1} -> ${s2}`,
                 census_open: censusOpen,
                 app_sync_ms: Math.round(Math.max(sync1, sync2 || 0) * 10) / 10 };
      });

      // EVERY PANE, which is a DIFFERENT GESTURE from the one above and is the one the campaign's
      // r100K figures (26.4 fps, 80.4% busy, 942 ms blocked) were measured on: studiobench's
      // `reasoning_toggle` clicks every trigger in the thread, this scene's clicks the first.
      // Same name, different work. Pagination can only ever act on panes that are OPEN, so the
      // one-pane gesture caps its possible effect at one pane's trace no matter how large the
      // thread is; this is the gesture where the mechanism has room to matter.
      mark("action:reasoning_toggle_all");
      await runAction("reasoning_toggle_all", async () => {
        const triggers = Array.from(
          document.querySelectorAll('[data-slot="reasoning-trigger"]'));
        if (triggers.length === 0) {
          return { detail: "no reasoning pane at this rung", not_applicable: true };
        }
        const open = () => document.querySelectorAll(
          '[data-slot="reasoning-root"][data-state="open"]').length;
        const before = open();
        const c0 = performance.now();
        for (const t of triggers) t.click();
        const sync1 = performance.now() - c0;
        await sleep(2500);
        const opened = open();
        const censusOpen = census();
        const c1 = performance.now();
        for (const t of Array.from(
          document.querySelectorAll('[data-slot="reasoning-trigger"]'))) t.click();
        const sync2 = performance.now() - c1;
        await sleep(2500);
        if (opened === before) throw new Error("no pane opened: " + before + " -> " + opened);
        return { detail: `open ${before} -> ${opened} of ${triggers.length}, then closed`,
                 census_open: censusOpen,
                 app_sync_ms: Math.round(Math.max(sync1, sync2) * 10) / 10 };
      });

      // ── FIDELITY, SETTLED. Not a performance window; do not score its frames. ───────────────
      //
      // Every census above is a SNAPSHOT taken 2,500 ms after the click, and syntax highlighting
      // on this app is asynchronous. That makes `highlight_spans` in `census_open` a race: the
      // same arm read 11,530 spans on one repetition and 11,094 on the next, and the JAMMED
      // control read 7,259 -- not because the jam changed what the app renders, but because a
      // blocked main thread had not finished highlighting when the snapshot was taken. A number
      // that moves with how busy the page happens to be cannot price a FIDELITY change.
      //
      // So this opens every pane again, waits for the span count to stop changing, and censuses
      // that. It runs AFTER every measured window, so it cannot perturb one. Its frame numbers
      // are recorded like any other action's and are deliberately never read by the criteria: the
      // only thing taken from here is `census_open`, which is what each arm's DOM SETTLES to.
      mark("action:reasoning_fidelity_settled");
      await runAction("reasoning_fidelity_settled", async () => {
        const triggers = Array.from(
          document.querySelectorAll('[data-slot="reasoning-trigger"]'));
        if (triggers.length === 0) {
          return { detail: "no reasoning pane at this rung", not_applicable: true };
        }
        for (const t of triggers) t.click();
        // Quiescence, not a fixed sleep: two consecutive identical readings of the quantity the
        // code policy acts on. Bounded, because at 500K a page this loaded may never settle, and
        // a census that never happened must be reported as such rather than waited for forever.
        const SETTLE_POLL_MS = 400, SETTLE_MAX_MS = 30000, SETTLE_STABLE_N = 3;
        const dl = performance.now() + SETTLE_MAX_MS;
        let last = -1, stable = 0, polls = 0, settled = false;
        while (performance.now() < dl) {
          await sleep(SETTLE_POLL_MS);
          polls += 1;
          const n = W.dom.highlightSpans();
          if (n === last) { stable += 1; } else { stable = 0; last = n; }
          if (stable >= SETTLE_STABLE_N) { settled = true; break; }
        }
        const censusOpen = census();
        const fences = fenceCensus();
        for (const t of Array.from(
          document.querySelectorAll('[data-slot="reasoning-trigger"]'))) t.click();
        await sleep(500);
        return { detail: `settled=${settled} after ${polls} polls, spans=${last}`,
                 settled, settle_polls: polls, census_open: censusOpen,
                 fence_census: fences };
      });

      mark("action:select_all_copy");
      await runAction("select_all_copy", async () => {
        const root = document.querySelector('[data-slot="reasoning-root"]')
          || document.querySelector('[data-role="assistant"]');
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

      if (o.skipSend) {
        // The jammed control does not need a reply: its job is to show that the frame channel
        // moves when the main thread cannot keep up, and idle/scroll already show that. Asking
        // it to stream as well only gives the jam a chance to starve the model chip restore,
        // which is what made the control fail and cost the whole run its verdict.
        const ph = (name, sl, el, extra) => Object.assign({
          phase: name, elapsed_ms: Math.round(el), raf: summarise(sl.g),
          raf_gaps_ms: sl.g.map((x) => Math.round(x * 10) / 10),
          busy: busyOver(sl.t, el, clamp), census: census() }, extra || {});
        post({
          __done: true, ok: true, ua: navigator.userAgent, rung: o.rung, skipped_send: true,
          engine_probe: {
            is_webkit_gtk_ua: /AppleWebKit/.test(navigator.userAgent) && /X11/.test(navigator.userAgent),
            vendor: navigator.vendor,
            has_chrome: typeof window.chrome !== "undefined",
            has_webkit_message_handlers:
              typeof (window.webkit && window.webkit.messageHandlers) !== "undefined",
            hardwareConcurrency: navigator.hardwareConcurrency, dpr: window.devicePixelRatio },
          build: {
            scripts: Array.from(document.querySelectorAll("script[src]")).map((x) => x.getAttribute("src")),
            css: Array.from(document.querySelectorAll('link[rel="stylesheet"]')).map((x) => x.getAttribute("href")) },
          url: location.href,
          mount: { ms: Math.round(mountMs), by: mountedBy, census: censusMounted,
                   last_marker: o.lastMarker },
          clamp, scroll_detail: scrollDetail, marks: W.marks, actions,
          phases: [ph("idle", idleS, idleEl), ph("scroll", scrollS, scrollEl,
                                                 { detail: scrollDetail })],
          final: census(),
        });
        return;
      }

      mark("send");
      const setter = Object.getOwnPropertyDescriptor(
        window.HTMLTextAreaElement.prototype, "value").set;
      // FOCUS FIRST. studiobench's own driver clicks the composer before filling it
      // (runtime/session.py::_send_turn) and records that click as the single largest cost in a
      // 500K cell. Setting .value on an unfocused textarea left the app's composer state
      // untouched here: the button existed, the click landed, and no request was ever made.
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
      // Fill, then CONFIRM the composer really holds the text before doing anything with it.
      // At 100K the app is busy enough that a fixed 250 ms wait after dispatching `input` is
      // sometimes not enough, and the send button then sends an empty composer, which looks
      // exactly like a send that was ignored.
      const fill = async (text) => {
        await focusComposer();
        setter.call(ta, text);
        ta.dispatchEvent(new Event("input", { bubbles: true }));
        const dl = performance.now() + 10000;
        while (performance.now() < dl && (ta.value || "") !== text) await sleep(100);
        await sleep(250);
        return (ta.value || "") === text;
      };
      // Some composers commit on Enter and not on the button, and a busy main thread makes the
      // difference visible. Both are tried before an attempt is called a failure.
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
      // WAIT FOR THE BUTTON TO BE ENABLED, not merely present.
      //
      // This was a real flake and it lied in the direction that matters. The model chip is
      // restored from localStorage asynchronously, and until it resolves the app keeps Send
      // disabled. With the composer already filled, a disabled Send means the MODEL is not
      // ready, not that the composer is empty -- and clicking it does nothing, silently. The
      // observed failures were 0K and 500K on one pass and 100K on another, i.e. a race, not a
      // rung effect. Left in, it would have deleted whichever rungs happened to lose the race
      // and left a ladder with holes in it that looked like data.
      const enabledSend = () => {
        const b = W.dom.sendButton();
        if (!b) return null;
        const off = b.disabled || b.getAttribute("aria-disabled") === "true";
        return off ? null : b;
      };
      const send = await waitFor(enabledSend, 120000, "an ENABLED send button (the model chip " +
                                 "restores from localStorage asynchronously)");
      const sendT = performance.now();
      let started = false;
      const attempts = [];
      W.sendAttempts = attempts;
      for (let attempt = 0; attempt < 6 && !started; attempt++) {
        const filled = attempt === 0 ? filled0 : await fill(o.prompt);
        const btn = enabledSend() || W.dom.sendButton() || send;
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
      const streamS = slice();

      mark("recover");
      const recT0 = performance.now();
      await sleep(o.recoverMs);
      const recEl = performance.now() - recT0;
      const recS = slice();
      mark("end");

      const phase = (name, s, elapsed, extra) => Object.assign({
        phase: name, elapsed_ms: Math.round(elapsed),
        raf: summarise(s.g),
        raf_gaps_ms: s.g.map((x) => Math.round(x * 10) / 10),
        busy: busyOver(s.t, elapsed, clamp),
        census: census(),
      }, extra || {});

      post({
        __done: true, ok: true, ua: navigator.userAgent, rung: o.rung,
        engine_probe: {
          is_webkit_gtk_ua: /AppleWebKit/.test(navigator.userAgent) && /X11/.test(navigator.userAgent),
          vendor: navigator.vendor,
          has_chrome: typeof window.chrome !== "undefined",
          has_webkit_message_handlers:
            typeof (window.webkit && window.webkit.messageHandlers) !== "undefined",
          hardwareConcurrency: navigator.hardwareConcurrency, dpr: window.devicePixelRatio,
        },
        build: {
          scripts: Array.from(document.querySelectorAll("script[src]")).map((s) => s.getAttribute("src")),
          css: Array.from(document.querySelectorAll('link[rel="stylesheet"]')).map((s) => s.getAttribute("href")),
        },
        url: location.href,
        mount: { ms: Math.round(mountMs), by: mountedBy, census: censusMounted,
                 last_marker: o.lastMarker },
        clamp,
        first_token_ms: firstTokenMs, composer_fill_ms: Math.round(composerMs),
        send_attempts: attempts,
        still_running_at_deadline: stillRunning,
        scroll_detail: scrollDetail,
        marks: W.marks, actions,
        phases: [
          phase("idle", idleS, idleEl),
          phase("scroll", scrollS, scrollEl, { detail: scrollDetail }),
          phase("stream", streamS, streamEl, { first_token_ms: firstTokenMs }),
          phase("recover", recS, recEl),
        ],
        final: census(),
      });
    } catch (e) {
      post({ __done: true, ok: false, rung: o.rung, phase: W.phase,
             error: String((e && e.message) || e),
             error_stack: String((e && e.stack) || ""),
             send_attempts: W.sendAttempts || null,
             marks: W.marks, url: location.href,
             dom: { composer: !!W.dom.composer(), send: !!W.dom.sendButton(),
                    running: W.dom.isRunning(), elements: W.dom.elements(),
                    messages: W.dom.messages(),
                    bodyText: (document.body.innerText || "").slice(0, 1200) } });
    }
  };
})();
