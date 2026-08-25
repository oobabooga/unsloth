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

  const census = () => ({
    elements: W.dom.elements(), messages: W.dom.messages(),
    assistant_chars: W.dom.assistantChars(),
    reasoning_roots: W.dom.reasoningRoots(), reasoning_open: W.dom.reasoningOpen(),
    code_blocks: W.dom.codeBlocks(), highlight_spans: W.dom.highlightSpans(),
    scroll_height: (document.scrollingElement || document.body).scrollHeight,
  });
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
      const fill = async (text) => {
        await focusComposer();
        setter.call(ta, text);
        ta.dispatchEvent(new Event("input", { bubbles: true }));
        await sleep(250);
      };
      const composerT0 = performance.now();
      await fill(o.prompt);
      const composerMs = performance.now() - composerT0;
      const send = await waitFor(W.dom.sendButton, 20000, "send button");
      const sendT = performance.now();
      let started = false;
      const attempts = [];
      for (let attempt = 0; attempt < 6 && !started; attempt++) {
        if (attempt > 0) { await fill(o.prompt); await sleep(1000 * attempt); }
        const btn = W.dom.sendButton() || send;
        attempts.push({ attempt, has_button: !!btn,
                        disabled: btn ? (btn.disabled || btn.getAttribute("aria-disabled")) : null,
                        composer_value_len: (ta.value || "").length });
        if (btn) btn.click();
        const dl = performance.now() + 45000;
        while (performance.now() < dl) {
          if (W.dom.isRunning()) { started = true; break; }
          await sleep(50);
        }
      }
      W.sendAttempts = attempts;
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
        marks: W.marks,
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
             error: String((e && e.stack) || e), marks: W.marks, url: location.href,
             dom: { composer: !!W.dom.composer(), send: !!W.dom.sendButton(),
                    running: W.dom.isRunning(), elements: W.dom.elements(),
                    messages: W.dom.messages(),
                    bodyText: (document.body.innerText || "").slice(0, 1200) } });
    }
  };
})();
