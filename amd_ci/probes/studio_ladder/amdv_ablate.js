// ABLATION scene: attribute the 500K scroll collapse to a mechanism, by removing candidate
// costs one at a time from a real seeded thread and re-running the identical gesture.
//
// Why ablation and not a trace. A 153.0 ms traced handler in this campaign measured ~26 ms when
// timed directly with tracing off, and a fix aimed at a candidate owning 21.4% of a traced
// commit measured completely flat. Traces nominate; removal is what convicts.
//
// The measured target, corpus 23cd2464, real WebKitGTK on gfx1151:
//   scroll at 500K = 2.4 fps at 94% busy, worst frame 3,092 ms, over 122,222 DOM elements,
//   while IDLE at the same rung is 61 fps at 9% busy. So the cost is in the scroll path, not in
//   holding the DOM, and any hypothesis predicting idle degradation is already refuted.
//
// Every arm runs the SAME gesture on the SAME page in one session, so mount variance cannot be
// mistaken for an arm effect. Order matters: reversible arms first, then a baseline repeat to
// prove the page has not drifted, then the destructive ones.
//
// TWO CONTROLS, because an ablation harness that cannot fail is worth as little as a frame
// clock that cannot fall:
//   * POSITIVE (detach): remove nearly every message. Frame rate MUST recover. If it does not,
//     the harness cannot detect a win and no other arm here means anything.
//   * NEGATIVE (noop_touch): walk the same nodes, add a class that styles nothing, change no
//     geometry. Frame rate must NOT recover. If it does, the arms are measuring the act of
//     mutating the DOM rather than what they removed.
(() => {
  if (window.__ab) return;
  const W = { gaps: [], ticks: [], marks: [], arms: [] };
  window.__ab = W;

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

  const q = (s) => document.querySelector(s);
  const post = (o) => {
    if (o && o.__done) W.result = o;
    try { window.webkit.messageHandlers.bench.postMessage(JSON.stringify(o)); } catch (e) {}
  };
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const waitFor = async (fn, ms, label) => {
    const dl = performance.now() + ms;
    while (performance.now() < dl) {
      let v; try { v = fn(); } catch (e) { v = null; }
      if (v) return v;
      await sleep(100);
    }
    throw new Error("timeout waiting for " + label);
  };

  const census = () => ({
    elements: document.getElementsByTagName("*").length,
    messages: document.querySelectorAll("[data-role]").length,
    reasoning_roots: document.querySelectorAll('[data-slot="reasoning-root"]').length,
    code_blocks: document.querySelectorAll("pre").length,
    highlight_spans: document.querySelectorAll("pre span").length,
    all_spans: document.getElementsByTagName("span").length,
    scroll_height: (document.scrollingElement || document.body).scrollHeight,
  });

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
      return { busy_pct: null, busy_pct_reason: clamp.reason || "no elapsed time" };
    }
    let b = 0;
    for (const g of t) { const x = g - clamp.clamp_ms; if (x > 0) b += x; }
    return { busy_pct: Math.round((b / el) * 1000) / 10, blocked_ms: Math.round(b) };
  };

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

  const cut = { g: 0, t: 0 };
  const slice = () => {
    const g = W.gaps.slice(cut.g), t = W.ticks.slice(cut.t);
    cut.g = W.gaps.length; cut.t = W.ticks.length;
    return { g, t };
  };

  // THE GESTURE. Identical in every arm: same distance, same step count, same pauses. If an arm
  // changed the gesture it would be measuring a different film, not a different page.
  const doScroll = async () => {
    const el = scrollerAt(window.innerWidth / 2, window.innerHeight * 0.5);
    if (!el) return "no scroller";
    const h0 = el.scrollHeight;
    for (let pass = 0; pass < 2; pass++) {
      for (let i = 0; i < 10; i++) { el.scrollTop = Math.max(0, el.scrollTop - 900); await sleep(80); }
      await sleep(250);
      for (let i = 0; i < 10; i++) { el.scrollTop = el.scrollTop + 900; await sleep(80); }
      await sleep(250);
    }
    el.scrollTop = el.scrollHeight;
    return `${el.tagName} h=${h0}`;
  };

  const style = (id, css) => {
    let s = document.getElementById(id);
    if (!s) { s = document.createElement("style"); s.id = id; document.head.appendChild(s); }
    s.textContent = css;
  };
  const unstyle = (id) => { const s = document.getElementById(id); if (s) s.remove(); };

  // ── the arms ──────────────────────────────────────────────────────────────────────────────
  // Each: {name, why, apply, revert (null = destructive), destructive}
  const ARMS = [
    { name: "baseline", why: "the measured collapse, unmodified",
      apply: async () => "none", revert: async () => {} },

    { name: "noop_touch",
      why: "NEGATIVE CONTROL. Touch every message node and add a class that styles nothing. "
         + "Same traversal, same mutation count, no removed work. Must NOT recover.",
      apply: async () => {
        const n = document.querySelectorAll("[data-role]");
        style("ab-noop", ".ab-noop-marker{}");
        n.forEach((e) => e.classList.add("ab-noop-marker"));
        return `touched ${n.length} messages`;
      },
      revert: async () => {
        document.querySelectorAll(".ab-noop-marker").forEach((e) =>
          e.classList.remove("ab-noop-marker"));
        unstyle("ab-noop");
      } },

    { name: "pointer_events_none",
      why: "task #56 measured one pointerover per scroll step costing 63 ms of React commit. "
         + "Disabling hit testing removes that path without changing what is rendered.",
      apply: async () => {
        style("ab-pe", "[data-role],[data-role] *{pointer-events:none !important}");
        return "pointer-events:none on every message subtree";
      },
      revert: async () => unstyle("ab-pe") },

    { name: "content_visibility",
      why: "task #129 measured a traversal regression whose crossover is near 500K: style and "
         + "layout walking the standing DOM. content-visibility:auto lets the engine skip "
         + "style, layout and paint for offscreen subtrees while leaving them mounted, so a "
         + "recovery here indicts per-frame traversal of offscreen content specifically.",
      apply: async () => {
        style("ab-cv", "[data-role]{content-visibility:auto;contain-intrinsic-size:auto 800px}");
        return "content-visibility:auto on every message";
      },
      revert: async () => unstyle("ab-cv") },

    { name: "contain_layout_paint",
      why: "Weaker than content-visibility and it does not skip the subtree: it only stops "
         + "layout and paint escaping it. Separates 'the engine crossed a containment boundary "
         + "it did not need to' from 'the offscreen work itself is the cost'.",
      apply: async () => {
        style("ab-ct", "[data-role]{contain:layout paint style}");
        return "contain:layout paint style on every message";
      },
      revert: async () => unstyle("ab-ct") },

    { name: "baseline_repeat",
      why: "DRIFT CHECK. The same arm as the first one, after every reversible arm has been "
         + "applied and reverted. If this does not match the first baseline, the arms above are "
         + "reporting page drift and not their own effect.",
      apply: async () => "none", revert: async () => {} },

    { name: "strip_highlight_spans",
      why: "#9567 fence deferral cut highlight spans 37,430 -> 2,341 at r100K; whether it is "
         + "doing anything at 500K is open. Replacing each pre's markup with its own text "
         + "removes every syntax span while keeping the characters and roughly the box.",
      destructive: true,
      apply: async () => {
        const pres = Array.from(document.querySelectorAll("pre"));
        let before = document.querySelectorAll("pre span").length;
        pres.forEach((e) => { e.textContent = e.textContent; });
        return `${pres.length} pre blocks, ${before} highlight spans -> `
             + `${document.querySelectorAll("pre span").length}`;
      }, revert: null },

    { name: "detach_messages",
      why: "POSITIVE CONTROL. Remove all but the last two messages. The frame rate MUST recover "
         + "to near the idle rate. If it does not, this harness cannot detect a win and every "
         + "other arm above is void.",
      destructive: true,
      apply: async () => {
        const n = Array.from(document.querySelectorAll("[data-role]"));
        const keep = new Set(n.slice(-2));
        let removed = 0;
        n.forEach((e) => { if (!keep.has(e)) { e.remove(); removed++; } });
        return `removed ${removed} of ${n.length} messages`;
      }, revert: null },
  ];

  W.run = async (opts) => {
    const o = Object.assign({ idleMs: 6000, lastMarker: null, mountTimeoutMs: 420000,
                              rung: "?", only: null }, opts || {});
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
      slice();

      // Calibrate on a dedicated idle window, never on a window that is also reported.
      W.marks.push({ name: "idle:calibrate", wall_ms: Date.now() });
      const cT0 = performance.now();
      await sleep(o.idleMs);
      clamp = calibrate(slice().t);
      const idleCalMs = performance.now() - cT0;

      W.marks.push({ name: "idle", wall_ms: Date.now() });
      const iT0 = performance.now();
      await sleep(o.idleMs);
      const iS = slice(), idleMs = performance.now() - iT0;
      const idle = { elapsed_ms: Math.round(idleMs), raf: summarise(iS.g),
                     eff_fps: Math.round((1000 * iS.g.length / idleMs) * 10) / 10,
                     busy: busyOver(iS.t, idleMs) };

      const chosen = ARMS.filter((a) => !o.only || o.only.indexOf(a.name) >= 0);
      for (const arm of chosen) {
        const before = census();
        let detail = "", ok = true;
        try { detail = await arm.apply(); } catch (e) { ok = false; detail = String(e); }
        // Let the mutation settle so its own cost is not billed to the gesture.
        await sleep(1200);
        const applied = census();
        slice();

        W.marks.push({ name: "arm:" + arm.name, wall_ms: Date.now() });
        const aT0 = performance.now();
        let gest = "";
        try { gest = await doScroll(); } catch (e) { ok = false; gest = String(e); }
        const el = performance.now() - aT0;
        const s = slice();

        W.arms.push({
          name: arm.name, why: arm.why, ok, apply_detail: detail, gesture: gest,
          destructive: Boolean(arm.destructive),
          elapsed_ms: Math.round(el),
          eff_fps: Math.round((1000 * s.g.length / el) * 10) / 10,
          frames: s.g.length, raf: summarise(s.g),
          raf_gaps_ms: s.g.map((x) => Math.round(x * 10) / 10),
          busy: busyOver(s.t, el),
          census_before: before, census_applied: applied, census_after: census(),
          wall_start_ms: Date.now() - Math.round(el), wall_end_ms: Date.now(),
        });

        if (arm.revert) {
          try { await arm.revert(); } catch (e) {}
          await sleep(800);
          // Back to the top, so the next arm starts where this one did.
          const sc = scrollerAt(window.innerWidth / 2, window.innerHeight * 0.5);
          if (sc) sc.scrollTop = sc.scrollHeight;
          await sleep(500);
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
             url: location.href, mount: { ms: Math.round(mountMs), census: census() },
             clamp, idle, idle_calibrate_ms: Math.round(idleCalMs),
             marks: W.marks, arms: W.arms, final: census() });
    } catch (e) {
      post({ __done: true, ok: false, rung: o.rung,
             error: String((e && e.message) || e), marks: W.marks, arms: W.arms,
             url: location.href,
             dom: { elements: document.getElementsByTagName("*").length,
                    messages: document.querySelectorAll("[data-role]").length,
                    bodyText: (document.body.innerText || "").slice(0, 800) } });
    }
  };
})();
