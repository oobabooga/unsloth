// Page-side bootstrap for the Unsloth DESKTOP ladder.
//
// This file is BAKED INTO THE BUNDLE at build time, and it has to be, which is worth stating
// plainly rather than hiding. The Tauri app loads its frontend from `frontendDist` embedded in
// the binary and served at `tauri://localhost`, whose CSP is `default-src 'self'` with no
// `unsafe-eval`. So there is no external eval channel, no remote script and no `?thread=` in
// the URL. Everything the harness wants the page to do has to already be in the page, and the
// only live channel is `fetch` to loopback, which the app's own CSP permits
// (`connect-src ... http://127.0.0.1:*`).
//
// What it therefore does NOT do: it does not change any application code. It is appended to
// index.html alongside the untouched production bundle, it never patches an app module, and
// with no control server answering it does nothing at all beyond one failed fetch every 250 ms.
// The pristine binary is built and launched in the same job precisely so that "Desktop
// functions" is a claim about an unmodified app, and only the ladder numbers come from this one.
//
// The scene itself (amdv_scene.js) is included VERBATIM from the web UI ladder, and is loaded
// before this file. Byte-identical phases, byte-identical busy calibration and byte-identical
// census are the whole point: a Desktop number is only comparable to the web UI number if the
// thing being counted is the same thing.

(function () {
  "use strict";

  // Replaced at injection time. A literal, not a lookup, because there is nothing to look it
  // up in: this runs before any app state exists.
  var CONTROL = "__AMDV_CONTROL_URL__";
  var POLL_MS = 250;

  var log = [];
  var sentResult = false;
  var started = false;

  function post(path, obj) {
    try {
      return fetch(CONTROL + path, {
        method: "POST",
        // text/plain, NOT application/json, and the body is still JSON text. An
        // application/json POST is not a CORS-simple request, so WebKit preflights every
        // single one with an OPTIONS round trip. text/plain is simple, needs no preflight,
        // and halves the loopback traffic this instrument adds to the thing it is measuring.
        headers: { "Content-Type": "text/plain;charset=UTF-8" },
        body: JSON.stringify(obj),
      }).catch(function () {});
    } catch (e) {
      return Promise.resolve();
    }
  }

  function note(tag, v) {
    log.push([Date.now(), tag, v]);
    post("/amdv/event", { tag: tag, v: v });
  }

  // ---- every diagnostic the webview will give us -------------------------------------
  //
  // In the web UI leg these came out of WebKitGTK's `console-message-sent` signal and its
  // `resource-load-started`/`failed` pair, because the harness owned the WebView. Inside the
  // Tauri shell nothing is listening, and a silently 404ing asset is the classic way a page
  // looks fine and is not. So they are captured in-page instead and shipped over the same
  // channel. Wrapping console is a modification of the ENVIRONMENT, not of the app, and it is
  // installed for the pristine binary too by simply not being there.
  ["log", "info", "warn", "error", "debug"].forEach(function (level) {
    var orig = console[level];
    console[level] = function () {
      try {
        var parts = [];
        for (var i = 0; i < arguments.length; i++) {
          var a = arguments[i];
          parts.push(typeof a === "string" ? a : (function () {
            try { return JSON.stringify(a); } catch (e) { return String(a); }
          })());
        }
        post("/amdv/event", { console: level, text: parts.join(" ").slice(0, 4000) });
      } catch (e) {}
      return orig.apply(console, arguments);
    };
  });
  window.addEventListener("error", function (e) {
    post("/amdv/event", {
      page_error: String((e && e.message) || e),
      source: e && e.filename, line: e && e.lineno,
      stack: e && e.error && e.error.stack ? String(e.error.stack).slice(0, 4000) : null,
    });
  });
  window.addEventListener("unhandledrejection", function (e) {
    post("/amdv/event", {
      unhandled_rejection: String((e && e.reason && e.reason.message) || (e && e.reason) || e),
      stack: e && e.reason && e.reason.stack ? String(e.reason.stack).slice(0, 4000) : null,
    });
  });
  // Failed subresources. `error` on window does not carry these for every element type, and a
  // missing chunk is exactly the failure that produces a half-rendered page with clean numbers.
  window.addEventListener("error", function (e) {
    var t = e && e.target;
    if (t && t !== window && (t.src || t.href)) {
      post("/amdv/event", { resource_failed: t.src || t.href, tag: t.tagName });
    }
  }, true);

  // The scene posts through `window.webkit.messageHandlers.bench` and swallows the throw when
  // it is absent. Providing it here means the scene's own progress notes reach the harness
  // live, instead of only its final payload. The scene is not edited to do this.
  try {
    if (window.webkit && window.webkit.messageHandlers &&
        !window.webkit.messageHandlers.bench) {
      Object.defineProperty(window.webkit.messageHandlers, "bench", {
        value: { postMessage: function (s) { post("/amdv/event", { bench: s.slice(0, 8000) }); } },
        configurable: true,
      });
      note("bench_handler", "installed");
    }
  } catch (e) {
    note("bench_handler_failed", String(e));
  }

  // ---- navigation ---------------------------------------------------------------------
  //
  // The web UI leg simply loaded `/chat?thread=<id>`. Desktop has no such lever, so the thread
  // has to be opened from inside the page. Several routes exist and they are NOT equally
  // trustworthy, so each is tried in the order the config gives, each is followed by the same
  // post-condition check, and WHICH one worked is recorded in the payload. A run that reached
  // the thread by a different route than another run is not the same experiment, and the
  // report has to be able to say so.
  // The one-shot readiness event the app dispatches when the REQUESTED thread's history has
  // loaded (studio/frontend/src/features/chat/runtime-provider.tsx:1320-1334; it deliberately
  // does not fire for the empty thread a runtime bootstraps on first). Latched here because
  // it is one-shot per provider and we may navigate more than once.
  var shellReady = false;
  window.addEventListener("unsloth:app-shell-ready", function () {
    shellReady = true;
    note("app_shell_ready_event", Date.now());
  });

  function threadRendered(threadId, lastMarker) {
    // Not "the URL changed": the URL changing is the thing attempted, not the result. And not
    // "[data-role] exists" on its own either -- ChatRuntimeProvider has NO `key`
    // (chat-page.tsx:3950), so switching threads does not remount it and the PREVIOUS thread's
    // messages can still be on screen. The seeder's last-turn marker is the only thing that
    // distinguishes this thread's DOM from the last one's, so it is required, not optional.
    if (document.querySelectorAll("[data-role]").length === 0) return false;
    if (lastMarker) return (document.body.innerText || "").indexOf(lastMarker) !== -1;
    return true;
  }

  function navStrategies(threadId) {
    return {
      // Raw pushState. @tanstack/history's createBrowserHistory MONKEY-PATCHES
      // window.history.pushState and turns a third-party call into a router navigation, so
      // this is a real navigation and not a URL edit. `history.state` is passed through so the
      // router's own __TSR_key/__TSR_index bookkeeping survives.
      //
      // No synthetic popstate afterwards, deliberately: the popstate handler differences
      // __TSR_index between the two states, and an event with state:null makes that
      // computation throw or mis-step. pushState alone is both sufficient and correct.
      history: function () {
        window.history.pushState(window.history.state, "",
                                 "/chat?thread=" + encodeURIComponent(threadId));
        return true;
      },
      // The sidebar row, which is what a user clicks
      // (app-sidebar.tsx:2490-2515). Most faithful, and the most fragile: the sidebar can be
      // unpinned, in which case the row is not in the DOM at all.
      click: function () {
        var el = document.querySelector(
          '[data-testid="recent-thread"][data-thread-id="' + threadId + '"]');
        if (!el) return false;
        el.click();
        return true;
      },
      // A full document load. Tauri's asset protocol falls back to index.html for an
      // unresolved path, so this works, and it is LAST for a reason: it re-runs the whole
      // startup path, and public/reload-snapshot.js then serializes and repaints a copy of the
      // previous shell over the top of the mount being measured.
      assign: function () {
        window.location.assign("/chat?thread=" + encodeURIComponent(threadId));
        return true;
      },
    };
  }

  function waitFor(pred, timeoutMs, label) {
    return new Promise(function (resolve) {
      var t0 = Date.now();
      (function tick() {
        var ok = false;
        try { ok = pred(); } catch (e) {}
        if (ok) return resolve({ ok: true, ms: Date.now() - t0, label: label });
        if (Date.now() - t0 > timeoutMs) return resolve({ ok: false, ms: Date.now() - t0, label: label });
        setTimeout(tick, 100);
      })();
    });
  }

  async function openThread(threadId, lastMarker, order, perStrategyMs) {
    var S = navStrategies(threadId);
    var tried = [];
    for (var i = 0; i < order.length; i++) {
      var name = order[i];
      if (!S[name]) { tried.push({ name: name, ran: false, reason: "unknown strategy" }); continue; }
      var ran = false, err = null;
      try { ran = S[name](); } catch (e) { err = String(e); }
      note("nav_attempt", { name: name, ran: ran, err: err });
      if (!ran) { tried.push({ name: name, ran: false, err: err }); continue; }
      var r = await waitFor(function () { return threadRendered(threadId, lastMarker); },
                            perStrategyMs, name);
      tried.push({ name: name, ran: true, rendered: r.ok, ms: r.ms, err: err,
                   shell_ready_event: shellReady });
      if (r.ok) return { ok: true, via: name, tried: tried };
    }
    return { ok: false, via: null, tried: tried };
  }

  // ---- the run ------------------------------------------------------------------------
  async function begin(cfg) {
    if (started) return;
    started = true;
    note("config", cfg);

    // The APP SHELL, not the startup screen. provider.tsx:739 renders StartupScreen until the
    // backend is running AND desktop auth has completed, and a scene that started measuring on
    // the startup screen would report an empty page at every rung, flat and meaningless.
    // Detected POSITIVELY: StartupScreen carries no data-testid or data-slot of its own, so
    // "not the startup screen" is not checkable, whereas the shell's sidebar wrapper
    // (components/ui/sidebar.tsx:189, rendered by RootLayout) is.
    var shell = await waitFor(function () {
      return !!document.querySelector('[data-slot="sidebar-wrapper"]');
    }, cfg.shellTimeoutMs || 300000, "app_shell");
    note("app_shell", shell);

    // The updater fires an HTTPS check 5 s after mount (hooks/use-tauri-update.ts:299-308) and
    // renders a banner. Left alone it lands inside the idle baseline window and shows up as
    // main-thread work that has nothing to do with thread size. Waiting it out is cheaper and
    // more honest than blocking it, and the wait is identical at every rung so it cannot
    // create a rung-dependent difference.
    if (cfg.settleMs) {
      note("settle_for_updater", cfg.settleMs);
      await new Promise(function (r) { setTimeout(r, cfg.settleMs); });
    }

    var nav = { ok: true, via: "none", tried: [] };
    if (cfg.threadId) {
      nav = await openThread(cfg.threadId, cfg.lastMarker,
                             cfg.navOrder || ["history", "click", "assign"],
                             cfg.navPerStrategyMs || 60000);
      note("nav", nav);
      if (!nav.ok) {
        post("/amdv/result", { __done: true, ok: false,
                               error: "could not open the seeded thread", nav: nav });
        sentResult = true;
        return;
      }
    }

    if (cfg.hogMs) {
      // The jammed positive control, installed exactly as the web UI leg installs it: a
      // synchronous spin on the main thread. Without a control that a healthy channel must
      // fall on, a flat frame rate cannot be told from a channel that cannot read anything
      // other than the display rate -- which is precisely how the web UI's first
      // presented-frame table came to be an artefact.
      setInterval(function () {
        var t = performance.now();
        while (performance.now() - t < cfg.hogMs) { /* spin */ }
      }, cfg.hogPeriodMs || 250);
      note("hog_installed", { ms: cfg.hogMs, period: cfg.hogPeriodMs || 250 });
    }

    if (!window.__av || typeof window.__av.run !== "function") {
      post("/amdv/result", { __done: true, ok: false, error: "scene not present", nav: nav });
      sentResult = true;
      return;
    }
    // Geometry, recorded because the shell entry RESIZES the native window: it is declared
    // 760x560 and not resizable (tauri.conf.json), and provider.tsx:257-372 makes it resizable,
    // restores a saved size and shows it as the shell mounts. Two rungs measured at different
    // window sizes are not the same experiment, so the size is reported rather than assumed
    // constant.
    window.__av.__desktop = {
      nav: nav, app_shell: shell, control: CONTROL, shell_ready_event: shellReady,
      viewport: { w: window.innerWidth, h: window.innerHeight, dpr: window.devicePixelRatio },
      href: location.href, ua: navigator.userAgent,
    };
    try {
      window.__av.run(cfg.runArgs || {});
    } catch (e) {
      post("/amdv/result", { __done: true, ok: false, error: "run threw: " + String(e), nav: nav });
      sentResult = true;
      return;
    }

    // The scene signals completion by setting W.result. Polling it is deliberate: the scene is
    // used unmodified, and its internal `post` is a closure that cannot be wrapped from here.
    (function drain() {
      if (sentResult) return;
      var r = window.__av && window.__av.result;
      if (r) {
        sentResult = true;
        r.__desktop = window.__av.__desktop;
        post("/amdv/result", r);
        return;
      }
      setTimeout(drain, 500);
    })();
  }

  // Poll for a config. The harness only publishes one once the backend is up, the rung is
  // seeded and the thread id is known, so this doubles as the readiness barrier.
  (function poll() {
    fetch(CONTROL + "/amdv/config", { cache: "no-store" })
      .then(function (r) { return r.json(); })
      .then(function (cfg) {
        if (cfg && cfg.ready) return begin(cfg);
        setTimeout(poll, POLL_MS);
      })
      .catch(function () { setTimeout(poll, POLL_MS); });
  })();

  post("/amdv/event", { boot: "installed", href: location.href, ts: Date.now() });
})();
