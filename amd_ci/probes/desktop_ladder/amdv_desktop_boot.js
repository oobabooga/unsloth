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
// The scene itself (chr_scene.js) is included VERBATIM and is loaded before this file.
// Byte-identical phases, busy calibration and census are the whole point: a Desktop number is
// only comparable to a web UI number if the thing being counted is the same thing.
//
// It is the REPAIRED scene rather than amdv_scene.js, and the difference is confined to the
// scroll gesture. The original assigns `el.scrollTop` while the thread viewport carries
// `scroll-smooth`, so each assignment animates and the next is computed from a position that
// has barely moved; measured in Chromium at r500K it commanded 54,000 px and travelled 6,610,
// never getting further than 1,107 px from the bottom of a 316,829 px thread. That is a jiggle
// reported as a traversal, and `scroll_detail` cannot reveal it because it records only the
// first and last position and both are the bottom either way. The repaired gesture scrolls
// instantly, dispatches a real WheelEvent so the app's autoscroll sees a user scroll, takes one
// step per painted frame, and RECORDS COMMANDED AND TRAVELLED PIXELS so the question is
// answered by the payload instead of assumed. Consequence for comparability, stated rather than
// buried: the idle and stream phases remain directly comparable to the web UI ladder, and the
// scroll phase is comparable only with the gesture named.

(function () {
  "use strict";

  // Replaced at injection time. A literal, not a lookup, because there is nothing to look it
  // up in: this runs before any app state exists.
  var CONTROL = "__AMDV_CONTROL_URL__";
  var POLL_MS = 250;

  var log = [];
  var sentResult = false;
  var started = false;

  // ── THE FENCE-DEFERRAL FLAG, READ SYNCHRONOUSLY, BEFORE ANY APP CODE RUNS ────────────────
  //
  // `fenceMode()` reads `globalThis.__UNSLOTH_DEFER_FENCE_HIGHLIGHT__`
  // (components/assistant-ui/code-fence-defer.tsx:78) and `resolveFenceMode` maps the BOOLEAN
  // `false` to "off" while ABSENT takes SHIP_DEFAULT, which is "defer"
  // (code-fence-mode.ts:26-56). So the ablation arm sets the boolean and the default arm sets
  // nothing at all; an unrecognised string would degrade to "off" and quietly measure the wrong
  // thing, which is why the value is written as a real boolean and never as a string.
  //
  // It cannot come from the control server: that is a fetch, and by the time it resolves the
  // app has already booted and read the flag. It comes from localStorage, which is readable
  // synchronously here in a classic script during parse, i.e. before the deferred module entry.
  // The harness seeds it and then reloads once, which is the same reload the provider seeding
  // already requires, so the ablation costs no extra page load.
  try {
    var fenceFlag = window.localStorage.getItem("__amdv_defer_fence");
    if (fenceFlag === "off") {
      window.__UNSLOTH_DEFER_FENCE_HIGHLIGHT__ = false;
    } else if (fenceFlag === "on") {
      window.__UNSLOTH_DEFER_FENCE_HIGHLIGHT__ = true;
    }
    window.__amdvFenceArm = fenceFlag || "default";
  } catch (e) {
    window.__amdvFenceArm = "unreadable";
  }

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

  function onRequestedThread(threadId) {
    try {
      return new URLSearchParams(window.location.search).get("thread") === threadId;
    } catch (e) {
      return (window.location.search || "").indexOf(encodeURIComponent(threadId)) !== -1;
    }
  }

  function threadRendered(threadId, lastMarker, expectMessages) {
    // ── THE EMPTY THREAD IS A REAL RUNG, NOT A DEGENERATE ONE ────────────────────────────
    //
    // This cost run 32808701910 every one of its 0K legs, all four of them, and with them both
    // of its controls. The post-condition below used to require at least one `[data-role]`
    // node unconditionally. A 0K thread is seeded with ZERO messages by construction (the
    // probe's SYNTHETIC_RUNGS path calls `seeder.create_thread` and stops), so it has no
    // `[data-role]` node and never will. Every strategy therefore "failed", the last of them
    // was `assign`, which reloads the document, which re-ran this whole file, which navigated
    // and failed again -- a ~130 s cycle that repeated until the rung timed out at 1,290 s with
    // "no result from the page". The page was fine; the post-condition was unsatisfiable.
    //
    // 0K is the 61 fps reference the entire collapse is measured against, so the fix is to make
    // an empty thread REPORTABLE rather than to delete the rung. What is checked instead is the
    // strongest thing that is true of an empty thread and false of a failed navigation:
    // the router really is on THIS thread id, the shell is up, the composer is live, and the
    // message list is empty -- which also rules out the specific failure the old comment was
    // guarding against, namely the PREVIOUS thread's messages still being on screen because
    // ChatRuntimeProvider has no `key` (chat-page.tsx:3950).
    if (typeof expectMessages === "number" && expectMessages === 0) {
      return onRequestedThread(threadId) &&
             !!document.querySelector('[data-slot="sidebar-wrapper"]') &&
             !!document.querySelector('textarea[aria-label="Message input"]') &&
             document.querySelectorAll("[data-role]").length === 0;
    }
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

  // `assign` is a FULL DOCUMENT LOAD, so it re-runs this file from the top. Without a latch
  // that survives the load, a navigation that cannot succeed becomes an infinite loop: try,
  // fail, assign, reload, try, fail, assign... at roughly 130 s a cycle, until the rung's own
  // timeout fires and reports "no result from the page" with no indication that anything was
  // retried. Run 32808701910 spent 4 x 1,290 s in exactly that loop. The latch is keyed on the
  // config stamp so it is per-run rather than per-profile, and the second pass reports a real
  // failure instead of reloading again.
  function assignAlreadyUsed(stamp) {
    try { return window.sessionStorage.getItem("__amdv_nav_assign") === String(stamp); }
    catch (e) { return false; }
  }
  function markAssignUsed(stamp) {
    try { window.sessionStorage.setItem("__amdv_nav_assign", String(stamp)); } catch (e) {}
  }

  async function openThread(threadId, lastMarker, order, perStrategyMs, expectMessages, stamp) {
    var S = navStrategies(threadId);
    var tried = [];
    for (var i = 0; i < order.length; i++) {
      var name = order[i];
      if (!S[name]) { tried.push({ name: name, ran: false, reason: "unknown strategy" }); continue; }
      if (name === "assign" && assignAlreadyUsed(stamp)) {
        tried.push({ name: name, ran: false,
                     reason: "a full reload was already spent on this run; refusing to loop" });
        note("nav_attempt", { name: name, ran: false, err: "assign already used this run" });
        continue;
      }
      var ran = false, err = null;
      if (name === "assign") markAssignUsed(stamp);
      // Cleared per attempt. The latch is set by the app's FIRST app-shell-ready, which fires
      // for the shell's own startup navigation long before ours, so an un-cleared latch made
      // the bounded wait below return in 0 ms and wait for nothing.
      shellReady = false;
      try { ran = S[name](); } catch (e) { err = String(e); }
      note("nav_attempt", { name: name, ran: ran, err: err });
      if (!ran) { tried.push({ name: name, ran: false, err: err }); continue; }
      var r = await waitFor(function () {
        return threadRendered(threadId, lastMarker, expectMessages);
      }, perStrategyMs, name);
      tried.push({ name: name, ran: true, rendered: r.ok, ms: r.ms, err: err,
                   shell_ready_event: shellReady });
      if (r.ok) {
        if (expectMessages === 0) {
          // An empty thread has no DOM of its own to prove the router really loaded it, so the
          // app's own one-shot readiness event (runtime-provider.tsx:1320-1334, dispatched when
          // the REQUESTED thread's history has loaded) is given a bounded chance to arrive and
          // is RECORDED either way. It is deliberately not required: the rung is the 61 fps
          // reference the whole collapse is measured against, and making it depend on an event
          // that has never been observed for an empty-but-real thread would be trading one
          // unsatisfiable post-condition for another.
          var sr = await waitFor(function () { return shellReady; }, 15000, "app_shell_ready");
          tried[tried.length - 1].shell_ready_waited_ms = sr.ms;
          tried[tried.length - 1].shell_ready_event = shellReady;
        }
        return { ok: true, via: name, tried: tried };
      }
    }
    return { ok: false, via: null, tried: tried };
  }

  // ---- the run ------------------------------------------------------------------------
  // localStorage the app reads on its FIRST paint: the pacer provider, its key, and the
  // last-selected checkpoint. The web UI ladder seeded these through a document-start user
  // script, which Desktop has no equivalent of, and they cannot simply be written late: the
  // model chip restores from storage asynchronously during boot, and a run whose composer never
  // gets an enabled send button measures a thread it never streamed into. So they are written
  // and the document is RELOADED once, before anything is measured and before the settle
  // window, with a stamp so the second boot does not loop.
  //
  // The auth tokens the web harness seeds are deliberately NOT written here. Under Tauri
  // `requireAuth` is a no-op and the app mints its own pair through desktop-login; overwriting
  // them would replace a live session with a stale one.
  function applyLocalStorage(cfg) {
    if (!cfg.localStorage) return false;
    var stamp = String(cfg.lsStamp || "");
    try {
      if (window.localStorage.getItem("__amdv_ls") === stamp) return false;
    } catch (e) { return false; }
    try {
      Object.keys(cfg.localStorage).forEach(function (k) {
        window.localStorage.setItem(k, cfg.localStorage[k]);
      });
      window.localStorage.setItem("__amdv_ls", stamp);
    } catch (e) {
      note("local_storage_failed", String(e));
      return false;
    }
    note("local_storage_seeded", Object.keys(cfg.localStorage));
    return true;
  }

  async function begin(cfg) {
    if (started) return;
    started = true;
    note("config", cfg);

    if (applyLocalStorage(cfg)) {
      note("reloading_once_after_seed", location.href);
      // A short delay so the note reaches the harness before the document goes away.
      setTimeout(function () { window.location.reload(); }, 400);
      return;
    }

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
                             cfg.navPerStrategyMs || 60000,
                             // The seeder's OWN count, straight from the backend it wrote to.
                             // 0 is a value here and not a missing field, so it is passed as a
                             // number and read as one.
                             typeof cfg.expectMessages === "number" ? cfg.expectMessages : null,
                             cfg.lsStamp || "");
      note("nav", nav);
      if (!nav.ok) {
        post("/amdv/result", { __done: true, ok: false,
                               error: "could not open the seeded thread", nav: nav,
                               expect_messages: cfg.expectMessages,
                               dom: { data_role: document.querySelectorAll("[data-role]").length,
                                      elements: document.getElementsByTagName("*").length,
                                      href: location.href } });
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
      // What the SEEDER put in the thread, carried through to the payload so the criteria can
      // check the mounted DOM against it rather than against a growth ratio.
      expect_messages: typeof cfg.expectMessages === "number" ? cfg.expectMessages : null,
      on_requested_thread: onRequestedThread(cfg.threadId || ""),
      // The arm actually in force, read back from the global rather than from the config that
      // asked for it. An arm that failed to apply is the one failure that would make an
      // ablation table read as "the flag does nothing".
      fence_arm: window.__amdvFenceArm,
      fence_global: typeof window.__UNSLOTH_DEFER_FENCE_HIGHLIGHT__,
      fence_value: window.__UNSLOTH_DEFER_FENCE_HIGHLIGHT__ === undefined
        ? null : window.__UNSLOTH_DEFER_FENCE_HIGHLIGHT__,
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
