#!/usr/bin/env python3
"""One studiobench LADDER RUNG, measured in REAL WebKitGTK.

The gap this fills. scripts/wk9477_bench.py drives real WebKitGTK but streams a long reply
into a FRESH thread: it varies reply length and has no thread-size knob at all. studiobench
varies thread size but drives Playwright, whose `webkit` on Linux its own source calls
"A PROXY FOR WebKitGTK ... not the GTK one Studio runs in on Linux". The user's report is
about thread size, in the GTK engine. Neither harness alone can express it.

So this one seeds a rung with studiobench's own Seeder and frozen corpus, then drives the
seeded thread through scripts/wkgtk_drive.py, which is the real libwebkit2gtk-4.1.

The streamed tail is the DEFAULT ladder tail (~6,000 chars at every rung). That is a known
defect of the ladder and it is deliberately not worked around here: --stream-tail-chars
silently under-delivers on the current main, so the only trustworthy axis today is the rung.
Vary one thing.

  python3 scripts/amdv_rung_bench.py --rung 100K --dist <dist> --home <home> \
      --port 5411 --display :77 --out outputs/amdv/r100K_rep1.json
"""
import argparse, hashlib, json, os, shutil, signal, subprocess, sys, time
from pathlib import Path

WS = Path(os.environ.get("UNSLOTH_WORKSPACE", "/mnt/disks/unslothai/daniel3/workspace_37"))

# The rung the user names as "0K" is not in studiobench's RUNGS, which start at 1K. It is an
# EMPTY thread, so it is constructed here rather than by plan_rung, and it is labelled
# synthetic in run_meta so nobody later compares it to a real ladder entry as though the
# corpus had produced it.
SYNTHETIC_RUNGS = {"0K"}

# The scene's per-wait budgets, here rather than inline, because the driver's own deadline is
# computed FROM them. One edit has to move both or the driver kills a session that was still
# inside the time it was given.
COMPOSER_MS = 180_000
MOUNT_MS = 420_000
SEND_MS = 240_000


def bundle_hash(dist: Path) -> str:
    h = hashlib.sha256()
    n = 0
    for f in sorted((dist / "assets").rglob("*")):
        if f.is_file() and f.suffix in (".js", ".css"):
            h.update(f.name.encode())
            h.update(f.read_bytes())
            n += 1
    idx = dist / "index.html"
    if idx.exists():
        h.update(idx.read_bytes())
    return f"{h.hexdigest()[:16]}({n} files)"


def wait_health(base, timeout=300):
    import urllib.request
    dl = time.time() + timeout
    while time.time() < dl:
        try:
            with urllib.request.urlopen(f"{base}/healthz", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", required=True)
    ap.add_argument("--rep", default="1")
    # WHICH SHIPPED BUILD THIS SESSION IS. It is a label only: the arm is decided entirely by
    # which `--dist` and which Studio install the caller hands over, and the scene reads the
    # arm's real state back out of the running page rather than trusting this string.
    ap.add_argument("--arm", default="?")
    ap.add_argument("--dist", required=True)
    ap.add_argument("--home", required=True, help="UNSLOTH_STUDIO_HOME, distinct per run")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--display", default=":77")
    ap.add_argument("--sb-root", default="", help="checkout that provides tests.studio.studiobench")
    ap.add_argument("--unsloth-bin", default="", help="the `unsloth` CLI to launch Studio with")
    ap.add_argument("--python-gi", default="/usr/bin/python3.12",
                    help="a python that can import gi; the venv python usually cannot")
    ap.add_argument("--idle-ms", type=int, default=6000)
    ap.add_argument("--recover-ms", type=int, default=6000)
    ap.add_argument("--accel", default="always")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-studio", action="store_true")
    # Explicit, because on the AMD CI runner this harness runs out of amd_ci/ and not out
    # of the workspace `scripts/` directory these default to.
    ap.add_argument("--scene", default="", help="path to amdv_scene.js")
    ap.add_argument("--entry", default="window.__av.run",
                    help="the page-side entry point the scene exposes")
    ap.add_argument("--driver", default="", help="path to wkgtk_drive.py")
    # A DELIBERATELY JAMMED ARM, and the only thing that makes a flat frame rate mean anything.
    # GdkFrameClock ticks under `begin_updating()` whether or not the page had anything new to
    # show, so "60 fps at every rung" is consistent with BOTH a compositor keeping up and a
    # channel that cannot read anything else. Blocking the main thread on purpose separates
    # those: a channel that resolves must read far below 60 here.
    ap.add_argument("--hog-ms", type=int, default=200)
    ap.add_argument("--hog-period-ms", type=int, default=250)
    ap.add_argument("--frame-clock", default="passive", choices=["updating", "passive"])
    # A CONTENT hash of the pinned instrument, supplied by the probe that cloned it once. The
    # resolved `__file__` already proves the import landed outside the arm's own tree; this
    # proves the tree behind that path was the same one for the first arm and the ninth. A path
    # assertion cannot see a directory that changed under it mid-run.
    ap.add_argument("--instrument-hash", default="")
    ap.add_argument("--skip-send", action="store_true",
                    help="stop after the action phases, without streaming a reply. Nothing in "
                         "the two-arm ladder passes this: the jam is a WINDOW inside every "
                         "session now rather than a session of its own, so there is no arm that "
                         "wants the film cut short. Kept because a scene-only rehearsal is "
                         "cheaper without a stream.")
    ap.add_argument("--studio-verbose", action="store_true", default=True)
    args = ap.parse_args()

    def P(v, default_parent=WS):
        return Path(v) if v.startswith("/") else (default_parent / v)

    dist, home, outp = P(args.dist), P(args.home), P(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    home.mkdir(parents=True, exist_ok=True)
    work = WS / "temp" / "amdv"
    work.mkdir(parents=True, exist_ok=True)
    # THE ARM IS PART OF THE TAG. It was not, and two arms at the same rung then wrote one
    # console log and one init script under the shared work directory. The result file is
    # unlinked before each run so a stale payload could not be read as this run's, but the
    # forensics were being overwritten, which is the half you need exactly when a session
    # fails.
    tag = f"{args.arm}_r{args.rung}_rep{args.rep}"

    sb_root = P(args.sb_root) if args.sb_root else (WS / "wt_mainchk")
    # THE INSTRUMENT MUST NOT CO-VARY WITH THE SUBJECT.
    #
    # This used to be handed the arm's OWN checkout, so a five-arm comparison would have measured
    # each arm with the pacer, seeder, corpus and scene that shipped alongside it. Two arms four
    # days apart would then differ by the change under test AND by the measuring device, and the
    # difference would be reported as the change. It did not silently happen only because an
    # August checkout has no `tests/studio/studiobench` at all and the import raised.
    #
    # `--sb-root` is now ONE pinned tree for every arm, chosen by the caller and never the arm's
    # own repo. `sys.path.insert(0, ...)` puts it ahead of anything the arm could shadow it with,
    # and the resolved file of the module that was actually imported is printed and recorded, so
    # "all arms used one instrument" is a checkable fact rather than an intention.
    sys.path.insert(0, str(sb_root))
    from tests.studio.studiobench import pacer as pacer_mod
    from tests.studio.studiobench.runtime import lifecycle
    from tests.studio.studiobench.runtime.seeder import Seeder
    from tests.studio.studiobench.fixture import corpus as corpus_mod
    from tests.studio.studiobench.fixture.corpus import plan_rung, RUNGS

    instrument_file = getattr(pacer_mod, "__file__", None)
    print(f"[{tag}] INSTRUMENT pacer={instrument_file}", flush=True)

    bh = bundle_hash(dist)
    print(f"[{tag}] sb_root={sb_root}", flush=True)
    print(f"[{tag}] dist={dist} BUNDLE_HASH={bh}", flush=True)

    corpus = corpus_mod.Corpus.load()
    print(f"[{tag}] corpus_hash={corpus.corpus_hash}", flush=True)

    if args.rung in SYNTHETIC_RUNGS:
        plan = None
        # The empty rung still needs something to stream, and it must be the SAME text the
        # seeded rungs stream or the stream phase is not comparable across the ladder. The
        # ladder's tail comes from plan_rung's own streamed_unit, so borrow 1K's.
        tail_unit = plan_rung(corpus, "1K").streamed_unit
        seed_target_chars = 0
    else:
        if args.rung not in RUNGS:
            print(f"[{tag}] FATAL: rung {args.rung!r} not in {sorted(RUNGS)} "
                  f"or {sorted(SYNTHETIC_RUNGS)}", flush=True)
            sys.exit(2)
        plan = plan_rung(corpus, args.rung)
        tail_unit = plan.streamed_unit
        seed_target_chars = plan.seeded_chars
        print(f"[{tag}] plan: target_chars={plan.target_chars} seeded={plan.seeded_chars} "
              f"units={len(plan.seeded_units)} streamed={plan.streamed_chars} "
              f"follow_ups={plan.follow_up_chars}", flush=True)

    reasoning = tail_unit.reasoning or ""
    content = tail_unit.content or ""
    print(f"[{tag}] streamed tail: reasoning={len(reasoning)} content={len(content)}", flush=True)

    pacer = pacer_mod.Pacer().start()
    pacer.state.model_ids = ["studiobench-pacer"]
    pacer.load(reasoning, content, cadence="field", tag=f"amdv-{tag}", model="studiobench-pacer")
    exp_ms = pacer.expected_duration_ms(reasoning, content, cadence="field")
    print(f"[{tag}] pacer={pacer.base_url} expected_stream_ms={exp_ms:.0f}", flush=True)

    # ── launch Studio on this arm's own home, port and dist ──
    logp = WS / "logs" / f"amdv_studio_{tag}_{int(time.time())}.log"
    logp.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["UNSLOTH_STUDIO_HOME"] = str(home)
    env.pop("UNSLOTH_API_ONLY", None)
    # The pacer's base URL is 127.0.0.1 and that SSRF guard would reject it.
    env.pop("UNSLOTH_STUDIO_BLOCK_PRIVATE_PROVIDER_URLS", None)
    shutil.rmtree(home / "auth", ignore_errors=True)

    # Studio picks the NEXT FREE PORT when the one it was given is taken, and says so only in
    # its own log. Observed here: a run asked for 5417, another Studio still held it, the
    # server came up on 5418, and every client call in this script went to the STALE server
    # on 5417 -- which answered /healthz happily and then 500'd on /api/providers/. A run that
    # measures somebody else's Studio is not a failed run, it is a wrong one, so this refuses
    # rather than adapts.
    if lifecycle.port_is_busy(args.port):
        print(f"[{tag}] FATAL: port {args.port} is already serving. Refusing to launch, because "
              f"Studio would silently move to the next free port and this script would then "
              f"measure whatever is on {args.port}.", flush=True)
        sys.exit(3)

    binp = args.unsloth_bin or str(WS / "bin" / "unsloth")
    cmd = ["setsid", "bash", "-c",
           f'"{binp}" studio -H 127.0.0.1 -p {args.port} -f "{dist}"'
           f'{" --verbose" if args.studio_verbose else ""} >> "{logp}" 2>&1']
    proc = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            start_new_session=True)
    base = f"http://127.0.0.1:{args.port}"
    if not wait_health(base):
        print(f"[{tag}] FATAL: studio never healthy; log {logp}", flush=True)
        print(logp.read_text(errors="ignore")[-4000:], flush=True)
        sys.exit(4)
    # Belt and braces on the same trap: the port guard above closes the race, this catches the
    # case where Studio moved anyway. Its banner names the port it actually bound.
    banner = logp.read_text(errors = "ignore")
    if f"127.0.0.1:{args.port}" not in banner:
        print(f"[{tag}] FATAL: Studio's own log never names port {args.port}, so the healthy "
              f"server on {base} is not the one this run launched.", flush=True)
        print(banner[-2000:], flush=True)
        sys.exit(4)
    print(f"[{tag}] studio healthy on its own port, log {logp}", flush=True)

    def teardown():
        if args.keep_studio:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass
        subprocess.run(["pkill", "-f", f"unsloth studio -H 127.0.0.1 -p {args.port}"],
                       capture_output=True)

    try:
        pw = lifecycle._read_bootstrap_password(home, logp, time.time() + 120) or ""
        auth = lifecycle.authenticate(base, "unsloth", pw)
        provider = lifecycle.pacer_provider(pacer.base_url, ["studiobench-pacer"])
        lifecycle.register_provider(base, auth, provider)
        ckpt = lifecycle.external_checkpoint_id(provider, "studiobench-pacer")
        print(f"[{tag}] authenticated, checkpoint={ckpt}", flush=True)

        # ── seed the rung ──
        seeder = Seeder(base_url = base, auth = auth, model_id = ckpt)
        seed_t0 = time.time()
        if plan is None:
            thread_id = seeder.create_thread(title = f"amdv {args.rung}")
            seeded = {"thread_id": thread_id, "messages": 0, "seeded_chars": 0,
                      "turns": 0, "last_marker": None, "seconds": 0.0}
        else:
            st = seeder.seed(plan)
            seeded = {"thread_id": st.thread_id, "messages": st.messages,
                      "seeded_chars": st.seeded_chars, "turns": st.turns,
                      "last_marker": st.last_marker, "seconds": st.seconds}
        print(f"[{tag}] seeded {seeded['messages']} messages / {seeded['seeded_chars']} chars "
              f"in {time.time()-seed_t0:.0f}s thread={seeded['thread_id']}", flush=True)

        # Read back what the SERVER thinks is in the thread. A seed that silently truncated
        # would otherwise be reported as a rung it is not, which is exactly the failure mode
        # --stream-tail-chars has on the reply axis.
        try:
            rows = seeder.read_back(seeded["thread_id"])
            read_back_msgs = len(rows)
        except Exception as e:  # noqa: BLE001
            rows, read_back_msgs = [], f"read_back failed: {type(e).__name__}: {e}"
        print(f"[{tag}] read back {read_back_msgs} messages", flush=True)

        init_js = lifecycle.seed_init_script(
            auth, [provider],
            extra_local_storage = {"unsloth_chat_last_external_checkpoint": ckpt},
        )
        scene_path = Path(args.scene) if args.scene else (WS / "scripts" / "amdv_scene.js")
        scene = scene_path.read_text()
        # NO PAGE-WIDE HOG IS INJECTED HERE. The predecessor pasted a `setInterval` spinner after
        # the scene, which jams the ENTIRE session including the mount and every scored window, so
        # a jammed run was a different run rather than a control inside a run. The scene owns the
        # jam now and confines it to one window. Anything appended here would contaminate every
        # cell in the table.
        initp = work / f"init_{tag}.js"
        initp.write_text(init_js + "\n" + scene + "\n")

        # THE JAM IS A WINDOW INSIDE THE SESSION, not a separate session.
        #
        # The ladder used to spend a whole extra run on one rung to prove the frame channel could
        # report a blocked main thread. That control then belonged to that rung, on that arm, and
        # every other cell was read on trust. Here the scene opens its own jammed idle window in
        # EVERY session, so the control is per rung AND per arm AND per repetition, on the same
        # page whose numbers it licenses, for about six seconds each. `--hog-ms 0` disables it,
        # which no scored configuration should ever do.
        max_ms = int(exp_ms * 4 + 240000)
        runp = work / f"run_{tag}.js"
        runp.write_text("%s(%s);" % (args.entry, json.dumps({
            "idleMs": args.idle_ms, "recoverMs": args.recover_ms,
            "maxMs": max_ms, "rung": args.rung, "arm": args.arm,
            "lastMarker": seeded["last_marker"],
            "mountTimeoutMs": MOUNT_MS,
            # Each wait gets its own budget. These used to be constants inside the scene that
            # no option could reach: the composer wait was pinned at 120 s while this line
            # asked for 420 s, and the send wait was pinned at 120 s, which is what a
            # rehearsal session actually died on. They are module constants now because the
            # DRIVER'S OWN DEADLINE has to be derived from them; see `total_timeout`.
            "composerTimeoutMs": COMPOSER_MS, "sendTimeoutMs": SEND_MS,
            "hogMs": args.hog_ms, "hogPeriodMs": args.hog_period_ms,
            "skipSend": bool(args.skip_send),
        })))

        conlog = outp.parent / f"console_{tag}.jsonl"
        resultp = work / f"result_{tag}.json"
        if resultp.exists():
            resultp.unlink()
        # THE DRIVER'S DEADLINE IS DERIVED FROM THE SCENE'S BUDGETS, not guessed from the
        # stream length alone.
        #
        # It used to be `exp_ms/1000*4 + 900`, about 950 s, while the waits the scene was
        # handed add up to 180 + 420 + 240 s before the stream cap of about 300 s even starts.
        # So a session that used its budgets was killed by `GLib.timeout_add(..., bail)`
        # inside the driver, which sets exit 3 and WRITES NOTHING to --out. The bench then
        # stores `{"__done": false}`, the completion gate fails, and an eight-hour run reports
        # INCONCLUSIVE with no attributable error: precisely the case the budgets were raised
        # to survive. Raising the budgets without raising this widened the hole from about 60 s
        # to about 220 s, which is how it was found.
        total_timeout = (COMPOSER_MS + MOUNT_MS + SEND_MS + max_ms) / 1000 + 120
        # `/chat/<id>` is a 404: the router owns only `/chat` and the thread is a SEARCH
        # PARAM (studio/frontend/src/features/chat/chat-page.tsx::validateChatSearch reads
        # `search.thread`). A path-style URL renders the not-found page, which still has a
        # body and would have been measured as an empty thread at every rung.
        url = f"{base}/chat?thread={seeded['thread_id']}"
        driver_path = Path(args.driver) if args.driver else (WS / "scripts" / "amdv_drive.py")
        drive = [args.python_gi, str(driver_path),
                 "--url", url, "--init-script", str(initp), "--script", str(runp),
                 "--out", str(resultp), "--timeout", str(total_timeout),
                 "--accel", args.accel, "--frame-clock", args.frame_clock,
                 "--console-log", str(conlog)]
        denv = dict(os.environ)
        denv["DISPLAY"] = args.display
        print(f"[{tag}] driving {url}, timeout {total_timeout:.0f}s", flush=True)
        t0 = time.time()
        r = subprocess.run(drive, env = denv, capture_output = True, text = True,
                           timeout = total_timeout + 180)
        print(f"[{tag}] driver exit={r.returncode} in {time.time()-t0:.0f}s", flush=True)
        print("\n".join(r.stdout.splitlines()[-20:]), flush=True)
        if r.returncode != 0:
            print("STDERR:", r.stderr[-3000:], flush=True)

        payload = json.loads(resultp.read_text()) if resultp.exists() else {"__done": False}
        stats = pacer.last_stats()
        payload["run_meta"] = {
            "rung": args.rung, "rep": args.rep, "arm": args.arm,
            "synthetic_rung": args.rung in SYNTHETIC_RUNGS,
            "dist": str(dist), "bundle_hash": bh, "sb_root": str(sb_root),
            "corpus_hash": corpus.corpus_hash,
            "instrument_pacer_file": instrument_file,
            "instrument_sb_root": str(sb_root),
            "instrument_hash": args.instrument_hash or None,
            "plan": None if plan is None else {
                "target_tokens": plan.target_tokens, "target_chars": plan.target_chars,
                "seeded_chars": plan.seeded_chars, "seeded_units": len(plan.seeded_units),
                "streamed_chars": plan.streamed_chars, "follow_up_chars": plan.follow_up_chars},
            "seed_target_chars": seed_target_chars,
            "seeded": seeded, "read_back_messages": read_back_msgs,
            "streamed_reasoning_chars": len(reasoning), "streamed_content_chars": len(content),
            "expected_stream_ms": exp_ms, "pacer": pacer.base_url,
            "pacer_stats": stats if isinstance(stats, dict) else str(stats),
            "studio_log": str(logp), "console_log": str(conlog),
            "skip_send": bool(args.skip_send), "studio_verbose": bool(args.studio_verbose), "studio_port": args.port, "url": url,
            "renderer": "REAL WebKitGTK via libwebkit2gtk-4.1 (PyGObject), NOT Playwright WebKit",
            "accel_policy": args.accel, "driver_exit": r.returncode,
            "scene": str(scene_path), "driver": str(driver_path),
            "hog_ms": args.hog_ms, "hog_period_ms": args.hog_period_ms,
            "frame_clock": args.frame_clock,
            "display": args.display, "ts": time.time(),
        }
        outp.write_text(json.dumps(payload))
        print(f"[{tag}] wrote {outp}", flush=True)
        if not payload.get("ok"):
            print(f"[{tag}] NOT OK: {str(payload.get('error'))[:600]}", flush=True)
        sys.exit(0 if payload.get("ok") else 5)
    finally:
        teardown()


if __name__ == "__main__":
    main()
