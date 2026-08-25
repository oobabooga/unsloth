#!/usr/bin/env python3
"""One studiobench LADDER RUNG, measured inside UNSLOTH DESKTOP -- the Tauri app itself.

The sibling of amdv_rung_bench.py, which measures the same rung in the same engine through a
PyGObject WebKitGTK window. Everything that CAN be identical is identical, deliberately, because
the only interesting question is what the DESKTOP SHELL adds:

  * the same studiobench frozen corpus, Seeder and plan_rung;
  * the same pacer, the same cadence, the same streamed tail;
  * the same scene (amdv_scene.js), byte for byte, so the phases, the busy calibration and the
    DOM census are the same quantities;
  * the same rung ladder and the same jammed positive control.

What is necessarily different, and why:

  1. **The frontend is not served over HTTP.** Tauri embeds `frontendDist` in the binary and
     serves it at `tauri://localhost`, so there is no `/chat?thread=<id>` URL to load. The
     thread is opened from inside the page instead, and WHICH route worked is recorded.
  2. **There is no external eval channel.** The CSP is `default-src 'self'` with no
     `unsafe-eval`, so the scene has to be compiled into the bundle. It is, by the build job,
     which produces a pristine binary and an instrumented one and keeps them apart.
  3. **The app owns its own backend lifecycle.** Rather than fight it, the backend is started
     first on a port the app scans (8888..8908) with the SAME studio root the app will use, so
     `desktop_preflight` returns `AttachedReady` and the app goes straight to the shell instead
     of its installer. That path is exercised by real users who start Studio from a terminal,
     and it is the only one that does not require the installer UI to be driven.
  4. **The rendering path is the app's to choose.** linux_webkit.rs decides before GTK init
     whether to apply a workaround, and main.rs only LOGS that decision when one is applied. So
     it is read back out of `/proc/<pid>/environ` of the live process, together with every input
     the decision was made from.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from desktop_lib import amdgpu_delta, sample_amdgpu, sh  # noqa: E402

from amdv_control import Control  # noqa: E402

SYNTHETIC_RUNGS = {"0K"}

# Symlinked from the one install into every run's own studio root. A run that shared `auth`,
# `logs`, `run` or `studio.db` with another run would mount the other run's threads, which is
# how a 0K rung quietly becomes a 500K one.
SHARED = ("unsloth_studio", "bin", "share", "llama.cpp", "whisper.cpp", "assets",
          "cache", "compiled_cache")
PRIVATE = ("auth", "logs", "run", "exports", "outputs", "runs", "rag")


def bundle_hash(binary: Path) -> str:
    h = hashlib.sha256()
    with open(binary, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def wait_health(base: str, timeout=600) -> dict:
    """Poll /api/health, which is what preflight/backend.rs:31 itself probes."""
    import urllib.request
    dl = time.time() + timeout
    last = None
    while time.time() < dl:
        try:
            with urllib.request.urlopen(f"{base}/api/health", timeout = 5) as r:
                body = json.loads(r.read().decode())
                if body.get("status") == "healthy":
                    return {"ok": True, "body": body}
                last = body
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
        time.sleep(2)
    return {"ok": False, "last": str(last)[:500]}


def attach_preconditions(base: str, studio_root: Path) -> dict:
    """Every condition preflight.rs needs for AttachedReady, checked before the app is launched.

    Checked rather than assumed because each one fails SILENTLY into the installer screen,
    which then looks like "Desktop does not work on this runner" and is nothing of the kind.
    """
    import urllib.request
    out: dict = {}
    try:
        with urllib.request.urlopen(f"{base}/api/health", timeout = 10) as r:
            h = json.loads(r.read().decode())
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    idf = studio_root / "share" / "studio_install_id"
    on_disk = idf.read_text().strip() if idf.is_file() else ""
    out["health"] = {k: h.get(k) for k in (
        "status", "service", "desktop_protocol_version", "desktop_manageability_version",
        "supports_desktop_auth", "supports_desktop_backend_ownership", "studio_root_id",
        "native_path_leases_supported", "desktop_owner")}
    out["studio_install_id_on_disk"] = on_disk[:16] + ("..." if on_disk else "")
    out["root_ids_match"] = bool(on_disk) and h.get("studio_root_id") == on_disk
    out["desktop_secret_present"] = (studio_root / "auth" / ".desktop_secret").is_file()
    # A stale owner file makes the app take the owned-backend path instead of attaching.
    out["stale_owner_file"] = (studio_root / "run" / "desktop_backend.json").is_file()
    # The 401 probe preflight itself performs. A 200 here would mean the backend accepts any
    # secret, which is a different and much worse problem.
    try:
        req = urllib.request.Request(
            f"{base}/api/auth/desktop-login", method = "POST",
            data = json.dumps({"secret": "definitely-not-the-secret"}).encode(),
            headers = {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout = 10) as r:
                out["bad_secret_status"] = r.status
        except Exception as e:  # noqa: BLE001
            out["bad_secret_status"] = getattr(e, "code", None) or str(e)[:120]
    except Exception as e:  # noqa: BLE001
        out["bad_secret_status"] = f"{type(e).__name__}: {e}"
    out["all_ok"] = bool(
        out["health"].get("status") == "healthy"
        and out["health"].get("service") == "Unsloth UI Backend"
        and out["health"].get("desktop_protocol_version") == 1
        and out["health"].get("supports_desktop_auth")
        and out["health"].get("supports_desktop_backend_ownership")
        and out["root_ids_match"] and out["desktop_secret_present"]
        and not out["stale_owner_file"] and out["bad_secret_status"] == 401)
    return out


def _desktop_login_status(base: str, secret: str) -> dict:
    """The exchange desktop_auth.rs performs. A 200 here is the app getting in."""
    import urllib.request
    if not secret:
        return {"status": None, "error": "no .desktop_secret on disk"}
    req = urllib.request.Request(
        f"{base}/api/auth/desktop-login", method = "POST",
        data = json.dumps({"secret": secret}).encode(),
        headers = {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout = 15) as r:
            return {"status": r.status, "has_token": '"access_token"' in r.read().decode()}
    except Exception as e:  # noqa: BLE001
        return {"status": getattr(e, "code", None), "error": str(e)[:200]}


def build_run_home(base_home: Path, run_home: Path) -> dict:
    studio_src = base_home / ".unsloth" / "studio"
    studio_dst = run_home / ".unsloth" / "studio"
    if run_home.exists():
        shutil.rmtree(run_home, ignore_errors = True)
    studio_dst.mkdir(parents = True, exist_ok = True)
    for sub in (".config", "xdg", ".local/share"):
        (run_home / sub).mkdir(parents = True, exist_ok = True)
    linked, made = [], []
    for name in SHARED:
        src = studio_src / name
        if src.exists() and not (studio_dst / name).exists():
            os.symlink(src, studio_dst / name)
            linked.append(name)
    for name in PRIVATE:
        (studio_dst / name).mkdir(parents = True, exist_ok = True)
        made.append(name)
    return {"home": str(run_home), "studio_root": str(studio_dst),
            "linked": linked, "private": made}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", required = True)
    ap.add_argument("--rep", default = "1")
    ap.add_argument("--binary", required = True, help = "the INSTRUMENTED Tauri binary")
    ap.add_argument("--base-home", required = True, help = "the home install.sh --local used")
    ap.add_argument("--run-home", required = True, help = "this run's private HOME")
    ap.add_argument("--sb-root", required = True, help = "checkout providing tests.studio.studiobench")
    ap.add_argument("--port", type = int, required = True, help = "8888..8908; the app scans these")
    ap.add_argument("--control-port", type = int, required = True)
    ap.add_argument("--display", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--idle-ms", type = int, default = 6000)
    ap.add_argument("--recover-ms", type = int, default = 6000)
    ap.add_argument("--settle-ms", type = int, default = 9000,
                    help = "waited after the shell mounts, before anything is measured. The "
                           "updater fires an HTTPS check 5 s after mount and renders a banner "
                           "(hooks/use-tauri-update.ts:299); left alone it lands inside the "
                           "idle baseline. Identical at every rung, so it cannot create a "
                           "rung-dependent difference.")
    ap.add_argument("--hog-ms", type = int, default = 0)
    ap.add_argument("--hog-period-ms", type = int, default = 250)
    ap.add_argument("--skip-send", action = "store_true")
    ap.add_argument("--software", action = "store_true",
                    help = "the NEGATIVE CONTROL leg: LIBGL_ALWAYS_SOFTWARE=1, everything "
                           "else identical. Without it the amdgpu counters are a number "
                           "rather than evidence.")
    args = ap.parse_args()

    outp = Path(args.out)
    outp.parent.mkdir(parents = True, exist_ok = True)
    tag = f"r{args.rung}_rep{args.rep}"
    binary = Path(args.binary)
    base_home = Path(args.base_home)
    run_home = Path(args.run_home)
    out: dict = {"rung": args.rung, "rep": args.rep, "tag": tag,
                 "binary": str(binary), "binary_sha256_16": bundle_hash(binary),
                 "software_control": bool(args.software),
                 "hog_ms": args.hog_ms, "port": args.port,
                 "control_port": args.control_port, "display": args.display}

    sys.path.insert(0, args.sb_root)
    from tests.studio.studiobench import pacer as pacer_mod
    from tests.studio.studiobench.runtime import lifecycle
    from tests.studio.studiobench.runtime.seeder import Seeder
    from tests.studio.studiobench.fixture import corpus as corpus_mod
    from tests.studio.studiobench.fixture.corpus import plan_rung, RUNGS

    corpus = corpus_mod.Corpus.load()
    out["corpus_hash"] = corpus.corpus_hash
    if args.rung in SYNTHETIC_RUNGS:
        plan = None
        tail_unit = plan_rung(corpus, "1K").streamed_unit
    else:
        if args.rung not in RUNGS:
            out["fatal"] = f"rung {args.rung!r} not in {sorted(RUNGS)} or {sorted(SYNTHETIC_RUNGS)}"
            outp.write_text(json.dumps(out))
            return 2
        plan = plan_rung(corpus, args.rung)
        tail_unit = plan.streamed_unit
    reasoning = tail_unit.reasoning or ""
    content = tail_unit.content or ""

    out["home"] = build_run_home(base_home, run_home)
    studio_root = Path(out["home"]["studio_root"])

    # Refuse rather than adapt if the port is taken: Studio silently moves to the next free
    # port and says so only in its own log, and a run that measures somebody else's backend is
    # not a failed run, it is a wrong one. The app would attach to whichever it found first.
    if lifecycle.port_is_busy(args.port):
        out["fatal"] = (f"port {args.port} is already serving; refusing, because Studio would "
                        f"move to the next free port and the app would attach to whatever is "
                        f"on {args.port}")
        outp.write_text(json.dumps(out))
        return 3

    pacer = pacer_mod.Pacer().start()
    pacer.state.model_ids = ["studiobench-pacer"]
    pacer.load(reasoning, content, cadence = "field", tag = f"amdvd-{tag}",
               model = "studiobench-pacer")
    exp_ms = pacer.expected_duration_ms(reasoning, content, cadence = "field")
    out["pacer"] = {"base_url": pacer.base_url, "expected_stream_ms": exp_ms}

    cli = studio_root / "unsloth_studio" / "bin" / "unsloth"
    if not cli.exists():
        cli = studio_root / "bin" / "unsloth"
    out["cli"] = str(cli)

    env = dict(os.environ)
    env["HOME"] = str(run_home)
    # The Tauri app hardcodes $HOME/.unsloth/studio and scrubs UNSLOTH_STUDIO_HOME from every
    # child it spawns, so the two sides agree only if this IS that directory.
    env["UNSLOTH_STUDIO_HOME"] = str(studio_root)
    env["XDG_RUNTIME_DIR"] = str(run_home / "xdg")
    env["XDG_CONFIG_HOME"] = str(run_home / ".config")
    env["XDG_DATA_HOME"] = str(run_home / ".local" / "share")
    env.pop("STUDIO_HOME", None)
    env.pop("UNSLOTH_API_ONLY", None)
    # The pacer's base URL is 127.0.0.1 and that SSRF guard would reject it.
    env.pop("UNSLOTH_STUDIO_BLOCK_PRIVATE_PROVIDER_URLS", None)
    # Setting these without matching run/desktop_backend.json metadata drives the app into
    # ExternalConflict instead of AttachedReady.
    for k in ("UNSLOTH_STUDIO_DESKTOP_OWNER_TOKEN", "UNSLOTH_STUDIO_DESKTOP_OWNER_KIND",
              "UNSLOTH_STUDIO_DESKTOP_OWNER_PID"):
        env.pop(k, None)

    # Maximum backend verbosity, through the env rather than --verbose: the CLI's system-dir
    # guard whitelists `studio --api-only -H .. -p ..` "and nothing else", so an extra flag on
    # that argv can turn a launch into a hard error.
    benv = dict(env)
    benv["LOG_LEVEL"] = "DEBUG"
    benv["UNSLOTH_STUDIO_ACCESS_LOG_DEDUP_MS"] = "0"
    benv["UNSLOTH_STUDIO_ACCESS_LOG_POLL_DEDUP_MS"] = "0"

    logdir = outp.parent / "logs"
    logdir.mkdir(parents = True, exist_ok = True)
    blog = logdir / f"backend_{tag}.log"
    base = f"http://127.0.0.1:{args.port}"
    state = {"backend": None}

    def start_backend(which: str):
        log = logdir / f"backend_{tag}_{which}.log"
        with open(log, "wb") as fh:
            p = subprocess.Popen(
                [str(cli), "studio", "--api-only", "-H", "127.0.0.1", "-p", str(args.port)],
                env = benv, cwd = str(run_home), stdin = subprocess.DEVNULL,
                stdout = fh, stderr = subprocess.STDOUT, start_new_session = True)
        state["backend"] = p
        return p, log

    def stop_backend():
        p = state.get("backend")
        if p is not None and p.poll() is None:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                for _ in range(30):
                    if p.poll() is not None:
                        break
                    time.sleep(1)
                if p.poll() is None:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass
        state["backend"] = None

    backend, blog = start_backend("seed")
    out["backend"] = {"pid": backend.pid, "log": str(blog)}
    health = wait_health(base)
    out["backend"]["health"] = health
    app = None
    control = None
    try:
        if not health.get("ok"):
            out["fatal"] = "backend never became healthy"
            out["backend"]["log_tail"] = blog.read_text(errors = "replace")[-6000:]
            return 4

        out["attach_preconditions"] = attach_preconditions(base, studio_root)

        # ── seed the rung, against the same backend the app will attach to ──
        pw = lifecycle._read_bootstrap_password(studio_root, blog, time.time() + 180) or ""
        auth = lifecycle.authenticate(base, "unsloth", pw)
        provider = lifecycle.pacer_provider(pacer.base_url, ["studiobench-pacer"])
        lifecycle.register_provider(base, auth, provider)
        ckpt = lifecycle.external_checkpoint_id(provider, "studiobench-pacer")
        seeder = Seeder(base_url = base, auth = auth, model_id = ckpt)
        t0 = time.time()
        if plan is None:
            thread_id = seeder.create_thread(title = f"amdv-desktop {args.rung}")
            seeded = {"thread_id": thread_id, "messages": 0, "seeded_chars": 0,
                      "turns": 0, "last_marker": None}
        else:
            st = seeder.seed(plan)
            seeded = {"thread_id": st.thread_id, "messages": st.messages,
                      "seeded_chars": st.seeded_chars, "turns": st.turns,
                      "last_marker": st.last_marker}
        seeded["seconds"] = round(time.time() - t0, 1)
        out["seeded"] = seeded
        try:
            out["read_back_messages"] = len(seeder.read_back(seeded["thread_id"]))
        except Exception as e:  # noqa: BLE001
            out["read_back_messages"] = f"read_back failed: {type(e).__name__}: {e}"

        # ── desktop auth, then a RESTART, then proof the running server accepts the secret ──
        #
        # The order is not arbitrary and each part earned its place. Seeding needs the admin
        # login, which needs the bootstrap password the server mints on its FIRST start against
        # an empty auth dir, so the backend has to run before provisioning. Provisioning then
        # writes both a hash into auth.db and the plaintext the app will read, and the server is
        # restarted rather than trusted to notice, because a cached credential would show up
        # only as the app sitting on its startup screen for the whole run with nothing in any
        # log saying why. Finally the real secret is POSTed here: a 200 is the same exchange
        # desktop_auth.rs is about to perform, so "the app will get in" is verified rather than
        # inferred.
        stop_backend()
        out["provision"] = sh([str(cli), "studio", "provision-desktop-auth"], timeout = 600,
                              env = env, cwd = str(run_home))
        backend, blog2 = start_backend("serve")
        out["backend"]["restarted_pid"] = backend.pid
        out["backend"]["log_serve"] = str(blog2)
        out["backend"]["health_after_restart"] = wait_health(base)

        secret_file = studio_root / "auth" / ".desktop_secret"
        out["desktop_login"] = _desktop_login_status(
            base, secret_file.read_text().strip() if secret_file.is_file() else "")
        out["attach_preconditions_after_provision"] = attach_preconditions(base, studio_root)
        if out["desktop_login"].get("status") != 200:
            out["fatal"] = ("the backend does not accept the provisioned desktop secret, so the "
                            "app would sit on its startup screen; refusing to measure a run "
                            "that cannot reach the shell")
            return 6

        # ── the control channel, then the app ──
        control = Control(args.control_port, outp.parent).start()
        control.store.set_config({
            "ready": True,
            "threadId": seeded["thread_id"],
            "lastMarker": seeded["last_marker"],
            "settleMs": args.settle_ms,
            "hogMs": args.hog_ms,
            "hogPeriodMs": args.hog_period_ms,
            "navOrder": ["history", "click", "assign"],
            "runArgs": {
                "idleMs": args.idle_ms, "recoverMs": args.recover_ms,
                "maxMs": int(exp_ms * 4 + 240000), "rung": args.rung,
                "lastMarker": seeded["last_marker"],
                "mountTimeoutMs": 420000,
                "skipSend": bool(args.skip_send),
            },
        })

        aenv = dict(env)
        aenv["DISPLAY"] = args.display
        # Without this GTK's gdk_x_io_error() reports a fatal X error through g_debug() and
        # then _exit(1)s, so the app dies printing nothing at all.
        aenv["G_MESSAGES_DEBUG"] = "all"
        aenv["RUST_BACKTRACE"] = "full"
        if args.software:
            aenv["LIBGL_ALWAYS_SOFTWARE"] = "1"
            aenv["GALLIUM_DRIVER"] = "llvmpipe"
        alog = logdir / f"desktop_{tag}.log"
        with open(alog, "wb") as fh:
            app = subprocess.Popen([str(binary)], env = aenv, cwd = str(run_home),
                                   stdin = subprocess.DEVNULL, stdout = fh,
                                   stderr = subprocess.STDOUT, start_new_session = True)
        out["app"] = {"pid": app.pid, "log": str(alog)}

        # The rendering path, read from the LIVE process rather than from a log line that is
        # only written when a workaround is applied.
        time.sleep(5)
        try:
            raw = Path(f"/proc/{app.pid}/environ").read_bytes().decode(errors = "replace")
            envmap = dict(i.split("=", 1) for i in raw.split("\0") if "=" in i)
        except Exception as e:  # noqa: BLE001
            envmap = {"_error": f"{type(e).__name__}: {e}"}
        out["app"]["renderer_env"] = {k: envmap.get(k) for k in (
            "WEBKIT_DISABLE_DMABUF_RENDERER", "WEBKIT_DMABUF_RENDERER_FORCE_SHM",
            "WEBKIT_FORCE_DMABUF_RENDERER", "UNSLOTH_WEBKIT_RENDERER_WORKAROUND",
            "LIBGL_ALWAYS_SOFTWARE", "GALLIUM_DRIVER", "APPIMAGE", "GDK_BACKEND")}
        out["app"]["nvidia_module_present"] = Path("/proc/driver/nvidia/version").exists()

        # Did the PAGE ever reach us? Distinguishes "the app never painted our script" from
        # "the scene ran and failed", which otherwise both present as a missing result.
        out["page_contact"] = control.wait_for_contact(300)

        # amdgpu, sampled early and again at the end and DIFFERENCED per drm-client-id, so the
        # figure is engine time accrued WHILE the ladder ran rather than a cumulative counter.
        out["amdgpu_early"] = sample_amdgpu(app.pid)

        total = exp_ms / 1000 * 4 + 1200
        result = control.wait_for_result(total)
        out["amdgpu_late"] = sample_amdgpu(app.pid)
        out["amdgpu"] = amdgpu_delta(out["amdgpu_early"], out["amdgpu_late"])
        out["app"]["alive_at_end"] = app.poll() is None
        out["app"]["exit_code"] = app.returncode

        out["windows"] = sh(["xwininfo", "-root", "-tree"], timeout = 60,
                            env = {"DISPLAY": args.display}).get("stdout", "")[-4000:]
        xwd = logdir / f"shot_{tag}.xwd"
        png = logdir / f"shot_{tag}.png"
        sh(["xwd", "-root", "-silent", "-out", str(xwd)], timeout = 120,
           env = {"DISPLAY": args.display})
        if xwd.is_file() and shutil.which("ffmpeg"):
            sh(["ffmpeg", "-y", "-loglevel", "error", "-i", str(xwd), str(png)], timeout = 180)
            out["screenshot"] = {"png": str(png),
                                 "bytes": png.stat().st_size if png.is_file() else 0}

        out["payload"] = result
        out["ok"] = bool(result and result.get("ok"))
        if not out["ok"]:
            out["error"] = (result or {}).get("error") if result else "no result from the page"
        out["page_events"] = str(outp.parent / "page_events.jsonl")
        return 0 if out["ok"] else 5
    finally:
        # By PID and by process group only. A pattern-matching killer would match this very
        # script's own command line, and a Studio or a Tauri app that outlives the run holds
        # VRAM and a port on a machine three other slots share.
        if app is not None and app.poll() is None:
            try:
                os.killpg(os.getpgid(app.pid), signal.SIGTERM)
                time.sleep(3)
                if app.poll() is None:
                    os.killpg(os.getpgid(app.pid), signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass
        stop_backend()
        if control is not None:
            control.stop()
        for key, path in (("log_tail", blog),
                          ("log_serve_tail", logdir / f"backend_{tag}_serve.log")):
            try:
                if Path(path).is_file():
                    out["backend"][key] = Path(path).read_text(errors = "replace")[-12000:]
            except Exception:  # noqa: BLE001
                pass
        try:
            out["app"] = out.get("app") or {}
            out["app"]["log_tail"] = (logdir / f"desktop_{tag}.log").read_text(
                errors = "replace")[-12000:]
            tl = Path(out["home"]["studio_root"]) / "tauri.log"
            out["app"]["tauri_log"] = tl.read_text(errors = "replace")[-12000:] \
                if tl.is_file() else None
        except Exception:  # noqa: BLE001
            pass
        outp.write_text(json.dumps(out))


if __name__ == "__main__":
    raise SystemExit(main())
