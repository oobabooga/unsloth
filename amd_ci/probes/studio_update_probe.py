#!/usr/bin/env python3
"""Probe: what does a no-op Studio update cost at this state, on this host?

Observes only. For one checkout it builds the wheel, installs Studio fresh from
that wheel with the checkout's own installer (the desktop path: `--tauri`,
isolated home, CONNECT proxy so every byte is attributed to a host), and then runs
the desktop's update command three times with nothing to update:

    settle   the first run by the freshly installed code (may legitimately settle
             manifest evidence; recorded, not judged)
    noop     the second run: what a user pays for an update that changes nothing
    offline  the third run with every outbound connection refused and UV_OFFLINE=1

Everything lands in the observation JSON: wall time, bytes down per host, which
update steps ran (from the log), the idempotency comparison of the venv before
and after, and the torch build (so a ROCm host can show whether the update kept
the ROCm wheel or repaired it away). The criteria module decides what any of it
means.

Drives the bisect harness (scripts/bisect/run_step.sh|ps1) vendored under
`--harness`; that harness is what measured the same thing on the hosted runners,
so the numbers are comparable.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

IS_WINDOWS = platform.system() == "Windows"

# Hosts whose bytes mean "a package or binary was fetched", as opposed to index
# metadata. The criteria judge these; the probe only records them.
PAYLOAD_HOSTS = ("files.pythonhosted.org", "objects.githubusercontent.com",
                 "release-assets.githubusercontent.com", "nodejs.org",
                 "repo.amd.com", "download.pytorch.org")

# Log lines that identify torch being (re)installed by the update, per installer.
TORCH_INSTALL_PATTERNS = (
    r"Installing PyTorch", r"installed PyTorch is not a ROCm build",
)


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding = "utf-8-sig"))
    except Exception:  # noqa: BLE001
        return None


def _run(cmd: list[str], *, cwd: Path | None, env: dict, log: Path, timeout: int) -> dict:
    """Run one harness command, capturing everything to `log`; never raises."""
    t0 = time.time()
    log.parent.mkdir(parents = True, exist_ok = True)
    with open(log, "ab") as fh:
        fh.write(("\n$ " + " ".join(cmd) + "\n").encode("utf-8"))
        fh.flush()
        try:
            p = subprocess.run(cmd, cwd = str(cwd) if cwd else None, env = env,
                               stdout = fh, stderr = subprocess.STDOUT, timeout = timeout)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = -9
            fh.write(b"\n[probe] TIMEOUT\n")
        except OSError as exc:
            rc = -1
            fh.write(f"\n[probe] OSError: {exc}\n".encode("utf-8"))
    return {"rc": rc, "seconds": round(time.time() - t0, 1), "cmd": " ".join(cmd)}


def _tail(path: Path, n: int = 40) -> str:
    try:
        lines = path.read_text(encoding = "utf-8", errors = "replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n:])


def _venv_paths(home: Path) -> tuple[Path, Path]:
    studio = home / ".unsloth" / "studio"
    venv = studio / "unsloth_studio"
    py = venv / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")
    return venv, py


def _torch_facts(py: Path, timeout: int = 240) -> dict:
    """Version and backend of the installed torch, from the venv itself.

    Imports torch, so on a ROCm host this also says whether the GPU is visible to
    the build the update left behind. Bounded: a hung import is recorded, not waited on.
    """
    if not py.is_file():
        return {"present": False, "error": "no venv python"}
    code = (
        "import json,sys\n"
        "try:\n"
        "    import torch\n"
        "    d={'present':True,'version':torch.__version__,'hip':getattr(torch.version,'hip',None),"
        "'cuda':getattr(torch.version,'cuda',None)}\n"
        "    try:\n"
        "        d['gpu_available']=bool(torch.cuda.is_available())\n"
        "        d['device_count']=int(torch.cuda.device_count())\n"
        "        d['device_names']=[torch.cuda.get_device_name(i) for i in range(d['device_count'])]\n"
        "    except Exception as e:\n"
        "        d['gpu_error']=repr(e)\n"
        "except Exception as e:\n"
        "    d={'present':False,'error':repr(e)}\n"
        "print(json.dumps(d))\n"
    )
    try:
        p = subprocess.run([str(py), "-I", "-c", code], capture_output = True, text = True,
                           timeout = timeout)
        line = (p.stdout or "").strip().splitlines()
        return json.loads(line[-1]) if line else {"present": False, "error": (p.stderr or "")[-500:]}
    except Exception as exc:  # noqa: BLE001
        return {"present": False, "error": repr(exc)}


def _torch_record_mtime(venv: Path) -> float | None:
    pattern = "Lib/site-packages/torch-*.dist-info/RECORD" if IS_WINDOWS \
        else "lib/python*/site-packages/torch-*.dist-info/RECORD"
    hits = glob.glob(str(venv / pattern))
    if not hits:
        return None
    return round(max(os.stat(h).st_mtime for h in hits), 3)


def _amd_escape(venv: Path, py: Path, timeout: int = 200) -> dict:
    """setup.sh's AMD fast-path escape, exactly as setup.sh invokes it (POSIX only).

    Exit 0 means "the installed torch is not a ROCm build on this AMD host, force
    the dependency pass"; non-zero keeps the fast path. Recorded verbatim.
    """
    if IS_WINDOWS:
        return {"applicable": False}
    hits = glob.glob(str(venv / "lib/python*/site-packages/studio/install_python_stack.py"))
    if not hits or not py.is_file():
        return {"applicable": True, "error": "installed install_python_stack.py not found"}
    try:
        p = subprocess.run([str(py), hits[0], "--amd-torch-needs-dependency-pass"],
                           capture_output = True, text = True, timeout = timeout)
        return {"applicable": True, "rc": p.returncode, "would_force_dependency_pass": p.returncode == 0,
                "stderr_tail": (p.stderr or "")[-600:], "stdout_tail": (p.stdout or "")[-300:]}
    except Exception as exc:  # noqa: BLE001
        return {"applicable": True, "error": repr(exc)}


def _step_summary(out_dir: Path, name: str) -> dict:
    """Fold one harness step's summary.json into the fields the criteria read."""
    d = out_dir / name
    s = _read_json(d / "summary.json")
    if not s:
        return {"present": False, "log_tail": _tail(d / "log.txt", 30)}
    proxy = s.get("proxy") or {}
    by_host = proxy.get("by_host") or {}
    hosts = {h: {"bytes_down": int(v.get("bytes_down", 0)), "connections": int(v.get("connections", v.get("count", 0)) or 0)}
             for h, v in by_host.items()}
    payload = sum(v["bytes_down"] for h, v in hosts.items() if any(h.endswith(x) for x in PAYLOAD_HOSTS))
    log_text = ""
    try:
        log_text = (d / "log.txt").read_text(encoding = "utf-8", errors = "replace")
    except OSError:
        pass
    torch_lines = [ln for ln in log_text.splitlines()
                   if any(re.search(pat, ln) for pat in TORCH_INSTALL_PATTERNS)]
    return {
        "present": True,
        "seconds": s.get("seconds_total"),
        "exit_code": s.get("exit_code"),
        "assert_ok": s.get("assert_ok"),
        "connections_attempted": s.get("connections_attempted"),
        "connections_refused": s.get("connections_refused"),
        "total_bytes_down": int(proxy.get("total_bytes_down", 0) or 0),
        "payload_bytes_down": payload,
        "by_host": hosts,
        "idempotent": s.get("idempotent"),
        "idempotent_relaxed": s.get("idempotent_relaxed"),
        "idempotency_reasons": s.get("idempotency_reasons"),
        "relaxed_reasons": s.get("relaxed_reasons"),
        "freeze_diff_empty": s.get("freeze_diff_empty"),
        "freeze_diff": s.get("freeze_diff"),
        "sidecars_changed": s.get("sidecars_changed"),
        "prebuilt_changed": s.get("prebuilt_changed"),
        "binaries_changed": s.get("binaries_changed"),
        "update_steps_ran": s.get("update_steps_ran"),
        "update_path": s.get("update_path"),
        "update_path_reason": s.get("update_path_reason"),
        "pypi_probe_answered": s.get("pypi_probe_answered"),
        "amd_escape_fired": "installed PyTorch is not a ROCm build" in log_text,
        "torch_install_lines": torch_lines[:12],
        "log_tail": "\n".join(log_text.splitlines()[-25:]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable,
                    help = "python that runs the harness helpers (stdlib only)")
    ap.add_argument("--harness", required = True, type = Path,
                    help = "directory holding run_step.sh/run_step.ps1, pr_wheel.py, pins/")
    ap.add_argument("--work", required = True, type = Path,
                    help = "scratch root; a short path, one subdirectory per state")
    ap.add_argument("--uv", default = "uv", help = "uv binary used to build the wheel")
    ap.add_argument("--pin-base", default = "807")
    ap.add_argument("--install-timeout", type = int, default = 3600)
    ap.add_argument("--update-timeout", type = int, default = 1800)
    ap.add_argument("--no-torch", action = "store_true",
                    help = "install without torch (a plumbing check, not the AMD question)")
    args = ap.parse_args()

    checkout = Path(args.checkout).resolve()
    harness = args.harness.resolve()
    work = (args.work / args.state).resolve()
    if work.exists():
        shutil.rmtree(work, ignore_errors = True)
    wheel_dir = work / "w"
    out_dir = work / "o"
    home = work / "u"           # the product's HOME / USERPROFILE
    for d in (wheel_dir, out_dir, home):
        d.mkdir(parents = True, exist_ok = True)
    probe_log = out_dir / "probe_driver.log"

    obs: dict = {
        "state": args.state, "checkout": str(checkout), "os": platform.system(),
        "machine": platform.node(), "work": str(work), "steps": {},
    }

    # ---- 1. the wheel this state produces, relabelled .post1 so the update resolves to it
    env = dict(os.environ)
    env["UV_CACHE_DIR"] = str(work / "uvc-build")
    r = _run([args.uv, "build", "--wheel", "--out-dir", str(wheel_dir)], cwd = checkout, env = env,
             log = probe_log, timeout = 1200)
    obs["steps"]["build"] = r
    if r["rc"] != 0:
        obs["error"] = "wheel build failed"
        obs["log_tail"] = _tail(probe_log)
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0
    pw = harness / "pr_wheel.py"
    r = _run([args.python, str(pw), "verify", "--source", str(checkout), "--wheel-dir", str(wheel_dir)],
             cwd = checkout, env = env, log = probe_log, timeout = 300)
    obs["steps"]["verify"] = r
    relabel_json = out_dir / "pr_wheel.json"
    with open(relabel_json, "wb") as fh:
        p = subprocess.run([args.python, str(pw), "relabel", "--wheel-dir", str(wheel_dir), "--post", "1"],
                           cwd = checkout, env = env, stdout = fh, stderr = subprocess.PIPE)
    obs["steps"]["relabel"] = {"rc": p.returncode, "stderr_tail": (p.stderr or b"")[-800:].decode("utf-8", "replace")}
    obs["wheel"] = _read_json(relabel_json)
    if p.returncode != 0:
        obs["error"] = "relabel failed"
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0

    # ---- 2. harness environment: isolated home, the checkout's installer, this state's wheel
    henv = dict(os.environ)
    henv.update({
        "OUT": str(out_dir), "SOURCE_ROOT": str(checkout), "PR_WHEEL_DIR": str(wheel_dir),
        "PR_PIN_BASE": args.pin_base, "BISECT_DIR": str(harness), "GEN_PINS": str(work / "pins_gen"),
        "EXPECT_IDEMPOTENT": "0",   # record, never abort: the base is EXPECTED to fail this
        "PY3": args.python,
    })
    if IS_WINDOWS:
        henv["USERPROFILE"] = str(home)
        henv["LOCALAPPDATA"] = str(home / "AppData" / "Local")
        henv["APPDATA"] = str(home / "AppData" / "Roaming")
        henv["HOME"] = str(home)
        Path(henv["LOCALAPPDATA"]).mkdir(parents = True, exist_ok = True)
        Path(henv["APPDATA"]).mkdir(parents = True, exist_ok = True)
        # run_step.ps1 calls the bare `python`; make it the harness python.
        henv["PATH"] = str(Path(args.python).parent) + os.pathsep + henv.get("PATH", "")
        base_cmd = ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                    "-File", str(harness / "run_step.ps1")]

        def step(argv: list[str], timeout: int) -> dict:
            return _run(base_cmd + argv, cwd = home, env = henv, log = probe_log, timeout = timeout)

        install_argv = ["-Cmd", "install", "-N", "pr"] + (["-NoTorch"] if args.no_torch else [])
        settle_argv = ["-Cmd", "noop-update", "-Label", "settle", "-Role", "settle"]
        noop_argv = ["-Cmd", "noop-update", "-Label", "second", "-Role", "noop"]
        offline_argv = ["-Cmd", "noop-update", "-Label", "third", "-Offline", "-Role", "offline"]
    else:
        henv["RUN_ROOT"] = str(work / "r")
        (work / "r").mkdir(exist_ok = True)
        home = work / "r" / "home"       # run_step.sh puts the product's HOME here
        base_cmd = ["bash", str(harness / "run_step.sh")]

        def step(argv: list[str], timeout: int) -> dict:
            return _run(base_cmd + argv, cwd = work, env = henv, log = probe_log, timeout = timeout)

        install_argv = ["install", "pr"] + (["--no-torch"] if args.no_torch else [])
        settle_argv = ["noop-update", "settle", "--role", "settle"]
        noop_argv = ["noop-update", "second", "--role", "noop"]
        offline_argv = ["noop-update", "third", "--offline", "--role", "offline"]

    venv, py = _venv_paths(home)

    # ---- 3. fresh install
    obs["steps"]["install"] = step(install_argv, args.install_timeout)
    obs["install"] = _step_summary(out_dir, "install-pr")
    obs["venv_python"] = str(py)
    obs["torch_after_install"] = _torch_facts(py)
    obs["torch_record_mtime_after_install"] = _torch_record_mtime(venv)
    obs["amd_escape_after_install"] = _amd_escape(venv, py)
    if obs["install"].get("exit_code") != 0 or not py.is_file():
        obs["error"] = "install failed"
        obs["log_tail"] = _tail(probe_log)
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0

    # ---- 4. three no-op updates on the desktop path
    obs["steps"]["settle"] = step(settle_argv, args.update_timeout)
    obs["settle"] = _step_summary(out_dir, "noop-update-settle")
    obs["torch_record_mtime_after_settle"] = _torch_record_mtime(venv)

    obs["steps"]["noop"] = step(noop_argv, args.update_timeout)
    obs["noop"] = _step_summary(out_dir, "noop-update-second")
    obs["torch_record_mtime_after_noop"] = _torch_record_mtime(venv)
    obs["torch_after_noop"] = _torch_facts(py)
    obs["amd_escape_after_noop"] = _amd_escape(venv, py)

    obs["steps"]["offline"] = step(offline_argv, args.update_timeout)
    off_name = "noop-update-third-offline" if (out_dir / "noop-update-third-offline").is_dir() else "noop-update-third"
    obs["offline"] = _step_summary(out_dir, off_name)
    obs["torch_record_mtime_after_offline"] = _torch_record_mtime(venv)
    obs["torch_after_offline"] = _torch_facts(py)

    try:
        obs["venv_bytes"] = sum(f.stat().st_size for f in venv.rglob("*") if f.is_file())
    except OSError:
        pass
    _keep_harness_output(out_dir, args.out.parent / f"harness_{args.state}")
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


def _keep_harness_output(out_dir: Path, dest: Path) -> None:
    """Copy the harness's per-step evidence (logs, summaries, proxy records, snapshots)
    next to the observation so the artifact carries the WHY, not only the numbers.
    Snapshot payloads can be large; only the small evidence files travel."""
    keep = {"summary.json", "log.txt", "proxy.jsonl", "steps.json", "idempotency.json",
            "extra.json", "state_before.json", "state_after.json", "diff.json", "pr_wheel.json"}
    try:
        for f in out_dir.rglob("*"):
            if f.is_file() and f.name in keep:
                rel = f.relative_to(out_dir)
                (dest / rel).parent.mkdir(parents = True, exist_ok = True)
                shutil.copy2(f, dest / rel)
    except OSError:
        pass


if __name__ == "__main__":
    sys.exit(main())
