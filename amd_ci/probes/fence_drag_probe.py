#!/usr/bin/env python3
"""Probe: does a scrollbar-thumb drag over a long code fence stall or crash the browser, per fence-mode arm?

For one state (a checkout): builds the Studio frontend twice, as checked out and with the scrollbar guard
from unslothai/unsloth#13919's review applied, serves each through the checkout's own backend (run.py
--frontend), and runs probes/fence_drag_cell.py against them in Microsoft Edge.
Base arms: default, guard. Head arms: default, window override, guard plus window override. Observes only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
NODE_ZIP = "https://nodejs.org/dist/v24.14.0/node-v24.14.0-win-x64.zip"
PASSWORD = "fdbench-Passw0rd!"


def log(msg: str) -> None:
    print(f"[fence_drag_probe] {msg}", flush = True)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def node_dir(work: Path) -> Path | None:
    found = shutil.which("node")
    if found:
        return Path(found).parent
    if os.name != "nt":
        return None
    dest = work / "node"
    exe = next(dest.glob("node-*/node.exe"), None) if dest.exists() else None
    if exe is None:
        dest.mkdir(parents = True, exist_ok = True)
        zpath = dest / "node.zip"
        urllib.request.urlretrieve(NODE_ZIP, zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(dest)
        exe = next(dest.glob("node-*/node.exe"))
    return exe.parent


def run(cmd: list[str], cwd: Path, env: dict, logf: Path, timeout: int) -> int:
    with open(logf, "a", encoding = "utf-8") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n")
        fh.flush()
        try:
            return subprocess.run(cmd, cwd = cwd, env = env, stdout = fh, stderr = subprocess.STDOUT,
                                  timeout = timeout).returncode
        except subprocess.TimeoutExpired:
            return -9


def build(front: Path, out: Path, env: dict, npm: str, npx: str, logf: Path) -> str | None:
    if not (front / "node_modules").exists():
        if run([npm, "ci", "--no-audit", "--no-fund"], front, env, logf, 1800) != 0:
            return "npm ci failed"
    if run([npx, "vite", "build", "--outDir", str(out), "--emptyOutDir"], front, env, logf, 1800) != 0:
        return "vite build failed"
    return None if (out / "index.html").exists() else "no index.html"


def serve(backend: Path, dist: Path, python: str, home: Path, logf: Path) -> tuple[subprocess.Popen, str]:
    shutil.rmtree(home, ignore_errors = True)
    home.mkdir(parents = True)
    port = free_port()
    env = {**os.environ, "UNSLOTH_STUDIO_HOME": str(home), "UNSLOTH_DISABLE_AUTO_UPDATES": "1",
           "UNSLOTH_DISABLE_SANDBOX_WARMUP": "1", "PYTHONIOENCODING": "utf-8"}
    fh = open(logf, "w", encoding = "utf-8")
    proc = subprocess.Popen([python, "run.py", "--host", "127.0.0.1", "--port", str(port), "--frontend", str(dist),
                             "--password", PASSWORD], cwd = backend, env = env, stdout = fh, stderr = subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 300
    while time.time() < deadline and proc.poll() is None:
        try:
            with urllib.request.urlopen(f"{base}/api/health", timeout = 5) as r:
                if r.status == 200:
                    return proc, base
        except Exception:  # noqa: BLE001
            time.sleep(2)
    return proc, ""


def stop(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output = True)
        else:
            proc.kill()
        try:
            proc.wait(timeout = 30)
        except Exception:  # noqa: BLE001
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--reps", type = int, default = 2)
    ap.add_argument("--stress", type = int, default = 15)
    args = ap.parse_args()

    checkout = Path(args.checkout)
    front, backend = checkout / "studio" / "frontend", checkout / "studio" / "backend"
    work = Path(os.environ.get("AMD_CI_WORK") or tempfile.gettempdir()) / f"fd_{args.state}"
    work.mkdir(parents = True, exist_ok = True)
    obs: dict = {"state": args.state, "arms": {}, "builds": {}, "channels": []}

    def done() -> int:
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0

    nd = node_dir(work)
    if nd is None:
        obs["error"] = "no node"
        return done()
    env = {**os.environ, "PATH": str(nd) + os.pathsep + os.environ.get("PATH", "")}
    npm = str(nd / ("npm.cmd" if os.name == "nt" else "npm"))
    npx = str(nd / ("npx.cmd" if os.name == "nt" else "npx"))
    obs["node"] = subprocess.run([str(nd / ("node.exe" if os.name == "nt" else "node")), "--version"],
                                 capture_output = True, text = True).stdout.strip()
    blog = work / "build.log"
    dists = {"plain": work / "dist_plain", "guard": work / "dist_guard"}
    err = build(front, dists["plain"], env, npm, npx, blog)
    if not err:
        patch = HERE / "fence_drag_guard.patch"
        # A Windows checkout can carry CRLF line ends, which the LF patch's context lines do not.
        flags = ["--ignore-whitespace", "--whitespace=nowarn"]
        applied = subprocess.run(["git", "apply", *flags, str(patch)], cwd = checkout, capture_output = True, text = True)
        if applied.returncode != 0:
            err = "guard patch did not apply: " + (applied.stderr or "")[-500:]
        else:
            err = build(front, dists["guard"], env, npm, npx, blog)
            subprocess.run(["git", "apply", "-R", *flags, str(patch)], cwd = checkout, capture_output = True)
    obs["builds"]["error"] = err
    obs["build_log_tail"] = blog.read_text(encoding = "utf-8", errors = "replace")[-1500:] if blog.exists() else ""
    if err:
        obs["error"] = err
        obs["build_log_tail"] = blog.read_text(encoding = "utf-8", errors = "replace")[-3000:]
        return done()

    subprocess.run([args.python, "-m", "pip", "install", "-q", "playwright<2"], capture_output = True)
    if args.state == "base":
        arms = [("default", "plain", None), ("guard", "guard", None)]
    else:
        arms = [("default", "plain", None), ("window", "plain", "window"), ("guard_window", "guard", "window")]

    servers = {}
    for dist in sorted({a[1] for a in arms}):
        proc, base = serve(backend, dists[dist], args.python, work / f"home_{dist}", work / f"serve_{dist}.log")
        servers[dist] = (proc, base)
        if not base:
            obs["error"] = f"backend for {dist} never became healthy"
            obs["serve_log_tail"] = (work / f"serve_{dist}.log").read_text(encoding = "utf-8", errors = "replace")[-3000:]
    try:
        if "error" not in obs:
            channels = ["msedge"] if os.name == "nt" else ["chromium"]
            for channel in channels:
                rows_path = work / f"rows_{channel}.jsonl"
                rows_path.unlink(missing_ok = True)
                cmd = [args.python, str(HERE / "fence_drag_cell.py"), "--reps", str(args.reps), "--throttle", "1",
                       "--throttle", "4", "--stress", str(args.stress), "--channel", channel, "--out", str(rows_path)]
                for label, dist, mode in arms:
                    cmd += ["--arm", f"{label}={servers[dist][1]}" + (f"@{mode}" if mode else "")]
                clog = work / f"cell_{channel}.log"
                rc = run(cmd, work, {**os.environ, "PYTHONIOENCODING": "utf-8"}, clog, 7200)
                rows = [json.loads(x) for x in rows_path.read_text(encoding = "utf-8").splitlines()] if rows_path.exists() else []
                obs["channels"].append({"channel": channel, "rc": rc, "rows": rows,
                                        "log_tail": clog.read_text(encoding = "utf-8", errors = "replace")[-2500:]})
    finally:
        for proc, _ in servers.values():
            stop(proc)
    return done()


if __name__ == "__main__":
    sys.exit(main())
