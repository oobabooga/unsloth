#!/usr/bin/env python3
"""Probe: PR 9477 piece 2, base vs head, at rung 500K, in real WebKitGTK on the gfx1151.

Observes only. criteria/studio_9477_piece2.py judges and enforces the VOID rule.

Why this run exists. Piece 2 was measured twice in headless Chromium and found no benefit, but
that venue sat at 1-2% busy and could not load the main thread, so "no benefit" there was not a
statement about the change. At r100K on THIS host the base arm does not exhibit the defect either
(25-27% busy, 209 ms worst frame), so a differential there would also be VOID. At r500K the base
arm DOES exhibit it: 2.4 fps at 94% busy on scroll, 877-910 ms worst frame at 86% busy on the
reasoning toggle. That is the one rung where the question can be asked.

**How the two states are built, and why not from branches.** Both states are the SAME upstream
commit, 0e1968dc, which is piece 2's merge base and is in unslothai/unsloth history. Head is that
checkout plus `studio_ladder/pr9477_piece2.patch`, which is `git diff base..62564233c`. So the
only difference between the arms is the patch, by construction rather than by hoping two branches
had not drifted. Pushing piece 2 as a branch to the fork was the obvious alternative and is
wrong: a full code branch carries its own .github/workflows and pushing it fires every one of
them on the fork.

Both states are installed and measured in ONE job, so this takes a single hold on the
amd-ci-gfx1151-gpu group rather than racing anything else on the box.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LADDER = HERE / "studio_ladder"
PATCH = LADDER / "pr9477_piece2.patch"
MERGE_BASE = "0e1968dc61b2692ec7718e84044cc87f9ed2d68d"

sys.path.insert(0, str(HERE))
from webkit_paint_probe import fetch_xvfb, inventory, start_xserver  # noqa: E402
from studio_ladder_probe import clone, gi_python, sh  # noqa: E402


def build_state(work: Path, name: str, repo_url: str, ref: str, patch: Path | None,
                install_timeout: int) -> dict:
    """One checkout, optionally patched, installed. Returns what it built."""
    root = work / f"state_{name}"
    out: dict = {"name": name, "ref": ref, "patched": bool(patch)}
    out.update(clone(repo_url, ref, root))
    # `clone` checks out `ref` only when it is not "main"; force it either way so the recorded
    # commit is the one that was measured.
    out["checkout"] = sh(["git", "checkout", "--detach", ref], cwd = str(root), timeout = 300)
    out["commit"] = (sh(["git", "rev-parse", "HEAD"], cwd = str(root),
                        timeout = 60).get("stdout") or "").strip()

    if patch is not None:
        # --check first: a patch that half-applies would produce an arm that is neither base nor
        # head, and it would still install and still measure.
        out["patch_check"] = sh(["git", "apply", "--check", "-v", str(patch)],
                                cwd = str(root), timeout = 300)
        out["patch_apply"] = sh(["git", "apply", "--stat", "--apply", str(patch)],
                                cwd = str(root), timeout = 300)
        out["patch_ok"] = out["patch_apply"].get("rc") == 0
        out["dirty_files"] = len([l for l in (sh(["git", "status", "--porcelain"],
                                                 cwd = str(root), timeout = 60)
                                              .get("stdout") or "").splitlines() if l.strip()])
    else:
        out["patch_ok"] = True
        out["dirty_files"] = 0

    home = work / f"home_{name}"
    home.mkdir(parents = True, exist_ok = True)
    t0 = time.time()
    inst = sh(["bash", "install.sh", "--local"], cwd = str(root), timeout = install_timeout,
              env = {"UNSLOTH_STUDIO_HOME": str(home)})
    inst["seconds"] = round(time.time() - t0, 1)
    inst["stdout"] = (inst.get("stdout") or "")[-3000:]
    out["install"] = inst

    dist = root / "studio" / "frontend" / "dist"
    out["dist"] = {"path": str(dist), "exists": dist.is_dir(),
                   "index_html": (dist / "index.html").is_file(),
                   "asset_files": len(list((dist / "assets").rglob("*")))
                   if (dist / "assets").is_dir() else 0}
    binp = None
    for c in [home / "bin" / "unsloth", *sorted(home.glob(".venv*/bin/unsloth"))]:
        if c.exists() and os.access(c, os.X_OK):
            binp = str(c)
            break
    out["unsloth_bin"] = binp
    out["home"] = str(home)
    out["repo_root"] = str(root)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--state", default = "host")
    ap.add_argument("--checkout", default = "")
    ap.add_argument("--work", default = os.environ.get("AMD_CI_WORK", "/tmp/studio_9477"))
    ap.add_argument("--repo", default = "https://github.com/unslothai/unsloth")
    ap.add_argument("--rung", default = "500K")
    ap.add_argument("--reps", type = int, default = 2)
    ap.add_argument("--first-port", type = int, default = 5491)
    ap.add_argument("--install-timeout", type = int, default = 3600)
    ap.add_argument("--rung-timeout", type = int, default = 2400)
    args = ap.parse_args()
    args.out.parent.mkdir(parents = True, exist_ok = True)

    work = Path(args.work) / "p9477"
    work.mkdir(parents = True, exist_ok = True)
    obs: dict = {"state": args.state, "rung": args.rung, "reps": args.reps,
                 "merge_base": MERGE_BASE, "patch": str(PATCH),
                 "patch_bytes": PATCH.stat().st_size if PATCH.is_file() else None}

    obs["inventory"] = inventory()
    if not obs["inventory"]["Xvfb"]:
        obs["fetch_xvfb"] = fetch_xvfb(work)
    xproc, xinfo = start_xserver(work, obs)
    obs["xserver"] = xinfo
    if not xinfo.get("display"):
        obs["fatal"] = "no display server could be started"
        args.out.write_text(json.dumps(obs, indent = 2))
        return 0

    py_gi, tried = gi_python()
    obs["gi_python"] = {"chosen": py_gi, "tried": tried}
    try:
        if py_gi is None:
            obs["fatal"] = "no python on this host can import gi + WebKit2 4.1"
            return 0
        if not PATCH.is_file():
            obs["fatal"] = f"the piece 2 patch is missing at {PATCH}"
            return 0

        obs["states"] = {}
        for name, patch in (("base", None), ("head", PATCH)):
            obs["states"][name] = build_state(work, name, args.repo, MERGE_BASE, patch,
                                              args.install_timeout)

        bad = [n for n, st in obs["states"].items()
               if not st["dist"]["exists"] or not st["unsloth_bin"] or not st["patch_ok"]]
        if bad:
            obs["fatal"] = f"states did not build: {bad}"
            return 0

        # The bundle hashes MUST differ. Two arms that built byte-identical frontends are one
        # arm measured twice, and it would read as a perfect null.
        obs["runs"] = []
        port = args.first_port
        # Interleaved, not blocked: base, head, base, head. A thermal or scheduling drift over
        # the job blocks would otherwise land entirely on whichever arm ran second.
        order = [(name, rep) for rep in range(1, args.reps + 1) for name in ("base", "head")]
        for name, rep in order:
            st = obs["states"][name]
            rhome = work / f"run_{name}_r{rep}"
            if rhome.exists():
                shutil.rmtree(rhome, ignore_errors = True)
            rhome.mkdir(parents = True, exist_ok = True)
            src_home = Path(st["home"])
            for d in ("assets", "bin", "cache", "compiled_cache", "llama.cpp", "share",
                      "unsloth_studio", "whisper.cpp"):
                s = src_home / d
                if s.exists() and not (rhome / d).exists():
                    os.symlink(s, rhome / d)
            for d in ("exports", "outputs", "logs", "runs", "rag", "auth"):
                (rhome / d).mkdir(parents = True, exist_ok = True)

            outp = work / "out" / f"{name}_rep{rep}.json"
            outp.parent.mkdir(parents = True, exist_ok = True)
            cmd = [sys.executable, str(LADDER / "amdv_rung_bench.py"),
                   "--rung", args.rung, "--rep", f"{name}{rep}",
                   "--dist", st["dist"]["path"], "--home", str(rhome),
                   "--port", str(port), "--display", xinfo["display"],
                   "--sb-root", st["repo_root"], "--unsloth-bin", st["unsloth_bin"],
                   "--python-gi", py_gi,
                   "--scene", str(LADDER / "amdv_scene.js"),
                   "--driver", str(LADDER / "amdv_drive.py"),
                   "--frame-clock", "updating",
                   "--out", str(outp)]
            t0 = time.time()
            r = sh(cmd, timeout = args.rung_timeout, env = {"UNSLOTH_WORKSPACE": str(work)})
            entry = {"arm": name, "rep": rep, "port": port,
                     "seconds": round(time.time() - t0, 1), "rc": r.get("rc"),
                     "error": r.get("error"),
                     "stdout_tail": "\n".join((r.get("stdout") or "").splitlines()[-25:]),
                     "stderr_tail": (r.get("stderr") or "")[-2000:]}
            if outp.is_file():
                try:
                    entry["payload"] = json.loads(outp.read_text())
                except Exception as e:  # noqa: BLE001
                    entry["payload_error"] = f"{type(e).__name__}: {e}"
            obs["runs"].append(entry)
            port += 1
            time.sleep(10)

        logs = work / "out" / "logs"
        logs.mkdir(parents = True, exist_ok = True)
        got = []
        for pat in ("*.log", "*.jsonl"):
            for f in list((work / "out").glob(pat)) + list(work.rglob("logs/" + pat)):
                try:
                    if f.parent == logs:
                        continue
                    dest = logs / (f.parent.name + "__" + f.name)
                    shutil.copy2(f, dest)
                    got.append(str(dest))
                except Exception:  # noqa: BLE001
                    pass
        obs["logs_collected"] = got
        return 0
    finally:
        if xproc is not None and xproc.poll() is None:
            try:
                os.kill(xproc.pid, signal.SIGTERM)
                time.sleep(2)
                if xproc.poll() is None:
                    os.kill(xproc.pid, signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass
            obs["xserver"]["stopped_pid"] = xproc.pid
        args.out.write_text(json.dumps(obs, indent = 2))


if __name__ == "__main__":
    sys.exit(main())
