#!/usr/bin/env python3
"""Probe: concurrent Studio image clients, base vs head, on this AMD GPU (observes only; judged by
criteria/imgconc_bench.py).

Head state only (base and merge record the environment): ABBA arms, one fresh process each, running
probes/imgconc/driver.py against the base checkout (sibling ``base``) and the head checkout. Each arm loads the model
through Studio's DiffusionBackend, serves the real image routes in process, and drives N clients, each bound to its
own managed account. Every arm's row goes into the observation as ``rows``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLANS = {
    # Prove the path end to end: one pass per arm, small images.
    "smoke": {"order": ["base", "head"], "clients": 4, "requests": 1, "size": 512},
    # ABBA, so each arm has an A/A pair.
    "full": {"order": ["base", "head", "head", "base"], "clients": 4, "requests": 4, "size": 768},
    # One ticket: the smoke first (into smoke_rows), then the full ABBA only if the smoke batched at the head and
    # every arm finished, at 1024 px when the smoke's timing projects it inside the budget, else 768 px.
    "staged": {"order": ["base", "head", "head", "base"], "clients": 4, "requests": 4, "size": 0},
}
FULL_BUDGET_S = 110 * 60


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--plan", default = "staged", choices = sorted(PLANS))
    ap.add_argument("--size", type = int, default = 0)
    ap.add_argument("--requests", type = int, default = 0)
    ap.add_argument("--repo", default = "black-forest-labs/FLUX.2-klein-4B")
    ap.add_argument("--family", default = "flux.2-klein")
    ap.add_argument("--account-role", default = "user", choices = ["user", "owner"])
    ap.add_argument("--arm-timeout", type = int, default = 60 * 60)
    a = ap.parse_args()
    out = Path(a.out).resolve()
    checkout = Path(a.checkout).resolve()
    out.parent.mkdir(parents = True, exist_ok = True)
    obs: dict = {"state": a.state, "python": sys.executable, "plan": a.plan}
    try:
        import torch
        obs["torch"] = torch.__version__
        obs["hip"] = getattr(torch.version, "hip", None)
        obs["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        obs["arch"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None) if obs["gpu"] else None
    except Exception as exc:  # noqa: BLE001
        obs["env_error"] = repr(exc)[:500]
    out.write_text(json.dumps(obs, indent = 1), encoding = "utf-8")
    if a.state != "head":
        return 0
    base = checkout.parent / "base"
    if not (base / "studio").is_dir():
        obs["driver_error"] = f"no base checkout at {base}"
        out.write_text(json.dumps(obs, indent = 1), encoding = "utf-8")
        return 0
    # lpips for the batched-vs-solo check; best effort, the pixel diff stands without it.
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "lpips"], capture_output = True,
                       text = True, timeout = 600)
    obs["lpips_install_rc"] = r.returncode
    # Under the uploaded out dir, so each arm's log reaches the artifact.
    work = out.parent / "imgconc"
    work.mkdir(parents = True, exist_ok = True)
    if a.plan != "staged":
        plan = PLANS[a.plan]
        size, requests = a.size or plan["size"], a.requests or plan["requests"]
        obs.update(size = size, requests = requests, clients = plan["clients"], order = plan["order"])
        _arms(a, obs, "rows", plan["order"], plan["clients"], requests, size, base, checkout, work, out, "")
        return 0
    smoke = PLANS["smoke"]
    srows = _arms(a, obs, "smoke_rows", smoke["order"], smoke["clients"], smoke["requests"], smoke["size"], base,
                  checkout, work, out, "smoke_")
    ok = len(srows) == 2 and all(r.get("rc") == 0 and not r.get("error") for r in srows)
    head = next((r for r in srows if r["arm"] == "head"), {})
    base_row = next((r for r in srows if r["arm"] == "base"), {})
    if not ok or (head.get("max_batch") or 0) < 2:
        obs["staged_stop"] = "smoke did not finish or did not batch at the head; full ABBA skipped"
        out.write_text(json.dumps(obs, indent = 1, default = str), encoding = "utf-8")
        return 0
    # Serial seconds per image at 512 px (the base arm renders one at a time), scaled by pixel count.
    per512 = (base_row.get("wall_s") or 0) / max(1, base_row.get("images") or 1)
    load = max(r.get("load_s") or 0 for r in srows)
    plan = PLANS["staged"]
    requests = a.requests or plan["requests"]
    images_per_arm = plan["clients"] * requests + 5 + 4  # timed + warm-up + solo checks
    choice = {}
    for size, scale in ((1024, 4.6), (768, 2.4)):
        choice[size] = round(len(plan["order"]) * (load + 60 + images_per_arm * per512 * scale))
    size = a.size or (1024 if choice[1024] <= FULL_BUDGET_S else 768)
    if not a.size and choice[768] > FULL_BUDGET_S:
        requests = max(1, int(requests * FULL_BUDGET_S / choice[768]))
    obs.update(size = size, requests = requests, clients = plan["clients"], order = plan["order"],
               projected_full_s = choice, smoke_s_per_image_512 = round(per512, 2))
    _arms(a, obs, "rows", plan["order"], plan["clients"], requests, size, base, checkout, work, out, "")
    return 0


def _arms(a, obs, key, order, clients, requests, size, base, checkout, work, out, prefix) -> list:
    rows = []
    for index, arm in enumerate(order):
        tree = (base if arm == "base" else checkout) / "studio" / "backend"
        label = f"{prefix}{index}_{arm}"
        row_out = work / f"{label}.json"
        cmd = [sys.executable, "-u", str(HERE / "imgconc" / "driver.py"), "--tree", str(tree), "--repo", a.repo,
               "--family", a.family, "--clients", str(clients), "--requests", str(requests),
               "--width", str(size), "--height", str(size), "--label", label, "--account-role", a.account_role,
               "--home", str(work / f"home_{label}"), "--out", str(row_out)]
        log = work / f"{label}.log"
        t0 = time.time()
        try:
            with open(log, "w", encoding = "utf-8") as fh:
                rc = subprocess.run(cmd, stdout = fh, stderr = subprocess.STDOUT, timeout = a.arm_timeout,
                                    cwd = str(work)).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
        row = json.loads(row_out.read_text(encoding = "utf-8")) if row_out.is_file() else {}
        row.update(arm = arm, order_index = index, rc = rc, arm_seconds = round(time.time() - t0, 1))
        # The coalescer's own log lines, as a cross-check of the recorded call sizes (warm-up included).
        text = log.read_text(encoding = "utf-8", errors = "replace") if log.is_file() else ""
        lines: dict = {}
        for m in re.finditer(r"images\.batch: (\d+) requests from (\d+) accounts render as one call of (\d+) images",
                             text):
            lines[m.group(3)] = lines.get(m.group(3), 0) + 1
        row["batch_log_lines"] = lines
        if rc != 0:
            row["log_tail"] = log.read_text(encoding = "utf-8", errors = "replace")[-4000:]
        rows.append(row)
        obs[key] = rows
        out.write_text(json.dumps(obs, indent = 1, default = str), encoding = "utf-8")
    return rows


if __name__ == "__main__":
    sys.exit(main())
