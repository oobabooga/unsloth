#!/usr/bin/env python3
"""Probe: are Wan2.2-TI2V-5B renders through Studio bit-reproducible across fresh processes on gfx1151, and if not,
where do processes first diverge and which switch removes it? Observes only; judged by criteria/wanrep.py.

Each arm is a fresh process (probes/wanrep/arm.py): one prompt + seed rendered ``--renders`` times (within-process
repeat), 832x480x17, 8 steps, INT8 attention off, every stage hashed. Arms come from ``--arms`` (JSON list or a
preset name). Per arm env overrides; ``ind`` names the arm whose TORCHINDUCTOR_CACHE_DIR it reuses (default: a fresh
one). Studio home (flash verdict cache, TunableOp tables) is shared by every arm of every state, as in a real install
(base and head run the same attention verdicts).
Public repos only, no token. Runs at the states listed in ``--states`` (default head only)."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOKEN_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_HUB_TOKEN")
OFF = {"int8_attention": "off"}
PRESETS = {
    "diagnose": [
        {"tag": "D1"}, {"tag": "D2"}, {"tag": "D3"},
        {"tag": "S1", "ind": "D1"},
        {"tag": "F1", "env": {"TORCHINDUCTOR_FORCE_FILTER_REDUCTION_CONFIGS": "1"}},
        {"tag": "F2", "env": {"TORCHINDUCTOR_FORCE_FILTER_REDUCTION_CONFIGS": "1"}},
        {"tag": "X1", "env": {"TORCHINDUCTOR_DETERMINISTIC": "1"}},
        {"tag": "X2", "env": {"TORCHINDUCTOR_DETERMINISTIC": "1"}},
        {"tag": "C1", "env": {"UNSLOTH_DIFFUSION_ROCM_COMPILE": "0"}, "block_hooks": True},
        {"tag": "C2", "env": {"UNSLOTH_DIFFUSION_ROCM_COMPILE": "0"}, "block_hooks": True},
        {"tag": "N1", "env": {"UNSLOTH_DIFFUSION_ROCM_FLASH": "0"}},
        {"tag": "N2", "env": {"UNSLOTH_DIFFUSION_ROCM_FLASH": "0"}},
        {"tag": "B1", "env": {"UNSLOTH_DIFFUSION_BG_COMPILE": "0"}},
        {"tag": "B2", "env": {"UNSLOTH_DIFFUSION_BG_COMPILE": "0"}},
        {"tag": "T1", "env": {"UNSLOTH_DIFFUSION_ROCM_TUNABLEOP": "0"}},
        {"tag": "T2", "env": {"UNSLOTH_DIFFUSION_ROCM_TUNABLEOP": "0"}},
    ],
    # base vs head proof: 4 fresh processes per config, INT8 attention off (P) and the Studio default Auto (A)
    "proof": [x for i in (1, 2, 3, 4) for x in ({"tag": f"P{i}"}, {"tag": f"A{i}", "auto": True})],
}


def _load(p: Path):
    import torch

    return torch.load(p, map_location = "cpu") if p.is_file() else None


def first_divergence(ref: list, other: list):
    """First entry of ref's log (render, stage, index, key) whose hash differs in (or is missing from) other."""
    om = {(r, s, i, k): h for r, s, i, k, h in other}
    for r, s, i, k, h in ref:
        o = om.get((r, s, i, k))
        if o != h:
            return {"render": r, "stage": s, "index": i, "key": k, "missing": o is None}
    if len(other) != len(ref):
        return {"length": [len(ref), len(other)]}
    return None


def magnitudes(tref: Path, tother: Path, ra: int, rb: int) -> dict:
    """Per-step rel L2 of the scheduler's model_output and sample, VAE input, frame MAD/PSNR: ref render ra vs rb."""
    import numpy as np
    import torch

    out: dict = {"steps": []}
    for i in range(64):
        a, b = _load(tref / f"r{ra}_s{i}.pt"), _load(tother / f"r{rb}_s{i}.pt")
        if a is None or b is None:
            break
        row = {}
        for k in ("model_output", "sample"):
            x, y = a[k], b[k]
            if x.shape != y.shape:
                row[k] = "shape"
                continue
            row[k] = float((x - y).norm() / x.norm().clamp_min(1e-30))
        out["steps"].append(row)
    a, b = _load(tref / f"r{ra}_vae_in_0.pt"), _load(tother / f"r{rb}_vae_in_0.pt")
    if a is not None and b is not None and a.shape == b.shape:
        out["vae_in_rel"] = float((a - b).norm() / a.norm().clamp_min(1e-30))
    fa, fb = tref / f"r{ra}_frames.npy", tother / f"r{rb}_frames.npy"
    if fa.is_file() and fb.is_file():
        x, y = np.load(fa).astype(np.float64), np.load(fb).astype(np.float64)
        if x.shape == y.shape:
            mse = float(((x - y) ** 2).mean())
            out["frames_mad"] = float(np.abs(x - y).mean())
            out["frames_max"] = float(np.abs(x - y).max())
            out["frames_psnr"] = None if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))
    return out


def run_state(a, checkout: Path, out: Path) -> dict:
    arms = PRESETS[a.arms] if a.arms in PRESETS else json.loads(a.arms)
    root = out.resolve().parents[2] / f"wanrep_root_{a.state}"
    root.mkdir(parents = True, exist_ok = True)
    tens = root / "tensors"
    arms_out = out.parent / f"wanrep_arms_{a.state}"
    arms_out.mkdir(parents = True, exist_ok = True)
    hf = Path(os.environ.get("WANREP_HF_HOME") or (out.resolve().parents[2] / "wanrep_hf"))
    base_env = {k: v for k, v in os.environ.items() if k not in TOKEN_VARS + ("PYTHONPATH",)}
    base_env.update({"HF_HOME": str(hf), "UNSLOTH_STUDIO_HOME": str(out.resolve().parents[2] / "wanrep_studio_home"), "PYTHONUNBUFFERED": "1",
                     "UNSLOTH_DISABLE_AUTO_UPDATES": "1", "HF_XET_HIGH_PERFORMANCE": "1"})
    base_env.pop("UNSLOTH_DIFFUSION_INT8_ATTENTION_KERNEL", None)
    backend = checkout.resolve() / "studio" / "backend"
    res: dict = {"state": a.state, "arms": {}, "order": [x["tag"] for x in arms]}
    for arm in arms:
        tag = arm["tag"]
        env = dict(base_env)
        env.update({k: str(v) for k, v in (arm.get("env") or {}).items()})
        ind = root / "ind" / (arm.get("ind") or tag)
        if not arm.get("ind"):
            shutil.rmtree(ind, ignore_errors = True)
        env["TORCHINDUCTOR_CACHE_DIR"] = str(ind)
        load = {**({} if arm.get("auto") else OFF), **(arm.get("load") or {})}
        cmd = [sys.executable, "-u", str(HERE / "wanrep" / "arm.py"), "--root", str(backend), "--tag", tag,
               "--out", str(arms_out), "--tensors", str(tens), "--load", json.dumps(load),
               "--gen", json.dumps({**json.loads(a.gen), **(arm.get("gen") or {})}), "--renders", str(a.renders)]
        if arm.get("block_hooks"):
            cmd.append("--block-hooks")
        if arm.get("det_algos"):
            cmd.append("--deterministic-algos")
        t0 = time.time()
        logp = arms_out / f"{tag}.log"
        with open(logp, "w", encoding = "utf-8") as fh:
            try:
                rc = subprocess.run(cmd, stdout = fh, stderr = subprocess.STDOUT, env = env, timeout = a.arm_timeout,
                                    cwd = str(root)).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        rp = arms_out / tag / "record.json"
        r = json.loads(rp.read_text(encoding = "utf-8")) if rp.is_file() else {"verdict": "missing"}
        r["rc"], r["arm_s"], r["spec"] = rc, round(time.time() - t0, 1), arm
        res["arms"][tag] = r
        print(f"ARM {tag} rc={rc} {r.get('verdict')} {r['arm_s']}s renders={[x.get('frames_sha') for x in r.get('renders', [])]}",
              flush = True)
    # comparisons: every arm vs the first arm of its group (same tag letter) and vs the first arm overall; in-arm repeat
    comp: dict = {}
    tags = [t for t in res["order"] if res["arms"][t].get("verdict") == "ok"]
    first_of: dict = {}
    for t in tags:
        first_of.setdefault(t[0], t)
    pairs = set()
    for t in tags:
        pairs.add((t, t, 0, 1))
        if first_of[t[0]] != t:
            pairs.add((first_of[t[0]], t, 0, 0))
            pairs.add((first_of[t[0]], t, 1, 1))
        if tags and t != tags[0]:
            pairs.add((tags[0], t, 1, 1))
    for x, y, ra, rb in sorted(pairs):
        if x == y and a.renders < 2:
            continue
        lx = [e for e in res["arms"][x].get("log", []) if e[0] == ra]
        ly = [[ra] + e[1:] for e in res["arms"][y].get("log", []) if e[0] == rb]
        d = first_divergence(lx, ly)
        try:
            mag = magnitudes(tens / x, tens / y, ra, rb)
        except Exception as exc:  # noqa: BLE001
            mag = {"error": repr(exc)[:300]}
        fx = (res["arms"][x]["renders"][ra] if len(res["arms"][x].get("renders", [])) > ra else {}).get("frames_sha")
        fy = (res["arms"][y]["renders"][rb] if len(res["arms"][y].get("renders", [])) > rb else {}).get("frames_sha")
        comp[f"{x}r{ra}|{y}r{rb}"] = {"identical": fx is not None and fx == fy, "first_divergence": d, **mag}
    res["comparisons"] = comp
    # Keep the logs in the observation only in summarised form (the full log stays in record.json artifacts).
    for t, r in res["arms"].items():
        r.pop("log", None)
        if isinstance(r.get("inductor_best_configs"), dict):
            r["n_best_configs"] = len(r["inductor_best_configs"])
    # differing inductor picks between arms that compiled the same kernels
    picks = {t: res["arms"][t].get("inductor_best_configs") or {} for t in tags}
    diff: dict = {}
    for t in tags:
        if t == tags[0]:
            continue
        p0, p1 = picks[tags[0]], picks[t]
        common = set(p0) & set(p1)
        diff[f"{tags[0]}|{t}"] = {"common": len(common),
                                  "differ": sorted(k for k in common if p0[k] != p1[k])[:50]}
    res["inductor_pick_diff"] = diff
    groups: dict = {}
    for t in res["order"]:
        r = res["arms"][t]
        g = groups.setdefault(t[0], {"tags": [], "frames": [], "steady_s": [], "first_s": [], "pins": [], "ok": []})
        rs = r.get("renders") or []
        g["tags"].append(t)
        g["ok"].append(r.get("verdict") == "ok" and len(rs) == a.renders)
        g["frames"].append(sorted({x.get("frames_sha") for x in rs}) if rs else None)
        g["first_s"].append(rs[0].get("wall_s") if rs else None)
        g["steady_s"].append([x.get("wall_s") for x in rs[1:]])
        g["pins"].append(r.get("reduction_pins"))
    res["groups"] = groups
    shutil.rmtree(tens, ignore_errors = True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--states", default = "head")
    ap.add_argument("--arms", default = "diagnose")
    ap.add_argument("--gen", default = "{}")
    ap.add_argument("--renders", type = int, default = 2)
    ap.add_argument("--arm-timeout", type = int, default = 2400)
    a = ap.parse_args()
    out = Path(a.out)
    if a.state not in a.states.split(","):
        out.write_text(json.dumps({"state": a.state, "skipped": f"measured at {a.states} only"}), encoding = "utf-8")
        return 0
    for k in TOKEN_VARS:
        os.environ.pop(k, None)
    res = run_state(a, Path(a.checkout), out)
    out.write_text(json.dumps(res, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
