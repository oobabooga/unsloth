#!/usr/bin/env python3
"""Probe: round-5 A1/A2 gfx1151 measurements (observes only; criteria/a12_measure.py judges nothing between states).

Runs on the head state only; arms that need main's code run from the base tree (states/base), so "studio@base" is
origin/main behaviour and the head tree adds only the opt-in K_T config key (default off).
Every arm is a fresh process; arms interleave (ABBA by round), 3 rounds = A/A spread. No token is read or passed.

  python a12_probe.py --state head --checkout <tree> --out obs.json [--items kt,hg,w8a8,lin,sdc,race,conv3d]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent / "a12"
SECRET_ENVS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN", "WANDB_API_KEY")
ROUNDS = 3


def child_env(extra = None):
    env = {k: v for k, v in os.environ.items() if k not in SECRET_ENVS and not k.startswith("GH_TOKEN")}
    env.update(extra or {})
    return env


def run(args, out: Path, log: Path, env = None, timeout = 3600) -> dict:
    t = time.time()
    with open(log, "ab") as fh:
        fh.write(f"\n=== {' '.join(map(str, args))}\n".encode())
        fh.flush()
        try:
            rc = subprocess.run([sys.executable, *map(str, args)], stdout = fh, stderr = subprocess.STDOUT,
                                env = child_env(env), timeout = timeout).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    res = {"rc": rc, "wall_s": round(time.time() - t, 1)}
    if out.is_file():
        try:
            res.update(json.loads(out.read_text(encoding = "utf-8")))
        except Exception as exc:  # noqa: BLE001
            res["parse_error"] = repr(exc)
    else:
        res["missing"] = True
    return res


def rounds(arms):
    for rnd in range(ROUNDS):
        for arm in (arms if rnd % 2 == 0 else arms[::-1]):
            yield rnd, arm


def cmp_saved(work, prefix, ref_name, names):
    """rel L2 / max abs / equal of each saved sampled-output tensor vs the reference arm's."""
    import torch
    out = {}
    ref_p = work / f"{prefix}_{ref_name}.pt"
    if not ref_p.is_file():
        return {"_missing_ref": str(ref_p)}
    ref = torch.load(ref_p)
    for n in names:
        p = work / f"{prefix}_{n}.pt"
        if p.is_file() and p != ref_p:
            x = torch.load(p)
            out[n] = {"equal": bool(torch.equal(x, ref)), "max_abs": float((x - ref).abs().max()),
                      "rel_l2": float((x - ref).norm() / ref.norm())}
    return out


def item_attn(name, impl, shapes, arms, ctx, obs):
    """arms: list of (arm label, tree backend, script arm)."""
    work, log = ctx["work"], ctx["log"]
    res = []
    for shape in shapes:
        saved = []
        for rnd, (label, tree, arm) in rounds(arms):
            o = work / f"{name}_{shape}_{label}_{rnd}.json"
            sv = f"{name}_{shape}_{label}_{rnd}"
            args = [HERE / "attn_bench.py", "--backend", tree, "--impl", impl, "--shape", shape, "--arm", arm,
                    "--out", o, "--save", work / f"{sv}.pt"]
            r = run(args, o, log, timeout = 1800)
            r.update({"round": rnd, "label": label})
            res.append(r)
            saved.append(f"{shape}_{label}_{rnd}")
            obs[name] = res
        ref = f"{shape}_{arms[0][0]}_0"
        obs.setdefault(f"{name}_cmp", {})[shape] = cmp_saved(work, name, ref, saved)
        for p in work.glob(f"{name}_{shape}_*.pt"):
            p.unlink()


def item_kt(ctx, obs):
    b, h = ctx["base"], ctx["head"]
    arms = [("studio_base", b, "studio"), ("studio_head", h, "studio"), ("kt", h, "kt"), ("kt_s2", h, "kt_s2"),
            ("s2", h, "s2"), ("bn64_s2", h, "bn64_s2"), ("kt_bn64_s2", h, "kt_bn64_s2")]
    item_attn("kt", "int8", ["h3_37720", "wan5b_27280", "klein_4608"], arms, ctx, obs)


def item_hg(ctx, obs):
    b = ctx["base"]
    item_attn("hg_int8", "int8", ["h3_37720", "wan5b_27280"],
              [("all", b, "studio")] + [(f"hg{g}", b, f"hg{g}") for g in (2, 4, 8, 16)], ctx, obs)
    item_attn("hg_sdpa", "sdpa", ["h3_37720", "wan5b_27280"],
              [("all", b, "all")] + [(f"hg{g}", b, f"hg{g}") for g in (2, 4, 8, 16)], ctx, obs)


def item_hgs(ctx, obs):
    """Head groups on Studio's own gfx1151 SDPA layout (q permuted view, k / v copied heads-major per call)."""
    b = ctx["base"]
    arms = [(x, b, x) for x in ("all", "hg2", "hg4", "hg8", "hg16", "qkvc_all", "qkvc_hg2")]
    item_attn("hgs", "sdpa_studio", ["h3_37720", "wan5b_27280"], arms, ctx, obs)
    arms = [(x, b, x) for x in ("all", "hg2", "hg4", "qkvc_hg2")]
    res = obs.pop("hgs")
    item_attn("hgs", "sdpa_studio", ["ltx_24576", "ltx_6144", "qwen21_4096"], arms, ctx, obs)
    obs["hgs"] = res + obs["hgs"]


def item_w8a8(ctx, obs):
    res = []
    for rnd in range(ROUNDS):
        o = ctx["work"] / f"w8a8_{rnd}.json"
        r = run([HERE / "w8a8_s1.py", "--backend", ctx["base"], "--out", o], o, ctx["log"], timeout = 3600)
        r["round"] = rnd
        res.append(r)
        obs["w8a8"] = res


def item_w8bo(ctx, obs):
    """n6#7: AMDGCN_USE_BUFFER_OPS=1 (Triton AMD buffer load / store ops) on Studio's W8A8 GEMM, env per process."""
    res = []
    for rnd, arm in rounds(["off", "on"]):
        o = ctx["work"] / f"w8bo_{arm}_{rnd}.json"
        env = {"AMDGCN_USE_BUFFER_OPS": "1" if arm == "on" else "0", "TRITON_CACHE_DIR": str(ctx["work"] / f"tc_{arm}")}
        r = run([HERE / "w8a8_s1.py", "--backend", ctx["base"], "--out", o, "--rounds", "5"], o, ctx["log"], env = env,
                timeout = 3600)
        r.update({"round": rnd, "arm": arm})
        res.append(r)
        obs["w8bo"] = res


def item_lin(ctx, obs):
    res = []
    for rnd in range(ROUNDS):
        o = ctx["work"] / f"lin_{rnd}.json"
        r = run([HERE / "lin_layout.py", "--out", o], o, ctx["log"], timeout = 3600)
        r["round"] = rnd
        res.append(r)
        obs["lin"] = res


def item_blk(ctx, obs):
    res = []
    for rnd in range(ROUNDS):
        o = ctx["work"] / f"blk_{rnd}.json"
        r = run([HERE / "krea_block.py", "--backend", ctx["base"], "--out", o], o, ctx["log"], timeout = 2400)
        r["round"] = rnd
        res.append(r)
        obs["blk"] = res


def item_lint(ctx, obs):
    """lin_layout under Studio's shipped gfx1151 TunableOp table (read only), Qwen-Image-2.1 / FLUX.3 klein shapes."""
    import glob
    tabs = sorted(glob.glob(str(Path(ctx["base"]) / "core" / "inference" / "rocm_gemm_tables" / "*hip713*.csv")))
    obs["lint_table"] = tabs
    if not tabs:
        return
    res = []
    for rnd in range(ROUNDS):
        for tab in ("on", "off") if rnd % 2 == 0 else ("off", "on"):
            o = ctx["work"] / f"lint_{tab}_{rnd}.json"
            args = [HERE / "lin_layout.py", "--out", o, "--models", "qwen21_1024,klein_1024"]
            if tab == "on":
                args += ["--tunableop", tabs[0]]
            r = run(args, o, ctx["log"], timeout = 3600)
            r.update({"round": rnd, "table": tab})
            res.append(r)
            obs["lint"] = res


def item_sdc(ctx, obs):
    res = []
    for rnd, arm in rounds(["ingraph", "disable"]):
        o = ctx["work"] / f"sdc_{arm}_{rnd}.json"
        r = run([HERE / "sdpa_compile.py", "--arm", arm, "--out", o], o, ctx["log"], timeout = 3600)
        r["round"] = rnd
        res.append(r)
        obs["sdc"] = res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--items", default = "kt,hg,w8a8,lin,sdc,race,conv3d")
    a = ap.parse_args()
    out = Path(a.out)
    obs: dict = {"state": a.state, "args": {"items": a.items}}

    def save():
        out.write_text(json.dumps(obs, indent = 1, default = str), encoding = "utf-8")
    if a.state != "head":
        obs["skipped"] = "measured from the head step only (base tree arms are run from there)"
        save()
        return 0
    head = Path(a.checkout).resolve()
    ctx = {"head": str(head / "studio" / "backend"), "base": str(head.parent / "base" / "studio" / "backend"),
           "work": out.parent / "a12work", "log": out.parent / "a12_children.log"}
    ctx["work"].mkdir(parents = True, exist_ok = True)
    obs["trees"] = {k: ctx[k] for k in ("head", "base")}
    obs["trees_exist"] = {k: Path(ctx[k]).is_dir() for k in ("head", "base")}
    try:
        import torch
        import triton
        obs["env"] = {"torch": torch.__version__, "hip": torch.version.hip, "triton": triton.__version__,
                      "arch": getattr(torch.cuda.get_device_properties(0), "gcnArchName", None),
                      "python": sys.version.split()[0]}
    except Exception as exc:  # noqa: BLE001
        obs["env_error"] = repr(exc)
    save()
    fns = {"kt": item_kt, "hg": item_hg, "hgs": item_hgs, "blk": item_blk, "w8bo": item_w8bo, "lint": item_lint, "w8a8": item_w8a8, "lin": item_lin, "sdc": item_sdc}
    ext = {}
    sys.path.insert(0, str(HERE))
    for modname in ("aiter_items", "r6_lift_items", "r6_items"):
        try:
            mod = __import__(modname)
            ext.update(mod.ITEMS)
        except Exception as exc:  # noqa: BLE001
            obs[f"{modname}_import_error"] = repr(exc)
    fns.update(ext)
    for item in a.items.split(","):
        t = time.time()
        try:
            if item in fns:
                fns[item](ctx, obs) if item not in ext else fns[item](ctx, obs, run, rounds, HERE)
            else:
                obs[f"{item}_error"] = "unknown item"
        except Exception:  # noqa: BLE001
            import traceback
            obs[f"{item}_error"] = traceback.format_exc()[-2000:]
        obs.setdefault("item_s", {})[item] = round(time.time() - t, 1)
        save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
