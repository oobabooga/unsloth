"""R6-A item 1 (ROCm gate lift of Studio's exact fused kernels): kernel / block level first, end to end only for a
part that is bit-exact and faster. ITEMS["lift"](ctx, obs, run, rounds, HERE). The head tree carries r6a_lift.diff;
the lift is on only where UNSLOTH_DIFFUSION_ROCM_FUSED_LIFT=1 is set (B = base tree = origin/main, no env)."""
from __future__ import annotations

import statistics as st
import sys
from pathlib import Path

LIFT = {"UNSLOTH_DIFFUSION_ROCM_FUSED_LIFT": "1"}


def _ok(r):
    return r and r.get("rc") == 0 and not r.get("error") and not r.get("missing")


def item_lift(ctx, obs, run, rounds, HERE):
    work, log, head, base = ctx["work"], ctx["log"], ctx["head"], ctx["base"]
    res = obs.setdefault("lift", [])
    for part in ("f3", "h3"):
        for rnd in range(3):
            o = work / f"lift_{part}_{rnd}.json"
            r = run([HERE / "fusedk.py", "--backend", head, "--part", part, "--out", o], o, log, env = LIFT,
                    timeout = 1800)
            r.update({"round": rnd})
            res.append(r)
    for tokens in (27280, 5070):
        for rnd, arm in rounds(["stock", "fused", "fused_norope"]):
            o = work / f"lift_wan_{tokens}_{arm}_{rnd}.json"
            tree, env = (base, {}) if arm == "stock" else (head, LIFT)
            r = run([HERE / "fusedk.py", "--backend", tree, "--part", "wanblk", "--arm", arm, "--tokens", tokens,
                     "--out", o, "--save", work / f"lift_wan_{tokens}_{arm}_{rnd}.pt"], o, log, env = env,
                    timeout = 1800)
            r.update({"round": rnd})
            res.append(r)
    obs["lift"] = res
    # gates for the end-to-end arms
    f3 = [r for r in res if r.get("part") == "f3" and _ok(r)]
    f3_pass = bool(f3) and all(((r.get("real_klein") or {}).get("bf16") or {}).get("equal") for r in f3) and \
        st.median([r["real_klein"]["bf16"]["stock"] / r["real_klein"]["bf16"]["fused"] for r in f3]) > 1.0
    wb = [r for r in res if r.get("part") == "wanblk" and r.get("tokens") == 5070 and _ok(r)]
    shas = {r.get("arm"): r.get("out_sha") for r in wb}
    med = {a: st.median([r["med_ms"] for r in wb if r.get("arm") == a] or [0]) for a in ("stock", "fused")}
    wan_pass = shas.get("fused") == shas.get("stock") and shas.get("stock") and med["fused"] and \
        med["stock"] / med["fused"] > 1.01
    obs["lift_e2e_gate"] = {"klein": f3_pass, "wan": bool(wan_pass), "wan_med": med}
    fams = [f for f, ok in (("klein", f3_pass), ("wan", wan_pass)) if ok]
    if not fams:
        return
    sys.path.insert(0, str(Path(HERE).parent))
    import wintiers_probe as WP
    import subprocess, json, os, shutil  # noqa: E401
    WP.ENGAGE_KEYS = WP.ENGAGE_KEYS + ("wan_fused", "flux3_qk_rope", "qk_norm")
    import os
    hf_home = Path(os.environ.get("AMD_CI_WORK") or work) / "hf_a12"
    arms_dir = work / "lift_arms"
    arms_dir.mkdir(parents = True, exist_ok = True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "lpips"], capture_output = True)
    trees = {"base": Path(base).parents[1], "head": Path(head).parents[1]}
    obs["lift_e2e"] = {}
    for fam in fams:
        spec = dict(WP.FAMILIES[fam])
        if fam == "klein":
            spec["load"] = {"model_kind": "pipeline"}  # Studio defaults on this card
        got = WP.fetch(spec["repo"], spec["ignore"], hf_home, obs.setdefault("lift_fetch", []))
        if got["rc"] != 0:
            continue
        for tag, tree, env in (("B1", "base", {}), ("H1", "head", LIFT), ("H2", "head", LIFT), ("B2", "base", {}),
                               ("B3", "base", {}), ("H3", "head", LIFT)):
            rec = WP.run_arm(trees[tree], fam, tag, spec, env, arms_dir, hf_home, 3, spec["warmup"], spec["gen"])
            obs["lift_e2e"][f"{fam}_{tag}"] = rec
        sc = subprocess.run([sys.executable, "-u", str(WP.TOOLS / "score.py"), "--out", str(arms_dir), "--family",
                             fam, "--ref", "B1", "--pairs", "H1:B1,B2:B1,H2:H1", "--kind", spec["kind"]],
                            capture_output = True, text = True, env = WP.arm_env(hf_home, {}), timeout = 3600)
        try:
            obs.setdefault("lift_scores", {})[fam] = json.loads((arms_dir / f"scores_{fam}.json").read_text(
                encoding = "utf-8"))
        except Exception as exc:  # noqa: BLE001
            obs.setdefault("lift_scores", {})[fam] = {"error": repr(exc)[:300], "tail": (sc.stdout + sc.stderr)[-800:]}
        shutil.rmtree(hf_home / "hub", ignore_errors = True)
        _ = os


ITEMS = {"lift": item_lift}



def _e2e(ctx, obs, HERE, key, fam, plan, gen_over, pairs, extra_keys):
    import json
    import shutil
    import subprocess
    sys.path.insert(0, str(Path(HERE).parent))
    import wintiers_probe as WP
    WP.ENGAGE_KEYS = tuple(WP.ENGAGE_KEYS) + tuple(extra_keys)
    import os
    # model weights outside the uploaded out/ tree (a 59 GB artifact otherwise)
    hf_home = Path(os.environ.get("AMD_CI_WORK") or ctx["work"]) / "hf_a12"
    arms_dir = ctx["work"] / f"{key}_arms"
    arms_dir.mkdir(parents = True, exist_ok = True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "lpips"], capture_output = True)
    trees = {"base": Path(ctx["base"]).parents[1], "head": Path(ctx["head"]).parents[1]}
    spec = dict(WP.FAMILIES[fam])
    gen = dict(spec["gen"], **gen_over)
    got = WP.fetch(spec["repo"], spec["ignore"], hf_home, obs.setdefault(f"{key}_fetch", []))
    if got["rc"] != 0:
        return
    res = obs.setdefault(key, {})
    for tag, tree, env in plan:
        res[tag] = WP.run_arm(trees[tree], fam, tag, spec, env, arms_dir, hf_home, 3, spec["warmup"], gen)
    sc = subprocess.run([sys.executable, "-u", str(WP.TOOLS / "score.py"), "--out", str(arms_dir), "--family", fam,
                         "--ref", plan[0][0], "--pairs", pairs, "--kind", spec["kind"]],
                        capture_output = True, text = True, env = WP.arm_env(hf_home, {}), timeout = 3600)
    try:
        obs[f"{key}_scores"] = json.loads((arms_dir / f"scores_{fam}.json").read_text(encoding = "utf-8"))
    except Exception as exc:  # noqa: BLE001
        obs[f"{key}_scores"] = {"error": repr(exc)[:300], "tail": (sc.stdout + sc.stderr)[-800:]}
    shutil.rmtree(hf_home / "hub", ignore_errors = True)


PITCH = {"UNSLOTH_DIFFUSION_ROCM_WEIGHT_PITCH": "64"}
KEYS = ("rocm_aiter_conv3d", "aiter conv3d", "rocm_weight_pitch", "weight pitch", "wan_fused", "flux3_qk_rope")


def item_wane2e(ctx, obs, run, rounds, HERE):
    """Wan 2.2 TI2V-5B at Studio defaults (832x480x49, 8 steps). B = main; T = head default (the new TunableOp table
    loads by itself); A = AITER conv3d VAE, P = Linear weight pitch +64 (both with the head's table off, i.e. main's
    GEMMs); TA = table + AITER conv3d. Order B1 T1 A1 P1 TA1 TA2 P2 A2 T2 B2; LPIPS vs B1."""
    sys.path.insert(0, str(HERE))
    import aiter_items
    w = aiter_items._setup(ctx, obs, HERE)
    a_env = {"UNSLOTH_DIFFUSION_ROCM_AITER_CONV3D": "1", "UNSLOTH_AITER_DIR": str(w / "aiter")}
    off = {"UNSLOTH_DIFFUSION_ROCM_TUNABLEOP": "0"}
    plan = [("B1", "base", {}), ("T1", "head", {}), ("A1", "head", {**a_env, **off}), ("P1", "head", {**PITCH, **off}),
            ("TA1", "head", a_env), ("TA2", "head", a_env), ("P2", "head", {**PITCH, **off}),
            ("A2", "head", {**a_env, **off}), ("T2", "head", {}), ("B2", "base", {})]
    _e2e(ctx, obs, HERE, "wane2e", "wan", plan, {"num_frames": 49},
         "T1:B1,A1:B1,P1:B1,TA1:B1,B2:B1,T2:T1,A2:A1,P2:P1,TA2:TA1", KEYS + ("tunableop", "TunableOp"))


def item_q21e2e(ctx, obs, run, rounds, HERE):
    """Qwen-Image-2.1 at Studio defaults 1024^2: B = main vs P = Linear weight pitch +64; B1 P1 P2 B2."""
    plan = [("B1", "base", {}), ("P1", "head", PITCH), ("P2", "head", PITCH), ("B2", "base", {})]
    _e2e(ctx, obs, HERE, "q21e2e", "q21", plan, {}, "P1:B1,B2:B1,P2:P1", KEYS)


ITEMS["wane2e"] = item_wane2e
ITEMS["q21e2e"] = item_q21e2e
