#!/usr/bin/env python3
"""Probe: round-7 A7 gfx1151 measurements (observes only; criteria/a7_measure.py gates that each item ran).

Runs on the head state only; arms that need main's code run from the base tree (states/base = origin/main), the head
tree adds two default-off knobs (int8 attention config key ``vblk``; UNSLOTH_NATIVE_CACHE_MMAP=read). Every arm is a
fresh process; no token is read or passed (public repos, token=False).

  python a7_probe.py --state head --checkout <tree> --out obs.json --items gemm,flydsl,vtb,conv,mmap,hsa

  gemm    ROCm large-GEMM silent-corruption probe (plain and TunableOp processes)          [Linux + Windows]
  flydsl  FlyDSL RDNA3 INT8 WMMA GEMM import + microbench vs Studio's Triton W8A8          [Linux]
  vtb     comfy-kitchen #368 block-contiguous V^T, H3 37.7k / Wan 5B 16.8k, short + long    [Linux]
  conv    temporal-unfolded batched Conv2d vs Conv3d, Wan 5B (fp32) + LTX-2.5 (bf16) VAE   [Linux + Windows]
  mmap    native build mapping ACCESS_COPY vs ACCESS_READ (+MIMALLOC_PURGE_DELAY=0), Qwen-2.1 + Wan A14B  [Linux]
  hsa     HSA_USE_SVM=0 / +HSA_ENABLE_SDMA=0 on a cold Wan 5B load + short render             [Linux]
  vaee2e  item 5 through Studio: Wan 5B render, main vs head + UNSLOTH_VIDEO_VAE_ROCM_UNFOLD=1   [Linux + Windows]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent / "a7"
SECRET_ENVS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_HUB_TOKEN", "GH_TOKEN",
               "GITHUB_TOKEN", "WANDB_API_KEY")
ROUNDS = 3
WIN = sys.platform == "win32"


def child_env(ctx, extra = None):
    env = {k: v for k, v in os.environ.items() if k not in SECRET_ENVS and not k.startswith("GH_TOKEN")}
    env.pop("PYTHONPATH", None)
    env.update({"HF_HOME": str(ctx["hf"]), "PYTHONUNBUFFERED": "1", "UNSLOTH_DISABLE_AUTO_UPDATES": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1", "PYTHONIOENCODING": "utf-8", "HF_HUB_DOWNLOAD_TIMEOUT": "120"})
    env.update(extra or {})
    return env


def run(ctx, args, out: Path, env = None, timeout = 3600) -> dict:
    t = time.time()
    with open(ctx["log"], "ab") as fh:
        fh.write(f"\n=== {' '.join(map(str, args))} env={env}\n".encode())
        fh.flush()
        try:
            rc = subprocess.run([sys.executable, "-u", *map(str, args)], stdout = fh, stderr = subprocess.STDOUT,
                                env = child_env(ctx, env), timeout = timeout).returncode
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


def fetch(ctx, repo, allow, obs, key):
    code = ("import sys, json, time; from huggingface_hub import snapshot_download\n"
            "t=time.time()\n"
            "for i in range(6):\n"
            "    try:\n"
            "        p=snapshot_download(sys.argv[1], allow_patterns=json.loads(sys.argv[2]) or None, token=False,"
            " max_workers=8); break\n"
            "    except Exception as e:\n"
            "        print('retry', i, repr(e)[:300], flush=True); time.sleep(20)\n"
            "else:\n"
            "    raise SystemExit(3)\n"
            "print('FETCHED', p, round(time.time()-t,1), flush=True)\n")
    t0 = time.time()
    r = subprocess.run([sys.executable, "-c", code, repo, json.dumps(allow)], capture_output = True, text = True,
                       encoding = "utf-8", errors = "replace", env = child_env(ctx), timeout = 7200)
    res = {"repo": repo, "allow": allow, "rc": r.returncode, "seconds": round(time.time() - t0, 1),
           "tail": (r.stdout + r.stderr)[-800:]}
    obs.setdefault(key, []).append(res)
    return res["rc"] == 0


def drop_repo(ctx, repo):
    shutil.rmtree(ctx["hf"] / "hub" / ("models--" + repo.replace("/", "--")), ignore_errors = True)


def repo_blobs(ctx, repo):
    d = ctx["hf"] / "hub" / ("models--" + repo.replace("/", "--")) / "blobs"
    return [str(p) for p in d.glob("*") if p.is_file()] if d.is_dir() else []


def drop_cache(files):
    for f in files:
        try:
            fd = os.open(f, os.O_RDONLY)
            try:
                os.fsync(fd)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except Exception:  # noqa: BLE001
            pass


def disk_free_gb(path):
    return round(shutil.disk_usage(path).free / 1e9, 1)


# ---------------------------------------------------------------- items

def item_gemm(ctx, obs):
    res = []
    for arm, env in (("plain", {}), ("tunableop", {"PYTORCH_TUNABLEOP_ENABLED": "1", "PYTORCH_TUNABLEOP_TUNING": "1",
                                                    "PYTORCH_TUNABLEOP_FILENAME": str(ctx["work"] / "tunableop_a7.csv")})):
        o = ctx["work"] / f"gemm_{arm}.json"
        r = run(ctx, [HERE / "gemm_probe.py", "--out", o], o, env = env, timeout = 1800)
        r["arm"] = arm
        res.append(r)
        obs["gemm"] = res


def item_flydsl(ctx, obs):
    w = ctx["work"]
    pk, src = w / "flydsl_pkgs", w / "FlyDSL_src"
    st = {}
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--target", str(pk), "flydsl==0.5.4"],
                       capture_output = True, text = True, env = child_env(ctx), timeout = 1800)
    st["pip"] = {"rc": r.returncode, "tail": (r.stdout + r.stderr)[-800:]}
    r = subprocess.run(["git", "clone", "-q", "https://github.com/ROCm/FlyDSL", str(src)], capture_output = True,
                       text = True, timeout = 900)
    st["clone"] = {"rc": r.returncode, "tail": (r.stdout + r.stderr)[-400:]}
    if r.returncode == 0:
        subprocess.run(["git", "-C", str(src), "checkout", "-q", "0768aa599c2962ff1d381e339d1476a28e3cf01c"],
                       capture_output = True, timeout = 300)
        st["src_head"] = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output = True,
                                        text = True).stdout.strip()
    obs["flydsl_setup"] = st
    res = []
    for rnd in range(2):
        o = w / f"flydsl_{rnd}.json"
        r = run(ctx, [HERE / "flydsl_bench.py", "--backend", ctx["base"], "--flydsl-src", src, "--out", o], o,
                env = {"PYTHONPATH": str(pk)}, timeout = 3600)
        r["round"] = rnd
        res.append(r)
        obs["flydsl"] = res
        if not r.get("imported"):
            break


def item_vtb(ctx, obs):
    work = ctx["work"]
    # (label, tree, attn_bench arm): long_base = origin/main's tree, proves the head's default-off path is main's bits
    arms = [("short", "head", "short"), ("short_vb", "head", "short_vb"), ("long", "head", "long"),
            ("long_vb", "head", "long_vb"), ("long_base", "base", "long")]
    res = []
    for shape in ("h3_37720", "wan5b_16800"):
        for rnd in range(ROUNDS):
            for label, tree, arm in (arms if rnd % 2 == 0 else arms[::-1]):
                o = work / f"vtb_{shape}_{label}_{rnd}.json"
                r = run(ctx, [HERE / "attn_bench.py", "--backend", ctx[tree], "--impl", "int8", "--shape", shape,
                              "--arm", arm, "--out", o, "--reps", "7"], o, timeout = 1800)
                r.update({"round": rnd, "label": label})
                res.append(r)
                obs["vtb"] = res


def item_conv(ctx, obs):
    res = []
    for model, repo in (("wan", "Wan-AI/Wan2.2-TI2V-5B-Diffusers"), ("ltx", "Lightricks/LTX-2.5-Diffusers")):
        if not fetch(ctx, repo, ["vae/*"], obs, "conv_fetch"):
            continue
        o = ctx["work"] / f"conv_{model}.json"
        r = run(ctx, [HERE / "conv_bench.py", "--model", model, "--out", o], o, timeout = 3600)
        res.append(r)
        obs["conv"] = res
        drop_repo(ctx, repo)


def item_mmap(ctx, obs):
    res = []
    obs["mmap_disk_free_gb"] = disk_free_gb(ctx["hf"])
    plans = [("q21", "Qwen/Qwen-Image-2.1", "qwen-image-2.1", 25), ("a14b", "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                                                                     "wan2.2-t2v-a14b", 75)]
    copy = {"UNSLOTH_NATIVE_CACHE_MMAP": "copy"}
    read = {"UNSLOTH_NATIVE_CACHE_MMAP": "read"}
    readp = {"UNSLOTH_NATIVE_CACHE_MMAP": "read", "MIMALLOC_PURGE_DELAY": "0"}
    for tag, repo, fam, need in plans:
        free = disk_free_gb(ctx["hf"])
        if free < need:
            obs.setdefault("mmap_skip", {})[tag] = f"disk free {free} GB < {need}"
            continue
        if not fetch(ctx, repo, ["transformer/*", "model_index.json"], obs, "mmap_fetch"):
            continue
        micro = [("C", copy), ("R", read), ("RP", readp), ("RP", readp), ("R", read), ("C", copy)]
        for i, (arm, env) in enumerate(micro):
            o = ctx["work"] / f"mmap_{tag}_micro_{arm}_{i}.json"
            args = [HERE / "mmap_bench.py", "--backend", ctx["head"], "--repo", repo, "--mode", "micro", "--out", o]
            if tag == "a14b":
                args += ["--max-gb", "20"]
            r = run(ctx, args, o, env = env, timeout = 3600)
            r.update({"model": tag, "arm": arm, "i": i})
            res.append(r)
            obs["mmap"] = res
        build = [("C", copy), ("R", read), ("R", read), ("C", copy)] if tag == "q21" else [("C", copy), ("R", read)]
        for i, (arm, env) in enumerate(build):
            o = ctx["work"] / f"mmap_{tag}_build_{arm}_{i}.json"
            r = run(ctx, [HERE / "mmap_bench.py", "--backend", ctx["head"], "--repo", repo, "--family", fam,
                          "--mode", "build", "--out", o], o, env = env, timeout = 5400)
            r.update({"model": tag, "arm": arm, "i": i})
            res.append(r)
            obs["mmap"] = res
        drop_repo(ctx, repo)


def kmsg_svm():
    for cmd in (["dmesg"], ["sudo", "-n", "dmesg"], ["journalctl", "-k", "--no-pager", "-q"]):
        try:
            r = subprocess.run(cmd, capture_output = True, text = True, timeout = 60)
            if r.returncode == 0 and r.stdout:
                lines = r.stdout.splitlines()
                return {"src": " ".join(cmd), "lines": len(lines),
                        "svm_restore": sum("svm_range_restore_work" in x for x in lines),
                        "svm_any": sum("svm" in x.lower() for x in lines),
                        "amdgpu_err": sum(("amdgpu" in x and ("error" in x.lower() or "fault" in x.lower())) for x in lines)}
        except Exception:  # noqa: BLE001
            continue
    return {"src": None}


def item_hsa(ctx, obs):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import wintiers_probe as WP
    repo = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    obs["hsa_kernel"] = os.uname().release if hasattr(os, "uname") else None
    if not fetch(ctx, repo, [], obs, "hsa_fetch"):
        return
    spec = dict(WP.FAMILIES["wan"])
    gen = {"width": 832, "height": 480, "num_frames": 17, "steps": 4}
    od = ctx["work"] / "hsa_arms"
    od.mkdir(parents = True, exist_ok = True)
    base_tree = Path(ctx["base"]).parents[1]
    svm = {"HSA_USE_SVM": "0"}
    both = {"HSA_USE_SVM": "0", "HSA_ENABLE_SDMA": "0"}
    plan = [("W0", {}), ("S1", {}), ("V1", svm), ("D1", both), ("D2", both), ("V2", svm), ("S2", {})]
    res = obs.setdefault("hsa", {})
    for tag, env in plan:
        drop_cache(repo_blobs(ctx, repo))
        k0 = kmsg_svm()
        rec = WP.run_arm(base_tree, "wan", tag, spec, env, od, ctx["hf"], 1, 0, gen)
        k1 = kmsg_svm()
        rec["kmsg_before"], rec["kmsg_after"] = k0, k1
        rec["env"] = env
        npz = od / f"wan_{tag}" / "r0.npz"
        if npz.is_file():
            try:
                import numpy as np
                rec["frames_md5"] = hashlib.md5(np.load(npz)["frames"].tobytes()).hexdigest()
            except Exception as exc:  # noqa: BLE001
                rec["frames_md5"] = f"err {exc!r}"[:200]
        rec.pop("log_tail", None)
        res[tag] = rec
        obs["hsa"] = res
    drop_repo(ctx, repo)


def item_vaee2e(ctx, obs):
    """Item 5 through Studio: Wan 2.2 TI2V-5B at 832x480x49, 8 steps, Studio defaults (its VAE tiling, MIOpen db).
    B = main; U = head tree with UNSLOTH_VIDEO_VAE_ROCM_UNFOLD=1. Order B1 U1 U2 B2; 1 warm-up + 2 timed renders per
    arm; LPIPS / PSNR vs B1 (wintiers/score.py)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import wintiers_probe as WP
    WP.ENGAGE_KEYS = tuple(WP.ENGAGE_KEYS) + ("vae_rocm_unfold", "vae_rocm_pertap", "miopen")
    repo = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    if not fetch(ctx, repo, [], obs, "vaee2e_fetch"):
        return
    spec = dict(WP.FAMILIES["wan"])
    gen = {"width": 832, "height": 480, "num_frames": 49, "steps": 8}
    od = ctx["work"] / "vaee2e_arms"
    od.mkdir(parents = True, exist_ok = True)
    trees = {"base": Path(ctx["base"]).parents[1], "head": Path(ctx["head"]).parents[1]}
    on = {"UNSLOTH_VIDEO_VAE_ROCM_UNFOLD": "1"}
    res = obs.setdefault("vaee2e", {})
    for tag, tree, env in (("B1", "base", {}), ("U1", "head", on), ("U2", "head", on), ("B2", "base", {})):
        rec = WP.run_arm(trees[tree], "wan", tag, spec, env, od, ctx["hf"], 2, 1, gen)
        rec.pop("log_tail", None)
        res[tag] = rec
        obs["vaee2e"] = res
    sc = subprocess.run([sys.executable, "-u", str(WP.TOOLS / "score.py"), "--out", str(od), "--family", "wan",
                         "--ref", "B1", "--pairs", "U1:B1,B2:B1,U2:U1", "--kind", "video"],
                        capture_output = True, text = True, encoding = "utf-8", errors = "replace",
                        env = WP.arm_env(ctx["hf"], {}), timeout = 3600)
    try:
        obs["vaee2e_scores"] = json.loads((od / "scores_wan.json").read_text(encoding = "utf-8"))
    except Exception as exc:  # noqa: BLE001
        obs["vaee2e_scores"] = {"error": repr(exc)[:300], "tail": (sc.stdout + sc.stderr)[-800:]}
    drop_repo(ctx, repo)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--items", default = "gemm,flydsl,vtb,conv,mmap,hsa")
    a = ap.parse_args()
    out = Path(a.out)
    obs: dict = {"state": a.state, "args": {"items": a.items}, "platform": sys.platform}

    def save():
        out.write_text(json.dumps(obs, indent = 1, default = str), encoding = "utf-8")
    if a.state != "head":
        obs["skipped"] = "measured from the head step only (base tree arms are run from there)"
        save()
        return 0
    head = Path(a.checkout).resolve()
    work_root = Path(os.environ.get("AMD_CI_WORK") or out.parent)
    ctx = {"head": str(head / "studio" / "backend"), "base": str(head.parent / "base" / "studio" / "backend"),
           "work": out.parent / "a7work", "log": out.parent / "a7_children.log", "hf": work_root / "hf_a7"}
    ctx["work"].mkdir(parents = True, exist_ok = True)
    ctx["hf"].mkdir(parents = True, exist_ok = True)
    obs["trees_exist"] = {k: Path(ctx[k]).is_dir() for k in ("head", "base")}
    obs["disk_free_gb"] = disk_free_gb(ctx["hf"])
    try:
        import torch
        obs["env"] = {"torch": torch.__version__, "hip": torch.version.hip,
                      "arch": getattr(torch.cuda.get_device_properties(0), "gcnArchName", None),
                      "python": sys.version.split()[0]}
        try:
            import triton
            obs["env"]["triton"] = triton.__version__
        except Exception as exc:  # noqa: BLE001
            obs["env"]["triton"] = repr(exc)[:100]
    except Exception as exc:  # noqa: BLE001
        obs["env_error"] = repr(exc)
    save()
    fns = {"gemm": item_gemm, "flydsl": item_flydsl, "vtb": item_vtb, "conv": item_conv, "mmap": item_mmap,
           "hsa": item_hsa, "vaee2e": item_vaee2e}
    for item in a.items.split(","):
        t = time.time()
        try:
            if item in fns:
                fns[item](ctx, obs)
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
