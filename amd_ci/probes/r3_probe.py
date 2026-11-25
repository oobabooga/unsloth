#!/usr/bin/env python3
"""Probe: four gfx1151 diffusion-inference measurements on Studio's tree (observes only; criteria/r3_measure.py).

Head state only (the base is main's parent; nothing differs between arms but env / config):
  gemm  R3-06  W8A8 Triton GEMM tile sweep, then confirm the swept per-M table in fresh processes vs Studio's tiles
  attn  R3-07  INT8 attention gfx1151 long config vs bn64 / 2 stages / waves_per_eu=2 (AITER #8442) and its parts
  vae   R3-05  Wan 2.2 5B + Qwen-Image-2.1 VAE decode: contiguous vs channels_last vs channels_last + MIOpen NHWC
  load  R3-08  cold safetensors load to the GPU: mmap vs the pread / read_file backend (safetensors #760)
Every arm is a fresh process; arms interleave (ABBA), 3+ repeats of the reference arm give the A/A spread.
Only public Hugging Face downloads; no token is read or passed (HF_TOKEN etc. are removed from child envs).

  python r3_probe.py --state head --checkout <tree> --out obs.json [--items gemm,attn,vae,load]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent / "r3"
SECRET_ENVS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN", "WANDB_API_KEY")


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


def gmean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def item_gemm(backend, work, log, obs):
    sweep = run([HERE / "gemm.py", "sweep", "--backend", backend, "--out", work / "gemm_sweep.json"],
                work / "gemm_sweep.json", log, timeout = 5400)
    obs["gemm_sweep"] = sweep
    rows = sweep.get("rows") or []
    if not rows:
        return
    sys.path.insert(0, str(HERE))
    from gemm import CANDS
    table = {}
    for cls, pred in (("short", lambda k: k <= 4096), ("long", lambda k: k > 4096)):
        ents = []
        for m in sorted({r["m"] for r in rows}):
            rs = [r for r in rows if r["m"] == m and pred(r["k"])]
            best, best_g = None, None
            names = set.intersection(*[set(r["res"]) for r in rs]) if rs else set()
            for name in names:
                ok = all(r["res"][name].get("exact") and r["res"][name].get("med") for r in rs)
                if not ok:
                    continue
                g = gmean([r["res"][name]["med"] / r["res"]["studio"]["med"] for r in rs])
                if best_g is None or g < best_g:
                    best, best_g = name, g
            cfg = rs[0]["studio"] if best in (None, "studio") else list(CANDS[best])
            ents.append({"m_le": m, "pick": best, "sweep_ratio": best_g, "cfg": cfg})
        table[cls] = ents
    (work / "gemm_table.json").write_text(json.dumps(table), encoding = "utf-8")
    obs["gemm_table"] = table
    conf = []
    for i, arm in enumerate(["aa", "proposed", "proposed", "aa", "aa", "proposed"]):
        o = work / f"gemm_confirm_{i}_{arm}.json"
        r = run([HERE / "gemm.py", "confirm", "--backend", backend, "--out", o, "--table", work / "gemm_table.json",
                 "--arm", arm], o, log, timeout = 3600)
        conf.append(r)
    obs["gemm_confirm"] = conf


def item_attn(backend, work, log, obs):
    arms = ["studio", "c_aiter", "c_bn64", "c_wpe2"]
    res = []
    for shape in ("wan5b_720p", "h3_35490", "h3_91728"):
        for rnd in range(3):
            order = arms if rnd % 2 == 0 else arms[::-1]
            for arm in order:
                o = work / f"attn_{shape}_{arm}_{rnd}.json"
                r = run([HERE / "attn.py", "--backend", backend, "--shape", shape, "--arm", arm, "--out", o], o, log,
                        timeout = 1800)
                r["round"] = rnd
                res.append(r)
                obs["attn"] = res


def snapshot(repo, patterns, dest):
    code = ("import sys\nfrom huggingface_hub import snapshot_download\n"
            "snapshot_download(sys.argv[1], allow_patterns = sys.argv[3].split(','), local_dir = sys.argv[2], "
            "token = False)\n")
    t = time.time()
    r = subprocess.run([sys.executable, "-c", code, repo, str(dest), ",".join(patterns)], env = child_env(),
                       capture_output = True, text = True, timeout = 7200)
    return {"repo": repo, "rc": r.returncode, "s": round(time.time() - t, 1), "tail": (r.stdout + r.stderr)[-800:]}


def psnr(a_path, b_path):
    import torch
    a, b = torch.load(a_path).float(), torch.load(b_path).float()
    if a.shape != b.shape:
        return None
    mse = float(((a - b) ** 2).mean())
    return float("inf") if mse == 0 else 10 * math.log10(4.0 / mse)


def item_vae(backend, work, log, obs):
    models = work / "models"
    dl = {"wan5b": snapshot("Wan-AI/Wan2.2-TI2V-5B-Diffusers", ["vae/*", "model_index.json"], models / "wan5b"),
          "qwen21": snapshot("Qwen/Qwen-Image-2.1", ["vae/*", "model_index.json"], models / "qwen21")}
    obs["vae_download"] = dl
    arms = ["contig", "cl", "cl_nhwc"]
    res = []
    for model in ("wan5b", "qwen21"):
        if dl[model]["rc"] != 0:
            continue
        for rnd in range(3):
            order = arms if rnd % 2 == 0 else arms[::-1]
            for arm in order:
                o = work / f"vae_{model}_{arm}_{rnd}.json"
                save = work / f"vae_{model}_{arm}_{rnd}.pt"
                env = {"PYTORCH_MIOPEN_SUGGEST_NHWC": "1"} if arm == "cl_nhwc" else {"PYTORCH_MIOPEN_SUGGEST_NHWC": "0"}
                if os.environ.get("R3_VAE_FIND_MODE"):
                    env["MIOPEN_FIND_MODE"] = os.environ["R3_VAE_FIND_MODE"]  # same for every arm
                r = run([HERE / "vae.py", "--backend", backend, "--model", model, "--dir", models / model, "--arm", arm,
                         "--out", o, "--save", save], o, log, env = env,
                         timeout = int(os.environ.get("R3_VAE_TIMEOUT", "3600")))
                r["round"] = rnd
                res.append(r)
                obs["vae"] = res
        ps = {}
        ref = work / f"vae_{model}_contig_0.pt"
        for arm in arms:
            for rnd in range(3):
                p = work / f"vae_{model}_{arm}_{rnd}.pt"
                if ref.is_file() and p.is_file() and (arm, rnd) != ("contig", 0):
                    ps[f"{arm}_{rnd}"] = psnr(ref, p)
        cl0 = work / f"vae_{model}_cl_0.pt"
        for rnd in range(3):
            p = work / f"vae_{model}_cl_nhwc_{rnd}.pt"
            if cl0.is_file() and p.is_file():
                ps[f"cl_nhwc_{rnd}_vs_cl_0"] = psnr(cl0, p)
        obs.setdefault("vae_psnr", {})[model] = ps
        for p in work.glob(f"vae_{model}_*.pt"):
            p.unlink()


def item_load(backend, work, log, obs):
    models = work / "models"
    models.mkdir(parents = True, exist_ok = True)
    free = shutil.disk_usage(models).free / 2 ** 30
    mem = [ln for ln in open("/proc/meminfo", encoding = "utf-8") if ln.startswith(("MemTotal", "MemAvailable"))]
    obs["load_host"] = {"disk_free_gib": round(free, 1), "meminfo": mem}
    cands = [("Wan-AI/Wan2.2-T2V-A14B-Diffusers", "transformer", 53.3), ("Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                                                                            "transformer", 18.7)]
    for repo, sub, gib in cands:
        if free > gib + 15:
            break
    else:
        obs["load_skip"] = f"disk free {free:.1f} GiB"
        return
    dest = models / "ckpt"
    obs["load_download"] = snapshot(repo, [f"{sub}/*"], dest)
    obs["load_model"] = f"{repo}/{sub}"
    from safetensors import safe_open
    sig = getattr(safe_open, "__text_signature__", "") or ""
    new_arm = None
    if "backend" in sig:
        import glob
        f = sorted(glob.glob(str(dest / sub / "*.safetensors")))[0]
        for name in ("read_file", "pread"):
            try:
                with safe_open(f, framework = "pt", backend = name):
                    new_arm = name
                    break
            except Exception:  # noqa: BLE001
                continue
    obs["load_backend_name"] = new_arm
    if new_arm is None:
        obs["load_skip"] = "installed safetensors has no read_file / pread backend"
        return
    res = []
    for i, arm in enumerate(["mmap", new_arm, new_arm, "mmap", "mmap", new_arm]):
        o = work / f"load_{i}_{arm}.json"
        r = run([HERE / "load.py", "--dir", dest / sub, "--arm", arm, "--out", o], o, log, timeout = 3600)
        res.append(r)
        obs["load"] = res
    shutil.rmtree(dest, ignore_errors = True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--items", default = "gemm,attn,vae,load")
    ap.add_argument("--vae-find-mode", default = "")
    ap.add_argument("--vae-timeout", default = "3600")
    a = ap.parse_args()
    if a.vae_find_mode:
        os.environ["R3_VAE_FIND_MODE"] = a.vae_find_mode
    os.environ["R3_VAE_TIMEOUT"] = a.vae_timeout
    obs_args = {"items": a.items, "vae_find_mode": a.vae_find_mode, "vae_timeout": a.vae_timeout}
    out = Path(a.out)
    obs: dict = {"state": a.state, "args": obs_args}

    def save():
        out.write_text(json.dumps(obs, indent = 1, default = str), encoding = "utf-8")
    if a.state != "head":
        obs["skipped"] = "measured on the head tree only (arms differ by env / config, not by code)"
        save()
        return 0
    backend = str(Path(a.checkout).resolve() / "studio" / "backend")
    work = out.parent / "r3work"
    work.mkdir(parents = True, exist_ok = True)
    log = out.parent / "r3_children.log"
    try:
        import safetensors
        import torch
        import triton
        obs["env"] = {"torch": torch.__version__, "hip": torch.version.hip, "triton": triton.__version__,
                      "safetensors": safetensors.__version__,
                      "arch": getattr(torch.cuda.get_device_properties(0), "gcnArchName", None),
                      "python": sys.version.split()[0]}
    except Exception as exc:  # noqa: BLE001
        obs["env_error"] = repr(exc)
    save()
    fns = {"gemm": item_gemm, "attn": item_attn, "vae": item_vae, "load": item_load}
    for item in a.items.split(","):
        t = time.time()
        try:
            fns[item](backend, work, log, obs)
        except Exception as exc:  # noqa: BLE001
            import traceback
            obs[f"{item}_error"] = traceback.format_exc()[-2000:]
        obs.setdefault("item_s", {})[item] = round(time.time() - t, 1)
        save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
