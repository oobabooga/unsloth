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


def _psnr_map(work, prefix, ref_name, names):
    out = {}
    ref = work / f"{prefix}_{ref_name}.pt"
    for n in names:
        p = work / f"{prefix}_{n}.pt"
        if ref.is_file() and p.is_file() and p != ref:
            out[n] = psnr(ref, p)
    return out


def item_h3vae(backend, work, log, obs):
    """base tree = stock ROCm decode; head tree = the fast path (default tier, and without the int8 decoder)."""
    base_backend = str(Path(backend).parents[2] / "base" / "studio" / "backend")
    models = work / "models"
    obs["h3vae_download"] = snapshot("MiniMaxAI/MiniMax-H3", ["vae/*", "model_index.json"], models / "h3")
    arms = [("stock", base_backend, {}), ("fast", backend, {}), ("fast_noint8", backend, {"UNSLOTH_H3_VAE_INT8": "0"})]
    res, names = [], []
    for rnd in range(3):
        order = arms if rnd % 2 == 0 else arms[::-1]
        for arm, tree, env in order:
            o = work / f"h3vae_{arm}_{rnd}.json"
            save = work / f"h3vae_{arm}_{rnd}.pt" if rnd < 2 else None
            args = [HERE / "h3vae.py", "--backend", tree, "--dir", models / "h3", "--arm", arm, "--out", o]
            if save:
                args += ["--save", save]
                names.append(f"{arm}_{rnd}")
            r = run(args, o, log, env = env, timeout = 2400)
            r["round"] = rnd
            res.append(r)
            obs["h3vae"] = res
    obs["h3vae_psnr_vs_stock0"] = _psnr_map(work, "h3vae", "stock_0", names)
    for p in work.glob("h3vae_*.pt"):
        p.unlink()


def item_qattn(backend, work, log, obs):
    models = work / "models"
    if not (models / "qwen21" / "vae").is_dir():
        obs["qattn_download"] = snapshot("Qwen/Qwen-Image-2.1", ["vae/*", "model_index.json"], models / "qwen21")
    res, names = [], []
    for rnd in range(3):
        for arm in (["stock", "studio"] if rnd % 2 == 0 else ["studio", "stock"]):
            o = work / f"qattn_{arm}_{rnd}.json"
            save = work / f"qattn_{arm}_{rnd}.pt"
            names.append(f"{arm}_{rnd}")
            r = run([HERE / "qvae_attn.py", "--backend", backend, "--dir", models / "qwen21", "--arm", arm, "--out", o,
                     "--save", save], o, log, timeout = 1800)
            r["round"] = rnd
            res.append(r)
            obs["qattn"] = res
    obs["qattn_psnr_vs_stock0"] = _psnr_map(work, "qattn", "stock_0", names)
    for p in work.glob("qattn_*.pt"):
        p.unlink()


def item_h3vis(backend, work, log, obs):
    models = work / "models"
    shard = "text_encoder/model-00014-of-00014.safetensors"
    obs["h3vis_download"] = snapshot("MiniMaxAI/MiniMax-H3", [shard, "text_encoder/config.json", "processor/*"],
                                     models / "h3")
    res = []
    for rnd in range(3):
        for arm in (["fp32", "bf16"] if rnd % 2 == 0 else ["bf16", "fp32"]):
            o = work / f"h3vis_{arm}_{rnd}.json"
            r = run([HERE / "h3vis.py", "--backend", backend, "--dir", models / "h3", "--arm", arm, "--out", o,
                     "--save", work / f"h3vis_{arm}_{rnd}.pt"], o, log, timeout = 1800)
            r["round"] = rnd
            res.append(r)
            obs["h3vis"] = res
    import torch
    cmp = {}
    for rnd in range(3):
        a, b = work / f"h3vis_fp32_{rnd}.pt", work / f"h3vis_bf16_{rnd}.pt"
        if a.is_file() and b.is_file():
            x, y = torch.load(a), torch.load(b)
            if x.shape == y.shape:
                mse = float(((x - y) ** 2).mean())
                peak = float(x.abs().max())
                cmp[str(rnd)] = {"rel_l2": float((x - y).norm() / x.norm()),
                                 "psnr_peak_db": float("inf") if mse == 0 else 10 * math.log10(peak ** 2 / mse),
                                 "cos_min": float(torch.nn.functional.cosine_similarity(x, y, dim = -1).min())}
    obs["h3vis_cmp"] = cmp
    for p in work.glob("h3vis_*.pt"):
        p.unlink()


E2E_MODELS = (("qwen21", "Qwen/Qwen-Image-2.1", {"width": 1024, "height": 1024, "steps": 25, "guidance": 1.0}),
              ("klein", "unsloth/flux-3-image-klein", {"width": 1024, "height": 1024, "steps": 16, "guidance": 1.0}))


def item_e2e(backend, work, log, obs):
    """R3-06 end to end + R3-08 through Studio's load: main (base tree) vs the branch (head tree), Studio defaults,
    fresh process per arm, cold page cache for the model before every arm, interleaved, 3 repeats."""
    base_backend = str(Path(backend).parents[2] / "base" / "studio" / "backend")
    hf_home = work.resolve().parents[2] / "r3_hf_home"  # the job root: never inside the uploaded out/
    env0 = {"HF_HOME": str(hf_home), "HF_HUB_DOWNLOAD_TIMEOUT": "300", "HF_HUB_ETAG_TIMEOUT": "60",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "UNSLOTH_DISABLE_AUTO_UPDATES": "1", "PYTHONUNBUFFERED": "1"}
    pre = []
    for _, repo, _ in E2E_MODELS:
        code = ("import sys\nfrom huggingface_hub import snapshot_download\n"
                "snapshot_download(sys.argv[1], token = False, max_workers = 8)\n")
        t = time.time()
        env = child_env(env0)
        env.pop("HF_ENDPOINT", None)
        r = subprocess.run([sys.executable, "-c", code, repo], env = env, capture_output = True, text = True,
                           timeout = 7200)
        pre.append({"repo": repo, "rc": r.returncode, "s": round(time.time() - t, 1), "tail": (r.stdout + r.stderr)[-600:]})
        obs["e2e_prefetch"] = pre
    rows = []
    od = work / "e2e"
    od.mkdir(parents = True, exist_ok = True)
    scratch = work.resolve().parents[2] / "r3_scratch"
    for rnd in range(3):
        for name, repo, gen in E2E_MODELS:
            order = [("main", base_backend), ("branch", backend)]
            if rnd % 2:
                order = order[::-1]
            for arm, tree in order:
                tag = f"{name}_{arm}_{rnd}"
                ind = scratch / f"ind_{tag}"
                env = dict(env0, TORCHINDUCTOR_CACHE_DIR = str(ind))
                env_full = child_env(env)
                env_full.pop("HF_ENDPOINT", None)
                t = time.time()
                with open(log, "ab") as fh:
                    try:
                        rc = subprocess.run([sys.executable, "-u", str(HERE / "e2e_arm.py"), "--root", tree,
                                             "--model", repo, "--tag", tag, "--out", str(od), "--gen", json.dumps(gen),
                                             "--renders", "4", "--cold"], stdout = fh, stderr = subprocess.STDOUT,
                                            env = env_full, timeout = 5400).returncode
                    except subprocess.TimeoutExpired:
                        rc = "timeout"
                rp = od / f"{tag}.json"
                rec = json.loads(rp.read_text(encoding = "utf-8")) if rp.is_file() else {"verdict": "missing"}
                rec.update(rc = rc, arm = arm, model = name, round = rnd, wall_s = round(time.time() - t, 1))
                rows.append(rec)
                obs["e2e"] = rows
                shutil.rmtree(ind, ignore_errors = True)


def item_h3img(backend, work, log, obs):
    """Coverage: MiniMax-H3-Image GGUF UD-Q4_K_XL at 1024 px through Studio (family default 6 steps, guidance 1),
    2 renders; load progress logged every minute so a stall is visible."""
    od = work / "h3img"
    od.mkdir(parents = True, exist_ok = True)
    env = child_env({"HF_HUB_DOWNLOAD_TIMEOUT": "300", "HF_HUB_ETAG_TIMEOUT": "60", "HF_XET_HIGH_PERFORMANCE": "1",
                     "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "UNSLOTH_DISABLE_AUTO_UPDATES": "1",
                     "PYTHONUNBUFFERED": "1",
                     "TORCHINDUCTOR_CACHE_DIR": str(work.resolve().parents[2] / "r3_scratch" / "ind_h3img")})
    env.pop("HF_ENDPOINT", None)
    cmd = [sys.executable, "-u", str(HERE / "img_arm.py"), "--root", backend, "--model", "unsloth/MiniMax-H3-Image-GGUF",
           "--tag", "I1", "--out", str(od), "--load", json.dumps({"gguf_filename": "minimax_h3_image_pruned-UD-Q4_K_XL.gguf"}),
           "--gen", json.dumps({"width": 1024, "height": 1024, "steps": 6, "guidance": 1.0}), "--renders", "2"]
    t = time.time()
    with open(od / "I1.log", "w", encoding = "utf-8", errors = "replace") as fh:
        try:
            rc = subprocess.run(cmd, stdout = fh, stderr = subprocess.STDOUT, env = env, timeout = 4 * 3600).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    rp = od / "I1" / "record.json"
    rec = json.loads(rp.read_text(encoding = "utf-8")) if rp.is_file() else {"verdict": "missing"}
    rec.update(rc = rc, wall_s = round(time.time() - t, 1))
    try:
        rec["log_tail"] = (od / "I1.log").read_text(encoding = "utf-8", errors = "replace").splitlines()[-40:]
    except Exception:  # noqa: BLE001
        pass
    obs["h3img"] = [rec]


def item_fadv(backend, work, log, obs):
    """R3-08 add-on (llama.cpp #32992): pread alone vs pread + per-tensor POSIX_FADV_DONTNEED, cold, interleaved."""
    item_load(backend, work, log, obs, arms = ("pread", "pread_fadv", "pread_fadv", "pread", "pread", "pread_fadv"),
              key = "fadv")


def _e2e_env(work, tag):
    env = child_env({"HF_HOME": str(work.resolve().parents[2] / "r3_hf_home"), "HF_HUB_DOWNLOAD_TIMEOUT": "300",
                     "HF_HUB_ETAG_TIMEOUT": "60", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "HF_XET_HIGH_PERFORMANCE": "1",
                     "UNSLOTH_DISABLE_AUTO_UPDATES": "1", "PYTHONUNBUFFERED": "1",
                     "TORCHINDUCTOR_CACHE_DIR": str(work.resolve().parents[2] / "r3_scratch" / f"ind_{tag}")})
    env.pop("HF_ENDPOINT", None)
    return env


def _e2e_run(tree, repo, tag, od, work, log, args, timeout = 5400, extra_env = None):
    t = time.time()
    env = _e2e_env(work, tag)
    env.update(extra_env or {})
    with open(log, "ab") as fh:
        try:
            rc = subprocess.run([sys.executable, "-u", str(HERE / "e2e_arm.py"), "--root", tree, "--model", repo, "--tag", tag,
                                 "--out", str(od), *args], stdout = fh, stderr = subprocess.STDOUT,
                                env = env, timeout = timeout).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    rp = od / f"{tag}.json"
    rec = json.loads(rp.read_text(encoding = "utf-8")) if rp.is_file() else {"verdict": "missing"}
    rec.update(rc = rc, wall_s = round(time.time() - t, 1))
    shutil.rmtree(work.resolve().parents[2] / "r3_scratch" / f"ind_{tag}", ignore_errors = True)
    return rec


def _prefetch(work, repos, obs, key, patterns = None):
    out = []
    for repo in repos:
        code = ("import json, sys\nfrom huggingface_hub import snapshot_download\n"
                "snapshot_download(sys.argv[1], token = False, max_workers = 8, allow_patterns = json.loads(sys.argv[2]))\n")
        t = time.time()
        pats = json.dumps((patterns or {}).get(repo))
        r = subprocess.run([sys.executable, "-c", code, repo, pats], env = _e2e_env(work, "pre"), capture_output = True,
                           text = True, timeout = 7200)
        out.append({"repo": repo, "rc": r.returncode, "s": round(time.time() - t, 1), "tail": (r.stdout + r.stderr)[-400:]})
        obs[key] = out


def item_pread2(backend, work, log, obs):
    """R3-08 follow-up: Qwen-Image-2.1 load only through Studio, cold, main vs branch with A/A (4 fresh processes per
    tree, M B B M M B B M), every loader open traced; plus one traced FLUX.3 klein load per tree."""
    base_backend = str(Path(backend).parents[2] / "base" / "studio" / "backend")
    _prefetch(work, ["Qwen/Qwen-Image-2.1", "unsloth/flux-3-image-klein"], obs, "pread2_prefetch")
    od = work / "pread2"
    od.mkdir(parents = True, exist_ok = True)
    rows = []
    for i, arm in enumerate("MBBMMBBM"):
        tree, name = (base_backend, "main") if arm == "M" else (backend, "branch")
        rec = _e2e_run(tree, "Qwen/Qwen-Image-2.1", f"q21load_{name}_{i}", od, work, log,
                       ["--renders", "0", "--cold", "--trace", "--checksum"], timeout = 2400)
        rec.update(arm = name, i = i, model = "qwen21")
        rows.append(rec)
        obs["pread2"] = rows
    for i, (tree, name) in enumerate(((base_backend, "main"), (backend, "branch"))):
        rec = _e2e_run(tree, "unsloth/flux-3-image-klein", f"kleinload_{name}", od, work, log,
                       ["--renders", "0", "--cold", "--trace", "--checksum"], timeout = 2400)
        rec.update(arm = name, i = i, model = "klein")
        rows.append(rec)
        obs["pread2"] = rows


def item_covimg(backend, work, log, obs):
    """Coverage cell(s) named in R3_COV (moss, krea) on the BASE tree (main) at Studio's
    family defaults: 2 renders, page cache cold before the load."""
    base_backend = str(Path(backend).parents[2] / "base" / "studio" / "backend")
    known = {"moss": ("moss_sft_1024", "OpenMOSS-Team/MOSS-Image-SFT", 1024, 1024),
             "krea": ("krea2_turbo_1024", "unsloth/Krea-2-Turbo", 1024, 1024)}
    cells = [known[n] for n in (os.environ.get("R3_COV") or "").split(",") if n in known]
    od = work / "cov"
    od.mkdir(parents = True, exist_ok = True)
    rows = []
    for tag, repo, w, h in cells:
        rec = _e2e_run(base_backend, repo, tag, od, work, log,
                       ["--renders", "2", "--cold", "--gen", json.dumps({"width": w, "height": h})], timeout = 4 * 3600)
        rec.update(model = repo)
        rows.append(rec)
        obs["covimg"] = rows


SMOKE = r"""
import json, os, sys, time
sys.path.insert(0, sys.argv[1])
out = {"tree": sys.argv[1]}
import torch
out["torch"], out["hip"], out["cuda"] = torch.__version__, getattr(torch.version, "hip", None), torch.cuda.is_available()
from core.inference import safetensors_pread as SP
out["pread_wanted"] = SP.pread_wanted()
out["rocm_integrated"] = getattr(SP, "rocm_integrated", lambda: None)()
out["fadvise_wanted"] = getattr(SP, "fadvise_wanted", lambda: None)()
out["igpu_pread_backend"] = getattr(SP, "igpu_pread_backend", lambda: None)()
x = torch.randn(4096, 4096, device = "cuda", dtype = torch.bfloat16)
for _ in range(3):
    x @ x
torch.cuda.synchronize(); t = time.perf_counter()
for _ in range(20):
    x @ x
torch.cuda.synchronize()
out["gemm_tflops"] = round(20 * 2 * 4096 ** 3 / (time.perf_counter() - t) / 1e12, 2)
try:
    import triton
    out["triton"] = triton.__version__
except Exception as e:
    out["triton"] = repr(e)[:200]
try:
    f = torch.compile(lambda a: (a * 2 + 1).sin())
    t = time.perf_counter(); y = f(torch.randn(1024, device = "cuda")); torch.cuda.synchronize()
    out["compile_ok"], out["compile_s"] = True, round(time.perf_counter() - t, 2)
except Exception as e:
    out["compile_ok"], out["compile_err"] = False, repr(e)[:400]
try:
    from core.inference.diffusion_speed import compile_eligible  # noqa: F401
    out["compile_eligible_import"] = True
except Exception as e:
    out["compile_eligible_import"] = repr(e)[:200]
print("SMOKE " + json.dumps(out), flush = True)
"""


def item_smoke(backend, work, log, obs):
    """Seconds-long gate before the large arms: imports, device, pread / fadvise switches, one GEMM, one compile, and
    (Linux branch) the pread unit tests on the real device."""
    base_backend = str(Path(backend).parents[2] / "base" / "studio" / "backend")
    rows = []
    for tree in (base_backend, backend):
        r = subprocess.run([sys.executable, "-c", SMOKE, tree], env = _e2e_env(work, "smoke"), capture_output = True,
                           text = True, timeout = 900)
        line = [x for x in r.stdout.splitlines() if x.startswith("SMOKE ")]
        rows.append(json.loads(line[-1][6:]) if line else {"tree": tree, "rc": r.returncode,
                                                            "tail": (r.stdout + r.stderr)[-1500:]})
        obs["smoke"] = rows
    if sys.platform.startswith("linux"):
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-o", "addopts=", "-p", "no:cacheprovider",
                            "tests/test_safetensors_pread_rocm_igpu.py", "tests/test_safetensors_pread.py"],
                           cwd = backend, env = _e2e_env(work, "smoke"), capture_output = True, text = True, timeout = 900)
        obs["smoke_pytest"] = {"rc": r.returncode, "tail": (r.stdout + r.stderr)[-800:]}
    ok = all(row.get("cuda") for row in rows)
    obs["smoke_ok"] = ok
    if not ok:
        raise RuntimeError("smoke failed: no device")


def item_pread3(backend, work, log, obs):
    """R3-08 + fadvise: Qwen-Image-2.1 cold load through Studio; one warm-up load (downloads, discarded), then 12 fresh
    interleaved processes, 4 each: main (M), branch (B), branch with UNSLOTH_SAFETENSORS_FADVISE=0 (K)."""
    base_backend = str(Path(backend).parents[2] / "base" / "studio" / "backend")
    od = work / "pread3"
    od.mkdir(parents = True, exist_ok = True)
    rows = []
    rec = _e2e_run(base_backend, "Qwen/Qwen-Image-2.1", "q21warm_main", od, work, log, ["--renders", "0"], timeout = 5400)
    rec.update(arm = "warm", i = -1)
    rows.append(rec)
    obs["pread3"] = rows
    for i, arm in enumerate("MBKKBMMBKKBM"):
        tree, name, env = {"M": (base_backend, "main", None), "B": (backend, "branch", None),
                           "K": (backend, "branch_nofadv", {"UNSLOTH_SAFETENSORS_FADVISE": "0"})}[arm]
        rec = _e2e_run(tree, "Qwen/Qwen-Image-2.1", f"q21load_{name}_{i}", od, work, log,
                       ["--renders", "0", "--cold", "--trace", "--checksum"], timeout = 2400, extra_env = env)
        rec.update(arm = name, i = i)
        rows.append(rec)
        obs["pread3"] = rows


def item_krea3(backend, work, log, obs):
    """Krea-2-Turbo at 1024 / 8 steps on main through Studio, one load and 3 renders (Studio engages its deferred
    speed profile on the 3rd), bf16 GEMM TFLOP/s before and after each render, speed status per render."""
    base_backend = str(Path(backend).parents[2] / "base" / "studio" / "backend")
    _prefetch(work, ["unsloth/Krea-2-Turbo", "unsloth/Krea-2-Turbo-FP8"], obs, "krea3_prefetch",
              patterns = {"unsloth/Krea-2-Turbo-FP8": ["*text_encoder-INT8-ConvRot*", "*.json", "*.md"]})
    od = work / "krea3"
    od.mkdir(parents = True, exist_ok = True)
    rec = _e2e_run(base_backend, "unsloth/Krea-2-Turbo", "krea2_turbo_1024_r3", od, work, log,
                   ["--renders", "3", "--gemm-probe", "--gen", json.dumps({"width": 1024, "height": 1024})],
                   timeout = 3 * 3600)
    obs["krea3"] = rec


def item_load(backend, work, log, obs, arms = None, key = "load"):
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
    seq = list(arms) if arms else ["mmap", new_arm, new_arm, "mmap", "mmap", new_arm]
    seq = [x.replace("pread", new_arm, 1) if x.startswith("pread") else x for x in seq]
    for i, arm in enumerate(seq):
        o = work / f"{key}_{i}_{arm}.json"
        r = run([HERE / "load.py", "--dir", dest / sub, "--arm", arm, "--out", o], o, log, timeout = 3600)
        res.append(r)
        obs[key] = res
    shutil.rmtree(dest, ignore_errors = True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--items", default = "gemm,attn,vae,load")
    ap.add_argument("--vae-find-mode", default = "")
    ap.add_argument("--vae-timeout", default = "3600")
    ap.add_argument("--cov", default = "", help = "covimg cells: comma list of moss,krea")
    a = ap.parse_args()
    if a.vae_find_mode:
        os.environ["R3_VAE_FIND_MODE"] = a.vae_find_mode
    os.environ["R3_VAE_TIMEOUT"] = a.vae_timeout
    if a.cov:
        os.environ["R3_COV"] = a.cov
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
    fns = {"gemm": item_gemm, "attn": item_attn, "vae": item_vae, "load": item_load, "h3vae": item_h3vae,
           "qattn": item_qattn, "h3vis": item_h3vis, "e2e": item_e2e, "h3img": item_h3img, "fadv": item_fadv, "pread2": item_pread2, "smoke": item_smoke, "pread3": item_pread3, "krea3": item_krea3,
           "covimg": item_covimg}
    for item in a.items.split(","):
        t = time.time()
        try:
            fns[item](backend, work, log, obs)
        except Exception as exc:  # noqa: BLE001
            import traceback
            obs[f"{item}_error"] = traceback.format_exc()[-2000:]
            if item == "smoke":
                obs.setdefault("item_s", {})[item] = round(time.time() - t, 1)
                save()
                return 0  # the small check failed: no large arms
        obs.setdefault("item_s", {})[item] = round(time.time() - t, 1)
        save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
