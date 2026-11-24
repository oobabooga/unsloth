#!/usr/bin/env python3
"""M13 / M21 probes (AMD gfx1151). Observes only; numbers are judged in the report.

  m13   Windows: Wan2.2-TI2V-5B through Studio at its default size (1280x704x121), default load, 2 steps (the decode is
        what is timed). Arms, each a fresh process:
          W0   default, warms downloads + inductor cache (untimed)
          F    MIOpen Find into an empty user db (MIOPEN_FIND_MODE=1, MIOPEN_FIND_ENFORCE=4): the first-run tuning cost;
               the db it writes is the tuned db under test (copied to the artifact)
          sentinel gate (job elapsed >= --warm-min and 3 readings within 4%)
          D0 / T0  default (empty user db) / tuned (a fresh copy of F's db), each with a COLD MIOpen kernel cache
          D1 T1 D2 T2 D3 T3  interleaved, warm kernel caches; a sentinel reading before every arm
          DL / TL  one render each at MIOPEN_LOG_LEVEL=5, streamed: immediate-mode solver_id counts (engagement)
        Every arm renders twice (first-in-process and steady decode). Frames md5 per render; LPIPS / PSNR of T vs D.
  net   M21 network: HF env, endpoint reachability and a throughput sample from the configured endpoint and the Hub.
  m21   M21: prefetch (long timeouts, retries), then Wan2.2-T2V-A14B arms from --arms.
Public repos only; every token variable is stripped from the children."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOKEN_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_HUB_TOKEN")
ARM = HERE / "mq" / "wan_arm.py"
SENT = HERE / "mq" / "sentinel.py"


def child_env(extra: dict | None = None, drop_miopen: bool = True) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in TOKEN_VARS + ("PYTHONPATH",)}
    if drop_miopen:  # the arm decides every MIOpen knob (Studio still sets MIOPEN_SEARCH_CUTOFF itself)
        env = {k: v for k, v in env.items() if not k.startswith("MIOPEN_")}
    env.update({"PYTHONUNBUFFERED": "1", "UNSLOTH_DISABLE_AUTO_UPDATES": "1", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1"})
    env.update({k: str(v) for k, v in (extra or {}).items()})
    return env


def run(cmd: list, log: Path, env: dict, timeout: float, cwd: str | None = None, count_re: str | None = None):
    """Run; with count_re, stream output and count regex group(1) values instead of keeping matching lines."""
    t0 = time.time()
    counts: dict = {}
    with open(log, "w", encoding = "utf-8", errors = "replace") as fh:
        if count_re is None:
            try:
                rc = subprocess.run(cmd, stdout = fh, stderr = subprocess.STDOUT, env = env, timeout = timeout,
                                    cwd = cwd).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        else:
            rx = re.compile(count_re)
            p = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, env = env, cwd = cwd,
                                 text = True, encoding = "utf-8", errors = "replace")
            rc = None
            for line in p.stdout:
                m = rx.search(line)
                if m:
                    counts[m.group(1)] = counts.get(m.group(1), 0) + 1
                elif "MIOpen" not in line:
                    fh.write(line)
                if time.time() - t0 > timeout:
                    p.kill()
                    rc = "timeout"
                    break
            rc = p.wait() if rc is None else rc
    return rc, round(time.time() - t0, 1), counts


def tail(p: Path, n: int = 40) -> list:
    try:
        return p.read_text(encoding = "utf-8", errors = "replace").splitlines()[-n:]
    except Exception:  # noqa: BLE001
        return []


def sentinel(work: Path, label: str, res: dict) -> dict:
    d = work / "sent"
    d.mkdir(parents = True, exist_ok = True)
    jp = d / f"{label}.json"
    env = child_env({"MIOPEN_USER_DB_PATH": str(d / "udb"), "MIOPEN_CUSTOM_CACHE_DIR": str(d / "kc")})
    rc, wall, _ = run([sys.executable, "-u", str(SENT), "--out", str(jp)], d / f"{label}.log", env, 600)
    r = json.loads(jp.read_text(encoding = "utf-8")) if jp.is_file() else {"error": tail(d / f"{label}.log", 10)}
    r["label"], r["rc"] = label, rc
    res.setdefault("sentinel", []).append(r)
    print("SENTINEL", label, r.get("gemm_ms"), r.get("conv_ms"), flush = True)
    return r


def lpips_net():
    try:
        import lpips

        return lpips.LPIPS(net = "alex", verbose = False)
    except Exception as exc:  # noqa: BLE001
        return f"unavailable: {exc!r}"[:300]


def video_metrics(a_path: Path, b_path: Path, net) -> dict:
    """uint8 clips (F, H, W, 3): max abs, PSNR, LPIPS (alex, mean over every 4th frame), identical flag."""
    import numpy as np
    import torch

    a = np.load(a_path)
    b = np.load(b_path)
    if a.shape != b.shape:
        return {"shape_mismatch": [list(a.shape), list(b.shape)]}
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    mse = float((((a.astype(np.float32) - b.astype(np.float32)) / 255.0) ** 2).mean())
    out = {"identical": bool(d.max() == 0), "max_abs_u8": int(d.max()), "mean_abs_u8": float(d.mean()),
           "psnr_db": round(10 * math.log10(1 / max(mse, 1e-20)), 2)}
    if isinstance(net, str):
        out["lpips"] = net
        return out
    vals = []
    with torch.no_grad():
        for i in range(0, a.shape[0], 4):
            x = torch.from_numpy(a[i]).permute(2, 0, 1)[None].float() / 127.5 - 1
            y = torch.from_numpy(b[i]).permute(2, 0, 1)[None].float() / 127.5 - 1
            vals.append(float(net(x, y).mean()))
    out["lpips"] = sum(vals) / len(vals)
    out["lpips_max_frame"] = max(vals)
    return out


# ---------------------------------------------------------------------------------------------------------------- m13
def cmd_m13(a) -> dict:
    work = Path(a.work)
    od = Path(a.out_dir)
    work.mkdir(parents = True, exist_ok = True)
    od.mkdir(parents = True, exist_ok = True)
    res: dict = {"arms": {}, "order": [], "job_start": a.job_start, "warm_min": a.warm_min}

    def save():
        Path(a.out).write_text(json.dumps(res, indent = 1, default = str), encoding = "utf-8")

    ind = work / "ind"  # one inductor cache for every arm (the DiT compile is not under test)
    gen = json.dumps({"steps": a.steps})
    db_find = work / "db_find"
    shutil.rmtree(db_find, ignore_errors = True)
    db_find.mkdir(parents = True)

    def fresh(name: str, src: Path | None = None) -> Path:
        p = work / "dbs" / name
        shutil.rmtree(p, ignore_errors = True)
        if src is not None:
            shutil.copytree(src, p)
        else:
            p.mkdir(parents = True)
        return p

    def arm(tag: str, env: dict, renders: int = 2, npy: bool = False, count: bool = False, sent: bool = True):
        if sent:
            sentinel(work, f"before_{tag}", res)
        cmd = [sys.executable, "-u", str(ARM), "--root", a.backend, "--model", a.model, "--tag", tag,
               "--out", str(od), "--gen", gen, "--renders", str(renders)]
        if npy:
            cmd += ["--save-npy", str(work / "npy" / f"{tag}.npy")]
        if tag in ("D1", "T1"):
            cmd += ["--save-png-every", "20"]
        e = child_env({"TORCHINDUCTOR_CACHE_DIR": str(ind), **env})
        rc, wall, counts = run(cmd, od / f"{tag}.log", e, a.arm_timeout, cwd = str(work),
                               count_re = r"solver_id = (\S+?),?\s" if count else None)
        rp = od / tag / "record.json"
        r = json.loads(rp.read_text(encoding = "utf-8")) if rp.is_file() else {"verdict": "missing"}
        r.pop("log_lines", None) if r.get("verdict") == "ok" and tag not in ("W0", "D1", "T1", "F") else None
        r["rc"], r["arm_s"], r["env_set"] = rc, wall, env
        if counts:
            r["solver_counts"] = counts
        if r.get("verdict") != "ok":
            r["log_tail"] = tail(od / f"{tag}.log", 60)
        res["arms"][tag] = r
        res["order"].append(tag)
        print(f"ARM {tag} rc={rc} {r.get('verdict')} {wall}s vae={[x.get('vae_s') for x in r.get('renders', [])]} "
              f"md5={[x.get('md5', '')[:8] for x in r.get('renders', [])]}", flush = True)
        save()
        return r

    kc = {k: work / f"kc_{k}" for k in ("W0", "F", "D", "T")}
    for p in kc.values():
        shutil.rmtree(p, ignore_errors = True)
    # warm-up / download arm (its own caches, untimed)
    arm("W0", {"MIOPEN_USER_DB_PATH": str(fresh("W0")), "MIOPEN_CUSTOM_CACHE_DIR": str(kc["W0"])}, renders = 1,
        sent = False)
    sentinel(work, "after_W0", res)
    # first-run tuning: Find into an empty user db (Studio leaves immediate mode off when MIOPEN_FIND_MODE is set)
    arm("F", {"MIOPEN_USER_DB_PATH": str(db_find), "MIOPEN_FIND_MODE": "1", "MIOPEN_FIND_ENFORCE": "4",
              "MIOPEN_CUSTOM_CACHE_DIR": str(kc["F"])}, renders = 1, sent = False)
    keep = od / "miopen_db"
    shutil.rmtree(keep, ignore_errors = True)
    shutil.copytree(db_find, keep)
    res["db_files"] = {str(p.relative_to(keep)): p.stat().st_size for p in keep.rglob("*") if p.is_file()}
    save()
    # sentinel gate: elapsed >= warm_min and the last 3 readings agree within 4%
    deadline = a.job_start + (a.warm_min + 30) * 60
    while True:
        r = sentinel(work, f"gate_{len(res.get('sentinel', []))}", res)
        g = [x.get("gemm_ms") for x in res["sentinel"][-3:] if x.get("gemm_ms")]
        c = [x.get("conv_ms") for x in res["sentinel"][-3:] if x.get("conv_ms")]
        steady = len(g) == 3 and len(c) == 3 and max(g) / min(g) < 1.04 and max(c) / min(c) < 1.04
        elapsed = (time.time() - a.job_start) / 60
        if (steady and elapsed >= a.warm_min) or time.time() > deadline:
            res["gate"] = {"steady": steady, "elapsed_min": round(elapsed, 1)}
            break
        time.sleep(60)
    save()
    # cold kernel caches: first-run decode cost with / without the shipped db
    arm("D0", {"MIOPEN_USER_DB_PATH": str(fresh("D0")), "MIOPEN_CUSTOM_CACHE_DIR": str(kc["D"])})
    arm("T0", {"MIOPEN_USER_DB_PATH": str(fresh("T0", db_find)), "MIOPEN_CUSTOM_CACHE_DIR": str(kc["T"])})
    for i in range(1, a.reps + 1):
        arm(f"D{i}", {"MIOPEN_USER_DB_PATH": str(fresh(f"D{i}")), "MIOPEN_CUSTOM_CACHE_DIR": str(kc["D"])},
            npy = i == 1)
        arm(f"T{i}", {"MIOPEN_USER_DB_PATH": str(fresh(f"T{i}", db_find)), "MIOPEN_CUSTOM_CACHE_DIR": str(kc["T"])},
            npy = i == 1)
    sentinel(work, "after_timed", res)
    # engagement: immediate-mode solver ids with / without the db
    arm("DL", {"MIOPEN_USER_DB_PATH": str(fresh("DL")), "MIOPEN_CUSTOM_CACHE_DIR": str(kc["D"]),
               "MIOPEN_LOG_LEVEL": "5"}, renders = 1, count = True, sent = False)
    arm("TL", {"MIOPEN_USER_DB_PATH": str(fresh("TL", db_find)), "MIOPEN_CUSTOM_CACHE_DIR": str(kc["T"]),
               "MIOPEN_LOG_LEVEL": "5"}, renders = 1, count = True, sent = False)
    # did the tuned arms write into their db copies? (shipping read-only matters)
    res["db_after_T1"] = {str(p.relative_to(work / "dbs" / "T1")): p.stat().st_size
                          for p in (work / "dbs" / "T1").rglob("*") if p.is_file()} if (work / "dbs" / "T1").is_dir() else None
    res["kernel_cache_bytes"] = {k: sum(p.stat().st_size for p in v.rglob("*") if p.is_file()) if v.is_dir() else None
                                 for k, v in kc.items()}
    # D1 vs T1 frames, F vs D1 is not comparable (one render); D A/A is md5
    net = lpips_net()
    npy = work / "npy"
    if (npy / "D1.npy").is_file() and (npy / "T1.npy").is_file():
        res["T1_vs_D1"] = video_metrics(npy / "T1.npy", npy / "D1.npy", net)
    save()
    shutil.rmtree(npy, ignore_errors = True)
    return res


# ---------------------------------------------------------------------------------------------------------------- net
def cmd_net(a) -> dict:
    import urllib.request

    res: dict = {"env": {k: v for k, v in os.environ.items() if k.startswith(("HF_", "HTTP", "HTTPS", "NO_PROXY",
                                                                                "http", "https", "no_proxy"))
                         and k not in TOKEN_VARS}}
    for k in list(res["env"]):
        if "TOKEN" in k.upper():
            res["env"].pop(k)
    Path(a.work).mkdir(parents = True, exist_ok = True)
    try:
        du = shutil.disk_usage(a.work)
        res["disk_free_gb"] = round(du.free / 1e9, 1)
    except Exception as exc:  # noqa: BLE001
        res["disk_free_gb"] = repr(exc)
    ep = os.environ.get("HF_ENDPOINT") or "https://huggingface.co"
    eps = {"configured": ep, "hub": "https://huggingface.co"}
    shard = "unsloth/Wan2.2-T2V-A14B-FP8/resolve/main/transformer/diffusion_pytorch_model-00003-of-00012.safetensors"
    for name, base in eps.items():
        r: dict = {"base": base}
        t0 = time.time()
        try:
            with urllib.request.urlopen(base.rstrip("/") + "/api/models/Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                                        timeout = 30) as resp:
                r["api_status"] = resp.status
            r["api_s"] = round(time.time() - t0, 2)
        except Exception as exc:  # noqa: BLE001
            r["api_error"] = repr(exc)[:300]
        t0 = time.time()
        got = 0
        try:
            req = urllib.request.Request(base.rstrip("/") + "/" + shard, headers = {"Range": "bytes=0-1073741823"})
            with urllib.request.urlopen(req, timeout = 120) as resp:
                r["dl_status"] = resp.status
                r["first_byte_s"] = round(time.time() - t0, 2)
                while time.time() - t0 < 120:
                    b = resp.read(8 << 20)
                    if not b:
                        break
                    got += len(b)
        except Exception as exc:  # noqa: BLE001
            r["dl_error"] = repr(exc)[:300]
        dt = time.time() - t0
        r["dl_bytes"], r["dl_s"] = got, round(dt, 1)
        r["mb_per_s"] = round(got / 1e6 / max(dt, 1e-6), 1)
        res[name] = r
        print("NET", name, json.dumps(r), flush = True)
    Path(a.out).write_text(json.dumps(res, indent = 1), encoding = "utf-8")
    return res


# ---------------------------------------------------------------------------------------------------------------- m21
def prefetch(a, res: dict) -> None:
    """snapshot_download with long timeouts and retries until --prefetch-min elapses."""
    from huggingface_hub import snapshot_download

    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(a.hf_timeout)
    os.environ["HF_HUB_ETAG_TIMEOUT"] = "60"
    plan = json.loads(a.prefetch)
    pf: dict = {"plan": plan, "tries": []}
    res["prefetch"] = pf
    end = time.time() + a.prefetch_min * 60
    for item in plan:
        ok = False
        while not ok and time.time() < end:
            t0 = time.time()
            try:
                p = snapshot_download(item["repo"], allow_patterns = item.get("allow"), token = False,
                                      max_workers = int(item.get("workers", 8)))
                ok = True
                size = sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file())
                pf["tries"].append({"repo": item["repo"], "ok": True, "s": round(time.time() - t0, 1),
                                    "gb": round(size / 1e9, 2)})
            except Exception as exc:  # noqa: BLE001
                pf["tries"].append({"repo": item["repo"], "ok": False, "s": round(time.time() - t0, 1),
                                    "error": repr(exc)[:600]})
                time.sleep(10)
            print("PREFETCH", json.dumps(pf["tries"][-1]), flush = True)
            Path(a.out).write_text(json.dumps(res, indent = 1, default = str), encoding = "utf-8")
        if not ok:
            pf["gave_up"] = item["repo"]
            break
    try:
        pf["disk_free_gb_after"] = round(shutil.disk_usage(a.work).free / 1e9, 1)
    except Exception:  # noqa: BLE001
        pass


def cmd_m21(a) -> dict:
    work = Path(a.work)
    od = Path(a.out_dir)
    work.mkdir(parents = True, exist_ok = True)
    od.mkdir(parents = True, exist_ok = True)
    res: dict = {"arms": {}, "order": []}
    if a.prefetch:
        prefetch(a, res)
    arms = json.loads(Path(a.arms).read_text(encoding = "utf-8"))
    for spec in arms:
        tag = spec["tag"]
        cmd = [sys.executable, "-u", str(ARM), "--root", a.backend, "--model", spec["model"], "--tag", tag,
               "--out", str(od), "--load", json.dumps(spec.get("load") or {}), "--gen", json.dumps(spec.get("gen") or {}),
               "--renders", str(spec.get("renders", 2)), "--noise-fp32", "--save-png-every", str(spec.get("png", 1))]
        e = child_env({"TORCHINDUCTOR_CACHE_DIR": str(work / "ind"), "HF_HUB_DOWNLOAD_TIMEOUT": str(a.hf_timeout),
                       "HF_HUB_ETAG_TIMEOUT": "60", **(spec.get("env") or {})}, drop_miopen = False)
        rc, wall, _ = run(cmd, od / f"{tag}.log", e, spec.get("timeout", a.arm_timeout), cwd = str(work))
        rp = od / tag / "record.json"
        r = json.loads(rp.read_text(encoding = "utf-8")) if rp.is_file() else {"verdict": "missing"}
        r["rc"], r["arm_s"], r["spec"] = rc, wall, spec
        if r.get("verdict") != "ok":
            r["log_tail"] = tail(od / f"{tag}.log", 80)
        res["arms"][tag] = r
        res["order"].append(tag)
        print(f"ARM {tag} rc={rc} {r.get('verdict')} {wall}s sps={[x.get('s_per_step') for x in r.get('renders', [])]}",
              flush = True)
        Path(a.out).write_text(json.dumps(res, indent = 1, default = str), encoding = "utf-8")
        if r.get("verdict") != "ok" and spec.get("stop_on_fail"):
            break
    return res


# ------------------------------------------------------------------------------------------------------------- vaeacc
def cmd_vaeacc(a) -> dict:
    """Same latent (a 20-step Studio render) decoded by Studio's VAE path with the default and the tuned db (2 fresh
    processes each, interleaved), and by stock diffusers fp32 on the CPU (the reference)."""
    work = Path(a.work)
    od = Path(a.out_dir)
    work.mkdir(parents = True, exist_ok = True)
    od.mkdir(parents = True, exist_ok = True)
    res: dict = {"arms": {}}
    va = HERE / "mq" / "vae_arm.py"
    lat = work / "latent.pt"
    npy = work / "npy"
    npy.mkdir(exist_ok = True)

    def fresh(name, src = None):
        p = work / "dbs" / name
        shutil.rmtree(p, ignore_errors = True)
        if src is not None:
            shutil.copytree(src, p)
        else:
            p.mkdir(parents = True)
        return p

    def go(tag, mode, env, timeout):
        rp = od / f"{tag}.json"
        cmd = [sys.executable, "-u", str(va), mode, "--root", a.backend, "--latent", str(lat), "--rec", str(rp),
               "--npy", str(npy / f"{tag}.npy"), "--steps", str(a.steps)]
        rc, wall, _ = run(cmd, od / f"{tag}.log", child_env({"TORCHINDUCTOR_CACHE_DIR": str(work / "ind"), **env}),
                          timeout, cwd = str(work))
        r = json.loads(rp.read_text(encoding = "utf-8")) if rp.is_file() else {"ok": False}
        r["rc"], r["wall_s"] = rc, wall
        if not r.get("ok"):
            r["log_tail"] = tail(od / f"{tag}.log", 40)
        res["arms"][tag] = r
        print("VAEACC", tag, rc, r.get("ok"), r.get("decode_s"), r.get("md5"), flush = True)
        Path(a.out).write_text(json.dumps(res, indent = 1, default = str), encoding = "utf-8")
        return r

    kd, kt = work / "kc_D", work / "kc_T"
    go("capture", "capture", {"MIOPEN_USER_DB_PATH": str(fresh("cap")), "MIOPEN_CUSTOM_CACHE_DIR": str(kd)}, 5400)
    if not lat.is_file():
        return res
    for i in (1, 2):
        # D: a user-set (empty) MIOPEN_USER_DB_PATH, so Studio declines = the behaviour before the shipped db.
        go(f"D{i}", "decode", {"MIOPEN_USER_DB_PATH": str(fresh(f"D{i}")), "MIOPEN_CUSTOM_CACHE_DIR": str(kd)}, 3600)
        # T: no MIOpen settings at all: the head tree's shipped db engages by itself (exact-stack match).
        go(f"T{i}", "decode", {"MIOPEN_CUSTOM_CACHE_DIR": str(kt)}, 3600)
    # K: the opt-out on the head tree (MIOpen's own default user db location)
    go("K1", "decode", {"UNSLOTH_DIFFUSION_ROCM_MIOPEN_DB": "0", "MIOPEN_CUSTOM_CACHE_DIR": str(kd)}, 3600)
    go("CPU", "cpu", {}, 3 * 3600)
    net = lpips_net()
    sc = {}
    for x, y in (("D2", "D1"), ("T1", "D1"), ("T2", "T1"), ("K1", "D1"), ("D1", "CPU"), ("T1", "CPU")):
        px, py = npy / f"{x}.npy", npy / f"{y}.npy"
        if px.is_file() and py.is_file():
            sc[f"{x}_vs_{y}"] = video_metrics(px, py, net)
            print("SCORE", x, y, json.dumps(sc[f"{x}_vs_{y}"]), flush = True)
    res["scores"] = sc
    Path(a.out).write_text(json.dumps(res, indent = 1, default = str), encoding = "utf-8")
    shutil.rmtree(npy, ignore_errors = True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices = ("m13", "m21", "net", "vaeacc"))
    ap.add_argument("--backend", default = "")
    ap.add_argument("--out", required = True)
    ap.add_argument("--out-dir", default = "")
    ap.add_argument("--work", required = True)
    ap.add_argument("--model", default = "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    ap.add_argument("--steps", type = int, default = 2)
    ap.add_argument("--reps", type = int, default = 3)
    ap.add_argument("--job-start", type = float, default = 0.0)
    ap.add_argument("--warm-min", type = float, default = 25.0)
    ap.add_argument("--arms", default = "")
    ap.add_argument("--db", default = "")
    ap.add_argument("--prefetch", default = "")
    ap.add_argument("--prefetch-min", type = float, default = 200.0)
    ap.add_argument("--hf-timeout", type = int, default = 120)
    ap.add_argument("--arm-timeout", type = float, default = 5400)
    a = ap.parse_args()
    for k in TOKEN_VARS:
        os.environ.pop(k, None)
    if not a.job_start:
        a.job_start = time.time()
    {"m13": cmd_m13, "m21": cmd_m21, "net": cmd_net, "vaeacc": cmd_vaeacc}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
