#!/usr/bin/env python3
"""WINDOWS variant (amd_overlay_win_nov27): ORDER MAPRX XRPAM (2 per arm), host memory via psutil, the standby
list purged before each load when the runner holds the privilege (r3/e2e_arm.py records whether it worked), ballast
held resident with a hard minimum working set + VirtualLock (best effort; the criteria check MemAvailable at load start),
smoke no longer requires the ROCm-integrated classification (recorded: if False the PR gate is inert on Windows).

Probe for PR 19936 (pread on ROCm integrated GPUs gated on host-memory pressure): Qwen-Image-2.1 cold loads through
Studio's DiffusionBackend, one fresh process per load, page cache of the hub blobs dropped before each (r3 e2e_arm.py).

Observes only (criteria/pr19936_gate.py judges). Runs on the head state; the base tree (merge base) is one of the arms,
so both trees are measured in the same job, on the same host, interleaved:
  M  base tree, UNSLOTH_SAFETENSORS_PREAD unset                    (A/A: 3 repeats)
  A  head tree, auto, no ballast                                  (expect mmap)
  P  head tree, auto, ballast holding MemAvailable below the gate (expect pread)
  R  head tree, UNSLOTH_SAFETENSORS_PREAD=pread                    (control)
  X  head tree, UNSLOTH_SAFETENSORS_PREAD=mmap                     (control)
Order MAPRX XRPAM MAPRX after one discarded warm-up load (download). Per load: reader decision (in-process
last_reader_decision and the logged `safetensors reader:` line), load s, peak RSS, page-cache growth, checksums.
No token is read or passed (HF_TOKEN etc. removed from child envs).

  python pr19936_probe.py --state head --checkout <tree> --out obs.json [--items smoke,gate]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent / "r3"
SECRET_ENVS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN", "WANDB_API_KEY")
REPO = "Qwen/Qwen-Image-2.1"
GiB = 1 << 30
ORDER = "MAPRX" + "XRPAM"
# target MemAvailable under ballast = checkpoint + headroom - MARGIN: the PR's gate is pread when
# MemAvailable - checkpoint < headroom, so the target sits MARGIN under the switch point (not at the edge)
MARGIN_GIB = 6.0


def child_env(work: Path, tag: str, extra = None) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in SECRET_ENVS and not k.startswith("GH_TOKEN")}
    root = work.resolve().parents[2]  # the job root: never inside the uploaded out/
    env.update({"HF_HOME": str(root / "r3_hf_home"), "HF_HUB_DOWNLOAD_TIMEOUT": "300", "HF_HUB_ETAG_TIMEOUT": "60",
                "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "HF_XET_HIGH_PERFORMANCE": "1",
                "UNSLOTH_DISABLE_AUTO_UPDATES": "1", "PYTHONUNBUFFERED": "1",
                "TORCHINDUCTOR_CACHE_DIR": str(root / "r3_scratch" / f"ind_{tag}")})
    env.pop("HF_ENDPOINT", None)
    env.pop("UNSLOTH_SAFETENSORS_PREAD", None)
    env.update(extra or {})
    return env


def meminfo() -> dict:
    out = {}
    if sys.platform == "win32":
        import psutil
        vm = psutil.virtual_memory()
        return {"MemAvailable": int(vm.available), "MemTotal": int(vm.total),
                "SwapTotal": int(psutil.swap_memory().total)}
    with open("/proc/meminfo", encoding = "ascii") as fh:
        for line in fh:
            name, _, rest = line.partition(":")
            out[name] = int(rest.split()[0]) * 1024
    return out


def arm_run(tree: str, tag: str, od: Path, work: Path, args: list, extra_env = None, timeout = 2400) -> dict:
    t = time.time()
    log = od / f"{tag}.log"
    with open(log, "wb") as fh:
        try:
            rc = subprocess.run([sys.executable, "-u", str(HERE / "e2e_arm.py"), "--root", tree, "--model", REPO,
                                 "--tag", tag, "--out", str(od), *args], stdout = fh, stderr = subprocess.STDOUT,
                                env = child_env(work, tag, extra_env), timeout = timeout).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    rp = od / f"{tag}.json"
    rec = json.loads(rp.read_text(encoding = "utf-8")) if rp.is_file() else {"verdict": "missing"}
    rec.pop("trace", None)
    text = log.read_text(encoding = "utf-8", errors = "replace")
    rec["reader_log_lines"] = [ln.strip()[-400:] for ln in text.splitlines() if "safetensors reader:" in ln]
    rec.update(rc = rc, wall_s = round(time.time() - t, 1))
    if rc != 0:
        rec["log_tail"] = text[-3000:]
    shutil.rmtree(work.resolve().parents[2] / "r3_scratch" / f"ind_{tag}", ignore_errors = True)
    return rec


def start_ballast(target_gib: float, floor_gib: float, work: Path) -> tuple:
    p = subprocess.Popen([sys.executable, "-u", str(HERE / "ballast.py"), "--target-gib", f"{target_gib:.2f}",
                          "--floor-gib", f"{floor_gib:.2f}"], stdin = subprocess.PIPE, stdout = subprocess.PIPE,
                         stderr = subprocess.STDOUT, env = child_env(work, "ballast"), text = True)
    ready = None
    deadline = time.time() + 600
    while time.time() < deadline:
        line = p.stdout.readline()
        if not line:
            break
        if line.startswith("READY "):
            ready = json.loads(line[6:])
            break
    return p, ready


def stop_ballast(p) -> dict:
    try:
        p.stdin.close()
    except Exception:  # noqa: BLE001
        pass
    try:
        p.wait(timeout = 60)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait(timeout = 60)
    t = time.time()
    m = meminfo()
    while time.time() - t < 60 and m["MemAvailable"] < 0.8 * m["MemTotal"]:
        time.sleep(1)
        m = meminfo()
    return {"rc": p.returncode, "recover_s": round(time.time() - t, 1), "mem_available_gib": round(m["MemAvailable"] / GiB, 2)}


SMOKE = r"""
import json, os, sys
sys.path.insert(0, sys.argv[1])
out = {"tree": sys.argv[1]}
import torch
out["torch"], out["hip"], out["cuda"] = torch.__version__, getattr(torch.version, "hip", None), torch.cuda.is_available()
out["arch"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None) if out["cuda"] else None
from core.inference import safetensors_pread as SP
from core.inference import host_pinning as HP
out["has_pressure_gate"] = hasattr(SP, "pressure_decision")
out["rocm_is_integrated"] = HP.rocm_is_integrated(torch.device("cuda", 0)) if hasattr(HP, "rocm_is_integrated") else None
out["device_integrated_status"] = str(HP.device_integrated_status(torch.device("cuda", 0))) if hasattr(HP, "device_integrated_status") else None
out["reader_mode"] = SP.reader_mode() if hasattr(SP, "reader_mode") else None
out["host_memory_gib"] = [round(x / 2**30, 2) for x in SP.host_memory()] if hasattr(SP, "host_memory") else None
if out["has_pressure_gate"]:
    out["decision_now_24gb"] = SP.pressure_decision(24270000000)
x = torch.randn(1024, 1024, device = "cuda")
out["mm_ok"] = bool(torch.isfinite(x @ x).all())
print("SMOKE " + json.dumps(out), flush = True)
"""


def item_smoke(backend, base_backend, work, obs):
    rows = []
    for tree in (base_backend, backend):
        r = subprocess.run([sys.executable, "-c", SMOKE, tree], env = child_env(work, "smoke"), capture_output = True,
                           text = True, timeout = 900)
        line = [x for x in r.stdout.splitlines() if x.startswith("SMOKE ")]
        rows.append(json.loads(line[-1][6:]) if line else {"tree": tree, "rc": r.returncode,
                                                            "tail": (r.stdout + r.stderr)[-1500:]})
    obs["smoke"] = rows
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-o", "addopts=", "-p", "no:cacheprovider",
                        "tests/test_safetensors_pread_rocm_igpu.py", "tests/test_safetensors_pread.py"],
                       cwd = backend, env = child_env(work, "smoke"), capture_output = True, text = True, timeout = 900)
    obs["smoke_pytest"] = {"rc": r.returncode, "tail": (r.stdout + r.stderr)[-1500:]}
    head = rows[-1]
    obs["smoke_ok"] = bool(all(x.get("cuda") and x.get("mm_ok") for x in rows) and head.get("has_pressure_gate"))
    obs["gate_engages_here"] = head.get("rocm_is_integrated") is True
    if not obs["smoke_ok"]:
        raise RuntimeError("smoke failed: device or head gate missing")


def item_gate(backend, base_backend, work, obs, save):
    od = work / "loads"
    od.mkdir(parents = True, exist_ok = True)
    rows = []
    obs["loads"] = rows
    warm = arm_run(base_backend, "q21warm_main", od, work, ["--renders", "0"], timeout = 5400)
    warm.update(arm = "warm", i = -1)
    rows.append(warm)
    save()
    hub = work.resolve().parents[2] / "r3_hf_home" / "hub"
    # every blob of the hub (the Linux probe's repo glob matched nothing, checkpoint_gib 0.0): the PR's checkpoint_bytes
    # counts the snapshot's safetensors, which is this minus the json / tokenizer files (errs a little high)
    ckpt = sum(p.stat().st_size for p in hub.glob("models--*/blobs/*") if p.is_file())
    m = meminfo()
    headroom = max(16 * GiB, int(m["MemTotal"] * 0.25))
    target = (ckpt + headroom) / GiB - MARGIN_GIB
    floor = target - 2.0
    obs["pressure_plan"] = {"checkpoint_gib": round(ckpt / GiB, 2), "headroom_gib": round(headroom / GiB, 2),
                            "switch_point_gib": round((ckpt + headroom) / GiB, 2), "ballast_target_gib": round(target, 2),
                            "mem_total_gib": round(m["MemTotal"] / GiB, 2),
                            "mem_available_idle_gib": round(m["MemAvailable"] / GiB, 2),
                            "swap_total_gib": round(m.get("SwapTotal", 0) / GiB, 2)}
    save()
    spec = {"M": (base_backend, "main", None), "A": (backend, "auto_plenty", None), "P": (backend, "auto_pressure", None),
            "R": (backend, "pread_forced", {"UNSLOTH_SAFETENSORS_PREAD": "pread"}),
            "X": (backend, "mmap_forced", {"UNSLOTH_SAFETENSORS_PREAD": "mmap"})}
    for i, arm in enumerate(ORDER):
        tree, name, env = spec[arm]
        ballast = None
        rec_extra = {}
        if arm == "P":
            ballast, ready = start_ballast(target, floor, work)
            rec_extra["ballast_ready"] = ready
        t = time.time()
        rec = arm_run(tree, f"q21_{name}_{i}", od, work, ["--renders", "0", "--cold", "--checksum"], extra_env = env)
        rec.update(arm = name, i = i, started_at = round(t, 1), ended_at = round(time.time(), 1), **rec_extra)
        if ballast is not None:
            rec["ballast_stop"] = stop_ballast(ballast)
        rows.append(rec)
        save()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--items", default = "smoke,gate")
    a = ap.parse_args()
    out = Path(a.out)
    obs: dict = {"state": a.state, "args": {"items": a.items}}

    def save():
        out.write_text(json.dumps(obs, indent = 1, default = str), encoding = "utf-8")
    if a.state != "head":
        obs["skipped"] = "measured from the head state: the base tree is the M arm of the same interleaved sequence"
        save()
        return 0
    backend = str(Path(a.checkout).resolve() / "studio" / "backend")
    base_backend = str(Path(backend).parents[2] / "base" / "studio" / "backend")
    obs["trees"] = {"head": backend, "base": base_backend}
    for k, tree in obs["trees"].items():
        try:
            obs.setdefault("tree_sha", {})[k] = subprocess.run(["git", "-C", tree, "rev-parse", "HEAD"],
                                                               capture_output = True, text = True).stdout.strip()
        except Exception as exc:  # noqa: BLE001
            obs.setdefault("tree_sha", {})[k] = repr(exc)
    obs["pr_head_expected_at_submit"] = "2890b3c6b669ea78aa545f45553b19bd38232287"
    obs["platform"] = sys.platform
    work = out.parent / "r3work"
    work.mkdir(parents = True, exist_ok = True)
    try:
        import safetensors
        import torch
        obs["env"] = {"torch": torch.__version__, "hip": torch.version.hip, "safetensors": safetensors.__version__,
                      "arch": getattr(torch.cuda.get_device_properties(0), "gcnArchName", None),
                      "python": sys.version.split()[0]}
    except Exception as exc:  # noqa: BLE001
        obs["env_error"] = repr(exc)
    save()
    for item in a.items.split(","):
        t = time.time()
        try:
            if item == "smoke":
                item_smoke(backend, base_backend, work, obs)
            elif item == "gate":
                item_gate(backend, base_backend, work, obs, save)
        except Exception:  # noqa: BLE001
            import traceback
            obs[f"{item}_error"] = traceback.format_exc()[-2000:]
            obs.setdefault("item_s", {})[item] = round(time.time() - t, 1)
            save()
            return 0  # a failed step stops the rest; the criteria gates turn it into a non-result
        obs.setdefault("item_s", {})[item] = round(time.time() - t, 1)
        save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
