#!/usr/bin/env python3
"""Probe for unsloth#18763 (NPU model list) and #18805 (Windows APU host shortfall).

Observes only. npu: the pinned FastFlowLM binary's raw `list --json` / `validate --json` /
`version --json` streams, then this checkout's Flm.catalog() on that real binary.
apu (Windows): real carve-out / WDDM / ROCm readings, then this checkout's
_launch_host_shortfall_message over a grid of model sizes, with host RAM held down so
Windows-visible RAM is the smaller pool, as on the reporter's 96/32 split. Only the GGUF
size is faked; every hardware reading is real.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import types
import urllib.request
import zipfile
import tarfile
from pathlib import Path

GiB = 1024**3
WIN = sys.platform == "win32"


def work_root() -> Path:
    base = os.environ.get("AMD_CI_WORK") or tempfile.gettempdir()
    root = Path(base) / "issues_probe"
    root.mkdir(parents = True, exist_ok = True)
    return root


def run(cmd, env = None, timeout = 180):
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output = True, text = True, encoding = "utf-8",
                           errors = "replace", env = env, timeout = timeout)
        return {"rc": p.returncode, "stdout": p.stdout, "stderr": p.stderr,
                "s": round(time.time() - t0, 2)}
    except Exception as e:
        return {"rc": None, "stdout": "", "stderr": f"{type(e).__name__}: {e}",
                "s": round(time.time() - t0, 2)}


def json_split(text: str) -> dict:
    """Where the first JSON object in stdout ends and what follows it."""
    start = text.find("{")
    if start < 0:
        return {"json_ok": False, "before": text[:400], "after": ""}
    try:
        value, end = json.JSONDecoder().raw_decode(text, start)
    except ValueError as e:
        return {"json_ok": False, "error": str(e), "before": text[:start][:400], "after": ""}
    return {"json_ok": True, "before": text[:start][:400], "after": text[end:][:2000],
            "n_models": len(value.get("models") or []) if isinstance(value, dict) else None,
            "json_chars": end - start}


def install_flm(checkout: Path) -> dict:
    """The pinned FastFlowLM at <root>/bin/flm/npu, as Studio's direct install lays it out."""
    pins = json.loads((checkout / "studio" / "lemonade_prebuilt_pins.json").read_text("utf-8"))
    flm = pins["fastflowlm"]
    asset = flm["assets"]["windows-x64" if WIN else "linux-x64"]
    root = work_root() / f"flm_{flm['version']}"
    target = root / "bin" / "flm" / "npu"
    binary = target / ("flm.exe" if WIN else "flm")
    info = {"version": flm["version"], "asset": asset["name"], "root": str(root)}
    if binary.is_file():
        return {**info, "binary": str(binary), "reused": True}
    archive = work_root() / asset["name"]
    if not archive.is_file():
        url = f"https://github.com/{flm['repo']}/releases/download/{flm['version']}/{asset['name']}"
        urllib.request.urlretrieve(url, archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    info["sha256_ok"] = digest == asset["sha256"]
    if not info["sha256_ok"]:
        return {**info, "error": f"sha256 {digest} != pin"}
    scratch = work_root() / "flm_extract"
    shutil.rmtree(scratch, ignore_errors = True)
    scratch.mkdir()
    if asset["name"].endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(scratch)
    else:
        with tarfile.open(archive) as t:
            try:
                t.extractall(scratch, filter = "fully_trusted")
            except TypeError:
                t.extractall(scratch)
    found = sorted(scratch.rglob(binary.name), key = lambda p: len(p.parts))
    if not found:
        return {**info, "error": f"no {binary.name} in the archive",
                "listing": [str(p.relative_to(scratch)) for p in list(scratch.rglob("*"))[:40]]}
    target.parent.mkdir(parents = True, exist_ok = True)
    shutil.rmtree(target, ignore_errors = True)
    shutil.move(str(found[0].parent), str(target))
    return {**info, "binary": str(binary), "reused": False, "found_at": str(found[0].relative_to(scratch))}


def npu_host_facts() -> dict:
    facts = {}
    if WIN:
        ps = ("Get-PnpDevice -PresentOnly | Where-Object { $_.FriendlyName -match 'NPU|IPU|Neural' }"
              " | Select-Object FriendlyName,Status,InstanceId | ConvertTo-Json -Compress")
        r = run(["powershell", "-NoProfile", "-Command", ps], timeout = 60)
        facts["pnp_npu"] = (r["stdout"] or r["stderr"]).strip()[:1500]
    else:
        facts["dev_accel"] = sorted(os.listdir("/dev/accel")) if os.path.isdir("/dev/accel") else []
        facts["amdxdna_loaded"] = os.path.isdir("/sys/module/amdxdna")
        r = run(["bash", "-c", "ulimit -l; uname -r"], timeout = 20)
        facts["memlock_kernel"] = r["stdout"].strip()
    return facts


def load_flm_host(checkout: Path):
    if importlib.util.find_spec("httpx") is None:
        # catalog() never touches httpx; a stub keeps the import from needing the Studio venv.
        sys.modules["httpx"] = types.ModuleType("httpx")
    path = checkout / "studio" / "backend" / "core" / "inference" / "flm_host.py"
    spec = importlib.util.spec_from_file_location("flm_host_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def probe_npu(checkout: Path) -> dict:
    out = {"host": npu_host_facts()}
    try:
        inst = install_flm(checkout)
    except Exception as e:
        return {**out, "install": {"error": f"{type(e).__name__}: {e}"}}
    out["install"] = inst
    if "binary" not in inst or "error" in inst:
        return out
    root = Path(inst["root"])
    env = dict(os.environ, FLM_MODEL_PATH = str(root / "flm"), FLM_DISABLE_UPDATE_CHECK = "1")
    raw = {}
    for label, args, e in (
        ("version", ["version", "--json"], env),
        ("validate", ["validate", "--json"], env),
        ("list", ["list", "--json"], env),
        ("list_update_check_on", ["list", "--json"],
         {k: v for k, v in env.items() if k != "FLM_DISABLE_UPDATE_CHECK"}),
    ):
        r = run([inst["binary"], *args], env = e, timeout = 180)
        raw[label] = {"rc": r["rc"], "s": r["s"], "stdout_chars": len(r["stdout"]),
                      "stderr": r["stderr"][:3000], **json_split(r["stdout"])}
        if label != "list":
            raw[label]["stdout_head"] = r["stdout"][:1500]
    out["raw"] = raw
    old = {k: os.environ.get(k) for k in ("FLM_MODEL_PATH", "FLM_DISABLE_UPDATE_CHECK")}
    os.environ.update(FLM_MODEL_PATH = str(root / "flm"), FLM_DISABLE_UPDATE_CHECK = "1")
    try:
        fh = load_flm_host(checkout)
        rows = fh.Flm(root).catalog()
        out["catalog"] = {"ok": True, "n": len(rows)}
    except BaseException as e:  # SystemExit from Flm() is a result too
        out["catalog"] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:1500]}
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return out


def ensure_backend_deps(checkout: Path) -> dict:
    backend = checkout / "studio" / "backend"
    sys.path.insert(0, str(backend))
    marker = work_root() / "backend_deps.ok"
    note = {}
    if not marker.is_file():
        req = backend / "requirements" / "studio.txt"
        r = run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req), "psutil"], timeout = 2400)
        note = {"pip_rc": r["rc"], "pip_tail": (r["stdout"] + r["stderr"])[-1500:], "pip_s": r["s"]}
        if r["rc"] == 0:
            marker.write_text("ok", encoding = "utf-8")
    return note


def write_tiny_gguf(path: Path) -> None:
    import numpy as np
    import gguf

    w = gguf.GGUFWriter(str(path), "llama")
    w.add_block_count(4)
    w.add_context_length(4096)
    w.add_embedding_length(64)
    w.add_head_count(4)
    w.add_feed_forward_length(128)
    w.add_tensor("token_embd.weight", np.zeros((256, 64), dtype = np.float32))
    for i in range(4):
        w.add_tensor(f"blk.{i}.attn_q.weight", np.zeros((64, 64), dtype = np.float32))
    w.add_tensor("output.weight", np.zeros((256, 64), dtype = np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


class Ballast:
    """Touched host pages, so Windows' available-RAM reading really drops."""

    def __init__(self) -> None:
        self.chunks = []

    def hold_down_to(self, target_free_gib: float, read_avail_mib, cap_gib: int = 56) -> dict:
        before = read_avail_mib()
        while len(self.chunks) < cap_gib:
            avail = read_avail_mib()
            if avail is None or avail / 1024 <= target_free_gib + 1:
                break
            try:
                buf = bytearray(GiB)
                mv = memoryview(buf)
                mv[::4096] = b"\x01" * len(range(0, GiB, 4096))
                self.chunks.append(buf)
            except MemoryError:
                break
        return {"avail_before_mib": before, "avail_after_mib": read_avail_mib(),
                "held_gib": len(self.chunks)}

    def release(self) -> None:
        self.chunks.clear()


def probe_apu(checkout: Path, ballast_gib: float | None = 10) -> dict:
    out = {"deps": ensure_backend_deps(checkout)}
    try:
        import torch

        out["torch"] = {"version": torch.__version__, "hip": getattr(torch.version, "hip", None),
                        "available": torch.cuda.is_available(),
                        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                        if torch.cuda.is_available() else []}
    except Exception as e:
        out["torch"] = {"error": f"{type(e).__name__}: {e}"}
    from core.inference.llama_cpp import LlamaCppBackend

    backend = LlamaCppBackend()
    ballast = Ballast()
    try:
        if ballast_gib is not None:
            out["ballast"] = ballast.hold_down_to(ballast_gib, backend._available_system_memory_mib)
        t0 = time.time()
        gpus = backend._get_gpu_memory()
        out["get_gpu_memory"] = {"rows": [list(r) for r in gpus], "s": round(time.time() - t0, 2)}
        unified = sorted(LlamaCppBackend._rocm_unified_memory_gpu_ids())
        out["unified_ids"] = unified
        idx = gpus[0][0] if gpus else 0
        out["carve_out_bytes"] = LlamaCppBackend._igpu_dedicated_memory_bytes([idx], rocm_fallback = False)
        out["wddm_used_bytes"] = backend._windows_apu_gpu_used_bytes()
        out["avail_mib"] = backend._available_system_memory_mib()
        gguf_path = work_root() / "tiny.gguf"
        write_tiny_gguf(gguf_path)
        backend._n_layers = 4
        argvs = {
            "full": ["llama-server", "-m", str(gguf_path), "--load-mode", "none", "-ngl", "99", "--fit", "off"],
            "partial": ["llama-server", "-m", str(gguf_path), "--load-mode", "none", "-ngl", "2", "--fit", "off"],
        }
        out["forces_full"] = {k: backend._launch_forces_full_offload(v, os.environ) for k, v in argvs.items()}
        carve = int(out["carve_out_bytes"] or 0)
        free_gib = sum(r[1] for r in gpus) / 1024
        top = max(carve / GiB, 32, free_gib + (out["avail_mib"] or 0) / 1024) + 16
        cells = []
        step = 4 if top <= 64 else 8
        for size_gib in range(4, int(top) + 1, step):
            size = size_gib * GiB
            backend._gguf_read_bytes = lambda _p, _s = size: _s
            for case, argv in argvs.items():
                cell = {"gib": size_gib, "case": case}
                try:
                    cell["credit"] = int(backend._windows_rocm_apu_carve_out_credit(
                        size, model_path = str(gguf_path), caps = {}, gpu_indices = [idx],
                        extra_args = argv, env = os.environ))
                except Exception as e:
                    cell["credit_error"] = f"{type(e).__name__}: {e}"
                t0 = time.perf_counter()
                try:
                    msg = backend._launch_host_shortfall_message(argv, gpus, os.environ)
                    cell["message"] = msg
                except Exception as e:
                    cell["error"] = f"{type(e).__name__}: {e}"
                cell["ms"] = round((time.perf_counter() - t0) * 1000, 1)
                cells.append(cell)
        out["cells"] = cells
    finally:
        ballast.release()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--parts", default = "npu,apu")
    ap.add_argument("--no-ballast", action = "store_true")
    args = ap.parse_args()
    obs = {"state": args.state, "platform": sys.platform}
    for part in args.parts.split(","):
        if part == "apu" and not WIN:
            continue
        try:
            if part == "npu":
                obs[part] = probe_npu(args.checkout.resolve())
            else:
                obs[part] = probe_apu(args.checkout.resolve(), None if args.no_ballast else 10)
        except BaseException as e:
            obs[part] = {"fatal": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-3000:]}
    args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
