#!/usr/bin/env python3
"""Probe: load a real FastFlowLM model on the NPU through Studio's own backend, then read what the
checkout's live-monitor code reports while it generates. Observes only.

Alongside, it reads the NPU adapter's GPU Engine counters itself, the source Task Manager's NPU graph
uses, so the checkout's reading can be compared with an independent one.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path


def _raw_npu_util(luid: int) -> float | None:
    key = f"luid_0x{luid >> 32:08x}_0x{luid & 0xFFFFFFFF:08x}"
    ps = (
        f"(Get-Counter '\\GPU Engine(*{key}*)\\Utilization Percentage' -ErrorAction SilentlyContinue)"
        ".CounterSamples|ForEach-Object{$_.InstanceName+'|'+$_.CookedValue.ToString([cultureinfo]::InvariantCulture)}"
    )
    r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                       capture_output = True, text = True, encoding = "utf-8", errors = "replace",
                       timeout = 30)
    per_engine: dict[str, float] = {}
    for line in r.stdout.splitlines():
        name, _, raw = line.strip().rpartition("|")
        m = re.search(r"_eng_(\d+)", name)
        try:
            v = float(raw)
        except ValueError:
            continue
        if m and v == v and v != float("inf"):
            per_engine[m.group(1)] = per_engine.get(m.group(1), 0.0) + v
    return max(per_engine.values()) if per_engine else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()

    home = Path(os.environ.get("RUNNER_TEMP") or args.out.parent) / f"npu_home_{args.state}"
    os.environ["UNSLOTH_STUDIO_HOME"] = str(home)
    backend_dir = args.checkout / "studio" / "backend"
    sys.path.insert(0, str(backend_dir))
    obs: dict = {"state": args.state, "home": str(home), "steps": []}

    def flush() -> None:
        args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")

    def step(name: str, **kw) -> None:
        obs["steps"].append({"t": round(time.time(), 1), "step": name, **kw})
        print(f"[{args.state}] {name} {kw}", flush = True)
        flush()

    be = None
    try:
        import utils.hardware.hardware as hw  # noqa: PLC0415

        obs["has_npu_load_percent"] = hasattr(hw, "npu_load_percent")
        if not obs["has_npu_load_percent"]:
            step("base has no NPU load reading; nothing to drive")
            flush()
            return 0
        import routes.inference as ri  # noqa: PLC0415
        from core.inference.npu_backend import get_npu_backend  # noqa: PLC0415

        obs["has_npu_resident_mib"] = hasattr(ri, "_npu_resident_mib")
        be = get_npu_backend()
        step("hardware", hardware = be.hardware())
        t0 = time.time()
        status = be.enable()
        step("enabled", seconds = round(time.time() - t0, 1), status = status)
        catalog = [m for m in be.catalog() if m.size_gb]
        catalog.sort(key = lambda m: (not m.downloaded, m.size_gb))
        model = catalog[0]
        step("picked", model = model.to_json(), n_catalog = len(catalog))
        if not model.downloaded:
            t0 = time.time()
            last = None
            for last in be.download(model.id):
                pass
            step("downloaded", seconds = round(time.time() - t0, 1), last = last)
        t0 = time.time()
        be.load(model.id)
        step("loaded", seconds = round(time.time() - t0, 1))
        obs["catalog_mib"] = int(model.size_gb * 1000**3 // (1024 * 1024))
        obs["resident_loaded"] = ri._npu_resident_mib()
        luid = None
        load_before = hw.npu_load_percent()
        luid = hw._npu_luid
        obs["luid"] = luid
        obs["idle_loaded_reading"] = load_before
        step("idle with model resident", reading = load_before, luid = luid,
             resident = obs["resident_loaded"])

        resident = be.resident()
        samples: list[dict] = []
        stop = threading.Event()

        def sampler() -> None:
            while not stop.is_set():
                ts = time.time()
                ours = hw.npu_load_percent()
                raw = _raw_npu_util(luid) if luid else None
                samples.append({"t": round(ts, 2), "ours": ours, "raw": raw,
                                "resident": ri._npu_resident_mib()})
                stop.wait(1.0)

        th = threading.Thread(target = sampler, daemon = True)
        th.start()
        import httpx  # noqa: PLC0415

        gens = []
        t_end = time.time() + 45
        while time.time() < t_end:
            t0 = time.time()
            r = httpx.post(
                f"{resident.base_url}/v1/chat/completions",
                headers = {"Authorization": f"Bearer {resident.api_key}"} if resident.api_key else {},
                json = {"model": model.id, "max_tokens": 384, "stream": False,
                        "messages": [{"role": "user", "content": "Write a long story about a lighthouse keeper."}]},
                timeout = 300,
            )
            body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            gens.append({"status": r.status_code, "seconds": round(time.time() - t0, 1),
                         "usage": body.get("usage"), "error": None if r.is_success else r.text[:300]})
            if not r.is_success:
                break
        stop.set()
        th.join(timeout = 60)
        obs["generations"] = gens
        obs["samples"] = samples
        step("generated", n = len(gens), n_samples = len(samples))
        be.unload()
        time.sleep(5)
        hw._npu_load_cache = None
        obs["after_unload_reading"] = hw.npu_load_percent()
        obs["after_unload_resident"] = ri._npu_resident_mib()
        step("unloaded", reading = obs["after_unload_reading"])
    except Exception as e:  # noqa: BLE001
        obs["error"] = f"{type(e).__name__}: {e}"
        obs["traceback"] = traceback.format_exc()[-3000:]
    finally:
        if be is not None:
            try:
                be.shutdown()
            except Exception as e:  # noqa: BLE001
                obs["shutdown_error"] = str(e)
        flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
