#!/usr/bin/env python3
"""Probe: the /api/system GPU payload this checkout's backend builds here, and what this checkout's
frontend (floating monitor total, VRAM tile capacity, carve-out reading) shows from it. Observes only."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

NODE_VERSION = "v24.11.1"

DRIVER = r"""
import { pathToFileURL } from "node:url";
import { readFileSync } from "node:fs";
const [src, payloadPath] = process.argv.slice(2);
const load = (p) => import(pathToFileURL(`${src}/${p}`).href);
const md = await load("hooks/gpu-memory-display.ts");
const gv = await load("hooks/gpu-vram.ts");
const mc = await load("components/monitor-cards.ts");
const { gpu, ram_available_gb } = JSON.parse(readFileSync(payloadPath, "utf8"));
const display = md.gpuMemoryDisplay(gpu);
const cards = mc.vramCards(display.usageDevices, display.usageGpu);
const totals = gv.gpuMemoryTotalsGb(display.usageDevices);
const used = gv.resolveGpuVramUsedGb(display.usageGpu);
const carve = typeof gv.carveOutVramGb === "function"
  ? gv.carveOutVramGb(display.usageDevices, used, ram_available_gb)
  : null;
const unifiedPool = totals.shared > 0 && totals.dedicated === 0;
// Same precedence as resources-tab.tsx vramCapacityLabel.
const tileCapacity = !carve && totals.shared > 0 && !unifiedPool
  ? `${totals.dedicated} + ${totals.shared} shared`
  : `${unifiedPool ? totals.shared : (carve?.totalGb ?? totals.total)}`;
console.log(JSON.stringify({
  has_carve_out_api: typeof gv.carveOutVramGb === "function",
  shared_devices: display.sharedDevices.map((d) => d.name),
  usage_devices: display.usageDevices.map((d) => d.name),
  totals, used_gb: used, carve_out: carve,
  floating_total_gb: cards.totalGb, floating_used_gb: cards.usedGb,
  tile_capacity: tileCapacity,
}));
"""


def backend_payload(checkout: Path) -> dict:
    backend = checkout / "studio" / "backend"
    sys.path.insert(0, str(backend))
    for stale in [m for m in sys.modules if m.startswith("utils")]:
        del sys.modules[stale]
    from utils.hardware import get_backend_visible_gpu_info, get_visible_gpu_utilization  # noqa: PLC0415

    vis = get_backend_visible_gpu_info() or {"available": False, "devices": []}
    util = get_visible_gpu_utilization() or {"devices": []}
    match = not vis.get("backend") or not util.get("backend") or vis.get("backend") == util.get("backend")
    by_idx = {d.get("index"): d for d in util.get("devices", [])} if match else {}
    devices = []
    # Mirrors main._get_cached_system_gpu_info's per-device merge.
    for dev in vis.get("devices", []):
        u = by_idx.get(dev.get("index"), {})
        total = u.get("vram_total_gb") or dev.get("memory_total_gb") or 0
        used = u.get("vram_used_gb", dev.get("vram_used_gb"))
        free = u.get("vram_free_gb", dev.get("vram_free_gb"))
        d = dict(dev)
        d["vram_used_gb"] = used
        d["vram_free_gb"] = free if free is not None else (round(total - used, 2) if total and used is not None else None)
        d["vram_utilization_pct"] = u.get("vram_utilization_pct", dev.get("vram_utilization_pct"))
        devices.append(d)
    same = match and {d.get("index") for d in util.get("devices", [])} == {d.get("index") for d in devices}
    return {
        **vis,
        "devices": devices,
        "vram_used_gb_aggregate": util.get("vram_used_gb_aggregate") if same else None,
    }


def find_node(work: Path) -> str:
    for cand in [shutil.which("node")]:
        if cand:
            try:
                v = subprocess.run([cand, "--version"], capture_output = True, text = True, timeout = 30).stdout.strip()
                major, minor = (int(x) for x in v.lstrip("v").split(".")[:2])
                if (major, minor) >= (22, 18):
                    return cand
            except Exception:  # noqa: BLE001
                pass
    win = platform.system() == "Windows"
    name = f"node-{NODE_VERSION}-{'win' if win else 'linux'}-x64"
    dest = work / name
    exe = dest / ("node.exe" if win else "bin/node")
    if not exe.exists():
        url = f"https://nodejs.org/dist/{NODE_VERSION}/{name}.{'zip' if win else 'tar.xz'}"
        arc = work / Path(url).name
        urllib.request.urlretrieve(url, arc)
        if win:
            zipfile.ZipFile(arc).extractall(work)
        else:
            tarfile.open(arc).extractall(work)
    return str(exe)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()
    obs: dict = {"state": args.state, "platform": platform.system()}
    work = Path(os.environ.get("RUNNER_TEMP") or args.out.parent) / "vram_probe_node"
    work.mkdir(parents = True, exist_ok = True)
    try:
        import psutil

        obs["ram_available_gb"] = round(psutil.virtual_memory().available / 1024**3, 2)
    except Exception as e:  # noqa: BLE001
        obs["ram_available_gb"] = None
        obs["ram_error"] = repr(e)
    try:
        obs["gpu"] = backend_payload(args.checkout)
    except Exception as e:  # noqa: BLE001
        obs["payload_error"] = f"{type(e).__name__}: {e}"
    if "gpu" in obs:
        try:
            node = find_node(work)
            payload = work / f"payload_{args.state}.json"
            payload.write_text(json.dumps({"gpu": obs["gpu"], "ram_available_gb": obs["ram_available_gb"]}), encoding = "utf-8")
            driver = work / "driver.mjs"
            driver.write_text(DRIVER, encoding = "utf-8")
            src = args.checkout / "studio" / "frontend" / "src"
            r = subprocess.run(
                [node, "--experimental-strip-types", "--no-warnings", str(driver), str(src), str(payload)],
                capture_output = True, text = True, encoding = "utf-8", timeout = 120,
            )
            obs["node_rc"] = r.returncode
            obs["node_stderr"] = r.stderr[-2000:]
            obs["display"] = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else None
        except Exception as e:  # noqa: BLE001
            obs["display_error"] = f"{type(e).__name__}: {e}"
    args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
