#!/usr/bin/env python3
"""Measure what the two Model Memory toggles actually do to a llama-server launch.

Issue 9549: "Studio loads the model into system RAM despite the VRAM-only checks".

For each cell of the 2x2 toggle matrix, run twice -- once with no user extras and
once with a user-supplied ``--mlock`` -- because "Don't reserve system RAM" is a
flag-STRIPPING policy. With no extras to strip, OFF/OFF, OFF/ON and ON/ON emit
byte-identical argv, so a bare 2x2 would show three copies of one cell and read as
"the toggles do nothing" for the wrong reason.

Everything is captured from the live child process, not from Studio's own account of
it: argv comes from /proc/<pid>/cmdline, residency from /proc/<pid>/status and
smaps_rollup, GPU memory from amdgpu sysfs. Studio's self-report
(GET /api/settings/model-memory) is recorded alongside so the two can disagree
visibly.

Usage:
    python3 probe_9549.py --base-url http://127.0.0.1:8901 --password ... \
        --model-copies /path/copy0.gguf /path/copy1.gguf ... --out results/
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import urllib.error
import urllib.request

# The 8 cells. `extras` is what the USER typed into the llama.cpp extra-args box;
# it is the only thing no_ram_reserve has to strip.
CELLS = [
    {"keep_resident": kr, "no_ram_reserve": nrr, "extras": extras}
    for extras in ([], ["--mlock"])
    for kr in (False, True)
    for nrr in (False, True)
]

# A manual layer count above any real block count, so placement is a fixed full
# offload in every cell and the toggles stay the only variable. -1 would hand
# placement to the fitter, which re-decides per launch.
FULL_OFFLOAD_NGL = 999
FIXED_CTX = 2048


class Client:
    def __init__(self, base_url: str, token: Optional[str] = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token

    def request(
        self,
        method: str,
        path: str,
        body: Optional[dict] = None,
        timeout: int = 60,
    ) -> Any:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode(errors="ignore")
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode(errors="ignore")
            raise RuntimeError(f"{method} {path} -> {exc.code}: {raw[:2000]}") from exc
        # /load pads its response with spaces past 15s so a proxy cannot time it
        # out (routes/inference.py:2114); that padding is not JSON, so strip it.
        raw = raw.strip()
        if not raw:
            return None
        return json.loads(raw)


def login(base_url: str, username: str, password: str) -> str:
    client = Client(base_url)
    payload = client.request(
        "POST", "/api/auth/login", {"username": username, "password": password}
    )
    token = payload.get("access_token")
    if not token:
        raise RuntimeError(f"no access_token in login response: {payload}")
    if payload.get("must_change_password"):
        # Every authenticated route answers 403 while this flag is set, and the
        # failure is quiet: /healthz is fine and only the next call reveals it.
        raise RuntimeError(
            "must_change_password is set -- launch Studio with --password on a "
            "fresh UNSLOTH_STUDIO_HOME so the flag is cleared at startup"
        )
    return token


# --------------------------------------------------------------------------- #
# Host-side measurement
# --------------------------------------------------------------------------- #
def find_llama_server_pids() -> list[int]:
    pids = []
    for entry in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            with open(entry, "rb") as handle:
                raw = handle.read()
        except OSError:
            continue  # exited between the glob and the open
        if b"llama-server" not in raw:
            continue
        try:
            pids.append(int(entry.split("/")[2]))
        except (IndexError, ValueError):
            continue
    return sorted(pids)


def proc_cmdline(pid: int) -> Optional[list[str]]:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            raw = handle.read()
    except OSError:
        return None
    return [part.decode(errors="ignore") for part in raw.split(b"\0") if part]


def _kv_file(path: str, keys: tuple[str, ...]) -> dict[str, int]:
    """Parse a `Key:   123 kB` file into {key: kB}."""
    out: dict[str, int] = {}
    try:
        text = Path(path).read_text(errors="ignore")
    except OSError:
        return out
    for line in text.splitlines():
        parts = line.split(":", 1)
        if len(parts) != 2:
            continue
        key = parts[0].strip()
        if key not in keys:
            continue
        match = re.search(r"(\d+)", parts[1])
        if match:
            out[key] = int(match.group(1))
    return out


def proc_memory(pid: int) -> dict[str, Any]:
    status = _kv_file(
        f"/proc/{pid}/status", ("VmRSS", "VmLck", "VmPin", "VmSwap", "RssFile", "RssAnon")
    )
    rollup = _kv_file(
        f"/proc/{pid}/smaps_rollup",
        ("Rss", "Pss", "Shared_Clean", "Private_Clean", "Private_Dirty", "Locked"),
    )
    return {"status_kb": status, "smaps_rollup_kb": rollup}


def host_memory() -> dict[str, int]:
    return _kv_file(
        "/proc/meminfo",
        ("MemTotal", "MemFree", "MemAvailable", "Cached", "Buffers", "Mlocked", "AnonPages"),
    )


def amdgpu_sysfs() -> dict[str, dict[str, Optional[int]]]:
    """Per-card VRAM carve-out and GTT.

    On Strix Halo these are the only placement signal there is: the carve-out and
    GTT are both system DRAM, so a rise in `mem_info_vram_used` does NOT mean the
    weights left system RAM. It only says which heap ggml allocated from.
    """
    out: dict[str, dict[str, Optional[int]]] = {}
    for device in sorted(glob.glob("/sys/class/drm/card*/device")):
        card = device.split("/")[4]
        entry: dict[str, Optional[int]] = {}
        for name in (
            "mem_info_vram_used",
            "mem_info_vram_total",
            "mem_info_vis_vram_used",
            "mem_info_gtt_used",
            "mem_info_gtt_total",
        ):
            try:
                entry[name] = int(Path(device, name).read_text().strip())
            except (OSError, ValueError):
                entry[name] = None
        if any(value is not None for value in entry.values()):
            out[card] = entry
    return out


def run_capture(cmd: list[str], timeout: int = 30) -> str:
    if shutil.which(cmd[0]) is None:
        return f"<{cmd[0]} not on PATH>"
    try:
        done = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired:
        return f"<{cmd[0]} timed out>"
    return (done.stdout or "") + (done.stderr or "")


def child_log_findings(studio_home: Path, since: float) -> dict[str, Any]:
    """Read the newest llama-server child log written after `since`."""
    logs = sorted(
        (p for p in studio_home.glob("logs/llama-server/llama-*.log")
         if p.stat().st_mtime >= since - 5),
        key=lambda p: p.stat().st_mtime,
    )
    if not logs:
        return {"log": None}
    text = logs[-1].read_text(errors="ignore")
    offload = re.findall(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers to GPU", text)
    return {
        "log": str(logs[-1]),
        "offloaded": offload[-1] if offload else None,
        "failed_to_mlock": "failed to mlock" in text,
        "mmap_disabled": "mmap = false" in text or "--no-mmap" in text,
        "tail": text[-4000:],
    }


def backend_log_argv(studio_home: Path, since: float) -> Optional[str]:
    """Studio's own redacted record of the launch, for cross-check against /proc."""
    logs = sorted(studio_home.glob("logs/server/server-*.log"), key=lambda p: p.stat().st_mtime)
    if not logs:
        return None
    hits = [
        line for line in logs[-1].read_text(errors="ignore").splitlines()
        if "Starting llama-server: " in line
    ]
    return hits[-1] if hits else None


# --------------------------------------------------------------------------- #
# One cell
# --------------------------------------------------------------------------- #
def run_cell(
    client: Client,
    cell: dict,
    model_path: str,
    studio_home: Path,
    index: int,
    settle_s: float,
) -> dict[str, Any]:
    label = (
        f"keep_resident={cell['keep_resident']} "
        f"no_ram_reserve={cell['no_ram_reserve']} "
        f"extras={cell['extras'] or '[]'}"
    )
    print(f"\n===== cell {index}: {label} =====", flush=True)

    before_pids = set(find_llama_server_pids())
    started = time.time()

    client.request(
        "PUT",
        "/api/settings/model-memory",
        {"keep_resident": cell["keep_resident"], "no_ram_reserve": cell["no_ram_reserve"]},
    )
    settings_before = client.request("GET", "/api/settings/model-memory")

    mem_before = host_memory()
    gpu_before = amdgpu_sysfs()

    load_body = {
        "model_path": model_path,
        "force_reload": True,
        "force_cancel_active": True,
        "gpu_memory_mode": "manual",
        "gpu_layers": FULL_OFFLOAD_NGL,
        "max_seq_length": FIXED_CTX,
        "n_parallel": 1,
        "cache_ram": 0,
        "llama_extra_args": list(cell["extras"]),
    }
    load_error = None
    load_response = None
    try:
        load_response = client.request("POST", "/api/inference/load", load_body, timeout=900)
    except Exception as exc:  # noqa: BLE001 - a failed cell must not kill the matrix
        load_error = str(exc)
        print(f"  LOAD FAILED: {load_error}", flush=True)

    # Weights page in for a while after /load returns; sample once things settle.
    time.sleep(settle_s)

    after_pids = set(find_llama_server_pids())
    fresh = sorted(after_pids - before_pids)
    pid = fresh[-1] if fresh else (sorted(after_pids)[-1] if after_pids else None)

    record: dict[str, Any] = {
        "index": index,
        "label": label,
        "keep_resident": cell["keep_resident"],
        "no_ram_reserve": cell["no_ram_reserve"],
        "extras": cell["extras"],
        "model_path": model_path,
        "load_error": load_error,
        "load_response": load_response,
        "settings_before_load": settings_before,
        "settings_after_load": client.request("GET", "/api/settings/model-memory"),
        "system": client.request("GET", "/api/system"),
        "pid": pid,
        # A pid that did not change means this cell measured the previous cell's
        # child, which would make every downstream number a duplicate.
        "pid_is_fresh": bool(fresh),
        "argv": proc_cmdline(pid) if pid else None,
        "proc_memory": proc_memory(pid) if pid else None,
        "meminfo_before": mem_before,
        "meminfo_after": host_memory(),
        "amdgpu_before": gpu_before,
        "amdgpu_after": amdgpu_sysfs(),
        "rocm_smi_showpids": run_capture(["rocm-smi", "--showpids"]),
        "child_log": child_log_findings(studio_home, started),
        "backend_log_line": backend_log_argv(studio_home, started),
    }
    if record["argv"]:
        record["argv_hash"] = hashlib.sha256(
            "\0".join(record["argv"]).encode()
        ).hexdigest()[:16]
        # The model path differs per cell (distinct copies defeat the page cache),
        # so hash the argv with it removed or every cell looks unique.
        scrubbed = [a for a in record["argv"] if not a.endswith(".gguf")]
        record["argv_hash_no_model"] = hashlib.sha256(
            "\0".join(scrubbed).encode()
        ).hexdigest()[:16]
        record["has_mlock"] = "--mlock" in record["argv"]
        record["has_no_mmap"] = "--no-mmap" in record["argv"]
        record["load_mode"] = next(
            (
                record["argv"][i + 1]
                for i, a in enumerate(record["argv"])
                if a == "--load-mode" and i + 1 < len(record["argv"])
            ),
            None,
        )

    print(
        f"  pid={pid} fresh={record['pid_is_fresh']} "
        f"mlock={record.get('has_mlock')} load_mode={record.get('load_mode')} "
        f"VmLck={(record.get('proc_memory') or {}).get('status_kb', {}).get('VmLck')} kB "
        f"VmRSS={(record.get('proc_memory') or {}).get('status_kb', {}).get('VmRSS')} kB",
        flush=True,
    )

    try:
        client.request(
            "POST", "/api/inference/unload", {"model_path": model_path}, timeout=300
        )
    except Exception as exc:  # noqa: BLE001
        record["unload_error"] = str(exc)
    time.sleep(3)
    return record


# --------------------------------------------------------------------------- #
def render_table(records: list[dict]) -> str:
    header = (
        "| # | keep_resident | no_ram_reserve | user extras | --mlock in argv | "
        "--load-mode | VmLck MB | VmRSS MB | RssFile MB | RssAnon MB | "
        "mlock_active | reload_required | vram_used MB | gtt_used MB | offloaded |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for rec in records:
        status = (rec.get("proc_memory") or {}).get("status_kb", {})
        settings = rec.get("settings_after_load") or {}
        gpu = rec.get("amdgpu_after") or {}
        first = next(iter(gpu.values()), {}) if gpu else {}

        def mb(value: Optional[int], divisor: int = 1024) -> str:
            return "-" if value is None else f"{value / divisor:.0f}"

        offload = (rec.get("child_log") or {}).get("offloaded")
        rows.append(
            "| {i} | {kr} | {nrr} | {ex} | {ml} | {lm} | {lck} | {rss} | {rf} | {ra} | "
            "{ma} | {rr} | {vram} | {gtt} | {off} |".format(
                i=rec["index"],
                kr=rec["keep_resident"],
                nrr=rec["no_ram_reserve"],
                ex=" ".join(rec["extras"]) or "-",
                ml=rec.get("has_mlock"),
                lm=rec.get("load_mode") or "-",
                lck=mb(status.get("VmLck")),
                rss=mb(status.get("VmRSS")),
                rf=mb(status.get("RssFile")),
                ra=mb(status.get("RssAnon")),
                ma=settings.get("mlock_active"),
                rr=settings.get("reload_required"),
                vram=mb(first.get("mem_info_vram_used"), 1024 * 1024),
                gtt=mb(first.get("mem_info_gtt_used"), 1024 * 1024),
                off="/".join(offload) if offload else "-",
            )
        )
    return header + "\n".join(rows) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--username", default="unsloth")
    parser.add_argument("--password", required=True)
    parser.add_argument("--studio-home", required=True)
    parser.add_argument("--model-copies", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--settle-s", type=float, default=20.0)
    args = parser.parse_args()

    if len(args.model_copies) < len(CELLS):
        # Sharing one file would leave the GGUF in page cache after cell 0, so
        # every later cell would fault warm and the RSS/MemAvailable deltas would
        # measure the cache, not the load.
        print(
            f"WARNING: {len(args.model_copies)} copies for {len(CELLS)} cells; "
            "later cells will load from a warm page cache",
            file=sys.stderr,
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    studio_home = Path(args.studio_home)

    token = login(args.base_url, args.username, args.password)
    client = Client(args.base_url, token)

    records = []
    for index, cell in enumerate(CELLS):
        model = args.model_copies[index % len(args.model_copies)]
        records.append(
            run_cell(client, cell, model, studio_home, index, args.settle_s)
        )
        (out_dir / f"cell_{index}.json").write_text(json.dumps(records[-1], indent=2))

    (out_dir / "all_cells.json").write_text(json.dumps(records, indent=2))
    table = render_table(records)
    (out_dir / "table.md").write_text(table)
    print("\n" + table, flush=True)

    # Verdicts. These are the checks that separate a real measurement from a
    # green run that proves nothing.
    problems = []
    if any(not rec.get("pid_is_fresh") for rec in records):
        problems.append("a cell reused the previous llama-server pid")
    if any(rec.get("load_error") for rec in records):
        problems.append("at least one load failed")
    for rec in records:
        offload = (rec.get("child_log") or {}).get("offloaded")
        if offload and offload[0] != offload[1]:
            problems.append(f"cell {rec['index']} was not fully offloaded: {offload}")

    bare = [rec for rec in records if not rec["extras"]]
    with_mlock = [rec for rec in records if rec["extras"]]
    summary = {
        "bare_argv_hashes": {rec["index"]: rec.get("argv_hash_no_model") for rec in bare},
        "mlock_extra_argv_hashes": {
            rec["index"]: rec.get("argv_hash_no_model") for rec in with_mlock
        },
        "problems": problems,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)

    if problems:
        print("\nPROBLEMS: " + "; ".join(problems), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
