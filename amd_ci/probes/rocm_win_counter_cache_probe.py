#!/usr/bin/env python3
"""Probe: how many `powershell Get-Counter` processes do Studio's Windows ROCm
counter reads start, inside and outside gpu_query.display_reads()?

Observes only; criteria/rocm_win_counter_cache.py judges. Written for PR 17306
(display polls read all three counters in ONE Get-Counter call and reuse it for
max(display TTL, 4x that call's duration)), but nothing in here knows which state
has the cache: it counts spawns (a call over all three counters is "combined",
with its duration recorded) and records values.

Per state, in one process importing THIS checkout's backend the way the app does
(studio/backend on sys.path, `utils.hardware.hardware`):

  env       pip-installs the checkout's requirements/studio.txt and, unless
            --rocm-torch-index is empty, a ROCm torch from that index, so the
            public display paths can reach the ROCm branch. Idempotent, so the
            second state finds it installed: both states run in the same env.
  direct    the three counter readers themselves (Dedicated Usage, Shared Usage,
            3D engine utilization): reachable on any Windows box, no torch.
            Phases, N iterations each:
              fresh            no scope (placement / fit reads)
              display          inside display_reads()
              display_fresh    inside display_reads() + fresh_reads()
              invalidation     display read, gpu_memory_events bump, display read
            plus a fresh read right after the display phase, to compare values.
  public    get_gpu_utilization / get_visible_gpu_utilization (the /hardware,
            /hardware/visible and /api/system display routes), each called the
            way the routes do (display_reads() around asyncio.to_thread). First a
            discovery pass outside any scope records which of them reach a
            Get-Counter at all on this box; only those are measured.
  pytest    the PR's tests through probes/pytest_probe.py (same probe the
            generic pipeline uses), result embedded under "pytest".

Output goes to --out as JSON, never stdout.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

_REAL_RUN = subprocess.run
_LOCK = threading.Lock()
_SPAWNS: list[dict] = []


def _counter_of(script: str) -> str:
    # One call over all three display counters (the head's display read) serves each of them.
    if "Get-Counter -Counter" in script or (
            "GPU Engine" in script and "Shared Usage" in script and "Dedicated Usage" in script):
        return "combined"
    if "GPU Engine" in script:
        return "util"
    if "Shared Usage" in script:
        return "vram_shared"
    if "Dedicated Usage" in script:
        return "vram_dedicated"
    return "other"


def _counting_run(*args, **kwargs):
    argv = args[0] if args else kwargs.get("args")
    try:
        flat = argv if isinstance(argv, str) else " ".join(str(a) for a in argv)
    except Exception:  # noqa: BLE001
        flat = ""
    if "Get-Counter" in flat:
        t0 = time.monotonic()
        try:
            return _REAL_RUN(*args, **kwargs)
        finally:
            with _LOCK:
                _SPAWNS.append({"counter": _counter_of(flat), "t": t0,
                                "dur_s": round(time.monotonic() - t0, 3)})
    return _REAL_RUN(*args, **kwargs)


def _count(since: int) -> dict:
    out: dict = {}
    with _LOCK:
        for s in _SPAWNS[since:]:
            out[s["counter"]] = out.get(s["counter"], 0) + 1
    return out


def _durations(since: int, counter: str) -> list[float]:
    with _LOCK:
        return [s["dur_s"] for s in _SPAWNS[since:] if s["counter"] == counter]


def _mark() -> int:
    with _LOCK:
        return len(_SPAWNS)


def _jsonable(v):
    try:
        return json.loads(json.dumps(v, default = str))
    except Exception:  # noqa: BLE001
        return repr(v)


def _pip(py: str, args: list[str], timeout: int) -> dict:
    t0 = time.monotonic()
    try:
        r = _REAL_RUN([py, "-m", "pip", "install", "-q", "--disable-pip-version-check", *args],
                      capture_output = True, text = True, encoding = "utf-8", errors = "replace",
                      timeout = timeout)
        return {"args": args, "rc": r.returncode, "s": round(time.monotonic() - t0, 1),
                "tail": (r.stdout + r.stderr)[-1500:]}
    except Exception as e:  # noqa: BLE001
        return {"args": args, "rc": None, "error": f"{type(e).__name__}: {e}"}


def _torch_info(py: str) -> dict:
    code = ("import json,torch;print('AMDCI'+json.dumps({'version':torch.__version__,"
            "'hip':getattr(torch.version,'hip',None),'cuda':torch.version.cuda,"
            "'available':torch.cuda.is_available(),'count':torch.cuda.device_count(),"
            "'name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))")
    try:
        r = _REAL_RUN([py, "-c", code], capture_output = True, text = True, encoding = "utf-8",
                      errors = "replace", timeout = 300)
        for line in r.stdout.splitlines():
            if line.startswith("AMDCI"):
                return json.loads(line[5:])
        return {"error": (r.stdout + r.stderr)[-800:]}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def setup_env(checkout: Path, rocm_index: str, obs: dict) -> None:
    py = sys.executable
    env: dict = {"python": py}
    req = checkout / "studio" / "backend" / "requirements" / "studio.txt"
    env["studio_reqs"] = _pip(py, ["-r", str(req)], 2400) if req.is_file() else {"error": f"missing {req}"}
    if env["studio_reqs"].get("rc") != 0:
        # The handful the hardware module and the tests' conftest import, so one
        # wheel failing to build does not take the whole probe with it.
        env["minimal_reqs"] = _pip(py, ["structlog", "httpx", "pydantic", "huggingface_hub",
                                        "fastapi", "pytest", "pytest-timeout"], 900)
    if rocm_index:
        before = _torch_info(py)
        if before.get("hip"):
            env["rocm_torch_install"] = {"skipped": "a ROCm torch is already installed"}
        else:
            env["rocm_torch_install"] = _pip(py, ["--index-url", rocm_index, "--pre", "torch"], 2700)
    env["torch"] = _torch_info(py)
    obs["env"] = env


def import_hardware(checkout: Path):
    backend = checkout / "studio" / "backend"
    if not backend.is_dir():
        raise SystemExit(f"no backend at {backend}")
    sys.path.insert(0, str(backend))
    os.chdir(backend)
    for stale in [m for m in sys.modules if m == "utils" or m.startswith("utils.")]:
        del sys.modules[stale]
    import utils.hardware.hardware as hw  # noqa: PLC0415
    from utils import gpu_memory_events  # noqa: PLC0415
    from utils.hardware import gpu_query  # noqa: PLC0415
    return hw, gpu_query, gpu_memory_events


def _direct_read_all(hw) -> dict:
    return {
        "vram_dedicated": hw._rocm_windows_perf_counter_vram_by_adapter(),
        "vram_shared": hw._rocm_windows_perf_counter_vram_by_adapter("Shared Usage"),
        "util": hw._rocm_windows_perf_counter_gpu_util_pct(),
    }


def _phase(fn, n: int, scope) -> dict:
    """Call fn n times inside scope(); spawns per counter, wall time, every value."""
    m = _mark()
    values, per_call = [], []
    t0 = time.monotonic()
    with scope():
        for _ in range(n):
            c0 = time.monotonic()
            try:
                values.append(_jsonable(fn()))
            except Exception as e:  # noqa: BLE001
                values.append({"__error__": f"{type(e).__name__}: {e}"})
            per_call.append(round(time.monotonic() - c0, 3))
    return {"n": n, "wall_s": round(time.monotonic() - t0, 3), "per_call_s": per_call,
            "spawns": _count(m), "combined_dur_s": _durations(m, "combined"), "values": values}


def _route_call(fn):
    """What the /hardware routes do: the display scope is entered on the loop thread
    and the read runs in asyncio.to_thread, which carries the context over."""
    async def go():
        return await asyncio.to_thread(fn)
    return asyncio.run(go())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--n", type = int, default = 10)
    ap.add_argument("--rocm-torch-index", default = "https://rocm.nightlies.amd.com/v2/gfx1151/",
                    help = "pip index for a ROCm torch; empty skips it (public paths then "
                           "likely never reach the ROCm branch, which is recorded)")
    ap.add_argument("--tests", nargs = "*", default = [
        "tests/test_rocm_windows_counter_display_cache.py",
        "tests/test_rocm_windows_vram_7072.py"])
    args = ap.parse_args()
    out = args.out.resolve()
    checkout = args.checkout.resolve()
    obs: dict = {"state": args.state, "n": args.n, "platform": sys.platform}

    def save() -> None:
        out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")

    try:
        setup_env(checkout, args.rocm_torch_index, obs)
        save()
        # pytest first, in its own interpreter, so this process's patching cannot touch it.
        if args.tests:
            pt_out = out.with_name(out.stem + ".pytest.json")
            probe = Path(__file__).resolve().parent / "pytest_probe.py"
            r = _REAL_RUN([sys.executable, str(probe), "--state", args.state,
                           "--checkout", str(checkout), "--out", str(pt_out),
                           "--tests", *args.tests],
                          capture_output = True, text = True, encoding = "utf-8",
                          errors = "replace", timeout = 3700)
            obs["pytest_probe_rc"] = r.returncode
            try:
                obs["pytest"] = json.loads(pt_out.read_text(encoding = "utf-8"))
            except Exception as e:  # noqa: BLE001
                obs["pytest"] = {"error": f"{type(e).__name__}: {e}",
                                 "probe_output": (r.stdout + r.stderr)[-1500:]}
            save()

        hw, gq, events = import_hardware(checkout)
        obs["hardware_file"] = hw.__file__
        obs["has_counter_cache"] = hasattr(hw, "_rocm_windows_counter_stdout")
        obs["has_combined_read"] = hasattr(hw, "_rocm_windows_combined_counter_ps")
        obs["display_ttl_s"] = gq.ttl_for(gq.DISPLAY)
        obs["gpu_query_cache_enabled"] = gq.enabled()
        # Patched on the module object hardware.py calls through (hw.subprocess is
        # the global subprocess module), so every Get-Counter spawn is counted.
        subprocess.run = _counting_run
        n = args.n
        nullscope = _NullScope

        # ---- direct: the counter readers themselves -----------------------------
        d: dict = {}
        d["fresh"] = _phase(lambda: _direct_read_all(hw), n, nullscope)
        d["display"] = _phase(lambda: _direct_read_all(hw), n, gq.display_reads)
        # The reference for the display values: a fresh read taken right after.
        d["fresh_after_display"] = _phase(lambda: _direct_read_all(hw), 1, nullscope)
        d["display_fresh"] = _phase(lambda: _direct_read_all(hw), n,
                                    lambda: _Both(gq.display_reads(), gq.fresh_reads()))
        # Invalidation: warm the display cache, bump the generation, read again.
        m = _mark()
        with gq.display_reads():
            hw._rocm_windows_perf_counter_vram_by_adapter()
            warm = _count(m)
            # Control: the same re-read without a bump, so a respawn after the bump
            # is attributable to the bump and not to TTL expiry.
            m1 = _mark()
            hw._rocm_windows_perf_counter_vram_by_adapter()
            control = _count(m1)
            m2 = _mark()
            events.invalidate_gpu_memory("amd-ci probe")
            hw._rocm_windows_perf_counter_vram_by_adapter()
            after = _count(m2)
        d["invalidation"] = {"warm_spawns": warm, "no_bump_spawns": control,
                             "after_bump_spawns": after}
        obs["direct"] = d
        save()

        # ---- public display paths ----------------------------------------------
        pub: dict = {}
        try:
            pub["device"] = str(hw.get_device())
        except Exception as e:  # noqa: BLE001
            pub["device_error"] = f"{type(e).__name__}: {e}"
        pub["is_rocm"] = bool(getattr(hw, "IS_ROCM", False))
        funcs = {"get_gpu_utilization": hw.get_gpu_utilization,
                 "get_visible_gpu_utilization": hw.get_visible_gpu_utilization}
        discovery: dict = {}
        for name, fn in funcs.items():
            # Twice: the first call may pay one-off detection that takes a different route.
            for attempt in ("first", "second"):
                m = _mark()
                try:
                    _route_call(fn)
                    err = None
                except Exception as e:  # noqa: BLE001
                    err = f"{type(e).__name__}: {e}"
                discovery.setdefault(name, {})[attempt] = {"spawns": _count(m), "error": err}
        pub["discovery"] = discovery
        engaged = [nm for nm, v in discovery.items() if sum(v["second"]["spawns"].values()) > 0]
        pub["engaged"] = engaged
        for name in engaged:
            fn = funcs[name]
            pub[name] = {
                "fresh": _phase(lambda fn=fn: _route_call(fn), n, nullscope),
                "display": _phase(lambda fn=fn: _route_call(fn), n, gq.display_reads),
                "fresh_after_display": _phase(lambda fn=fn: _route_call(fn), 1, nullscope),
            }
        obs["public"] = pub
    except BaseException as e:  # noqa: BLE001
        obs["error"] = f"{type(e).__name__}: {e}"
        obs["traceback"] = traceback.format_exc()[-4000:]
    finally:
        subprocess.run = _REAL_RUN
        with _LOCK:
            obs["all_spawns"] = len(_SPAWNS)
        save()
    return 0


class _NullScope:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Both:
    def __init__(self, a, b):
        self.a, self.b = a, b

    def __enter__(self):
        self.a.__enter__()
        self.b.__enter__()
        return self

    def __exit__(self, *exc):
        self.b.__exit__(*exc)
        self.a.__exit__(*exc)
        return False


if __name__ == "__main__":
    sys.exit(main())
