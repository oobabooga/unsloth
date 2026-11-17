#!/usr/bin/env python3
"""run_cell.py + whole-cell memory monitor + prof_hook on both stacks. Same CLI as run_cell.py.

- Every render writes its label (r0 = warm-up, r1..rN = new prompts, then long / short pairs) to
  $PROF_OUT/current_render, which prof_hook tags every denoiser / text-encoder / VAE call with (device-event timing).
- After the timed protocol, ONE extra untimed render of a new prompt ("prof") runs with denoiser call PROF_STEP and
  the first text-encoder / VAE call under torch.profiler, so the profiler never inflates a timed number.
- A 4 Hz monitor records the whole cell (load included): device VRAM + GTT used from amdgpu sysfs (delta over the
  reading before load), this process tree's RSS (the baseline's server is a child), and MemAvailable's drop.
Studio: the denoiser / text encoder / VAE are found on the loaded pipeline objects and wrapped per instance.
The baseline gets the hook from prof_comfy_node.py inside its server process.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import prof_hook  # noqa: E402
import run_cell  # noqa: E402
from backends import base as B  # noqa: E402


def _sysfs_used_mib():
    import glob

    for dev in sorted(glob.glob("/sys/class/drm/card[0-9]*/device")):
        try:
            rd = lambda n: int(Path(dev, n).read_text(encoding = "utf-8").strip())  # noqa: E731
            return (rd("mem_info_vram_used") + rd("mem_info_gtt_used")) >> 20
        except (OSError, ValueError):
            continue
    return None


def _memavail_mib():
    try:
        for line in open("/proc/meminfo", encoding = "utf-8"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) >> 10
    except OSError:
        return None
    return None


class Monitor(threading.Thread):
    def __init__(self):
        super().__init__(daemon = True)
        self.stop_ev = threading.Event()
        self.base_dev = _sysfs_used_mib()
        self.base_avail = _memavail_mib()
        self.peak_dev = self.base_dev or 0
        self.min_avail = self.base_avail or 0
        self.peak_rss = 0

    def run(self):
        import psutil

        me = psutil.Process()
        tl = None
        if os.environ.get("PROF_OUT"):
            Path(os.environ["PROF_OUT"]).mkdir(parents = True, exist_ok = True)
            tl = open(Path(os.environ["PROF_OUT"]) / "mem_timeline.csv", "a", encoding = "utf-8")
            tl.write("t,dev_mib,tree_rss_mib,memavail_mib\n")
        while not self.stop_ev.is_set():
            d = _sysfs_used_mib()
            if d is not None:
                self.peak_dev = max(self.peak_dev, d)
            a = _memavail_mib()
            if a is not None:
                self.min_avail = min(self.min_avail, a) if self.min_avail else a
            try:
                rss = me.memory_info().rss
                for c in me.children(recursive = True):
                    try:
                        rss += c.memory_info().rss
                    except Exception:  # noqa: BLE001
                        pass
                self.peak_rss = max(self.peak_rss, rss >> 20)
            except Exception:  # noqa: BLE001
                rss = 0
            if tl is not None:
                tl.write(f"{time.time():.2f},{d or 0},{rss >> 20},{a or 0}\n")
            self.stop_ev.wait(0.25)
        if tl is not None:
            tl.close()

    def result(self) -> dict:
        return {"cell_dev_base_gib": round((self.base_dev or 0) / 1024, 3),
                "cell_dev_peak_delta_gib": round(max(0, self.peak_dev - (self.base_dev or 0)) / 1024, 3),
                "cell_tree_rss_peak_gib": round(self.peak_rss / 1024, 3),
                "cell_memavail_drop_gib": round(max(0, (self.base_avail or 0) - (self.min_avail or 0)) / 1024, 3)}


def _set_render(label: str) -> None:
    if prof_hook.enabled():
        p = Path(os.environ["PROF_OUT"])
        p.mkdir(parents = True, exist_ok = True)
        (p / "current_render").write_text(label, encoding = "utf-8")
        with open(p / "renders.jsonl", "a", encoding = "utf-8") as f:
            f.write(json.dumps({"render": label, "t": round(time.time(), 3)}) + "\n")


def _attach_studio() -> dict:
    """Wrap the loaded pipeline's denoiser forward, text encoder forward and VAE decode (per instance)."""
    import torch

    seen, got = set(), {"denoiser": 0, "te": 0, "vae": 0, "pipes": []}
    for obj in gc.get_objects():
        try:
            if not (hasattr(obj, "components") and hasattr(obj, "vae") and type(obj).__name__.endswith("Pipeline")):
                continue
        except Exception:  # noqa: BLE001
            continue
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        got["pipes"].append(type(obj).__name__)
        for attr, method, label in (("transformer", "forward", "studio_denoiser"), ("unet", "forward", "studio_denoiser"),
                                    ("text_encoder", "forward", "studio_te"), ("vae", "decode", "studio_vae")):
            m = getattr(obj, attr, None)
            if isinstance(m, torch.nn.Module):
                if prof_hook.attach_instance(m, method, label):
                    got[label.split("_")[1]] += 1
    return got


_orig_get = B.get_backend
_ORDER = {"n": 0}


def _get_backend(name):
    cls = _orig_get(name)
    if getattr(cls, "_prof_patched", False):
        return cls
    o_load, o_render, o_status = cls.load, cls.render, cls.status

    def load(self):
        st = o_load(self)
        if name == "studio" and prof_hook.enabled():
            got = _attach_studio()
            print(f"prof_hook: Studio attach {got}", flush = True)
            if isinstance(st, dict):
                st["prof_attach"] = got
        return st

    def render(self, row, steps):
        _set_render(row.get("_prof_label") or f"r{_ORDER['n']}")
        if not row.get("_prof_label"):
            _ORDER["n"] += 1
        return o_render(self, row, steps)

    def status(self):
        # run_cell calls status() once after the timed renders: the extra profiled render goes here
        if _ORDER["n"] > 0 and prof_hook.enabled() and not getattr(self, "_prof_done", False):
            self._prof_done = True
            row = {"id": "prof", "prompt": os.environ.get("PROF_PROMPT", "a red bicycle leaning on a stone wall at dusk, "
                                                                            "film photo"),
                   "seed": 4242, "_prof_label": "prof"}
            steps = int(os.environ.get("PROF_STEPS_TOTAL", "0")) or int(self.cell.get("steps") or 1)
            (Path(os.environ["PROF_OUT"]) / "prof_step").write_text(
                str(max(0, min(int(os.environ.get("PROF_STEP", "5")), steps // 2))), encoding = "utf-8")
            try:
                t = time.perf_counter()
                self.render(row, steps)  # the patched render: writes the 'prof' label first
                self.sync()
                print(f"prof_hook: profiled render {time.perf_counter() - t:.2f}s", flush = True)
            except Exception as exc:  # noqa: BLE001
                print(f"prof_hook: profiled render failed: {exc!r}", flush = True)
            _set_render("after")
        return o_status(self)

    cls.load, cls.render, cls.status = load, render, status
    cls._prof_patched = True
    return cls


B.get_backend = _get_backend
run_cell.get_backend = _get_backend


def _arg(flag):
    a = sys.argv
    return a[a.index(flag) + 1] if flag in a else None


if __name__ == "__main__":
    mon = Monitor()
    mon.start()
    t0 = time.time()
    rc = 1
    try:
        rc = run_cell.main()
    finally:
        prof_hook.flush()
        mon.stop_ev.set()
        mon.join(2)
        tag, out = _arg("--tag"), _arg("--out")
        if tag and out:
            rp = Path(out) / tag / "record.json"
            try:
                rec = json.loads(rp.read_text(encoding = "utf-8"))
                rec.update(mon.result())
                rec["cell_wall_s"] = round(time.time() - t0, 1)
                rp.write_text(json.dumps(rec, indent = 1, default = str), encoding = "utf-8")
            except Exception as exc:  # noqa: BLE001
                print(f"prof_cell: could not amend {rp}: {exc!r}", flush = True)
    raise SystemExit(rc)
