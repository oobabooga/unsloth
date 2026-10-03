"""Shared plumbing for diffusion_bench: prompts, cell specs, GPU sampling and gating, host memory,
environment fingerprint, media saving and the record file every backend writes.

Nothing here imports torch at module level, so the matrix driver and the scorer can run from any
Python (the render itself always runs in the cell's own venv, one cell per process).
"""

from __future__ import annotations

import base64
import io
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Optional

PKG = Path(__file__).resolve().parent
WS = Path(os.environ.get("WORKSPACE") or os.getcwd()).resolve()
RECORD = "record.json"

# Defaults for a cell. A spec's "defaults" block and each cell override these, in that order.
CELL_DEFAULTS: dict = {
    "kind": "image",  # image | video
    "width": 1024,
    "height": 1024,
    "steps": 25,
    "guidance": 1.0,
    "negative_prompt": None,
    "frames": None,  # video only
    "fps": None,
    "prompts": "image_24",  # a name under prompts/ or a path to a json/txt file
    "ids": None,  # explicit prompt ids instead of the first n
    "n": 24,
    "warmup": 1,
    "short_steps": 5,  # s/step = (median long - median short) / (steps - short_steps)
    "short_n": 3,
    "save_frames_every": 1,  # video: keep every k-th frame in frames.npz for scoring
    "options": {},
    "env": {},
}


# ---------------------------------------------------------------------------------------------- specs
def merge_cell(defaults: dict, cell: dict) -> dict:
    out = {**CELL_DEFAULTS, **(defaults or {}), **cell}
    out["options"] = {**CELL_DEFAULTS["options"], **(defaults or {}).get("options", {}), **cell.get("options", {})}
    out["env"] = {**(defaults or {}).get("env", {}), **cell.get("env", {})}
    return out


def load_spec(path: str | Path) -> dict:
    spec = json.loads(Path(path).read_text(encoding = "utf-8"))
    spec.setdefault("name", Path(path).stem)
    spec.setdefault("defaults", {})
    spec.setdefault("cells", [])
    tags = [c["tag"] for c in spec["cells"]]
    dup = {t for t in tags if tags.count(t) > 1}
    if dup:
        raise ValueError(f"duplicate cell tags in {path}: {sorted(dup)}")
    resolve_model_aliases(spec)
    return spec


def model_env_key(alias: str) -> str:
    """DBENCH_MODEL_<ALIAS>: the per-host override of a spec model alias (a local dir or another repo id)."""
    return "DBENCH_MODEL_" + "".join(ch if ch.isalnum() else "_" for ch in alias).upper()


def resolve_model(value: Any, models: dict) -> tuple:
    """``"@alias"`` -> (path or repo id, provenance). Order: $DBENCH_MODEL_<ALIAS>, the alias's "local" path when it
    exists on this host, then its "repo" id. Anything not starting with "@" is returned unchanged."""
    if not (isinstance(value, str) and value.startswith("@")):
        return value, None
    alias = value[1:]
    if alias not in models:
        raise KeyError(f"model alias {value!r} is not in the spec's \"models\" table: {sorted(models)}")
    entry = models[alias] or {}
    env = os.environ.get(model_env_key(alias))
    if env:
        return expand_env(env), {"alias": alias, "source": "env", "repo": entry.get("repo")}
    local = entry.get("local")
    if local and Path(expand_env(local)).exists():
        return expand_env(local), {"alias": alias, "source": "local", "repo": entry.get("repo")}
    if entry.get("repo"):
        return entry["repo"], {"alias": alias, "source": "repo", "revision": entry.get("revision")}
    raise FileNotFoundError(f"model alias {value!r}: no {model_env_key(alias)}, no local path, no repo id")


def resolve_model_aliases(spec: dict) -> dict:
    """Replace "@alias" in "model" and in string-valued options with the host's path or repo id, in place. The
    spec's "models" table maps alias -> {"repo", "local", "revision"}; the resolved source is kept in
    cell["model_source"] so a record says whether it ran from a local copy or a repo id."""
    models = spec.get("models") or {}
    if not models:
        return spec
    for block in [spec.get("defaults") or {}] + list(spec.get("cells") or []):
        if "model" in block:
            block["model"], src = resolve_model(block["model"], models)
            if src:
                block["model_source"] = src
        for key, val in list((block.get("options") or {}).items()):
            if isinstance(val, str) and val.startswith("@") and val[1:] in models:
                block["options"][key] = resolve_model(val, models)[0]
    return spec


def expand_env(value: Any) -> Any:
    """``$WORKSPACE`` / ``~`` / env vars inside strings, recursively."""
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value.replace("$WORKSPACE", str(WS))))
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    return value


# ---------------------------------------------------------------------------------------------- prompts
def load_prompts(name: str, n: Optional[int] = None, ids: Optional[Iterable[str]] = None) -> list[dict]:
    """Rows of {id, prompt, seed, split}. ``name`` is a file under prompts/ (without .json) or a path.
    A .txt file gives one prompt per line with seeds 20260920+i."""
    path = Path(expand_env(name))
    if not path.exists():
        path = PKG / "prompts" / f"{name}.json"
    if path.suffix == ".txt":
        lines = [ln.strip() for ln in path.read_text(encoding = "utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
        rows = [{"id": f"p{i:02d}", "prompt": p, "seed": 20260920 + i, "split": "eval"} for i, p in enumerate(lines)]
    else:
        data = json.loads(path.read_text(encoding = "utf-8"))
        rows = data["rows"] if isinstance(data, dict) else data
    if ids:
        want = list(ids)
        by_id = {r["id"]: r for r in rows}
        missing = [i for i in want if i not in by_id]
        if missing:
            raise KeyError(f"prompt ids not in {path.name}: {missing}")
        return [by_id[i] for i in want]
    return rows[:n] if n else rows


# ---------------------------------------------------------------------------------------------- GPU
def gpu_vendor() -> str:
    if shutil.which("nvidia-smi"):
        return "nvidia"
    if shutil.which("amd-smi") or shutil.which("rocm-smi"):
        return "amd"
    return "none"


def visible_gpu() -> Optional[str]:
    """The first device of the process's mask (index or UUID), as the SMI tools take it. None when the mask
    is set but empty (a CPU-only run), so nothing samples a device the process cannot see."""
    for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        raw = os.environ.get(key)
        if raw is not None:
            raw = raw.strip()
            return raw.split(",")[0].strip() if raw and raw != "-1" else None
    return "0"


def _amdgpu_sysfs(device: Optional[str]) -> dict:
    """{used_mib, total_mib, util, name, vram_mib, gtt_mib} from /sys/class/drm/card*/device for the device-th amdgpu
    card (render-node order), or {} when sysfs has no amdgpu memory counters."""
    try:
        cards = sorted(p for p in Path("/sys/class/drm").glob("card[0-9]*")
                       if (p / "device" / "mem_info_gtt_used").is_file() and "-" not in p.name)
        if not cards:
            return {}
        d = cards[min(int(device or 0), len(cards) - 1)] / "device"

        def rd(name: str) -> int:
            try:
                return int((d / name).read_text(encoding = "utf-8").strip())
            except Exception:  # noqa: BLE001
                return 0

        vram, gtt = rd("mem_info_vram_used"), rd("mem_info_gtt_used")
        total = rd("mem_info_vram_total") + rd("mem_info_gtt_total")
        return {"used_mib": (vram + gtt) >> 20, "total_mib": total >> 20, "util": rd("gpu_busy_percent"),
                "name": "amd", "vram_mib": vram >> 20, "gtt_mib": gtt >> 20}
    except Exception:  # noqa: BLE001
        return {}


def gpu_query(device: Optional[str] = None) -> dict:
    """{used_mib, total_mib, util, name} for one device, or {} when there is no SMI to ask."""
    device = device or visible_gpu()
    vendor = gpu_vendor()
    if device is None:
        return {}
    try:
        if vendor == "nvidia":
            out = subprocess.run(
                ["nvidia-smi", "-i", device, "--query-gpu=memory.used,memory.total,utilization.gpu,name",
                 "--format=csv,noheader,nounits"], capture_output = True, text = True, timeout = 10,
            ).stdout.strip().splitlines()[0]
            used, total, util, name = [x.strip() for x in out.split(",", 3)]
            return {"used_mib": int(used), "total_mib": int(total), "util": int(util), "name": name}
        if vendor == "amd":
            # amdgpu sysfs: VRAM + GTT. On a unified-memory APU (Strix Halo) most allocations land in GTT, which
            # amd-smi's used_vram does not count, so the SMI view reads ~0 there.
            sysfs = _amdgpu_sysfs(device)
            if sysfs:
                return sysfs
        if vendor == "amd" and shutil.which("amd-smi"):
            raw = subprocess.run(["amd-smi", "metric", "-g", device, "--json"], capture_output = True,
                                 text = True, timeout = 10).stdout
            data = json.loads(raw)
            data = data[0] if isinstance(data, list) else data
            mem = data.get("mem_usage") or data.get("memory_usage") or {}
            used = mem.get("used_vram", {}).get("value") if isinstance(mem.get("used_vram"), dict) else mem.get("used_vram")
            total = mem.get("total_vram", {}).get("value") if isinstance(mem.get("total_vram"), dict) else mem.get("total_vram")
            util = (data.get("usage") or {}).get("gfx_activity", {})
            util = util.get("value") if isinstance(util, dict) else util
            return {"used_mib": int(used or 0), "total_mib": int(total or 0), "util": int(util or 0), "name": "amd"}
        if vendor == "amd":
            raw = subprocess.run(["rocm-smi", "-d", device, "--showmeminfo", "vram", "--showuse", "--json"],
                                 capture_output = True, text = True, timeout = 10).stdout
            card = next(iter(json.loads(raw).values()))
            used = int(card.get("VRAM Total Used Memory (B)", 0)) >> 20
            total = int(card.get("VRAM Total Memory (B)", 0)) >> 20
            return {"used_mib": used, "total_mib": total, "util": int(card.get("GPU use (%)", 0)), "name": "amd"}
    except Exception:  # noqa: BLE001 - sampling is best effort
        return {}
    return {}


class GpuSampler(threading.Thread):
    """Peak device memory as the driver sees it (every process on the device, allocator caches included),
    sampled every ``interval`` s. The only fair VRAM number across frameworks that manage memory themselves
    (ComfyUI, sd.cpp); pair it with torch's own peak where the backend is in process."""

    def __init__(self, interval: float = 0.25, device: Optional[str] = None):
        super().__init__(daemon = True)
        self.extra_pids = lambda: []  # set by the runner once the backend exists
        self.interval, self.device = interval, device or visible_gpu()
        self.peak_mib = 0
        self.own_peak_mib = 0
        self.tree_rss_peak_mib = 0  # RSS of this process + children + backend servers (ComfyUI's lives there)
        self.baseline_mib = gpu_query(self.device).get("used_mib", 0)
        self._stop = threading.Event()

    def _own_mib(self) -> int:
        """VRAM held by this process and its children (servers a backend launched), NVIDIA only. On a shared
        GPU the device-wide figure moves with other tenants; this one does not."""
        if gpu_vendor() != "nvidia":
            return 0
        try:
            import psutil

            me = psutil.Process()
            pids = {me.pid} | {c.pid for c in me.children(recursive = True)} | set(self.extra_pids() or [])
            out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                                 capture_output = True, text = True, timeout = 10).stdout
            return sum(int(used) for pid, used in (ln.split(",") for ln in out.strip().splitlines() if "," in ln)
                       if int(pid) in pids)
        except Exception:  # noqa: BLE001
            return 0

    def run(self) -> None:
        while not self._stop.is_set():
            q = gpu_query(self.device)
            if q:
                self.peak_mib = max(self.peak_mib, q["used_mib"])
            self.own_peak_mib = max(self.own_peak_mib, self._own_mib())
            self.tree_rss_peak_mib = max(self.tree_rss_peak_mib, self._tree_rss_mib())
            self._stop.wait(self.interval)

    def _tree_rss_mib(self) -> int:
        try:
            import psutil

            me = psutil.Process()
            procs = [me] + me.children(recursive = True)
            for pid in self.extra_pids() or []:
                try:
                    p = psutil.Process(pid)
                    procs += [p] + p.children(recursive = True)
                except Exception:  # noqa: BLE001
                    pass
            seen, total = set(), 0
            for p in procs:
                if p.pid in seen:
                    continue
                seen.add(p.pid)
                try:
                    total += p.memory_info().rss
                except Exception:  # noqa: BLE001
                    pass
            return total >> 20
        except Exception:  # noqa: BLE001
            return 0

    def reset(self) -> None:
        # Seeded with a reading, so a cell shorter than one interval does not report 0.
        q = gpu_query(self.device)
        self.peak_mib = q.get("used_mib", 0) if q else 0
        self.own_peak_mib = self._own_mib()

    def stop(self) -> None:
        self._stop.set()


def gpu_gate(max_util: int = 10, min_free_mib: int = 0, timeout_s: float = 1800, poll_s: float = 20,
             device: Optional[str] = None, log=print) -> dict:
    """Wait until the device is quiet before a timed cell. Returns the last reading plus ``gate``:
    ok, timeout (ran anyway; the record is flagged), or no_smi."""
    device = device or visible_gpu()
    deadline = time.time() + timeout_s
    while True:
        q = gpu_query(device)
        if not q:
            return {"gate": "no_smi"}
        free = q["total_mib"] - q["used_mib"]
        if q["util"] <= max_util and free >= min_free_mib:
            return {**q, "gate": "ok"}
        if time.time() > deadline:
            log(f"gpu gate TIMEOUT on {device}: util {q['util']}% free {free} MiB; running anyway, flagged")
            return {**q, "gate": "timeout"}
        time.sleep(poll_s)


# ---------------------------------------------------------------------------------------------- host memory
def host_memory() -> dict:
    """RSS split into anonymous and file-backed pages (Linux), plus the lifetime peak. File-backed pages are
    the mmapped safetensors the kernel can drop under pressure; anonymous pages are what a copy costs."""
    out: dict = {}
    try:
        import psutil

        out["rss_gib"] = round(psutil.Process().memory_info().rss / 2**30, 3)
        vm = psutil.virtual_memory()
        out["host_available_gib"] = round(vm.available / 2**30, 2)
    except Exception:  # noqa: BLE001
        pass
    status = Path("/proc/self/status")
    if status.exists():
        fields = {ln.split(":")[0]: ln.split()[1] for ln in status.read_text().splitlines() if ln.startswith(("Rss", "VmHWM"))}
        for key, name in (("RssAnon", "rss_anon_gib"), ("RssFile", "rss_file_gib"), ("RssShmem", "rss_shmem_gib"),
                          ("VmHWM", "peak_rss_gib")):
            if key in fields:
                out[name] = round(int(fields[key]) / 2**20, 3)
    elif sys.platform != "win32":
        import resource

        scale = 2**30 if sys.platform == "darwin" else 2**20
        out["peak_rss_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale, 3)
    return out


# ---------------------------------------------------------------------------------------------- environment
def _version(mod: str) -> Optional[str]:
    try:
        from importlib.metadata import version

        return version(mod)
    except Exception:  # noqa: BLE001
        return None


def git_rev(path: str | Path) -> Optional[str]:
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "--short=10", "HEAD"], capture_output = True,
                             text = True, timeout = 10)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def env_fingerprint(trees: Optional[dict] = None, with_torch: bool = True) -> dict:
    """What produced a record: interpreter, platform, packages, device and driver, git revisions."""
    fp: dict = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "packages": {m: _version(m) for m in ("torch", "diffusers", "transformers", "accelerate", "torchao",
                                               "flashinfer-python", "sageattention", "unsloth", "unsloth_zoo",
                                               "comfy-kitchen", "lpips")},
        "gpu": gpu_query(),
        "visible_devices": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")
                            if os.environ.get(k) is not None},
    }
    fp["packages"] = {k: v for k, v in fp["packages"].items() if v}
    if with_torch and "torch" in sys.modules:
        torch = sys.modules["torch"]
        fp["torch_cuda"] = getattr(torch.version, "cuda", None)
        fp["torch_hip"] = getattr(torch.version, "hip", None)
        try:
            if torch.cuda.is_available():
                fp["device_name"] = torch.cuda.get_device_name(0)
                fp["device_capability"] = list(torch.cuda.get_device_capability(0))
                fp["gcn_arch"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None)
        except Exception:  # noqa: BLE001
            pass
    if trees:
        fp["trees"] = {name: git_rev(path) for name, path in trees.items() if path}
    return fp


def scrub_tokens() -> None:
    """A render never needs a token (models are local or public), and one in the environment ends up in logs
    and child processes. Callers that must download pass it explicitly to the downloader instead."""
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        os.environ.pop(key, None)


# ---------------------------------------------------------------------------------------------- media
def to_pil(obj: Any):
    """PIL image from a PIL image, a (data-)URL/base64 string, a numpy HWC array, or a file path."""
    from PIL import Image

    if hasattr(obj, "save") and hasattr(obj, "convert"):
        return obj
    if isinstance(obj, (bytes, bytearray)):
        return Image.open(io.BytesIO(obj))
    if isinstance(obj, str):
        if Path(obj).exists():
            return Image.open(obj)
        return Image.open(io.BytesIO(base64.b64decode(obj.split(",", 1)[-1])))
    import numpy as np

    arr = np.asarray(obj)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).round().astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    return Image.fromarray(arr)


def save_image(obj: Any, path: Path) -> Path:
    to_pil(obj).convert("RGB").save(path)
    return path


def frames_array(frames: Any):
    """uint8 [T, H, W, 3] from a list of PIL frames / arrays or one array (floats in [0, 1] are scaled)."""
    import numpy as np

    if isinstance(frames, (list, tuple)):
        arr = np.stack([np.asarray(to_pil(f).convert("RGB")) if not isinstance(f, np.ndarray) else f for f in frames])
    else:
        arr = np.asarray(frames)
    while arr.ndim > 4:
        arr = arr[0]
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).round().astype(np.uint8)
    return arr


def save_video(frames: Any, stem: Path, fps: int = 16, every: int = 1) -> dict:
    """frames.npz (every k-th frame, what the scorer reads) and an .mp4 when imageio can write one."""
    import numpy as np

    arr = frames_array(frames)
    np.savez_compressed(f"{stem}.npz", frames = arr[::max(1, every)])
    out = {"file": f"{stem.name}.npz", "frames": int(arr.shape[0]), "shape": list(arr.shape[1:])}
    try:
        import imageio.v3 as iio

        iio.imwrite(f"{stem}.mp4", arr, fps = fps or 16, codec = "libx264")
        out["mp4"] = f"{stem.name}.mp4"
    except Exception:  # noqa: BLE001 - scoring only needs the npz
        pass
    return out


# ---------------------------------------------------------------------------------------------- records
def median(values: list) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return round(statistics.median(vals), 4) if vals else None


def finalize_record(rec: dict) -> dict:
    """Summary numbers every scorer reads, derived identically for every backend."""
    cell = rec.get("cell", {})
    walls = [r["wall_s"] for r in rec.get("renders", [])]
    rec["wall_s_median"] = median(walls)
    rec["step_s_reported_median"] = median([r.get("step_s") for r in rec.get("renders", [])])
    long_, short = rec.get("long", []), rec.get("short", [])
    if long_ and short and cell.get("steps") and cell.get("short_steps") and cell["steps"] > cell["short_steps"]:
        rec["step_s_derived"] = round((statistics.median(long_) - statistics.median(short))
                                      / (cell["steps"] - cell["short_steps"]), 4)
    rec["steady_s_median"] = median(long_)
    peaks = [r.get("peak_alloc_gib") for r in rec.get("renders", []) if r.get("peak_alloc_gib") is not None]
    if peaks:
        rec["peak_alloc_gib"] = max(peaks)
    return rec


def write_record(out: Path, rec: dict) -> Path:
    out.mkdir(parents = True, exist_ok = True)
    path = out / RECORD
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, indent = 1, default = str))
    os.replace(tmp, path)
    return path


def read_record(cell_dir: Path) -> Optional[dict]:
    path = Path(cell_dir) / RECORD
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None


def record_ok(rec: Optional[dict]) -> bool:
    return bool(rec) and rec.get("verdict") == "ok"


def log(msg: str) -> None:
    print(f"[diffusion_bench {time.strftime('%H:%M:%S')}] {msg}", flush = True)
