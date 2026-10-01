"""Unsloth Studio's own image / video backends, imported in this process: the exact DiffusionBackend /
VideoBackend objects the Images and Video pages drive, without HTTP, the gallery, or the engine router.

Cell fields:
  model                 local path or repo id
  kind                  image | video
  options.studio_src    the Studio source (setup_studio.resolve_tree: path, git ref, pr/N, pypi); default
                        $DIFFUSION_BENCH_STUDIO_SRC, else the backend directory already importable in this venv
  options.<load key>    any of LOAD_KEYS, passed straight to load_pipeline / begin_load
  options.load          dict passthrough for anything else the loader takes
  options.generate      dict passthrough for generate() (e.g. {"batch_size": 2} or {"flow_shift": 5.0})
  options.local_files_only  default True (a bench never downloads mid-timing)
  options.load_via      image only: "load_pipeline" (default, synchronous) or "begin_load" (the route's path:
                        background thread + load_progress polling, same as video)
  options.studio_home   UNSLOTH_STUDIO_HOME for compile caches etc. (default under $WORKSPACE/temp)
  options.video_frames  "raw" (default: the frames Studio hands its MP4 encoder, lossless) or "mp4" (decode the
                        H.264 file Studio produced, i.e. what the gallery / HTTP backend delivers)

``load()`` returns the status fields that say what engaged (offload_policy, speed_optims, quant, attention...).
"""

from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path
from typing import Any

from .base import Backend, Render

LOAD_KEYS = ("model_kind", "family_override", "base_repo", "gguf_filename", "memory_mode", "speed_mode",
             "cpu_offload", "transformer_quant", "text_encoder_quant", "transformer_quant_fast_accum",
             "transformer_prequant_path", "attention_backend", "transformer_cache", "transformer_cache_threshold",
             "loras", "h3_task", "gpu_ids")
VIDEO_ONLY = ("h3_task",)
IMAGE_ONLY = ("cpu_offload", "transformer_quant_fast_accum", "transformer_prequant_path", "loras")
STATUS_KEYS = ("loaded", "repo_id", "family", "base_repo", "device", "dtype", "model_kind", "gguf_filename",
               "gguf_variant", "engine", "cpu_offload", "offload_policy", "vae_tiling", "memory_mode", "speed_mode",
               "speed_optims", "text_encoder_quant", "transformer_quant", "attention_backend", "transformer_cache",
               "transformer_cache_stats", "resolved", "workflows", "defaults", "supports_cfg", "has_audio")


def _jsonable(value: Any, depth: int = 0) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if depth > 4:
        return str(value)[:200]
    if isinstance(value, dict):
        return {str(k): _jsonable(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v, depth + 1) for v in list(value)[:64]]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(), depth + 1)
    return str(value)[:200]


def status_subset(st: dict) -> dict:
    return {k: _jsonable(st.get(k)) for k in STATUS_KEYS if k in (st or {})}


def studio_import_path(opts: dict) -> dict:
    """Resolve the Studio tree for this process and put its backend first on sys.path (ahead of anything on
    PYTHONPATH named ``models`` / ``core`` / ``utils``). Returns the resolve_tree dict."""
    here = Path(__file__).resolve().parents[1]
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    import setup_studio

    src = opts.get("studio_src") or os.environ.get("DIFFUSION_BENCH_STUDIO_SRC")
    if src:
        tree = setup_studio.resolve_tree(src, python = sys.executable)
    else:
        tree = setup_studio.resolve_tree("pypi", python = sys.executable)
    backend = tree["studio_backend"]
    if backend in sys.path:
        sys.path.remove(backend)
    sys.path.insert(0, backend)
    for name in ("models", "core", "utils", "routes", "auth", "storage", "hub", "state"):
        mod = sys.modules.get(name)
        if mod is None:
            continue
        paths = [getattr(mod, "__file__", None) or ""] + [str(p) for p in (getattr(mod, "__path__", None) or [])]
        if not any(p.startswith(backend) for p in paths if p):
            del sys.modules[name]  # a same-named package from elsewhere on the path was imported first
    home = opts.get("studio_home") or os.environ.get("UNSLOTH_STUDIO_HOME")
    if not home:
        import common as C

        home = str(C.WS / "temp" / "diffusion_bench" / "studio_home_inproc")
    os.environ["UNSLOTH_STUDIO_HOME"] = str(home)
    Path(home).mkdir(parents = True, exist_ok = True)
    return tree


def load_kwargs(opts: dict, kind: str) -> dict:
    kw = {k: opts[k] for k in LOAD_KEYS if k in opts and opts[k] is not None}
    kw.update(opts.get("load") or {})
    drop = IMAGE_ONLY if kind == "video" else VIDEO_ONLY
    for k in drop:
        kw.pop(k, None)
    if kw.get("loras"):
        kw["loras"] = [tuple(x) if isinstance(x, (list, tuple)) else (x["id"], float(x.get("weight", 1.0)))
                       for x in kw["loras"]]
    return kw


def wait_ready(backend: Any, timeout_s: float, poll_s: float = 0.5) -> None:
    deadline = time.time() + timeout_s
    null_since = None
    while True:
        p = backend.load_progress() or {}
        phase = p.get("phase")
        if phase == "ready":
            return
        if phase == "error":
            raise RuntimeError(f"load failed: {p.get('error')}")
        if phase is None:
            null_since = null_since or time.time()
            if time.time() - null_since > 30:
                raise RuntimeError("load thread ended without a model or an error")
        if time.time() > deadline:
            raise TimeoutError(f"load not ready after {timeout_s}s: {p}")
        time.sleep(poll_s)


class StudioBackend(Backend):
    name = "studio"
    in_process = True

    def __init__(self, cell: dict, out: Path):
        super().__init__(cell, out)
        self.tree = studio_import_path(self.opts)
        self.kind = cell.get("kind", "image")
        self.engine: Any = None
        self._captured: dict = {}

    # ------------------------------------------------------------------------------------------ lifecycle
    def load(self) -> dict:
        model = self.cell["model"]
        kw = load_kwargs(self.opts, self.kind)
        local = bool(self.opts.get("local_files_only", True))
        timeout = float(self.opts.get("load_timeout_s", 3600))
        if self.kind == "video":
            from core.inference import video as V

            orig = V.VideoBackend._encode_mp4
            captured = self._captured

            def grab(video_frames, fps, audio, pipe, **k):
                captured["frames"], captured["fps"] = video_frames, fps
                return orig(video_frames, fps, audio, pipe, **k)

            V.VideoBackend._encode_mp4 = staticmethod(grab)
            self.engine = V.VideoBackend()
            self.engine.begin_load(model, local_files_only = local, **kw)
            wait_ready(self.engine, timeout)
            st = self.engine.status()
        else:
            from core.inference.diffusion import DiffusionBackend

            self.engine = DiffusionBackend()
            if self.opts.get("load_via") == "begin_load":
                self.engine.begin_load(model, local_files_only = local, **kw)
                wait_ready(self.engine, timeout)
                st = self.engine.status()
            else:
                st = self.engine.load_pipeline(model, local_files_only = local, **kw) or {}
                st = {**st, **(self.engine.status() or {})}
        return {**status_subset(st), "load_kwargs": _jsonable(kw), "studio_rev": self.tree.get("rev")}

    def render(self, row: dict, steps: int) -> Render:
        gen = dict(self.opts.get("generate") or {})
        c = self.cell
        if self.kind == "video":
            self._captured.clear()
            res = self.engine.generate(prompt = row["prompt"], negative_prompt = c.get("negative_prompt"),
                                       width = c.get("width"), height = c.get("height"), num_frames = c.get("frames"),
                                       fps = c.get("fps"), steps = steps, guidance = c.get("guidance"),
                                       seed = row["seed"], **gen)
            frames = self._captured.get("frames")
            if frames is None or self.opts.get("video_frames") == "mp4":
                from studio_client import decode_mp4

                frames, _ = decode_mp4(res["mp4_bytes"])
            extra = {k: res.get(k) for k in ("seed", "width", "height", "num_frames", "fps", "duration_s",
                                              "flow_shift", "offload_policy")}
            return Render(frames = frames, fps = res.get("fps"), extra = extra)
        res = self.engine.generate(prompt = row["prompt"], negative_prompt = c.get("negative_prompt"),
                                   width = c.get("width"), height = c.get("height"), steps = steps,
                                   guidance = c.get("guidance"), seed = row["seed"], **gen)
        images = res["images"]
        extra = {"seed": res.get("seed")}
        if len(images) > 1:
            extra["batch"] = len(images)
            extra["seeds"] = res.get("seeds")
        return Render(image = images[0], extra = extra)

    def status(self) -> dict:
        return status_subset(self.engine.status()) if self.engine is not None else {}

    def close(self) -> None:
        if self.engine is not None:
            try:
                self.engine.unload()
            finally:
                self.engine = None
                gc.collect()
                torch = sys.modules.get("torch")
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def trees(self) -> dict:
        return {"studio": self.tree.get("tree")}
