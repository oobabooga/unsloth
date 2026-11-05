"""The contract every backend implements. ``run_cell.py`` owns the timing protocol, so a backend only knows
how to load a model, render one prompt, and report what it engaged; it never times itself end to end.

A backend is constructed with the merged cell dict and its output directory, then:

    status = backend.load()            # dict: what actually engaged (quant, offload, attention, compile, ...)
    result = backend.render(row, steps) # Render: media plus optional per-render facts
    backend.status()                   # optional, re-read after the renders (a lever can disengage mid-run)
    backend.close()                    # always called; must free the device and kill any child process

``render`` must not return before the device work is finished (synchronise), so wall time is honest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class Render:
    """One render. ``image`` for image cells (PIL / array / base64 / path), ``frames`` for video cells."""

    image: Any = None
    frames: Any = None
    fps: Optional[int] = None
    step_s: Optional[float] = None  # the backend's own per-step figure, when it reports one (ComfyUI, sd.cpp)
    peak_alloc_gib: Optional[float] = None  # torch allocator peak for this render, in-process backends only
    extra: dict = field(default_factory = dict)


class Backend:
    name = "base"
    #: in-process backends import torch in this process, so torch peak / host RSS are this model's own
    in_process = False

    def __init__(self, cell: dict, out: Path):
        self.cell, self.out = cell, Path(out)
        self.opts = dict(cell.get("options") or {})

    # ------------------------------------------------------------------------------------------ lifecycle
    def load(self) -> dict:
        raise NotImplementedError

    def render(self, row: dict, steps: int) -> Render:
        raise NotImplementedError

    def status(self) -> dict:
        return {}

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------------------------------ helpers
    def extra_pids(self) -> list:
        """Processes outside this one's tree whose VRAM belongs to the cell (e.g. an attached or re-parented
        Studio server). Children of this process are counted already."""
        return []

    def trees(self) -> dict:
        """Source trees to fingerprint (git revision) in the record."""
        return {}

    def sync(self) -> None:
        """Wait for queued device work; a no-op without torch."""
        import sys

        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()

    def reset_peak(self) -> None:
        import sys

        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def peak_alloc_gib(self) -> Optional[float]:
        import sys

        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 2**30, 3)
        return None


REGISTRY = {
    # name -> "module:Class", imported lazily so a backend's dependencies load only in its own venv
    "fake": "backends.fake:FakeBackend",
    "studio": "backends.studio_inproc:StudioBackend",
    "studio_http": "backends.studio_http:StudioHttpBackend",
    "diffusers": "backends.diffusers_ref:DiffusersBackend",
    "comfyui": "backends.comfyui:ComfyUIBackend",
    "sdcpp": "backends.sdcpp:SdCppBackend",
}


def get_backend(name: str):
    import importlib

    if name not in REGISTRY:
        raise KeyError(f"unknown backend {name!r}; known: {sorted(REGISTRY)}")
    mod, _, cls = REGISTRY[name].partition(":")
    return getattr(importlib.import_module(mod), cls)
