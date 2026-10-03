"""A backend with no model: deterministic noise images / clips from the seed, with an optional per-cell
perturbation. Exercises the whole driver, matrix, and scorer without a GPU (selftest.py)."""

from __future__ import annotations

import time

from .base import Backend, Render


class FakeBackend(Backend):
    name = "fake"

    def load(self) -> dict:
        self.noise = float(self.opts.get("noise", 0.0))
        self.delay = float(self.opts.get("step_delay_s", 0.001))
        if self.opts.get("fail_on_load"):
            raise RuntimeError("fake backend asked to fail on load")
        return {"backend": "fake", "noise": self.noise}

    def render(self, row: dict, steps: int) -> Render:
        import numpy as np

        time.sleep(self.delay * steps)
        rng = np.random.default_rng(row["seed"])
        h, w = self.cell["height"] // 8, self.cell["width"] // 8
        shape = (self.cell.get("frames") or 1, h, w, 3)
        base = rng.random(shape)
        if self.noise:
            base = np.clip(base + np.random.default_rng(row["seed"] + 1).normal(0, self.noise, shape), 0, 1)
        arr = (base * 255).astype("uint8")
        if self.cell["kind"] == "video":
            return Render(frames = arr, fps = self.cell.get("fps") or 8, step_s = self.delay)
        return Render(image = arr[0], step_s = self.delay)
