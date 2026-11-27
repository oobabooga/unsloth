#!/usr/bin/env python3
"""Probe: round-9d gfx1151 C6-2 / C6-6 compiled Wan 5B block (observes only; criteria/r9d_measure.py gates that each listed item ran and
tabulates the numbers with a verdict line per item against its pre-registered kill line).

Head state only: base = origin/main (the --patch-base), head = base + the job's patch. Items run from the tree they
need (most from the base tree = main; patched arms from the head tree). Nothing is judged between states.
Items (modules): r9d_items.py (Linux). The first item must be a smoke item:
if it fails, no large arm runs. No token is read or passed (public unsloth/* repos, token=False).

  python r9_probe.py --state head --checkout <tree> --out obs.json --items wsmoke,sdma,...
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def registry() -> tuple:
    items, errors = {}, {}
    for mod in ("r9d_items",):
        try:
            m = importlib.import_module(mod)
            items.update(getattr(m, "ITEMS", {}))
        except Exception:  # noqa: BLE001
            errors[mod] = traceback.format_exc()[-800:]
    return items, errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--items", required = True)
    ap.add_argument("--main-tree", default = "base", choices = ("base", "head"),
                    help = "which state is origin/main: base (a --patch job) or head (a --merged job on main's tip)")
    a = ap.parse_args()
    out = Path(a.out)
    obs: dict = {"state": a.state, "args": {"items": a.items, "main_tree": a.main_tree}}

    def save():
        out.write_text(json.dumps(obs, indent = 1, default = str), encoding = "utf-8")
    if a.state != "head":
        obs["skipped"] = "measured on the head state only (arms pick base or head trees themselves)"
        save()
        return 0
    head = Path(a.checkout).resolve()
    backend = str(head / "studio" / "backend")
    base_backend = str(head.parent / "base" / "studio" / "backend")
    root = out.resolve().parents[2]  # the job root ($AMD_CI_WORK); never inside the uploaded out/
    ctx = {"backend": backend, "base_backend": base_backend, "work": out.parent / "r9work",
           "scratch": root / "r9_scratch", "hf": root / "r9_hf_home", "log": out.parent / "r9_children.log",
           "py": sys.executable, "win": sys.platform == "win32", "save": save}
    ctx["main_backend"] = backend if a.main_tree == "head" else base_backend
    for k in ("work", "scratch", "hf"):
        Path(ctx[k]).mkdir(parents = True, exist_ok = True)
    items, errors = registry()
    obs["module_errors"] = errors
    try:
        import torch
        obs["env"] = {"torch": torch.__version__, "hip": getattr(torch.version, "hip", None),
                      "arch": getattr(torch.cuda.get_device_properties(0), "gcnArchName", None),
                      "python": sys.version.split()[0], "platform": sys.platform,
                      "base_exists": Path(base_backend).is_dir(), "runner": os.environ.get("RUNNER_NAME") or os.environ.get("COMPUTERNAME")}
    except Exception as exc:  # noqa: BLE001
        obs["env_error"] = repr(exc)
    save()
    names = [x for x in a.items.split(",") if x]
    for i, item in enumerate(names):
        t = time.time()
        if item not in items:
            obs[f"{item}_error"] = f"unknown item (modules: {sorted(items)}; import errors: {sorted(errors)})"
            save()
            continue
        try:
            items[item](ctx, obs)
        except Exception:  # noqa: BLE001
            obs[f"{item}_error"] = traceback.format_exc()[-2500:]
            if i == 0:
                obs.setdefault("item_s", {})[item] = round(time.time() - t, 1)
                save()
                return 0  # the smoke step failed: no large arms
        obs.setdefault("item_s", {})[item] = round(time.time() - t, 1)
        save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
