"""Shared gates for the batch3 criteria (imported by sibling modules)."""
from __future__ import annotations


def states(obs):
    return {k: v for k, v in obs.items() if not k.startswith("_")}


def base_gates(obs):
    out = []
    for name, o in states(obs).items():
        err = o.get("error")
        out.append((f"{name}: probe child produced JSON", not err, str(err)[-300:] if err else f"rc={o.get('rc')}"))
        out.append((f"{name}: torch is a ROCm build", bool(o.get("hip")), f"{o.get('torch')} hip={o.get('hip')}"))
        out.append((f"{name}: gfx1151 visible", "gfx1151" in str(o.get("arch")), str(o.get("arch"))))
        f = str(o.get("module_file")).replace("\\", "/")
        out.append((f"{name}: checkout's own module", f"/{name}/" in f, f))
    return out
