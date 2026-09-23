#!/usr/bin/env python3
"""Criteria: on a GPU outside gfx101x, does the head leave Triton's buffer ops exactly as
the base does? PR #11615 must only act on gfx101x; on gfx1151 every observation matches.

Pairs with probes/buffer_ops_gate_probe.py.
"""

from __future__ import annotations

TITLE = "gfx101x buffer-ops gate on a non-gfx101x GPU, base versus head"
MODE = "regression"
NEEDS: list[str] = ["rocm", "gpu"]


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        out.append((f"{name} imported unsloth", "import_error" not in o and "error" not in o,
                    o.get("import_error") or o.get("error") or o.get("unsloth_file", "?")))
        out.append((f"{name} imported THIS checkout", bool(o.get("checkout_imported")), o.get("unsloth_file", "?")))
        out.append((f"{name} saw a GPU arch", bool(o.get("archs")), str(o.get("archs"))))
        out.append((f"{name} ran the kernel", "kernel_correct" in o, o.get("kernel_error", "ok")))
    arch = (obs.get("head") or {}).get("archs") or []
    out.append(("no gfx101x visible (this criteria is the negative control)",
                not any(str(a).lower().startswith("gfx101") for a in arch), str(arch)))
    return out


def table(obs: dict) -> str:
    rows = ["| state | archs | AMDGCN_USE_BUFFER_OPS | TRITON_CACHE_DIR | TORCHINDUCTOR_CACHE_DIR | use_buffer_ops | kernel | compile | inductor dirs |",
            "|---|---|---|---|---|---|---|---|---|"]
    for n in ("base", "head", "merge"):
        o = obs.get(n)
        if not o:
            continue
        e = o.get("env") or {}
        tail = lambda v: "unset" if v is None else "`..." + str(v)[-40:] + "`"
        rows.append(f"| {n} | {o.get('archs')} | {e.get('AMDGCN_USE_BUFFER_OPS')} | {tail(e.get('TRITON_CACHE_DIR'))} "
                    f"| {tail(e.get('TORCHINDUCTOR_CACHE_DIR'))} | {o.get('use_buffer_ops')} | {o.get('kernel_correct')}/1024 "
                    f"| {o.get('compile_ok', o.get('compile_error'))} | {o.get('inductor_dirs')} |")
    return "\n".join(rows)


def _key(o):
    e = o.get("env") or {}
    return {"AMDGCN_USE_BUFFER_OPS": e.get("AMDGCN_USE_BUFFER_OPS"),
            "TRITON_CACHE_DIR": e.get("TRITON_CACHE_DIR"),
            "inductor_is_no_buffer_ops": str(e.get("TORCHINDUCTOR_CACHE_DIR") or "").endswith("_no_buffer_ops"),
            "use_buffer_ops": o.get("use_buffer_ops"),
            "kernel_correct": o.get("kernel_correct"),
            "compile_ok": o.get("compile_ok"),
            "inductor_dir_names": sorted((o.get("inductor_dirs") or {}).keys())}


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    b, h = _key(base), _key(head)
    moved = {k: (b[k], h[k]) for k in b if b[k] != h[k]}
    if moved:
        return True, "head differs from base on a non-gfx101x GPU: " + ", ".join(f"{k}: {v[0]} -> {v[1]}" for k, v in moved.items())
    if h["kernel_correct"] != 1024:
        return False, f"identical at both states, but the kernel wrote {h['kernel_correct']}/1024 at both (pre-existing)"
    return False, "identical at both states: knob unset, caches untouched, kernel 1024/1024, compile ok"
