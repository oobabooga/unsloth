#!/usr/bin/env python3
"""Criteria: do base and head route THIS host to the same torch index and constraints?

For changes that must not move the routing of the host running them (the PR under
test widens a reroute to other arches; on gfx1151 nothing may change). Regression
mode: the verdict is NO_REGRESSION only if every routed field is identical, and the
gates make sure the comparison is not vacuous:

  * both states produced observations from both sides (install.sh and
    install_python_stack.py), with no probe error;
  * install.sh at BOTH states resolved the gfx1151 per-arch index, so the probe
    measured the real Strix route and not a CPU / CUDA / generic fallback that
    would be identical for the uninteresting reason;
  * install_python_stack.py saw gfx1151 as the runtime target;
  * the fresh-install pass of _ensure_rocm_torch emitted a torch install at both
    states, so its routing decision was actually exercised.

Pairs with probes/rocm_routing_probe.py.
"""

from __future__ import annotations

import re

TITLE = "Torch index routing on real gfx1151 detection, base versus head"
MODE = "regression"
# What the change touches: Linux ROCm routing, including RDNA 4 (discrete) cards and
# mixed-arch hosts where masks pick the target. Only the first three exist here.
NEEDS = ["linux", "rocm", "gpu", "discrete_gpu", "multi_gpu_amd"]

EXPECT_GFX = "gfx1151"
_PER_ARCH = re.compile(r"/gfx1151/?$")

SH_FIELDS = ("TORCH_INDEX_URL", "_torch_index_leaf", "TORCH_CONSTRAINT",
             "TORCHVISION_CONSTRAINT", "TORCHAUDIO_CONSTRAINT", "_torch_index_pinned",
             "_amd_gpu_radeon", "_gfx_rocm64_target", "_runtime_gfx")
PY_FIELDS = ("rocm_version", "has_rocm_gpu", "has_usable_nvidia_gpu", "runtime_gfx",
             "runtime_gfx_target", "strix_needs_amd_arch_index", "amd_arch_index_url",
             "runtime_gfx_in_floor_set", "reroute_pending_installed", "reroute_pending_fresh")


def _sh(o: dict) -> dict:
    return ((o or {}).get("install_sh") or {}).get("fields") or {}


def _py(o: dict) -> dict:
    return (o or {}).get("python_stack") or {}


def _ensure(o: dict, which: str) -> dict:
    return _py(o).get(which) or {}


def routed(o: dict) -> dict:
    """Every field that is a routing decision, flattened for comparison."""
    out = {f"install.sh {k}": _sh(o).get(k) for k in SH_FIELDS}
    out.update({f"python {k}": _py(o).get(k) for k in PY_FIELDS})
    for which in ("ensure_installed", "ensure_fresh"):
        e = _ensure(o, which)
        out[f"python {which} torch calls"] = e.get("torch_calls")
        out[f"python {which} error"] = e.get("error")
    return out


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        sh = o.get("install_sh") or {}
        py = _py(o)
        probe_ok = (o.get("_probe_rc") == 0 and not o.get("_missing_output")
                    and not o.get("_parse_error"))
        out.append((f"{name} probe wrote observations", probe_ok,
                    f"rc={o.get('_probe_rc')}"
                    + ("; missing output" if o.get("_missing_output") else "")
                    + (f"; {o['_parse_error']}" if o.get("_parse_error") else "")))
        sh_ok = bool(sh.get("fields")) and not sh.get("error") and sh.get("rc") == 0
        out.append((f"{name} install.sh span resolved", sh_ok,
                    sh.get("error") or f"rc={sh.get('rc')}"))
        py_ok = bool(py) and not py.get("error") and not any(
            _ensure(o, w).get("error") for w in ("ensure_installed", "ensure_fresh"))
        out.append((f"{name} install_python_stack routing ran", py_ok,
                    py.get("error") or _ensure(o, "ensure_installed").get("error")
                    or _ensure(o, "ensure_fresh").get("error") or "ok"))
        url = _sh(o).get("TORCH_INDEX_URL") or ""
        out.append((f"{name} install.sh chose the {EXPECT_GFX} per-arch index",
                    bool(_PER_ARCH.search(url)), f"`{url or 'none'}`"))
        rg = py.get("runtime_gfx")
        out.append((f"{name} python runtime target is {EXPECT_GFX}", rg == EXPECT_GFX,
                    f"{rg}; host rocminfo gfx "
                    f"{((o.get('host') or {}).get('rocminfo') or {}).get('gfx')}"))
        fresh = _ensure(o, "ensure_fresh").get("torch_calls") or []
        out.append((f"{name} fresh-install routing emitted a torch install", bool(fresh),
                    f"index `{_ensure(o, 'ensure_fresh').get('torch_index_url')}`"))
    return out


def table(obs: dict) -> str:
    b, h = routed(obs.get("base") or {}), routed(obs.get("head") or {})
    rows = ["| field | base | head | same |", "|---|---|---|---|"]
    for k in b:
        bv, hv = b[k], h.get(k)
        short = lambda v: str(v).replace("|", "\\|")[:160]  # noqa: E731
        rows.append(f"| {k} | `{short(bv)}` | `{short(hv)}` | {'yes' if bv == hv else 'NO'} |")
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    b, h = routed(base), routed(head)
    moved = [k for k in b if b[k] != h.get(k)]
    if moved:
        return True, "routing moved at the head on this host: " + ", ".join(f"`{k}`" for k in moved)
    return False, (f"all {len(b)} routed fields identical; both states route this host to "
                   f"`{_sh(base).get('TORCH_INDEX_URL')}` with `{_sh(base).get('TORCH_CONSTRAINT')}`")
