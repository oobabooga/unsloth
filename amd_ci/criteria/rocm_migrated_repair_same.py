#!/usr/bin/env python3
"""Criteria: routing identical base vs head, AND install.sh's migrated-venv ROCm repair
leaves a matching gfx1151 rocm7.13 venv alone at the head.

Extends criteria/rocm_routing_same.py (all of its routed fields and gates are kept) for
the PR that added _amd_arch_index_routed / _amd_arch_index_family and a migrated repair
that reinstalls when the venv torch is below rocm7.13 or its `rocm` meta-package names
another AMD family. Regression mode. NO_REGRESSION needs:

  * every rocm_routing_same routed field identical base vs head;
  * the repair fields both checkouts share (index leaf, rocm-family flag, rocm64 floor,
    the `_venv_torch_rocm_below <venv> 7 13` answer, the REINSTALL decision) identical;
  * the head's repair does NOT reinstall.

Non-vacuity gates (a failure is INCONCLUSIVE, not a pass): the repair block ran to the
end at both states; the head block actually carries the family check; the head routed to
the gfx1151 per-arch index (_amd_arch_index_routed=true, family gfx1151); and the runner's
venv really is the AMD gfx1151 rocm7.13+ wheel (torch version rocm >= 7.13, `rocm`
metadata names gfx1151). If the venv is anything else the measurement did not exercise the
"matching venv is left alone" case and the verdict must say so rather than pass.

Pairs with probes/rocm_migrated_repair_probe.py.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rocm_routing_same as rr  # noqa: E402

TITLE = ("Torch index routing and migrated-venv repair on real gfx1151 detection, "
         "base versus head")
MODE = "regression"
NEEDS = list(rr.NEEDS)
EXPECT_GFX = rr.EXPECT_GFX
_ROCM_VER = re.compile(r"rocm(\d+)\.(\d+)")

SHARED_REPAIR_FIELDS = ("_torch_index_leaf", "_torch_index_is_rocm_family",
                        "_gfx_rocm64_target", "_gfx_rocm64_floor_maj",
                        "_gfx_rocm64_floor_min")


def _rep(o: dict) -> dict:
    return (o or {}).get("migrated_repair") or {}


def _vt(o: dict) -> dict:
    return (o or {}).get("venv_torch") or {}


def repair_fields(o: dict) -> dict:
    r = _rep(o)
    f = r.get("fields") or {}
    out = {f"repair {k}": f.get(k) for k in SHARED_REPAIR_FIELDS}
    out["repair venv below rocm7.13"] = r.get("below_7_13")
    out["repair would REINSTALL"] = r.get("reinstall")
    out["repair substeps"] = r.get("substeps")
    return out


def info_fields(o: dict) -> dict:
    """Head-only or informational: shown in the table, judged by gates, not by equality
    (the base has no _amd_arch_index_* variables and no family helper)."""
    r, vt = _rep(o), _vt(o)
    f = r.get("fields") or {}
    return {
        "repair _amd_arch_index_routed": f.get("_amd_arch_index_routed"),
        "repair _amd_arch_index_family": f.get("_amd_arch_index_family"),
        "repair _venv_torch_amd_family": r.get("venv_family"),
        "repair block has family check": r.get("block_has_family_check"),
        "venv torch version": vt.get("torch_version"),
        "venv torch hip": vt.get("torch_hip"),
        "venv rocm requires": vt.get("rocm_requires"),
        "venv python": r.get("venv_python"),
    }


def _venv_is_gfx1151_713(o: dict) -> tuple[bool, str]:
    vt, r = _vt(o), _rep(o)
    tv = vt.get("torch_version") or ""
    m = _ROCM_VER.search(tv)
    ver_ok = bool(m) and (int(m.group(1)), int(m.group(2))) >= (7, 13)
    reqs = " ".join(vt.get("rocm_requires") or []).lower()
    fam_ok = f"libraries-{EXPECT_GFX}" in reqs.replace("_", "-")
    ok = ver_ok and fam_ok and r.get("below_7_13") == "no"
    return ok, (f"torch `{tv or vt.get('torch_error') or 'none'}`, rocm requires "
                f"`{vt.get('rocm_requires')}`, below_7_13 `{r.get('below_7_13')}`")


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = list(rr.gates(obs))
    for name in ("base", "head"):
        o = obs.get(name) or {}
        r = _rep(o)
        out.append((f"{name} migrated repair block ran to completion",
                    bool(r.get("block_completed")) and not r.get("error") and r.get("rc") == 0,
                    r.get("error") or f"rc={r.get('rc')}"))
        rf = (r.get("fields") or {}).get("_torch_index_is_rocm_family")
        # install.sh only enters the repair block under this flag; off it, REINSTALL is moot.
        out.append((f"{name} index is a ROCm family (install.sh reaches the repair block)",
                    rf == "true", f"`{rf}`"))
        ok, detail = _venv_is_gfx1151_713(o)
        out.append((f"{name} venv torch is the AMD {EXPECT_GFX} rocm7.13+ wheel", ok, detail))
    h = obs.get("head") or {}
    hr = _rep(h)
    hf = hr.get("fields") or {}
    out.append(("head repair block carries the per-arch family check",
                bool(hr.get("block_has_family_check")),
                f"{hr.get('block_has_family_check')}"))
    out.append((f"head routed to the {EXPECT_GFX} per-arch family",
                hf.get("_amd_arch_index_routed") == "true"
                and hf.get("_amd_arch_index_family") == EXPECT_GFX,
                f"routed `{hf.get('_amd_arch_index_routed')}`, family "
                f"`{hf.get('_amd_arch_index_family')}`"))
    out.append((f"head _venv_torch_amd_family reads {EXPECT_GFX} off the real venv",
                hr.get("venv_family") == EXPECT_GFX, f"`{hr.get('venv_family')}`"))
    return out


def _row(k, bv, hv, judged: bool) -> str:
    short = lambda v: str(v).replace("|", "\\|")[:160]  # noqa: E731
    same = ("yes" if bv == hv else "NO") if judged else "(info)"
    return f"| {k} | `{short(bv)}` | `{short(hv)}` | {same} |"


def table(obs: dict) -> str:
    b, h = obs.get("base") or {}, obs.get("head") or {}
    rows = [rr.table(obs)]
    rb, rh = repair_fields(b), repair_fields(h)
    rows += [_row(k, rb[k], rh.get(k), True) for k in rb]
    ib, ih = info_fields(b), info_fields(h)
    rows += [_row(k, ib[k], ih.get(k), False) for k in ib]
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    worse, detail = rr.head_is_worse(base, head)
    if worse:
        return True, detail
    rb, rh = repair_fields(base), repair_fields(head)
    moved = [k for k in rb if rb[k] != rh.get(k)]
    if moved:
        return True, ("migrated repair moved at the head on this host: "
                      + ", ".join(f"`{k}` ({rb[k]!r} -> {rh.get(k)!r})" for k in moved))
    if _rep(head).get("reinstall"):
        return True, ("head migrated repair would REINSTALL a venv already on "
                      f"{_vt(head).get('torch_version')}: {_rep(head).get('substeps')}")
    return False, (detail + f"; migrated repair identical ({len(rb)} fields) and does not "
                   f"reinstall the venv `{_vt(head).get('torch_version')}` "
                   f"(family `{_rep(head).get('venv_family')}`, routed family "
                   f"`{(_rep(head).get('fields') or {}).get('_amd_arch_index_family')}`)")
