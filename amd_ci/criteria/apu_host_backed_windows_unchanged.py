#!/usr/bin/env python3
"""Criteria: does unsloth#11451 move anything on Windows? It must not.

Both hunks are inside `IS_ROCM and platform.system() == "Linux"`. The Windows
sibling three lines down -- `_windows_rocm_shared_pool_host_gb_by_index`, which
reads the DirectX registry -- is untouched, and so is the loop that consumes it.
So the claim this leg exists to test is a NEGATIVE one, and it is stronger than
"the tests still pass": the device dicts `get_backend_visible_gpu_info()` returns
on a Windows Strix Halo, and the Settings > System tile string derived from them,
must be IDENTICAL at base and head. Not similar. Identical.

Regression mode rather than differential, deliberately. A differential asks the
base to exhibit a defect, and on Windows there is no defect to exhibit: the two
states agreeing IS the result. Running this leg as a differential would report
VOID -- technically true and useless, because VOID means "nothing was learned"
and here something was.

What it cannot answer, and says so rather than inventing a pass: whether these
boxes carry a ROCm-enabled torch at all. If torch has no visible device the
backend returns no ROCm rows, the comparison is between two empty lists, and two
empty lists are always identical. That is vacuous, so it is a GATE, and a
Windows leg that cannot see the GPU lands on INCONCLUSIVE.

Pairs with probes/studio_apu_host_backed_probe.py.
"""

from __future__ import annotations

import json

TITLE = ("unsloth#11451 control: a Linux-only branch must leave the Windows "
         "Strix Halo readout byte-identical")
MODE = "regression"

# The change is Linux-only, so the Windows leg's own reach is what has to be
# declared: it is a Windows ROCm WDDM question, on an integrated part, and the
# same untouched loop also serves discrete and multi-GPU inventories.
NEEDS = ["windows", "windows_rocm_wddm", "rocm", "gpu", "integrated_gpu",
         "discrete_gpu", "multi_gpu"]

# The fields a consumer reads. Compared verbatim; a float that moved in the last
# decimal place is a difference, because nothing on this platform should have
# touched it at all.
COMPARED_FIELDS = ("index", "name", "memory_total_gb", "shared_memory",
                   "shared_memory_host_backed_gb",
                   "shared_memory_host_backed_gb_key_present",
                   "shared_memory_host_backed_gb_is_null", "unified_memory")


def _info(obs: dict):
    """The WHOLE `get_backend_visible_gpu_info()` return, not only its device
    rows. On a box with no ROCm torch the device list is empty and comparing two
    empty lists proves nothing, but the surrounding dict (`available`, `backend`,
    the reason it is empty) is still a real answer that the change must not have
    altered. Compared as well as gated on, so the artifact records what Windows
    actually returned rather than only that it returned nothing useful."""
    entry = (obs.get("backend") or {}).get("get_backend_visible_gpu_info") or {}
    return entry.get("value") if entry.get("ok") else {"_error": entry.get("error")}


def _payload(obs: dict) -> dict:
    """Everything that must not move: the backend answer, the device rows and the
    rendered tile."""
    render = obs.get("render") or {}
    totals = render.get("totals_python_port")
    totals = totals.get("value") if isinstance(totals, dict) and "value" in totals else None
    ts = (render.get("typescript") or {}).get("totals")
    return {
        "backend_info": _info(obs),
        "devices": [{f: d.get(f) for f in COMPARED_FIELDS}
                    for d in (obs.get("devices") or [])],
        "totals_port": totals,
        "totals_typescript": ts,
        "labels": render.get("labels_typescript") or render.get("labels_python_port"),
    }


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        system = o.get("platform_system")
        out.append((f"{name}: ran on Windows", system == "Windows",
                    f"platform.system()={system!r} on host {o.get('hostname')!r}"))

        backend = o.get("backend") or {}
        err = backend.get("import_error") or (
            backend.get("get_backend_visible_gpu_info") or {}).get("error")
        out.append((f"{name}: the backend answered", not err,
                    str(err)[:300] if err else
                    f"hardware.py at {backend.get('hardware_file')}"))

        # The vacuity that matters here. Two empty device lists always match.
        devices = o.get("devices") or []
        torch = o.get("torch") or {}
        out.append((f"{name}: a GPU row exists to compare", len(devices) >= 1,
                    f"{len(devices)} device(s); torch "
                    + (f"error: {str(torch.get('error'))[:200]}" if torch.get("error")
                       else f"{torch.get('version')}, hip={torch.get('hip')!r}, "
                            f"device_count={torch.get('device_count')}, "
                            f"is_available={torch.get('is_available')}")))
    return out


def table(obs: dict) -> str:
    rows = ["| state | devices | tile totals (dedicated / shared / total) | Settings tile |",
            "|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if not o:
            continue
        p = _payload(o)
        totals = p["totals_typescript"] or p["totals_port"] or {}
        labels = p["labels"] or {}
        rows.append(f"| {name} | `{json.dumps(p['devices'])}` "
                    f"| {totals.get('dedicated')} / {totals.get('shared')} "
                    f"/ {totals.get('total')} "
                    f"| `{labels.get('resources_tab', '?')}` |")
    adapters = (((obs.get("head") or {}).get("raw") or {})
                .get("windows_adapters") or {}).get("value")
    notes = []
    for name in ("base", "head"):
        o = obs.get(name)
        if o:
            notes.append(f"`get_backend_visible_gpu_info()` at the {name}: "
                         f"`{json.dumps(_info(o), default = repr)[:800]}`")
    if adapters:
        notes.append("Windows adapter inventory read independently of Studio: "
                     f"`{json.dumps(adapters, default = repr)[:600]}`")
    return "\n".join(rows) + "\n\n" + "\n\n".join(notes)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    b, h = _payload(base), _payload(head)
    differing = [k for k in b if b[k] != h[k]]
    if differing:
        return True, ("a Linux-only change moved the Windows readout. Differing: "
                      + ", ".join(f"`{k}`: base `{json.dumps(b[k])}` vs head "
                                  f"`{json.dumps(h[k])}`" for k in differing))
    return False, ("the Windows device rows and the tile string derived from them are "
                   "byte-identical at base and head, which is what a branch guarded by "
                   "`platform.system() == \"Linux\"` is required to do: "
                   f"`{json.dumps(b['devices'])}`")
