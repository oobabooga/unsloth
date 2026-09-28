#!/usr/bin/env python3
"""Criteria for unslothai/unsloth#7102 (pairs with probes/torchao_export_gate_probe.py).

Defect at the base: on Windows ROCm torch.distributed is absent and Studio's stub makes
Int8WeightOnlyConfig() return None, yet the export gate still offers the torchao formats and a
real torchao export of a real LoRA model fails. The exact error text depends on the transformers
and torchao versions installed, so it is recorded, not matched.
Fixed at the head: the gate withholds them, a forced torchao export is refused with a
Windows ROCm message before any work, and export_capability() reports win32_rocm True.
"""

from __future__ import annotations

TITLE = "Studio torchao export gate on Windows ROCm, base versus head"
MODE = "differential"
NEEDS: list[str] = ["windows", "rocm", "windows_rocm_wddm", "gpu", "nvidia"]


def _w(o: dict) -> dict:
    return (o or {}).get("worker") or {}


def _p(o: dict) -> dict:
    return (o or {}).get("premise") or {}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        p, w = _p(o), _w(o)
        out.append((f"{name}: probe produced results", bool(p) and bool(w) and "error" not in p and "error" not in w,
                    o.get("error") or p.get("error") or w.get("error") or "ok"))
        out.append((f"{name}: Windows host with a ROCm torch", p.get("platform") == "win32" and bool(p.get("hip") or "rocm" in str(p.get("torch", "")).lower()),
                    f"platform={p.get('platform')} torch={p.get('torch')} hip={p.get('hip')}"))
        out.append((f"{name}: GPU visible to torch", bool(p.get("cuda_available")),
                    f"device={p.get('device')} arch={p.get('gcn_arch')}"))
        # Without a real unsloth the base gate would read False for the wrong reason.
        out.append((f"{name}: unsloth imports with the torchao export path", w.get("unsloth_has_torchao_method") is True,
                    w.get("unsloth_import_error") or f"unsloth {w.get('unsloth_version')}"))
        out.append((f"{name}: Studio torchao stub active", w.get("torchao_is_stub") is True,
                    w.get("stub_error") or f"torchao_is_stub={w.get('torchao_is_stub')}"))
        # A load failure must not pass for an export failure.
        out.append((f"{name}: real model loaded and export attempted", "e2e_torchao_export" in w,
                    w.get("e2e_error") or f"model on {w.get('e2e_model_device')}"))
    return out


def table(obs: dict) -> str:
    rows = ["| fact | base | head |", "|---|---|---|"]

    def cell(o, f):
        try:
            v = f(o)
        except Exception as e:  # noqa: BLE001
            v = f"n/a ({type(e).__name__})"
        return str(v).replace("|", "\\|").replace("\n", " ")[:220]

    b, h = obs.get("base") or {}, obs.get("head") or {}
    facts = [
        ("torch", lambda o: _p(o).get("torch")),
        ("torch.version.hip", lambda o: _p(o).get("hip")),
        ("device / arch", lambda o: f"{_p(o).get('device')} / {_p(o).get('gcn_arch')}"),
        ("torch.distributed.is_available()", lambda o: _p(o).get("dist_available")),
        ("torch._C._distributed_c10d present", lambda o: _p(o).get("has_c10d_ext")),
        ("real `import torchao`", lambda o: _p(o).get("real_torchao")),
        ("stubbed Int8WeightOnlyConfig()", lambda o: _w(o).get("int8_config")),
        ("TorchAoConfig(quant_type=<that>)", lambda o: _w(o).get("torchaoconfig_error")),
        ("_torchao_export_supported()", lambda o: _w(o).get("gate_torchao_export_supported")),
        ("forced torchao_int8 export (placeholder model)", lambda o: _w(o).get("forced_torchao_export") or _w(o).get("export_error")),
        ("real LoRA model on", lambda o: _w(o).get("e2e_model_device")),
        ("real torchao_int8 export of a LoRA model", lambda o: _w(o).get("e2e_torchao_export") or _w(o).get("e2e_error")),
        ("export_capability()", lambda o: _w(o).get("export_capability") or _w(o).get("capability_error")),
    ]
    for label, f in facts:
        rows.append(f"| {label} | {cell(b, f)} | {cell(h, f)} |")
    return "\n".join(rows)


def base_shows_defect(base: dict) -> tuple[bool, str]:
    p, w = _p(base), _w(base)
    e2e = w.get("e2e_torchao_export") or {}
    checks = {
        "torch.distributed unavailable": p.get("dist_available") is False,
        "stubbed config is None": w.get("int8_config") == "None",
        "gate still offers torchao": w.get("gate_torchao_export_supported") is True,
        "real torchao export of a LoRA model fails": e2e.get("ok") is False,
    }
    missing = [k for k, v in checks.items() if not v]
    return (not missing), ("all of: " + ", ".join(checks)) if not missing else "missing: " + ", ".join(missing)


def head_is_fixed(head: dict) -> tuple[bool, str]:
    w = _w(head)
    forced = w.get("forced_torchao_export") or {}
    checks = {
        "gate withholds torchao": w.get("gate_torchao_export_supported") is False,
        "forced request refused with a Windows ROCm message": forced.get("ok") is False and "Windows ROCm" in str(forced.get("message", "")),
        "export_capability reports win32_rocm": (w.get("export_capability") or {}).get("win32_rocm") is True,
        "real LoRA export refused with a Windows ROCm message": (w.get("e2e_torchao_export") or {}).get("ok") is False
        and "Windows ROCm" in str((w.get("e2e_torchao_export") or {}).get("message", "")),
    }
    missing = [k for k, v in checks.items() if not v]
    return (not missing), ("all of: " + ", ".join(checks)) if not missing else "missing: " + ", ".join(missing)
