#!/usr/bin/env python3
"""Criteria for unslothai/unsloth#7102, real-torchao version (pairs with probes/torchao_export_real_probe.py).

Defect at the base: stock torchao cannot import on Windows ROCm (no torch.distributed), yet the
export gate offers the torchao formats and a real torchao export of a real LoRA model fails.
Fixed at the head: the export worker loads real torchao, the formats are advertised, and both
torchao_int8 and torchao_fp8 exports succeed and reload (without Unsloth) as quantized weights
with finite logits.
"""

from __future__ import annotations

TITLE = "Studio torchao INT8 / FP8 export on Windows ROCm, base versus head"
MODE = "differential"
NEEDS: list[str] = ["windows", "rocm", "windows_rocm_wddm", "gpu", "nvidia"]


def _p(o):
    return (o or {}).get("premise") or {}


def _w(o):
    return (o or {}).get("worker") or {}


def _exports(o):
    return _w(o).get("exports") or {}


def gates(obs):
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        p, w = _p(o), _w(o)
        out.append((f"{name}: Windows host with a ROCm torch and a visible GPU",
                    p.get("platform") == "win32" and bool(p.get("hip")) and bool(p.get("cuda_available")),
                    f"torch={p.get('torch')} hip={p.get('hip')} device={p.get('device')} arch={p.get('gcn_arch')}"))
        out.append((f"{name}: torch.distributed absent (the premise)", p.get("dist_available") is False,
                    f"dist_available={p.get('dist_available')}; stock torchao: {p.get('stock_torchao')}"))
        out.append((f"{name}: unsloth imported and a LoRA model loaded", "unsloth_file" in w and "model_error" not in w,
                    w.get("unsloth_import_error") or w.get("model_error") or w.get("error") or str(w.get("unsloth_file"))))
        out.append((f"{name}: both exports attempted", set(_exports(o)) == {"torchao_int8", "torchao_fp8"},
                    str(sorted(_exports(o)))))
    return out


def table(obs):
    rows = ["| fact | base | head |", "|---|---|---|"]

    def cell(o, f):
        try:
            v = f(o)
        except Exception as e:  # noqa: BLE001
            v = f"n/a ({type(e).__name__})"
        return str(v).replace("|", "\\|").replace("\n", " ")[:260]

    facts = [
        ("torch / arch", lambda o: f"{_p(o).get('torch')} / {_p(o).get('gcn_arch')}"),
        ("torch.distributed.is_available()", lambda o: _p(o).get("dist_available")),
        ("stock `import torchao`", lambda o: _p(o).get("stock_torchao")),
        ("plain Core `import unsloth`: torchao", lambda o: {k: (o or {}).get("core", {}).get(k) for k in ("unsloth_import_error", "torchao_file", "torchao_version", "configs")}),
        ("export worker loads torchao via", lambda o: _w(o).get("worker_loader")),
        ("torchao in the worker", lambda o: f"stub={_w(o).get('torchao_is_stub')} version={_w(o).get('torchao_version')}"),
        ("_torchao_export_supported()", lambda o: _w(o).get("gate_torchao_export_supported")),
        ("export_capability()", lambda o: _w(o).get("export_capability") or _w(o).get("capability_error")),
        ("torchao_int8 export", lambda o: _exports(o).get("torchao_int8")),
        ("torchao_fp8 export", lambda o: _exports(o).get("torchao_fp8")),
        ("reload without Unsloth", lambda o: (o or {}).get("verify", "not run")),
    ]
    b, h = obs.get("base") or {}, obs.get("head") or {}
    for label, f in facts:
        rows.append(f"| {label} | {cell(b, f)} | {cell(h, f)} |")
    return "\n".join(rows)


def base_shows_defect(base):
    w, e = _w(base), _exports(base)
    checks = {
        "gate offers torchao": w.get("gate_torchao_export_supported") is True,
        "a real torchao export fails": any(v.get("ok") is False for v in e.values()),
    }
    missing = [k for k, v in checks.items() if not v]
    return (not missing), ("all of: " + ", ".join(checks)) if not missing else "missing: " + ", ".join(missing)


def head_is_fixed(head):
    w, e, v = _w(head), _exports(head), (head or {}).get("verify") or {}

    def reloaded(alias, qtype):
        row = v.get(alias)
        return isinstance(row, dict) and row.get("finite") is True and qtype in row.get("param_types", [])

    checks = {
        "real torchao loaded in the worker": w.get("torchao_is_stub") is False,
        "plain Core import unsloth gets real torchao": isinstance((head or {}).get("core", {}).get("configs"), list),
        "formats advertised": (w.get("export_capability") or {}).get("torchao_export_supported") is True,
        "torchao_int8 export ok": (e.get("torchao_int8") or {}).get("ok") is True,
        "torchao_fp8 export ok": (e.get("torchao_fp8") or {}).get("ok") is True,
        "int8 reloads as Int8Tensor, finite": reloaded("torchao_int8", "Int8Tensor"),
        "fp8 reloads as Float8Tensor, finite": reloaded("torchao_fp8", "Float8Tensor"),
    }
    missing = [k for k, v in checks.items() if not v]
    return (not missing), ("all of: " + ", ".join(checks)) if not missing else "missing: " + ", ".join(missing)
