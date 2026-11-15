#!/usr/bin/env python3
"""Criteria (W5): fused CE loss, GA normalisation, offload_layers block swap and the gpt-oss MXFP4
merge_and_unload export on tiny models, gfx1151 ROCm, two version corners (A: transformers 4.57.6 / TRL 0.22.2 /
PEFT 0.18.0; B: transformers 5.23.0 / TRL 1.18.0 / PEFT 0.21.3). Base = unsloth-zoo main, head = main plus the
MXFP4 export fix (1176505a8); unsloth is main at both states.

Regression mode. Head is WORSE when a (corner, model, cell) that passed at the base does not pass at the head.
Cells that fail at both states are reported as pre-existing; a cell that fails at the base and passes at the head
is reported as fixed (the gpt-oss export cell is expected to be that one on corner B, if this ROCm stack loads the
MXFP4 checkpoint at all).

Gates: each state imported its own unsloth_zoo and ran on a gfx1151 GPU; at the base, corner B ran the core
llama cells (fused_ce, ga_norm, offload) to a verdict, and the offload cell actually swapped (copies counted).
"""
from __future__ import annotations

TITLE = "W5: fused CE / GA / offload_layers / MXFP4 export on gfx1151 ROCm, two corners, base vs head"
MODE = "regression"
NEEDS = ["rocm", "gpu"]


def _cells(o: dict) -> dict:
    out = {}
    for corner, c in ((o or {}).get("corners") or {}).items():
        for model, rec in ((c or {}).get("summary") or {}).items():
            if rec.get("import_error"):
                out[(corner, model, "import")] = {"ok": False, "error": rec["import_error"]}
            for cell, r in (rec.get("cells") or {}).items():
                out[(corner, model, cell)] = r
    return out


def _arch(o: dict) -> str:
    for c in ((o or {}).get("corners") or {}).values():
        for rec in ((c or {}).get("summary") or {}).values():
            a = (rec.get("env") or {}).get("arch")
            if a:
                return str(a)
    return ""


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        ck = str(o.get("checkout") or "<none>")
        f = str(o.get("unsloth_zoo_file") or "")
        out.append((f"{name} venv built", not (o.get("venv") or {}).get("error"),
                    str((o.get("venv") or {}).get("error") or (o.get("venv") or {}).get("pip_list"))[:300]))
        out.append((f"{name} imported its own unsloth_zoo", bool(f) and f.startswith(ck),
                    f or str(o.get("unsloth_zoo_import_error"))[:300]))
        out.append((f"{name} ran on gfx1151", "gfx1151" in _arch(o), _arch(o) or "no device recorded"))
    base = _cells(obs.get("base"))
    for cell in ("fused_ce", "ga_norm", "offload"):
        r = base.get(("B", "llama", cell))
        out.append((f"base corner B llama {cell} reached a verdict", r is not None and r.get("ok") is not None,
                    str(r)[:300]))
    r = base.get(("B", "llama", "offload")) or {}
    out.append(("base corner B llama offload swapped layers", bool(r.get("engaged")),
                f"copies={r.get('copies_total')} host_bytes={r.get('host_bytes')} swapped={r.get('swapped')}"))
    return out


def _fmt(r) -> str:
    if r is None:
        return "-"
    ok = r.get("ok")
    s = "P" if ok is True else ("n/a" if ok is None else "F")
    for k in ("rel", "loss_rel", "grad_rel", "logits_rel", "copies_total"):
        if k in r and isinstance(r[k], (int, float)):
            s += f" {k}={r[k]:.2g}"
    if ok is False:
        s += " " + str(r.get("error") or r.get("refused") or r.get("leftover_keys") or "")[:90]
    return s


def table(obs: dict) -> str:
    b, h = _cells(obs.get("base")), _cells(obs.get("head"))
    rows = ["| corner | model | cell | base | head |", "|---|---|---|---|---|"]
    for k in sorted(set(b) | set(h)):
        rows.append(f"| {k[0]} | {k[1]} | {k[2]} | {_fmt(b.get(k))} | {_fmt(h.get(k))} |")
    vers = {}
    for c, cc in ((obs.get("head") or {}).get("corners") or {}).items():
        for rec in ((cc or {}).get("summary") or {}).values():
            if rec.get("env"):
                vers[c] = {k: rec["env"].get(k) for k in ("torch", "hip", "transformers", "trl", "peft", "arch")}
                break
    return "\n".join(rows) + "\n\nVersions (head): " + str(vers)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    b, h = _cells(base), _cells(head)
    worse = sorted(k for k in b if b[k].get("ok") is True and (h.get(k) or {}).get("ok") is not True)
    fixed = sorted(k for k in h if h[k].get("ok") is True and (b.get(k) or {}).get("ok") is False)
    both = sorted(k for k in b if b[k].get("ok") is False and (h.get(k) or {}).get("ok") is False)
    fmt = lambda ks: ", ".join("/".join(k) for k in ks) or "none"  # noqa: E731
    detail = f"newly passing at the head: {fmt(fixed)}; failing at both states (pre-existing): {fmt(both)}"
    if worse:
        return True, f"passed at the base, not at the head: {fmt(worse)}; " + detail
    return False, "no cell that passed at the base fails at the head; " + detail
