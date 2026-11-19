#!/usr/bin/env python3
"""Criteria (#18001): the head's paging margin never gives the iGPU less than the base's, and the
head's llama-server still loads where the base's did."""

from __future__ import annotations

TITLE = "Paged-load fit headroom on a Windows Vulkan iGPU, base versus head"
MODE = "regression"
NEEDS: list[str] = ["gpu", "windows"]


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        out.append((f"{name} probe ran", not o.get("error"), o.get("error") or "ok"))
        out.append((f"{name} found one Vulkan iGPU", len(o.get("igpu_ids") or []) == 1,
                    f"rows={o.get('rows')}"))
        srv = o.get("server") or {}
        out.append((f"{name} fitter reported a placement", bool(srv.get("fit_result")),
                    f"fit={srv.get('fit_result')} listening={srv.get('listening')}"))
    return out


def table(obs: dict) -> str:
    rows = ["| state | Vulkan free (row) | carve-out GiB | headroom MiB | --fit-target | fit placement | listening |",
            "|---|---|---|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if not o:
            continue
        co = o.get("carve_out_bytes")
        srv = o.get("server") or {}
        rows.append(f"| {name} | {o.get('rows')} | {co / 1024**3 if co else co} | {o.get('headroom_mib')} "
                    f"| {o.get('effective_fit_target')} | {srv.get('fit_result')} | {srv.get('listening')} |")
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    bt, ht = base.get("effective_fit_target"), head.get("effective_fit_target")
    if bt is None or ht is None:
        return True, "a state produced no fit target"
    if ht > bt:
        return True, f"head keeps a larger margin ({ht} > {bt} MiB)"
    bs, hs = base.get("server") or {}, head.get("server") or {}
    if bs.get("listening") and not hs.get("listening"):
        return True, "head llama-server did not come up where base did"
    return False, f"fit target base {bt} MiB, head {ht} MiB; head listening={hs.get('listening')}"
