#!/usr/bin/env python3
"""Criteria: Windows + CUDA 13.0 torch wiring of Studio's SSM kernel install, 9f5b13325 (base) vs
7092b2bdf (head). Regression mode: the head is worse if a cu13 win_amd64 cell that resolved to a
live URL at the base does not at the head, an install that succeeded at the base fails at the head,
or an extension that imported after the base's install does not after the head's."""

from __future__ import annotations

TITLE = "Windows + CUDA 13.0 torch: SSM kernel wheel resolution, install and import (no NVIDIA driver)"
MODE = "regression"
NEEDS: list[str] = ["windows", "nvidia"]


def _d(r) -> dict:
    return (r or {}).get("data") or {}


def _cells(o: dict) -> dict:
    out = {}
    for c in _d(o.get("urls")).get("cells") or []:
        for n in ("causal_conv1d", "mamba_ssm", "flash_attn"):
            v = c.get(n) or {}
            out[(c["torch"], c["py"], n)] = (v.get("url"), v.get("status"))
    return out


def gates(obs: dict):
    out = []
    for s in ("base", "head"):
        o = obs.get(s) or {}
        t = (_d(o.get("torch")) if isinstance(_d(o.get("torch")), dict) else None)
        tt = (o.get("torch") or {}).get("tail", "")
        out.append((f"{s} URL matrix ran", bool(_cells(o)), f"rc={(o.get('urls') or {}).get('rc')} "
                    f"{'' if _cells(o) else (o.get('urls') or {}).get('tail', '')[-300:]}"))
        out.append((f"{s} venv torch is CUDA 13.0", "13.0" in str((o.get("torch") or {}).get("data")),
                    f"{(o.get('torch') or {}).get('data')} {tt[-200:] if not (o.get('torch') or {}).get('data') else ''}"))
        out.append((f"{s} started without SSM kernels", "rc=" in str(o.get("reset_before")), str(o.get("reset_before"))[:200]))
    return out


def table(obs: dict) -> str:
    lines = ["| torch | py | " + " | ".join(f"{s} cc1d / mamba / fa" for s in ("base", "head")) + " |",
             "|---|---|---|---|"]
    cb, ch = _cells(obs.get("base") or {}), _cells(obs.get("head") or {})
    keys = sorted({(t, p) for (t, p, _) in list(cb) + list(ch)})
    for t, p in keys:
        row = []
        for c in (cb, ch):
            row.append(" / ".join(f"{c.get((t, p, n), (None, None))[1]}" for n in ("causal_conv1d", "mamba_ssm", "flash_attn")))
        lines.append(f"| {t} | {p} | {row[0]} | {row[1]} |")
    lines.append("")
    lines.append("| state | model | error | seconds | installed | status messages |")
    lines.append("|---|---|---|---|---|---|")
    for s in ("base", "head"):
        o = obs.get(s) or {}
        for k, r in (o.get("install") or {}).items():
            d = _d(r)
            lines.append(f"| {s} | {k} | {d.get('error') or (r or {}).get('tail', '')[-150:] if not d else d.get('error')} | "
                         f"{d.get('seconds')} | {d.get('returned')} | {'; '.join(map(str, d.get('status') or []))[:200]} |")
    lines.append("")
    for s in ("base", "head"):
        lines.append(f"- {s} kernels after install: {_d((obs.get(s) or {}).get('kernels'))}")
    return "\n".join(lines)


def head_is_worse(base: dict, head: dict):
    bad = []
    cb, ch = _cells(base), _cells(head)
    for k, (url, st) in cb.items():
        if st == 200 and (ch.get(k) or (None, None))[1] != 200:
            bad.append(f"{k}: base {st}, head {ch.get(k)}")
    for k, r in (base.get("install") or {}).items():
        if _d(r) and _d(r).get("error") is None and _d((head.get("install") or {}).get(k)).get("error"):
            bad.append(f"install {k}: head error {_d(head['install'][k]).get('error')}")
    kb, kh = _d(base.get("kernels")), _d(head.get("kernels"))
    for m in ("import_causal_conv1d", "import_causal_conv1d_cuda", "import_mamba_ssm", "import_selective_scan_cuda"):
        if kb.get(m) is True and kh.get(m) is not True:
            bad.append(f"{m}: base True, head {kh.get(m)}")
    return bool(bad), "; ".join(bad) or "head resolves, installs and imports everything the base did"
