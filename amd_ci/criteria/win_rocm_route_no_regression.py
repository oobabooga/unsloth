#!/usr/bin/env python3
"""Criteria: the head's Windows ROCm torch route works on this card at least as well as the base's.

Pairs with probes/win_rocm_route_probe.py. Regression mode: base is the route the
installer picks today, head the route the change picks; head is worse if a smoke step
that passed at the base fails, the numbers drift past tolerance, or fp16 matmul
throughput drops more than 20%.
"""

from __future__ import annotations

import math

TITLE = "Windows ROCm torch route, installed and run, base versus head"
MODE = "regression"
NEEDS = ["gpu", "windows", "rocm", "windows_rocm_wddm"]

_STEPS = ("matmul", "grouped_mm", "torchvision", "torchaudio", "train")


def _smoke(o: dict) -> dict:
    return (o or {}).get("smoke") or {}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        out.append((f"{name} route resolved", bool(o.get("index_url")),
                    o.get("error") or f"{o.get('index_url')} {o.get('specs')}"))
        out.append((f"{name} trio installed", o.get("install_rc") == 0,
                    f"rc={o.get('install_rc')} in {o.get('install_s')}s; "
                    + (", ".join(o.get("freeze") or []) or str(o.get("install_tail", ""))[-300:])))
        s = _smoke(o)
        out.append((f"{name} torch sees the GPU", bool(s.get("cuda_available")),
                    f"torch {s.get('torch')} hip {s.get('hip')} device {s.get('device')} "
                    f"arch {s.get('arch')}" if s else str(o.get("smoke_error", ""))[-300:]))
    head_url = str((obs.get("head") or {}).get("index_url") or "")
    out.append(("head took the multi-arch route", "whl-multi-arch" in head_url, head_url))
    return out


def table(obs: dict) -> str:
    rows = ["| state | index | torch | arch | matmul err | fp16 TFLOPS | grouped_mm err | train loss first -> last | failed steps |",
            "|---|---|---|---|---|---|---|---|---|"]
    for name in ("base", "head"):
        o = obs.get(name) or {}
        s = _smoke(o)
        mm = s.get("matmul") or {}
        gm = s.get("grouped_mm") or {}
        tr = s.get("train") or {}
        failed = [k for k in _STEPS if isinstance(s.get(k), dict) and "error" in s[k]]
        rows.append(
            f"| {name} | {o.get('index_url')} | {s.get('torch')} | {s.get('arch')} "
            f"| {mm.get('max_abs_err')} | {round(mm.get('tflops_4096_fp16') or 0, 2)} "
            f"| {gm.get('max_abs_err_vs_bmm', gm.get('error'))} "
            f"| {tr.get('first')} -> {tr.get('last')} | {', '.join(failed) or 'none'} |")
    for name in ("base", "head"):
        tf = ((_smoke(obs.get(name) or {}).get("matmul") or {}).get("tflops_fp16")) or {}
        if tf:
            rows.append("")
            rows.append(f"{name} fp16 TFLOPS median [min, max] of 7: " + ", ".join(
                f"n={n}: {v['median']:.2f} [{v['min']:.2f}, {v['max']:.2f}]" for n, v in tf.items()))
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    b, h = _smoke(base), _smoke(head)
    problems = []
    for k in _STEPS:
        b_ok = isinstance(b.get(k), dict) and "error" not in b[k]
        h_ok = isinstance(h.get(k), dict) and "error" not in h[k]
        if b_ok and not h_ok:
            problems.append(f"`{k}` passes at base, fails at head: {(h.get(k) or {}).get('error')}")
    bm, hm = b.get("matmul") or {}, h.get("matmul") or {}
    if "max_abs_err" in hm and hm["max_abs_err"] > max(2 * bm.get("max_abs_err", 0), 0.1):
        problems.append(f"fp16 matmul error {hm['max_abs_err']} vs base {bm.get('max_abs_err')}")
    if bm.get("tflops_4096_fp16") and hm.get("tflops_4096_fp16") is not None \
            and hm["tflops_4096_fp16"] < 0.8 * bm["tflops_4096_fp16"]:
        problems.append(f"fp16 TFLOPS {hm['tflops_4096_fp16']:.2f} vs base {bm['tflops_4096_fp16']:.2f}")
    bg, hg = b.get("grouped_mm") or {}, h.get("grouped_mm") or {}
    if "max_abs_err_vs_bmm" in hg and hg["max_abs_err_vs_bmm"] > max(2 * bg.get("max_abs_err_vs_bmm", 0), 0.1):
        problems.append(f"grouped_mm error {hg['max_abs_err_vs_bmm']} vs base {bg.get('max_abs_err_vs_bmm')}")
    ht = h.get("train") or {}
    if "last" in ht and (not math.isfinite(ht["last"]) or ht["last"] >= ht["first"]):
        problems.append(f"head training did not reduce loss: {ht['first']} -> {ht['last']}")
    if problems:
        return True, "; ".join(problems)
    return False, "every smoke step that runs at the base runs at the head, within tolerance"
