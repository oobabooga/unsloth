#!/usr/bin/env python3
"""Criteria for probes/rotq_bench_probe.py: a measurement of the RDNA W8A8 rotate+quant and GEMM kernels at FLUX.3
Klein's shapes. Nothing is judged between states (the head alone is measured): the gates prove the kernels ran and
the table carries the numbers; the verdict is always NO_REGRESSION once the gates pass."""

from __future__ import annotations

TITLE = "gfx1151: RDNA W8A8 rotate+quant / GEMM kernels at FLUX.3 Klein shapes (timing, exclusive GPU)"
MODE = "regression"
NEEDS = ["gpu", "rocm", "exclusive_gpu", "nvidia", "windows", "multi_gpu"]


def _rows(head: dict, kind: str) -> list:
    return [r for r in head.get("rows") or [] if r.get("kind") == kind]


def gates(obs: dict) -> list:
    head = obs.get("head") or {}
    return [("head measured", head.get("verdict") == "ok", str(head.get("error") or head.get("verdict"))[:200]),
            ("rotq rows", len(_rows(head, "rotq")) >= 4, str(len(_rows(head, "rotq")))),
            ("gemm rows", len(_rows(head, "gemm")) >= 4, str(len(_rows(head, "gemm"))))]


def head_is_worse(base: dict, head: dict) -> tuple:
    return False, "measurement only (head tree)"


def table(obs: dict) -> str:
    head = obs.get("head") or {}
    L = [f"torch {head.get('torch')} triton {head.get('triton')} arch {head.get('arch')} table {head.get('gemm_table')}",
         "", "| rotq rows x K | shipped ms | inductor ref ms | no-rotation quant ms | best tile (ms) | shipped acc | "
         "inductor acc |", "|---|---|---|---|---|---|---|"]
    for r in _rows(head, "rotq"):
        ms = r.get("ms") or {}
        L.append(f"| {r['m']} x {r['k']} | {ms.get('shipped')} | {ms.get('inductor_ref')} | "
                 f"{ms.get('plain_quant_no_rotation')} | {r.get('best')} | {r.get('shipped_acc')} | "
                 f"{r.get('inductor_acc')} |")
    L += ["", "| gemm rows x K x N | w8a8_mm ms | bf16 linear ms | TFLOPS |", "|---|---|---|---|"]
    for r in _rows(head, "gemm"):
        L.append(f"| {r['m']} x {r['k']} x {r['n']} | {r['ms'].get('w8a8_mm')} | {r['ms'].get('bf16_linear')} | "
                 f"{r.get('tflops')} |")
    return "\n".join(L)
