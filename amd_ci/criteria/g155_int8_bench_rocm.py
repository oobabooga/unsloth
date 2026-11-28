#!/usr/bin/env python3
"""Criteria: are int8 GEMMs faster than bf16 on gfx1151 for the rotated INT8 LoRA kernels (G155)?

Pairs with probes/g155_int8_bench_probe.py. The probe imports nothing from the checkout, so base and
head run identical kernels: the two states are an A/A pair, and that is what is judged.

  gates()        non-vacuity: each state's sweep finished on ROCm gfx1151, every cell has bf16
                 timings, both int8 paths were probed (an int8 path that errors is a FINDING and is
                 reported in the table, not a gate failure), and the bf16 reference reproduces
                 between the two runs within BF16_AA_TOL (median over cells).
  head_is_worse  the int8 result did not reproduce: the head run's best int8 fwd or dgrad is more
                 than INT8_AA_TOL slower than the base run's on the median cell. With identical code
                 that means the int8 number is noise, and the table should not be quoted.
  table()        the answer: per cell bf16 vs best int8 (GEMM only, and with the activation quant
                 kernel added), plus geometric-mean speedups.
"""

from __future__ import annotations

import math

TITLE = "G155 rotated INT8 LoRA GEMMs: int8 vs bf16 on gfx1151 (A/A pair, identical kernels)"
MODE = "regression"
NEEDS: list[str] = ["rocm", "gpu", "exclusive_gpu", "discrete_gpu", "nvidia"]

BF16_AA_TOL = 0.15
INT8_AA_TOL = 0.20


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def _median(xs):
    xs = sorted(xs)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def _geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else None


def _rel_dev(base: dict, head: dict, field: str):
    devs = []
    for key, b in (base.get("shapes") or {}).items():
        h = (head.get("shapes") or {}).get(key) or {}
        bv, hv = _num(b.get(field)), _num(h.get(field))
        if bv and hv:
            devs.append((hv - bv) / bv)
    return devs


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        rc = o.get("kbench_rc")
        out.append((f"{name}: kbench sweep finished", rc == 0 and bool(o.get("finished")),
                    f"rc={rc}, {o.get('kbench_seconds')} s, finished={o.get('finished')}"
                    + ("" if rc == 0 else f"; log tail: {str(o.get('kbench_log_tail', ''))[-300:]!r}")))
        arch = str(o.get("arch") or (o.get("env") or {}).get("arch") or "")
        out.append((f"{name}: ran on ROCm gfx1151", bool(o.get("hip")) and arch.startswith("gfx1151"),
                    f"device={o.get('device')}, arch={arch or None}, hip={o.get('hip')}, "
                    f"torch={o.get('torch')}, triton={o.get('triton')}, dtype={o.get('dtype')}"))
        shapes = o.get("shapes") or {}
        bad = [k for k, s in shapes.items() if not (_num(s.get("bf16_fwd_us")) and _num(s.get("bf16_dgrad_us")))]
        out.append((f"{name}: every cell has bf16 timings", bool(shapes) and not bad,
                    f"{len(shapes)} cells" + (f"; without bf16: {', '.join(bad)}" if bad else "")))
        probes = o.get("probes") or {}
        out.append((f"{name}: both int8 paths probed", {"int_mm", "triton_int8_dot"} <= set(probes),
                    f"torch._int_mm: {probes.get('int_mm')}; Triton int8 tl.dot: {probes.get('triton_int8_dot')}"))
    base, head = obs.get("base") or {}, obs.get("head") or {}
    devs = [abs(d) for f in ("bf16_fwd_us", "bf16_dgrad_us") for d in _rel_dev(base, head, f)]
    med = _median(devs)
    out.append(("bf16 reference reproduces between the two runs",
                med is not None and med <= BF16_AA_TOL,
                f"median |head-base|/base over {len(devs)} bf16 timings = "
                + (f"{med:.1%}" if med is not None else "n/a") + f" (tolerance {BF16_AA_TOL:.0%})"))
    return out


def _fmt(v, spec = ".0f"):
    v = _num(v)
    return format(v, spec) if v is not None else "-"


def _fmtx(v):
    v = _num(v)
    return f"{v:.2f}x" if v is not None else "-"


def table(obs: dict) -> str:
    rows = []
    for name in ("base", "head"):
        o = obs.get(name)
        if not o:
            continue
        p = o.get("probes") or {}
        rows.append(f"**{name} run** ({o.get('kbench_seconds')} s; {o.get('device')} {o.get('arch')}, "
                    f"torch {o.get('torch')}, hip {o.get('hip')}, triton {o.get('triton')}; "
                    f"torch._int_mm: `{p.get('int_mm')}`, Triton int8 dot: `{p.get('triton_int8_dot')}`)")
        rows.append("")
        rows.append("| N x K @ M | bf16 fwd us | int8 fwd us (best) | fwd x | fwd x +quant | "
                    "bf16 dgrad us | int8 dgrad us (best) | dgrad x | dgrad x +quant | quant x / dy us |")
        rows.append("|---|---|---|---|---|---|---|---|---|---|")
        g = {"f": [], "fq": [], "d": [], "dq": []}
        for key, s in sorted((o.get("shapes") or {}).items()):
            if s.get("cell_error") and not s.get("bf16_fwd_us"):
                rows.append(f"| {key} | cell error: {str(s['cell_error'])[:120]} |||||||||")
                continue
            g["f"].append(_num(s.get("speedup_fwd_gemm")))
            g["fq"].append(_num(s.get("speedup_fwd_with_quant")))
            g["d"].append(_num(s.get("speedup_dgrad_gemm")))
            g["dq"].append(_num(s.get("speedup_dgrad_with_quant")))
            rows.append(
                f"| {key} | {_fmt(s.get('bf16_fwd_us'))} | {_fmt(s.get('int8_fwd_us'))} `{s.get('int8_fwd_best') or '-'}` "
                f"| {_fmtx(s.get('speedup_fwd_gemm'))} | {_fmtx(s.get('speedup_fwd_with_quant'))} "
                f"| {_fmt(s.get('bf16_dgrad_us'))} | {_fmt(s.get('int8_dgrad_us'))} `{s.get('int8_dgrad_best') or '-'}` "
                f"| {_fmtx(s.get('speedup_dgrad_gemm'))} | {_fmtx(s.get('speedup_dgrad_with_quant'))} "
                f"| {_fmt(s.get('quant_x_us'))} / {_fmt(s.get('quant_dy_us'))} |")
        rows.append("")
        rows.append("Geomean int8 speedup over bf16 (x > 1 means int8 faster): "
                    + ", ".join(f"{lbl} {_fmtx(_geomean(g[k]))}" for k, lbl in
                                (("f", "fwd GEMM"), ("fq", "fwd + quant"), ("d", "dgrad GEMM"), ("dq", "dgrad + quant"))))
        rows.append("")
    rows.append("Full per-tile timings: `kbench_base.json`, `kbench_head.json` in this directory.")
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    parts, worse = [], False
    for field, lbl in (("int8_fwd_us", "int8 fwd"), ("int8_dgrad_us", "int8 dgrad")):
        devs = _rel_dev(base, head, field)
        med = _median(devs)
        if med is None:
            parts.append(f"{lbl}: no timing at both states (no working int8 path, see the table)")
            continue
        parts.append(f"{lbl}: median (head-base)/base = {med:+.1%} over {len(devs)} cells")
        if med > INT8_AA_TOL:
            worse = True
    if worse:
        return True, ("the int8 timings did not reproduce between two runs of identical kernels (tolerance "
                      f"{INT8_AA_TOL:.0%}): " + "; ".join(parts) + ". Treat the int8 numbers as noise")
    return False, ("identical kernels at both states, int8 timings reproduce: " + "; ".join(parts)
                   + ". The answer to int8 vs bf16 is the speedup columns above")
