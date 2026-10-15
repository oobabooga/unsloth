#!/usr/bin/env python3
"""Criteria: IQ1_M on the ROCm MMQ path (llama.cpp ggml-hip). Judges only.

Defect (base): IQ1_M has no MMQ kernel, so batched IQ1_M matmuls fall back to
dequantize + hipBLAS, which for MoE (MUL_MAT_ID) is a per-expert loop. Observable as
IQ1_M batched MUL_MAT_ID being much slower than IQ1_S (which has MMQ) in the same build.

Fixed (head): every IQ1_M MUL_MAT / MUL_MAT_ID eval case passes against the CPU
backend, IQ1_M batched MUL_MAT_ID is in IQ1_S's time class, and it is faster than base.

Thresholds are fixed here, before any run, and are not tuned after seeing results.
"""

from __future__ import annotations

import math

TITLE = "IQ1_M MMQ on gfx1151 (ROCm): base vs head"
MODE = "differential"
# Everything the change touches. The patch also adds config rows for RDNA2/3/4, CDNA
# and GCN, and the CUDA DP4A/MMA paths; this host answers only RDNA3.5 under ROCm on Linux.
NEEDS = ["rocm", "gpu", "discrete_gpu", "windows", "windows_rocm_wddm", "nvidia", "multi_gpu"]

BATCHED = ["32", "128", "512"]       # MMQ range (n = 1 is MMVQ in both states)
DEFECT_RATIO = 1.5                   # base: iq1_m / iq1_s time on batched MUL_MAT_ID, geomean
FIXED_RATIO = 1.5                    # head: same ratio must be at or below this
MIN_SPEEDUP = 1.2                    # head iq1_m vs base iq1_m, batched MUL_MAT_ID, geomean


def _geo(xs: list[float]) -> float | None:
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def _ratio(state: dict, a: str = "iq1_m", b: str = "iq1_s", kind: str = "mmid") -> float | None:
    pa = ((state.get("perf") or {}).get(f"{kind}:{a}") or {}).get("us_per_run") or {}
    pb = ((state.get("perf") or {}).get(f"{kind}:{b}") or {}).get("us_per_run") or {}
    rs = [pa[n] / pb[n] for n in BATCHED if n in pa and n in pb and pb[n] > 0]
    return _geo(rs)


def _eval_ok(state: dict, key: str) -> bool:
    e = (state.get("eval") or {}).get(key) or {}
    return e.get("total", 0) > 0 and e.get("passed") == e.get("total") and e.get("n_fail_lines", 1) == 0


_CTX: dict = {"base_iq1m": {}}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    states = {n: v for n, v in obs.items() if not n.startswith("_")}
    out = []
    out.append(("both states probed", {"base", "head"} <= set(states), ", ".join(states)))
    out.append(("ROCm device is gfx1151 in every state",
                all(v.get("gfx1151") for v in states.values()),
                ", ".join(f"{k}={v.get('gfx1151')}" for k, v in states.items())))
    libs = {k: v.get("hip_lib") or "" for k, v in states.items()}
    out.append(("each state ran its OWN libggml-hip",
                all(v and v.startswith(states[k].get("libdir", "?")) for k, v in libs.items())
                and len(set(libs.values())) == len(libs),
                ", ".join(f"{k}={v}" for k, v in libs.items())))
    cnt = {f"{k}:{e}": (v.get("eval") or {}).get(e, {}).get("total", 0)
           for k, v in states.items() for e in ("MUL_MAT:iq1_m", "MUL_MAT_ID:iq1_m")}
    out.append(("IQ1_M eval cases ran (> 0) in every state", all(c > 0 for c in cnt.values()),
                ", ".join(f"{k}={c}" for k, c in cnt.items())))
    ctrl = {k: _eval_ok(v, "MUL_MAT:iq1_s") and _eval_ok(v, "MUL_MAT_ID:iq1_s") for k, v in states.items()}
    out.append(("control: IQ1_S eval passes in every state (same MMQ kernel both sides)",
                all(ctrl.values()), ", ".join(f"{k}={c}" for k, c in ctrl.items())))
    perf_ok = {k: _ratio(v) is not None for k, v in states.items()}
    out.append(("batched MUL_MAT_ID timings exist for iq1_m and iq1_s in every state",
                all(perf_ok.values()), ", ".join(f"{k}={c}" for k, c in perf_ok.items())))
    _CTX["base_iq1m"] = (((states.get("base") or {}).get("perf") or {}).get("mmid:iq1_m") or {}).get("us_per_run") or {}
    return out


def table(obs: dict) -> str:
    rows = ["| state | eval MUL_MAT iq1_m | eval MUL_MAT_ID iq1_m | eval iq1_s (control) | "
            "MUL_MAT_ID us/run iq1_m (n=1/32/128/512) | iq1_s | iq1_m/iq1_s batched | "
            "dense MUL_MAT us/run iq1_m (n=16/128/512) | iq1_s |",
            "|---|---|---|---|---|---|---|---|---|"]
    for name, v in obs.items():
        if name.startswith("_"):
            continue
        e = v.get("eval") or {}
        f = lambda k: f"{e.get(k, {}).get('passed', 0)}/{e.get(k, {}).get('total', 0)}"
        p = v.get("perf") or {}
        g = lambda k, ns: " / ".join(f"{(p.get(k) or {}).get('us_per_run', {}).get(n, float('nan')):.0f}" for n in ns)
        r = _ratio(v)
        r_str = f"{r:.2f}" if r is not None else "-"
        rows.append(f"| {name} | {f('MUL_MAT:iq1_m')} | {f('MUL_MAT_ID:iq1_m')} | "
                    f"{f('MUL_MAT:iq1_s')}, {f('MUL_MAT_ID:iq1_s')} | "
                    f"{g('mmid:iq1_m', ['1'] + BATCHED)} | {g('mmid:iq1_s', ['1'] + BATCHED)} | {r_str} | "
                    f"{g('dense:iq1_m', ['16', '128', '512'])} | {g('dense:iq1_s', ['16', '128', '512'])} |")
    rows.append("")
    rows.append(f"Defect: base iq1_m/iq1_s batched MUL_MAT_ID time ratio (geomean over n={BATCHED}) >= {DEFECT_RATIO}. "
                f"Fixed: all IQ1_M eval cases pass, head ratio <= {FIXED_RATIO}, and head iq1_m >= {MIN_SPEEDUP}x "
                f"faster than base iq1_m (geomean). MUL_MAT_ID shape: 64 experts, top-8, m 512, k 1024.")
    return "\n".join(rows)


def base_shows_defect(base: dict):
    r = _ratio(base)
    return (r is not None and r >= DEFECT_RATIO), f"base iq1_m/iq1_s = {r}"


def head_is_fixed(head: dict):
    evals = _eval_ok(head, "MUL_MAT:iq1_m") and _eval_ok(head, "MUL_MAT_ID:iq1_m")
    r = _ratio(head)
    hm = ((head.get("perf") or {}).get("mmid:iq1_m") or {}).get("us_per_run") or {}
    bm = _CTX["base_iq1m"]
    sp = _geo([bm[n] / hm[n] for n in BATCHED if n in hm and n in bm and hm[n] > 0])
    ok = evals and r is not None and r <= FIXED_RATIO and sp is not None and sp >= MIN_SPEEDUP
    return ok, f"evals={evals} ratio={r} speedup_vs_base={sp}"
