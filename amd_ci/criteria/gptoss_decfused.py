#!/usr/bin/env python3
"""Criteria: GPT-OSS fused decode kernels forced on for ROCm (gfx1151). Pairs with probes/gptoss_decfused_probe.py.

Non-vacuity gates (INCONCLUSIVE unless all hold): AMD gfx1151 GPU; model downloaded; base default env keeps the fused
path off on ROCm and head with UNSLOTH_GPT_OSS_FUSED_DECODE=1 turns it on; every bench arm exited 0 with timings;
forced arms engaged both kernels on every token, kill-switch and base arms engaged neither; corr ran.
head_is_worse (REGRESSION = the kernels are NOT fit to enable on ROCm) if any of:
  - the unit suites (forced on) fail
  - corr: step-0 logits fused vs eager not bit-identical, or on vs on / off vs off not bit-identical (nondeterminism)
  - corr: teacher-forced argmax agreement fused vs eager below 0.9
  - decode ms/token: median of F arms above the median of E arms by more than max(2%, E1-E2 spread)
"""

from __future__ import annotations

import statistics

TITLE = "GPT-OSS fused decode kernels on gfx1151 (forced on vs kill switch)"
MODE = "regression"
NEEDS = ["rocm", "gpu", "discrete_gpu", "windows", "nvidia", "multi_gpu", "mig", "xpu", "mlx"]


def _head(obs):
    return obs.get("head") or {}


def _row(arm, b):
    return ((arm or {}).get("result") or {}).get(b) or {}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    h = _head(obs)
    arms = h.get("arms") or {}
    be = h.get("backend") or {}
    out = [("model downloaded", (h.get("download") or {}).get("rc") == 0, str(h.get("download"))[:300]),
           ("base, default env: fused path off on ROCm", (be.get("base:None") or {}).get("enabled") is False,
            str(be.get("base:None"))),
           ("head, UNSLOTH_GPT_OSS_FUSED_DECODE=1: fused path on", (be.get("head:1") or {}).get("enabled") is True,
            str(be.get("head:1"))),
           ("all 5 bench arms present", len(arms) == 5, ", ".join(arms))]
    for name, a in arms.items():
        r = a.get("result") or {}
        ok = a.get("rc") == 0 and all(_row(a, b).get("decode_ms_per_token") for b in ("b1", "b4"))
        out.append((f"{name}: exited 0 with timings", ok, f"rc={a.get('rc')} {'' if ok else str(a.get('tail'))[-400:]}"))
        gpu = f"{r.get('gpu')} {r.get('arch')}"
        out.append((f"{name}: AMD gfx1151", "gfx1151" in gpu or "8060S" in gpu, gpu))
        calls = _row(a, "b1").get("fused_calls_per_token") or {}
        if a.get("flag") == "1":
            out.append((f"{name}: both kernels engaged per token", calls.get("add_rms_norm", 0) > 10 and
                        calls.get("sink_scale_output", 0) > 1, str(calls)))
        else:
            out.append((f"{name}: no fused calls", calls.get("add_rms_norm", 0) == 0 and
                        calls.get("sink_scale_output", 0) == 0, str(calls)))
    c = (h.get("corr") or {}).get("result") or {}
    out.append(("corr ran", bool(c.get("b1")) and (h.get("corr") or {}).get("rc") == 0,
                str((h.get("corr") or {}).get("tail"))[-300:]))
    return out


def _med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def _perf(h):
    arms = h.get("arms") or {}
    out = {}
    for b in ("b1", "b4"):
        out[b] = {n: _row(arms.get(n), b).get("decode_ms_per_token") for n in ("E1", "F1", "F2", "E2", "B1")}
    return out


def table(obs: dict) -> str:
    h = _head(obs)
    be = h.get("backend") or {}
    lines = [f"Model `{h.get('model')}`; base `{h.get('base_sha')}`; head = base + force-on override", "",
             f"Backend (head, forced): {be.get('head:1')}", "",
             f"Unit suites (forced on): rc={(h.get('pytest') or {}).get('rc')} "
             f"{(h.get('pytest') or {}).get('summary')} failed={(h.get('pytest') or {}).get('failed')}", "",
             "| batch | E1 | F1 | F2 | E2 | base (default) | F / E (median) | E spread |", "|---|---|---|---|---|---|---|---|"]
    for b, d in _perf(h).items():
        e, f = _med([d["E1"], d["E2"]]), _med([d["F1"], d["F2"]])
        sp = abs(d["E1"] - d["E2"]) / min(d["E1"], d["E2"]) if d["E1"] and d["E2"] else None
        lines.append(f"| {b} | {d['E1']} | {d['F1']} | {d['F2']} | {d['E2']} | {d['B1']} | "
                     f"{f / e:.3f} | {sp:.2%} |" if e and f and sp is not None else f"| {b} | {d} | - | - |")
    arms = h.get("arms") or {}
    for n in ("E1", "F1"):
        r = _row(arms.get(n), "b1")
        lines.append(f"- {n} b1: kernels/token {r.get('kernels_per_token')}, launches/token {r.get('launches_per_token')}, "
                     f"fused calls/token {r.get('fused_calls_per_token')}")
    c = (h.get("corr") or {}).get("result") or {}
    for b in ("b1", "b4"):
        r = c.get(b) or {}
        lines.append(f"- corr {b}: calls {r.get('calls')}; " + "; ".join(
            f"{k}: step0 {v.get('forced_max_abs_logit_diff_step0')} bit={v.get('forced_bit_identical')} "
            f"argmax {v.get('forced_argmax_agree_frac')} greedy_first_diff {v.get('greedy_first_diff')}"
            for k, v in r.items() if isinstance(v, dict)))
    return "\n".join(lines)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    why = []
    pt = head.get("pytest") or {}
    if pt.get("rc") != 0 or pt.get("failed"):
        why.append(f"unit suites: rc={pt.get('rc')} {pt.get('failed')}")
    c = (head.get("corr") or {}).get("result") or {}
    for b in ("b1", "b4"):
        for k, v in (c.get(b) or {}).items():
            if not isinstance(v, dict):
                continue
            mixed = ("E" in k) and ("F" in k)
            if v.get("forced_max_abs_logit_diff_step0") != 0.0:
                why.append(f"corr {b} {k}: step-0 logits differ by {v.get('forced_max_abs_logit_diff_step0')}")
            if not mixed and not v.get("forced_bit_identical"):
                why.append(f"corr {b} {k}: same path not deterministic")
            if mixed and (v.get("forced_argmax_agree_frac") or 0) < 0.9:
                why.append(f"corr {b} {k}: argmax agreement {v.get('forced_argmax_agree_frac')}")
    for b, d in _perf(head).items():
        if None in (d["E1"], d["E2"], d["F1"], d["F2"]):
            why.append(f"{b}: missing timings {d}")
            continue
        spread = abs(d["E1"] - d["E2"]) / min(d["E1"], d["E2"])
        tol = max(0.02, spread)
        e, f = _med([d["E1"], d["E2"]]), _med([d["F1"], d["F2"]])
        if f > e * (1 + tol):
            why.append(f"{b}: fused {f:.2f} ms/token vs eager {e:.2f} (> {tol:.1%})")
    return bool(why), "; ".join(why) or "forced-on kernels correct and not slower than the kill switch"
