#!/usr/bin/env python3
"""Criteria: fp16 fused linear losses on gfx1151, zoo main default vs HIP half-result GEMMs (probes/fl_fp16_diag_probe.py).
"base" = fp16 default, "head" = the halfout diagnostic patch; all arms ran in the head invocation.

Gates: every arm exited 0 with a result on gfx1151; the patch arms report patch = halfout.
head_is_worse (REGRESSION = the half-result idea does NOT hold) if any of: a halfout accuracy case errors or is
non-finite at scale 1024 or 65536 while the default is finite; a halfout error vs fp64 exceeds the default's by more
than 25% + 1e-3; halfout median ms is not at least 20% below the default's for an op."""

from __future__ import annotations

import statistics

TITLE = "fp16 fused linear losses on gfx1151: default (fp32-upcast GEMMs) vs half-result GEMMs"
MODE = "regression"
NEEDS = ["rocm", "gpu", "discrete_gpu", "windows", "nvidia", "multi_gpu", "mig", "xpu", "mlx"]
OPS = ("ce", "logprobs", "jsd")


def _arms(obs):
    return (obs.get("head") or {}).get("arms") or {}


def _res(obs, n):
    return (_arms(obs).get(n) or {}).get("result") or {}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    arms = _arms(obs)
    out = [("all 10 arms present", len(arms) == 10, ", ".join(arms))]
    for name, a in arms.items():
        r = a.get("result") or {}
        out.append((f"{name}: exited 0 with a result", a.get("rc") == 0 and bool(r),
                    f"rc={a.get('rc')} {'' if a.get('rc') == 0 and r else str(a.get('tail'))[-400:]}"))
        if r.get("gpu"):
            out.append((f"{name}: gfx1151", any(s in r["gpu"] for s in ("8060S", "gfx1151", "Radeon")), r["gpu"]))
        if "halfout" in name:
            out.append((f"{name}: patch applied", r.get("patch") == "halfout", str(r.get("patch"))))
    return out


def _ms(obs, name, op):
    return (_res(obs, name).get(op) or {}).get("ms")


def table(obs: dict) -> str:
    lines = ["| op | fp16 default ms | fp16 halfout ms | bf16 default ms | fp16 plain torch ms | peak MiB default / halfout / bf16 |",
             "|---|---|---|---|---|---|"]
    for op in OPS:
        pk = lambda n: (_res(obs, n).get(op) or {}).get("peak_mib")  # noqa: E731
        lines.append(f"| {op} | {[_ms(obs, f'perf_fp16_default_{i}', op) for i in (1, 2)]} | "
                     f"{[_ms(obs, f'perf_fp16_halfout_{i}', op) for i in (1, 2)]} | {_ms(obs, 'perf_bf16_default_1', op)} | "
                     f"{_ms(obs, 'perf_fp16_ref', op)} | {pk('perf_fp16_default_1')} / {pk('perf_fp16_halfout_1')} / {pk('perf_bf16_default_1')} |")
    lines += ["", "| scale | op | head | default loss / grad_h / grad_w err | halfout loss / grad_h / grad_w err |", "|---|---|---|---|---|"]
    f = lambda x: "-" if x is None else f"{x:.2e}"  # noqa: E731
    for sc in ("1024", "65536"):
        d, hcases = _res(obs, f"acc_default_s{sc}").get("cases") or [], _res(obs, f"acc_halfout_s{sc}").get("cases") or []
        for a, b in zip(d, hcases):
            fa, fb = a.get("fused") or {}, b.get("fused") or {}
            lines.append(f"| {sc} | {a['op']} | {'trainable' if a['trainable_head'] else 'frozen'} | "
                         f"{fa.get('error') or ' / '.join(f(fa.get(k)) for k in ('loss_abs', 'grad_h_rel', 'grad_w_rel'))} fin={fa.get('finite')} | "
                         f"{fb.get('error') or ' / '.join(f(fb.get(k)) for k in ('loss_abs', 'grad_h_rel', 'grad_w_rel'))} fin={fb.get('finite')} |")
    return "\n".join(lines)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    obs = {"head": head, "base": base}
    why = []
    for sc in ("1024", "65536"):
        d, hcases = _res(obs, f"acc_default_s{sc}").get("cases") or [], _res(obs, f"acc_halfout_s{sc}").get("cases") or []
        for a, b in zip(d, hcases):
            fa, fb = a.get("fused") or {}, b.get("fused") or {}
            tag = f"s{sc} {a['op']} {'trainable' if a['trainable_head'] else 'frozen'}"
            if fb.get("error") or (fa.get("finite") and not fb.get("finite")):
                why.append(f"{tag}: halfout {fb.get('error') or 'non-finite'}")
                continue
            for k in ("loss_abs", "grad_h_rel", "grad_w_rel"):
                if fa.get(k) is not None and fb.get(k) is not None and fb[k] > fa[k] * 1.25 + 1e-3:
                    why.append(f"{tag} {k}: halfout {fb[k]:.2e} vs default {fa[k]:.2e}")
    for op in OPS:
        dm = [_ms(obs, f"perf_fp16_default_{i}", op) for i in (1, 2)]
        hm = [_ms(obs, f"perf_fp16_halfout_{i}", op) for i in (1, 2)]
        if None in dm or None in hm:
            why.append(f"{op}: missing timings default {dm} halfout {hm}")
        elif statistics.median(hm) > 0.8 * statistics.median(dm):
            why.append(f"{op}: halfout {hm} ms not 20% below default {dm}")
    return bool(why), "; ".join(why) or "half-result GEMMs: finite, no accuracy loss, >= 20% faster on every op"
