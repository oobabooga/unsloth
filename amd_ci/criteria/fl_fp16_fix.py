#!/usr/bin/env python3
"""Criteria: fp16 fused linear losses on gfx1151, zoo main (base) vs main + bounded half-result HIP fp16 GEMMs (head).
Pairs with probes/fl_fp16_fix_probe.py (all arms in the head invocation).

Gates: head tree carries the fix and base does not; every arm exited 0 with a result on gfx1151 (pytest: rc recorded,
judged below); fused ops engaged in the accuracy arms.
head_is_worse (REGRESSION) if any of: a head accuracy case errors or is non-finite where base is finite (scale 1024 or
65536); a head error vs fp64 above base's by more than 25% + 1e-3; head median fp16 ms above base median by more than
max(5%, base A/A spread) for an op; head peak memory above base's by more than 5%; the head pytest run failed."""

from __future__ import annotations

import statistics

TITLE = "fp16 fused linear losses on gfx1151: zoo main vs bounded half-result HIP fp16 GEMMs"
MODE = "regression"
NEEDS = ["rocm", "gpu", "discrete_gpu", "windows", "nvidia", "multi_gpu", "mig", "xpu", "mlx"]
OPS = ("ce", "logprobs", "jsd")


def _h(obs):
    return obs.get("head") or {}


def _arms(obs):
    return _h(obs).get("arms") or {}


def _res(obs, n):
    return (_arms(obs).get(n) or {}).get("result") or {}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    h, arms = _h(obs), _arms(obs)
    out = [("head has the fix, base does not", h.get("head_has_fix") is True and h.get("base_has_fix") is False,
            f"head {h.get('head_has_fix')} base {h.get('base_has_fix')} base sha {h.get('base_sha')} head status {h.get('head_status')}"),
           ("all 11 arms present", len(arms) == 11, ", ".join(arms))]
    for name, a in arms.items():
        if name.startswith("pytest"):
            out.append((f"{name}: ran", a.get("rc") in (0, 1) and bool(a.get("summary")), f"rc={a.get('rc')} {a.get('summary')}"))
            continue
        r = a.get("result") or {}
        out.append((f"{name}: exited 0 with a result", a.get("rc") == 0 and bool(r),
                    f"rc={a.get('rc')} {'' if a.get('rc') == 0 and r else str(a.get('tail'))[-400:]}"))
        if r.get("gpu"):
            out.append((f"{name}: gfx1151", any(s in r["gpu"] for s in ("8060S", "gfx1151", "Radeon")), r["gpu"]))
    for sc in ("1024", "65536"):
        cs = _res(obs, f"acc_head_s{sc}").get("cases") or []
        out.append((f"acc head s{sc}: fused ops engaged", all((c.get("fused") or {}).get("calls") for c in cs) and bool(cs), str(len(cs))))
    return out


def _ms(obs, n, op):
    return (_res(obs, n).get(op) or {}).get("ms")


def _pk(obs, n, op):
    return (_res(obs, n).get(op) or {}).get("peak_mib")


def table(obs: dict) -> str:
    lines = [f"base `{_h(obs).get('base_sha')}` (zoo main), head = base + fix", "",
             "| op | fp16 base ms | fp16 head ms | head / base | bf16 base ms | fp16 plain torch ms | peak MiB base / head / plain |",
             "|---|---|---|---|---|---|---|"]
    for op in OPS:
        b = [_ms(obs, f"perf_fp16_base_{i}", op) for i in (1, 2)]
        hh = [_ms(obs, f"perf_fp16_head_{i}", op) for i in (1, 2)]
        ratio = f"{statistics.median(hh) / statistics.median(b):.3f}" if None not in b + hh else "-"
        lines.append(f"| {op} | {b} | {hh} | {ratio} | {_ms(obs, 'perf_bf16_base', op)} | {_ms(obs, 'perf_fp16_ref', op)} | "
                     f"{_pk(obs, 'perf_fp16_base_1', op)} / {_pk(obs, 'perf_fp16_head_1', op)} / {_pk(obs, 'perf_fp16_ref', op)} |")
    f = lambda x: "-" if x is None else f"{x:.2e}"  # noqa: E731
    lines += ["", "| scale | V | op | head | base loss / grad_h / grad_w err | head loss / grad_h / grad_w err | plain torch |",
              "|---|---|---|---|---|---|---|"]
    for sc in ("1024", "65536"):
        for a, b in zip(_res(obs, f"acc_base_s{sc}").get("cases") or [], _res(obs, f"acc_head_s{sc}").get("cases") or []):
            fa, fb, rf = a.get("fused") or {}, b.get("fused") or {}, a.get("ref") or {}
            g = lambda x: x.get("error") or (" / ".join(f(x.get(k)) for k in ("loss_abs", "grad_h_rel", "grad_w_rel")) + f" fin={x.get('finite')}")  # noqa: E731
            lines.append(f"| {sc} | {a['v']} | {a['op']} | {'trainable' if a['trainable_head'] else 'frozen'} | {g(fa)} | {g(fb)} | {g(rf)} |")
    p = _arms(obs).get("pytest_head") or {}
    lines += ["", f"pytest (head tree, gfx1151) rc={p.get('rc')}:", "", "```", *(p.get("summary") or []), "```"]
    return "\n".join(lines)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    obs = {"head": head, "base": base}
    why = []
    for sc in ("1024", "65536"):
        for a, b in zip(_res(obs, f"acc_base_s{sc}").get("cases") or [], _res(obs, f"acc_head_s{sc}").get("cases") or []):
            fa, fb = a.get("fused") or {}, b.get("fused") or {}
            tag = f"s{sc} V={a['v']} {a['op']} {'trainable' if a['trainable_head'] else 'frozen'}"
            if fb.get("error") or (fa.get("finite") and not fb.get("finite")):
                why.append(f"{tag}: head {fb.get('error') or 'non-finite'}")
                continue
            for k in ("loss_abs", "grad_h_rel", "grad_w_rel"):
                if fa.get(k) is not None and fb.get(k) is not None and fb[k] > fa[k] * 1.25 + 1e-3:
                    why.append(f"{tag} {k}: head {fb[k]:.2e} vs base {fa[k]:.2e}")
    for op in OPS:
        b = [_ms(obs, f"perf_fp16_base_{i}", op) for i in (1, 2)]
        hh = [_ms(obs, f"perf_fp16_head_{i}", op) for i in (1, 2)]
        if None in b or None in hh:
            why.append(f"perf {op}: missing base {b} head {hh}")
            continue
        tol = max(0.05, abs(b[0] - b[1]) / min(b))
        if statistics.median(hh) > statistics.median(b) * (1 + tol):
            why.append(f"perf {op}: head {hh} ms vs base {b}")
        pb, ph = _pk(obs, "perf_fp16_base_1", op), _pk(obs, "perf_fp16_head_1", op)
        if pb is not None and ph is not None and ph > pb * 1.05:
            why.append(f"perf {op}: head peak {ph} MiB vs base {pb}")
    p = _arms(obs).get("pytest_head") or {}
    if p.get("rc") != 0:
        why.append(f"pytest on the head tree: rc={p.get('rc')} {(p.get('summary') or [])[-6:]}")
    return bool(why), "; ".join(why) or "head no worse than base on accuracy (scale 1024 and 65536), memory and pytest, and not slower"
