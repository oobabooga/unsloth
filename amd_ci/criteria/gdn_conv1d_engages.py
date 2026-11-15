#!/usr/bin/env python3
"""Criteria for probes/gdn_conv1d_probe.py: the base leaves the Qwen3.5 GatedDeltaNet conv on the torch
(MIOpen) path on ROCm; the head routes it through zoo's Triton kernel with torch-path parity, no fallback
and no failing GDN conv test. Gated on a ROCm torch, gfx1151 and each state importing its own zoo."""

from __future__ import annotations

TITLE = "GDN causal conv1d on ROCm gfx1151: Triton engagement, parity and step time, base vs head"
MODE = "differential"
NEEDS = ["gpu", "rocm", "windows"]
LOSS_RTOL = 1e-4
GRAD_RTOL = 1e-4
OP_RTOL = {"torch.float32": 1e-5, "torch.bfloat16": 2e-2}


def _states(obs):
    return {k: v for k, v in obs.items() if not k.startswith("_")}


def gates(obs):
    out = []
    for name, o in _states(obs).items():
        out.append((f"{name}: ROCm torch with a visible GPU", bool(o.get("hip")) and bool(o.get("cuda_available")),
                    f"torch {o.get('torch')}, hip {o.get('hip')}, {o.get('device')}"))
        out.append((f"{name}: gfx1151", str(o.get("arch") or "").startswith("gfx1151"), str(o.get("arch"))))
        out.append((f"{name}: imports its own unsloth_zoo", bool(o.get("zoo_from_checkout")), str(o.get("zoo_file"))))
        out.append((f"{name}: no causal_conv1d package (zoo's kernel is the candidate)",
                    o.get("real_causal_conv1d") is False, str(o.get("real_causal_conv1d"))))
        errs = {k: v.strip().splitlines()[-1] for k, v in (o.get("errors") or {}).items() if k != "pytest"}
        out.append((f"{name}: setup, parity and benches ran", not errs, "; ".join(f"{k}: {v}" for k, v in errs.items()) or "ran"))
        p = o.get("parity") or {}
        out.append((f"{name}: kill-switch arm is the torch path", (p.get("stats_kill_switch") or {}).get("triton_fwd", 1) == 0,
                    str(p.get("stats_kill_switch"))))
    return out


def table(obs):
    s = _states(obs)
    rows = ["| measurement | " + " | ".join(s) + " |", "|---|" + "---|" * len(s)]

    def row(label, f):
        cells = []
        for o in s.values():
            try:
                cells.append(f(o))
            except Exception:
                cells.append("-")
        rows.append(f"| {label} | " + " | ".join(cells) + " |")

    row("versions", lambda o: f"torch {o['torch']}, triton {o['triton']}, transformers {o['transformers']}")
    row("modeling causal_conv1d_fn patched", lambda o: str(o["modeling_conv_patched"]))
    row("fp32 tiny model counters", lambda o: str(o["parity"]["stats_default"]))
    row("fp32 loss default / kill switch", lambda o: f"{o['parity']['loss_default']:.6f} / {o['parity']['loss_kill_switch']:.6f}")
    row("fp32 max grad rel diff vs torch path", lambda o: f"{o['parity']['max_grad_rel_diff']:.2e}")
    row("bf16 4B-width model, B=1 T=2048, counters", lambda o: str(o["model_bench"]["stats"]))
    row("bf16 4B-width model, median s per fwd+bwd step", lambda o: f"{o['model_bench']['median_s']:.3f} "
        f"(min {o['model_bench']['step_s'][0]:.3f}, max {o['model_bench']['step_s'][-1]:.3f})")
    row("conv op D=8192 T=2048 bf16 fwd+bwd, channel-first, Triton ms", lambda o: f"{o['op']['timing_bf16_fwd_bwd']['triton']['median_ms']:.2f}")
    row("conv op D=8192 T=2048 bf16 fwd+bwd, channel-first, torch ms", lambda o: f"{o['op']['timing_bf16_fwd_bwd']['torch']['median_ms']:.2f}")
    row("conv op, model layout (channel-last), Triton ms", lambda o: f"{o['op']['timing_bf16_fwd_bwd']['triton_channel_last']['median_ms']:.2f}")
    row("conv op, model layout (channel-last), torch ms", lambda o: f"{o['op']['timing_bf16_fwd_bwd']['torch_channel_last']['median_ms']:.2f}")
    for dt in OP_RTOL:
        row(f"conv op {dt} rel err vs fp64 (Triton / torch)", lambda o, dt = dt: ", ".join(
            f"{k} {o['op'][dt]['triton_rel'][k]:.1e} / {o['op'][dt]['torch_rel'][k]:.1e}" for k in ("y", "dx", "dw", "db")))
    row("GDN conv tests", lambda o: f"rc {o['pytest']['rc']}, {o['pytest']['passed']} passed, "
        f"{len(o['pytest']['failed'])} failed, {o['pytest']['skipped']} skipped")
    row("failing tests", lambda o: ", ".join(f"`{t.split('::')[-1]}`" for t in o["pytest"]["failed"]) or "none")
    return "\n".join(rows)


def base_shows_defect(base):
    st = (base.get("model_bench") or {}).get("stats") or {}
    return (not base.get("modeling_conv_patched")) and st.get("triton_fwd", 1) == 0


def head_is_fixed(head):
    p = head.get("parity") or {}
    st = p.get("stats_default") or {}
    mb = (head.get("model_bench") or {}).get("stats") or {}
    op = head.get("op") or {}
    rel = p.get("loss_kill_switch") or 0
    ok_loss = abs(p.get("loss_default", 0) - rel) <= LOSS_RTOL * max(abs(rel), 1e-12)
    ok_op = all(max(op[dt]["triton_rel"].values()) <= max(tol, 2 * max(op[dt]["torch_rel"].values()))
                for dt, tol in OP_RTOL.items()) if op else False
    pt = head.get("pytest") or {}
    return bool(
        head.get("modeling_conv_patched")
        and st.get("triton_fwd", 0) > 0 and st.get("triton_bwd", 0) > 0 and st.get("fallback", 1) == 0
        and mb.get("triton_fwd", 0) > 0 and mb.get("fallback", 1) == 0
        and ok_loss and p.get("max_grad_rel_diff", 1) <= GRAD_RTOL and ok_op
        and pt.get("rc") == 0 and not pt.get("failed") and pt.get("passed", 0) > 0
    )
