"""Shared judging helpers for the ROCm SDK-wheel detection criteria (s10_sdk_win.py, s10_sdk_lnx.py)."""

from __future__ import annotations

# What every changed gate must answer on a ROCm build (the head on any ROCm wheel; base too when hip is set).
ROCM_ANSWERS = {
    "fla_fp32_gfx11_guard": True, "fla_kda_withheld": True, "fla_fp32_tensor_routed": True, "fla_hopper_dev0": False,
    "zoo_flex_static": True, "chunked_logprob_fp32_out_gemm": False, "fused_linear_hip_tiles": True,
    "fused_linear_bwd_rows": 1024, "fused_linear_is_hip": True, "preference_loss_tried_fp32_out_gemm": False,
    "moe_fp32_out_gemm": False, "moe_torch_is_cuda_build": False, "core_prefix_flex_static": True,
    "core_inkling_auto_takes_flex": False,
}
# The defect on an SDK wheel at the base: the gates that test only torch.version.hip read "not ROCm".
SDK_BASE_DEFECT = {
    "fla_fp32_gfx11_guard": False, "zoo_flex_static": False, "chunked_logprob_fp32_out_gemm": True,
    "fused_linear_hip_tiles": False, "moe_fp32_out_gemm": True, "core_prefix_flex_static": False,
}
OP_TOL = 1e-4


def states(obs):
    return [k for k in obs if not k.startswith("_")]


def g(o):
    return ((o.get("gates") or {}).get("gates")) or {}


def facts(o):
    return ((o.get("gates") or {}).get("facts")) or {}


def gate_mismatch(o, want):
    have = g(o)
    return {k: have.get(k) for k, v in want.items() if have.get(k) != v}


def common_gates(obs):
    out = []
    for st in states(obs):
        o = obs[st]
        cell = o.get("gates") or {}
        f = facts(o)
        ck = o.get("checkout") or ""
        zf = str(cell.get("zoo_file") or "")
        out.append((f"{st} gates cell finished", cell.get("stage") == "done" and len(g(o)) == len(ROCM_ANSWERS),
                    f"stage {cell.get('stage')} steps {cell.get('steps')} rc {cell.get('rc')}"))
        out.append((f"{st} imported the state's unsloth_zoo", bool(zf) and zf.startswith(ck), zf[:200]))
        cf = str(cell.get("core_prefix_file") or "")
        out.append((f"{st} loaded the state's core files", f"s10_core" in cf and cf.replace("\\", "/").split("s10_core/")[-1]
                    .startswith(st + "/"), cf[-120:]))
        out.append((f"{st} ran on a ROCm torch on gfx1151", "rocm" in str(f.get("torch", "")).lower()
                    and "gfx1151" in str(f.get("arch")), f"torch {f.get('torch')} hip {f.get('hip')} arch {f.get('arch')}"))
    return out


def pytest_regression(base, head, keys = ("pytest_zoo", "pytest_core")):
    """Failed-at-head ids that did not fail at base, plus any head failure in the files new at head."""
    new = []
    for k in keys:
        b, h = base.get(k) or {}, head.get(k) or {}
        bf = set(b.get("failed") or []) | set(b.get("errors") or [])
        hf = set(h.get("failed") or []) | set(h.get("errors") or [])
        new.extend(sorted(hf - bf))
    return new


def pytest_table(obs, keys = ("pytest_zoo", "pytest_core")):
    rows = ["| state | suite | file | rc | passed | failed | skipped |", "|---|---|---|---|---|---|---|"]
    for st in states(obs):
        for k in keys:
            for f, r in (((obs[st].get(k) or {}).get("per_file")) or {}).items():
                rows.append(f"| {st} | {k} | {f} | {r.get('rc')} | {r.get('n_passed')} | {r.get('n_failed')} | {r.get('n_skipped')} |")
    for st in states(obs):
        for k in keys:
            fl = sorted(set((obs[st].get(k) or {}).get("failed") or []) | set((obs[st].get(k) or {}).get("errors") or []))
            if fl:
                rows.append(f"- {st} {k} failing: " + ", ".join(fl)[:1500])
    return "\n".join(rows)


def gates_table(obs):
    keys = list(ROCM_ANSWERS)
    rows = ["| gate | ROCm answer | " + " | ".join(states(obs)) + " |", "|---|---|" + "---|" * len(states(obs))]
    for k in keys:
        rows.append(f"| {k} | {ROCM_ANSWERS[k]} | " + " | ".join(str(g(obs[s]).get(k)) for s in states(obs)) + " |")
    for s in states(obs):
        f = facts(obs[s])
        rows.append(f"\n- {s}: torch {f.get('torch')}, torch.version.hip {f.get('hip')!r}, arch {f.get('arch')}, "
                    f"triton {f.get('triton')}, op_fp32 {_op_summary(obs[s].get('op_fp32') or {})}")
    return "\n".join(rows)


def op_ok(c):
    err = c.get("err_vs_fp64") or {}
    return (c.get("stage") == "done" and c.get("rc") == 0 and len(err) == 6
            and all(v.get("finite") and v.get("max_abs", 1.0) <= OP_TOL for v in err.values())
            and (c.get("torch_path_calls") or 0) > 0)


def op_dead(c):
    return bool(c) and not c.get("not_run") and (c.get("rc") not in (0,) or c.get("stage") != "done")


def _op_summary(c):
    if c.get("not_run"):
        return "not run (" + c["not_run"] + ")"
    err = c.get("err_vs_fp64") or {}
    return (f"rc {c.get('rc')} stage {c.get('stage')} torch_path_calls {c.get('torch_path_calls')} "
            f"max err {max((v.get('max_abs', 0) for v in err.values()), default = None)}")
