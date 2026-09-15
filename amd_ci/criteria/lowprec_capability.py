#!/usr/bin/env python3
"""Criteria: judge the low-precision capability inventory on gfx1151.

**This run is not a differential and does not pretend to be one.** There is no
defect to reproduce: the question is "what low-precision paths exist on this
hardware, and what does Studio pick when it sees them". The two states straddle
one commit at the tip of origin/main that touches AMD GPU-NAME recognition and
nothing in the quant selector, so base and head are expected to answer alike and
a difference between them would be a surprise, not the result. MODE is therefore
"regression" and the verdict line says so in words.

A criteria module that can only ever say NO_REGRESSION is a green-tick generator,
which is the thing this toolkit exists to prevent. Two things stop that here:

  * the gates are non-vacuity gates with real force. If torch did not import, or
    the part was not gfx1151, or Studio's quant module did not load, or the
    selector produced no answer, the verdict is INCONCLUSIVE and the job goes
    red. An inventory that inventoried nothing must not read as a pass.
  * head_is_worse has a real predicate. The brief's most valuable question is
    whether an explicit `nvfp4` request on a Strix Halo box yields a clean
    decline or a traceback. A selector call that RAISES where the other state
    returned is a genuine regression on a user-facing path, and it is reported
    as one.
"""

from __future__ import annotations

TITLE = "Low-precision paths on gfx1151, and Studio's diffusion quant selection"
MODE = "regression"

# Authored, not computed: every capability the QUESTION touches, so the report
# bounds itself. amd_fp4_matrix_cores is the load-bearing one. gfx1151 is
# RDNA3.5, whose WMMA carries FP16/BF16/INT8/INT4 and no float4 path at all;
# AMD's FP4 matrix cores are CDNA4/gfx950 (MI355X class) and this pool cannot
# reach one. So whatever this run observes about FP4, it can say nothing
# whatever about MXFP4 PERFORMANCE on AMD.
NEEDS = [
    "rocm", "gpu",
    "amd_fp4_matrix_cores", "amd_fp8_matrix_cores",
    "nvidia", "mig", "gpu_partitions",
    "windows", "windows_rocm_wddm", "windows_docker",
    "multi_gpu", "multi_gpu_amd", "discrete_gpu",
    "xpu", "mlx",
]

# Studio calls that must produce an ANSWER rather than an exception. These are
# reachable from the UI: a user picking a precision lands on exactly these.
_USER_FACING = (
    "auto_scheme[no-family]",
    "explicit[nvfp4]",
    "explicit[mxfp8]",
    "explicit[fp8]",
    "explicit[int8]",
    "explain[nvfp4]",
)


def _states(obs: dict) -> dict:
    return {k: v for k, v in obs.items() if not k.startswith("_")}


def _sub(state: dict, section: str) -> dict:
    """The recorded value of one probe section, or {} if the section itself blew up."""
    rec = state.get(section) or {}
    if isinstance(rec, dict) and rec.get("ok") and isinstance(rec.get("value"), dict):
        return rec["value"]
    return {}


def _val(rec):
    """Unwrap an attempt() record to its value, or None when it failed."""
    if isinstance(rec, dict) and rec.get("ok"):
        return rec.get("value")
    return None


def _raised(rec) -> bool:
    return isinstance(rec, dict) and rec.get("ok") is False


def _arch(state: dict) -> str:
    ident = _sub(state, "identity")
    props = _val(ident.get("device_properties")) or {}
    return str(props.get("gcnArchName") or "")


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    states = _states(obs)
    if not states:
        return [("any state was probed at all", False, "no states in the observation set")]

    # torch, and a real device
    ok, ev = True, []
    for name, st in states.items():
        ident = _sub(st, "identity")
        hip = ident.get("torch_version_hip")
        ev.append(f"{name}: hip={hip}")
        if not hip:
            ok = False
    out.append(("ROCm torch imported at every state", ok, "; ".join(ev)))

    # the part really is the one we were asked about
    ok, ev = True, []
    for name, st in states.items():
        a = _arch(st)
        ev.append(f"{name}: {a or 'unknown'}")
        if "gfx1151" not in a:
            ok = False
    out.append(("every state saw gfx1151", ok, "; ".join(ev)))

    # KFD topology readable: the arch claim's independent check
    ok, ev = True, []
    for name, st in states.items():
        v = _sub(st, "identity").get("kfd_gfx_target_version") or []
        ev.append(f"{name}: {v}")
        if not v:
            ok = False
    out.append(("KFD gfx_target_version was readable", ok, "; ".join(ev)))

    # Studio's selector module actually loaded from each worktree
    ok, ev = True, []
    for name, st in states.items():
        imported = (_sub(st, "studio").get("module_import") or {}).get("ok")
        ev.append(f"{name}: {'yes' if imported else 'NO'}")
        if not imported:
            ok = False
    out.append(("Studio quant module imported from each checkout", ok, "; ".join(ev)))

    # and produced an answer for auto. An exception here is a crash, but a
    # MISSING record means the probe never got that far, which is vacuity.
    ok, ev = True, []
    for name, st in states.items():
        rec = _sub(st, "studio").get("auto_scheme[no-family]")
        present = isinstance(rec, dict) and "ok" in rec
        ev.append(f"{name}: {'recorded' if present else 'MISSING'}")
        if not present:
            ok = False
    out.append(("the auto selector was actually exercised", ok, "; ".join(ev)))

    # the fp8 inventory ran; without it there is no capability table
    ok, ev = True, []
    for name, st in states.items():
        f8 = _sub(st, "float8")
        ev.append(f"{name}: {len(f8)} dtypes")
        if len(f8) < 4:
            ok = False
    out.append(("all four float8 dtypes were interrogated", ok, "; ".join(ev)))

    return out


def _fp8_row(state: dict, dtype: str) -> str:
    e = _sub(state, "float8").get(dtype) or {}
    if not e.get("exists"):
        return "absent from this torch"
    bits = []
    bits.append("constructs" if _val(e.get("construct_cuda")) else "construct FAILS")
    for label, key in (("per-tensor", "scaled_mm_per_tensor"), ("per-row", "scaled_mm_per_row")):
        rec = e.get(key) or {}
        if rec.get("ok"):
            bits.append(f"{label} OK")
        else:
            msg = ((rec.get("error") or {}).get("message") or "").strip().replace("|", "/")
            bits.append(f"{label} FAILS: {msg[:150]}")
    return "; ".join(bits)


def table(obs: dict) -> str:
    states = _states(obs)
    # Both states are expected to agree; report the head and note any divergence.
    ref_name = "head" if "head" in states else next(iter(states))
    ref = states[ref_name]
    rows: list[str] = []

    ident = _sub(ref, "identity")
    props = _val(ident.get("device_properties")) or {}
    rows += [
        f"### Device identity (state `{ref_name}`)", "",
        "| field | value |", "|---|---|",
        f"| torch | {ident.get('torch_version')} |",
        f"| torch.version.hip | {ident.get('torch_version_hip')} |",
        f"| torch.version.cuda | {ident.get('torch_version_cuda')} |",
        f"| gcnArchName | {props.get('gcnArchName')} |",
        f"| device name | {_val(ident.get('device_name'))} |",
        f"| get_device_capability | {_val(ident.get('device_capability'))} |",
        f"| KFD gfx_target_version | {ident.get('kfd_gfx_target_version')} |",
        f"| is_integrated | {props.get('is_integrated')} |",
        f"| ROCm version files | {ident.get('rocm_version_files')} |",
        f"| HSA/HIP env in play | {ident.get('hsa_env')} |",
        "",
        "### float8: does it exist, and does a matmul accept it", "",
        "| dtype | outcome |", "|---|---|",
    ]
    for dt in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz"):
        rows.append(f"| `torch.{dt}` | {_fp8_row(ref, dt)} |")

    fp4 = _sub(ref, "fp4")
    rows += ["", "### 4-bit", "", "| question | answer |", "|---|---|"]
    for name, present in (fp4.get("torch_fp4_dtypes") or {}).items():
        rows.append(f"| `torch.{name}` exists | {present} |")
    rows.append(f"| torchao imports | {'yes, ' + str(fp4.get('torchao_version')) if 'torchao_version' in fp4 else 'NO: ' + str((fp4.get('torchao_import') or {}).get('message'))} |")
    for label, key in (("NVFP4 config entry point", "entrypoint_NVFP4DynamicActivationNVFP4WeightConfig"),
                       ("MX config entry point", "entrypoint_MXDynamicActivationMXWeightConfig"),
                       ("MXFP4InferenceConfig entry point", "entrypoint_MXFP4InferenceConfig")):
        rec = fp4.get(key) or {}
        rows.append(f"| {label} | {'present' if rec.get('ok') else 'ABSENT: ' + str((rec.get('error') or {}).get('message'))[:120]} |")
    for label, key in (("apply NVFP4 to a Linear", "apply_nvfp4"),
                       ("apply MXFP4 to a Linear", "apply_mxfp4")):
        rec = fp4.get(key) or {}
        if rec.get("ok"):
            v = rec["value"] or {}
            if "config_error" in v:
                mode = f"config refused: {(v['config_error'] or {}).get('message', '')[:120]}"
            elif v.get("weight_type_changed"):
                mode = (f"APPLIED (weight {v.get('weight_type_before')} -> "
                        f"{v.get('weight_type_after')}), forward finite={v.get('forward_finite')}")
            else:
                mode = (f"SILENT NO-OP: quantize_ returned cleanly, weight stayed "
                        f"{v.get('weight_type_after')}, forward finite={v.get('forward_finite')}")
        else:
            mode = f"RAISED {(rec.get('error') or {}).get('type')}: {(rec.get('error') or {}).get('message', '')[:160]}"
        rows.append(f"| {label} | {mode} |")

    work = _sub(ref, "working")
    rows += ["", "### What does work", "", "| path | result |", "|---|---|"]
    for label, key in (("`torch._int_mm`", "torch_int_mm"),
                       ("`torchao safe_int_mm`", "torchao_safe_int_mm")):
        rec = work.get(key) or {}
        rows.append(f"| {label} | {'OK ' + str(_val(rec)) if rec.get('ok') else 'FAILS: ' + str((rec.get('error') or {}).get('message', ''))[:140]} |")
    for shape, per_dtype in (work.get("gemm") or {}).items():
        for dtype_name, rec in per_dtype.items():
            v = _val(rec)
            cell = (f"{v['tflops']:.2f} TFLOP/s ({v['seconds_per_iter'] * 1e3:.2f} ms)"
                    if v else f"FAILS: {str((rec.get('error') or {}).get('message', ''))[:120]}")
            rows.append(f"| {dtype_name} GEMM {shape} | {cell} |")
    rec = work.get("gemm_int8_4096") or {}
    v = _val(rec)
    if v:
        int8_cell = f"{v['tflops']:.2f} TOP/s ({v['seconds_per_iter'] * 1e3:.2f} ms)"
    else:
        int8_cell = "FAILS: " + str((rec.get("error") or {}).get("message", ""))[:120]
    rows.append(f"| int8 GEMM 4096x4096x4096 | {int8_cell} |")

    rows += ["", "### Studio's diffusion quant selection", "",
             "| call | base | head |", "|---|---|---|"]
    studio = {n: _sub(st, "studio") for n, st in states.items()}

    def cell(name: str, key: str) -> str:
        rec = (studio.get(name) or {}).get(key)
        if rec is None:
            return "not recorded"
        if rec.get("ok"):
            return f"`{rec.get('value')!r}`"[:200]
        e = rec.get("error") or {}
        return f"**RAISED** {e.get('type')}: {str(e.get('message'))[:120]}"

    keys = ["dense_transformer_supported", "capability_tuple", "is_consumer_gpu",
            "auto_ladder", "torchao_unavailable_reason",
            "auto_scheme[no-family]", "auto_candidates[no-family]",
            "auto_scheme[qwen-image]", "auto_scheme[z-image]",
            "explicit[nvfp4]", "explain[nvfp4]", "scheme_supported[nvfp4]",
            "explicit[mxfp8]", "explicit[fp8]", "explicit[int8]",
            "scheme_supported[fp8]", "scheme_supported[int8]"]
    for key in keys:
        rows.append(f"| `{key}` | {cell('base', key)} | {cell('head', key)} |")

    hw = _val((studio.get(ref_name) or {}).get("hardware")) or {}
    rows += ["", f"Host classification (state `{ref_name}`): "
                 f"`IS_ROCM={hw.get('IS_ROCM')}`, gpu_summary "
                 f"`{str(_val(hw.get('gpu_summary')))[:300]}`.",
             "",
             f"`dense_quant_host_capable` present in the tree: "
             f"{(studio.get(ref_name) or {}).get('dense_quant_host_capable_present')} "
             f"(the brief names it; the selector's real entry points are "
             f"`select_transformer_quant_scheme`, `dense_transformer_supported` and "
             f"`torchao_unavailable_reason`).",
             "",
             "**No differential happened here.** Base and head straddle a single "
             "tip-of-main commit that does not touch the quant selector, so the two "
             "columns above are expected to be identical and their agreement is a "
             "consistency check, not a finding. This run is a capability inventory."]
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    """A user-facing selector call that RAISES at the head where the base
    answered. That is the only thing this run treats as a regression, and it is
    precisely the unsafe case: a Strix Halo user must get a clean decline, not a
    traceback."""
    b, h = _sub(base, "studio"), _sub(head, "studio")
    broke = [k for k in _USER_FACING if _raised(h.get(k)) and not _raised(b.get(k))]
    if broke:
        detail = "; ".join(
            f"{k} raised {((h.get(k) or {}).get('error') or {}).get('type')}: "
            f"{str(((h.get(k) or {}).get('error') or {}).get('message'))[:160]}"
            for k in broke)
        return True, f"user-facing selector calls crash at the head but not the base: {detail}"

    # Not a regression, but the finding that matters most, so it is stated in
    # the verdict line rather than left for a reader to dig out of the table.
    crashing = [k for k in _USER_FACING if _raised(h.get(k))]
    if crashing:
        return False, (
            "no regression between the states, but these user-facing selector calls "
            "raise at BOTH: " + ", ".join(crashing) +
            ". That is a pre-existing crash on this hardware, not something this "
            "run introduced, and it still means a user on this box does not get a "
            "clean decline")

    nvfp4 = (h.get("explicit[nvfp4]") or {}).get("value", "unset")
    auto = (h.get("auto_scheme[no-family]") or {}).get("value", "unset")
    return False, (
        f"this is a capability inventory, not a differential: base and head are "
        f"adjacent commits that do not touch quant selection and they agree. Every "
        f"user-facing selector call returned an answer rather than raising. "
        f"`auto` resolves to {auto!r} and an explicit `nvfp4` request returns "
        f"{nvfp4!r} on this gfx1151 host")
