#!/usr/bin/env python3
"""Criteria for unslothai/unsloth#12044 on native Windows 11 x64, Python 3.12. Judges only.

Defect: the released unsloth / unsloth-zoo pair cannot be installed on Windows next to
torch==2.14.0+cu130 (resolver refuses, or torch is moved off 2.14.0+cu130).
Base = 2026.9.14 pair (cap <2.13.0), head = 2026.10.2 pair (cap <2.15.0).

The head is "fixed" only if, on top of resolving with torch kept:
  * triton-windows resolves to 3.8.x (the minor triton-windows pairs with torch 2.14),
  * `import triton` works, and `import unsloth` gets as far as the accelerator check
    (this box has no NVIDIA GPU, so the expected end is "cannot find any torch accelerator"),
  * leg 2 (NVIDIA spoof over the real gfx1151, ROCm torch 2.11): the tiny GPT-OSS
    q_proj/v_proj LoRA trains 5 finite steps in BOTH arms (default, UNSLOTH_COMPILE_DISABLE=1),
    step-1 lora_B grads nonzero on the first and last layer, LoRA weights move, experts do not,
    nothing but LoRA trainable.
Leg 2 is wiring evidence only: HIP kernels underneath, not CUDA.
"""

from __future__ import annotations

import math

TITLE = "Issue 12044: released unsloth pair + torch 2.14.0+cu130 on native Windows; tiny GPT-OSS LoRA under NVIDIA spoof"
MODE = "differential"
NEEDS: list[str] = ["windows", "nvidia", "discrete_gpu"]

TORCH = "2.14.0+cu130"
PAIR = {"base": "2026.9.14", "head": "2026.10.2"}
# The accelerator gate words its refusal by what it sees: a box with an AMD GPU and a CUDA torch
# build gets the ROCm wording. Both are the gate itself, which is where a GPU-less import must end.
NO_GPU_MSGS = ("cannot find any torch accelerator", "has no usable hip accelerator")


def _l1(o):
    return (o or {}).get("leg1") or {}


def _insp(o):
    i = _l1(o).get("inspect") or {}
    return i if isinstance(i, dict) else {}


def _rc(step):
    return (step or {}).get("rc")


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        l1 = _l1(o)
        ok = _rc(l1.get("venv")) == 0 and _rc(l1.get("torch_install")) == 0
        ev = (f"python {str(o.get('python', '?')).split()[0]}, platform {o.get('platform')}, "
              f"venv rc {_rc(l1.get('venv'))}, torch {TORCH} install rc {_rc(l1.get('torch_install'))}")
        if l1.get("probe_error"):
            ev += f"; probe error {l1['probe_error']}"
        if not ok and l1.get("torch_install"):
            ev += f"; tail: {str(l1['torch_install'].get('tail', ''))[-300:]}"
        out.append((f"{name}: torch {TORCH} installed on this Windows box before the pair", ok, ev))
        out.append((f"{name}: probe ran on Windows Python 3.12",
                    o.get("platform") == "win32" and str(o.get("python", "")).startswith("3.12"),
                    f"platform {o.get('platform')}, python {str(o.get('python', '?'))[:20]}"))
    l2 = (obs.get("head") or {}).get("leg2") or {}
    ti = l2.get("torch_import") or {}
    spoof_ok = (ti.get("ok") is True and ti.get("hip") is None and bool(ti.get("real_hip"))
                and ti.get("is_available") is True and str(ti.get("cuda", "")).startswith("12"))
    out.append(("head leg 2: ROCm torch installed and the NVIDIA spoof engaged over a real HIP device",
                _rc(l2.get("torch_install")) == 0 and spoof_ok,
                f"rocm torch install rc {_rc(l2.get('torch_install'))}, torch {ti.get('version')}, "
                f"reported cuda {ti.get('cuda')}, reported hip {ti.get('hip')}, real hip "
                f"{ti.get('real_hip')}, available {ti.get('is_available')}, device "
                f"{ti.get('device_name')}; {l2.get('probe_error') or ''}"))
    return out


def base_shows_defect(base: dict):
    l1, i = _l1(base), _insp(base)
    rc = _rc(l1.get("pair_install"))
    torch_v, uns = i.get("v_torch"), i.get("v_unsloth")
    if rc != 0:
        # Only a resolver conflict against the torch pin is the defect; anything else (a version
        # that does not exist, a network error) means the base never tested the cap: VOID.
        if l1.get("resolver_conflict"):
            return True, f"pip refused the {PAIR['base']} pair with torch {TORCH} (rc {rc}, resolver conflict)"
        return False, f"base pair install failed without a resolver conflict (rc {rc}): harness, not the cap"
    if torch_v != TORCH:
        return True, f"pair installed but torch moved to {torch_v}"
    if uns != PAIR["base"]:
        return True, f"unsloth resolved to {uns}, not {PAIR['base']}"
    return False, f"base pair installed with torch {torch_v} kept"


def _leg1_head_problems(head: dict) -> list[str]:
    l1, i = _l1(head), _insp(head)
    bad = []
    if _rc(l1.get("pair_install")) != 0:
        bad.append(f"pair install rc {_rc(l1.get('pair_install'))}")
    if i.get("v_torch") != TORCH:
        bad.append(f"torch is {i.get('v_torch')}")
    if i.get("v_unsloth") != PAIR["head"] or i.get("v_unsloth_zoo") != PAIR["head"]:
        bad.append(f"unsloth {i.get('v_unsloth')} / zoo {i.get('v_unsloth_zoo')}")
    tw = i.get("v_triton-windows")
    if not (tw or "").startswith("3.8"):
        bad.append(f"triton-windows {tw} (torch 2.14 pairs with 3.8.x)")
    imps = l1.get("imports") or {}
    if not (imps.get("triton") or {}).get("ok"):
        bad.append(f"import triton failed: {(imps.get('triton') or {}).get('exc', '?')[:200]}")
    u = imps.get("unsloth") or {}
    if u.get("ok"):
        bad.append("import unsloth succeeded on a box with no NVIDIA GPU (unexpected; check the spoof was stripped)")
    elif not any(m in str(u.get("exc", "")).lower() for m in NO_GPU_MSGS):
        bad.append(f"import unsloth failed BEFORE the accelerator check: {u.get('exc_type')}: "
                   f"{str(u.get('exc', ''))[:300]}")
    return bad


def _arm_problems(name: str, a: dict) -> list[str]:
    if not a:
        return [f"{name}: no observation"]
    if not a.get("ok"):
        return [f"{name}: failed at stage {a.get('stage')}: {a.get('exc_type')}: {str(a.get('exc', ''))[:300]}"]
    bad = []
    losses = a.get("losses") or []
    if len(losses) != 5 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in losses):
        bad.append(f"{name}: losses {losses}")
    nz = a.get("step1_lora_B_nonzero") or {}
    if not nz or not all(nz.values()):
        bad.append(f"{name}: step-1 lora_B grads {a.get('step1_lora_grad_norms')}")
    if not (a.get("lora_delta_max") or 0) > 0:
        bad.append(f"{name}: LoRA weights did not move")
    if a.get("experts_max_change") not in (0, 0.0):
        bad.append(f"{name}: expert weights changed by {a.get('experts_max_change')}")
    if a.get("trainable_non_lora"):
        bad.append(f"{name}: non-LoRA trainable {a.get('trainable_non_lora')[:4]}")
    return bad


def _leg2_problems(head: dict) -> list[str]:
    l2 = head.get("leg2") or {}
    if l2.get("probe_error"):
        return [f"leg 2 probe error {l2['probe_error']}"]
    arms = l2.get("arms") or {}
    bad = []
    for arm in ("default", "compile_disabled"):
        bad += _arm_problems(arm, arms.get(arm) or {})
    return bad


def head_is_fixed(head: dict):
    bad = _leg1_head_problems(head) + _leg2_problems(head)
    if bad:
        return False, "; ".join(bad)
    return True, "head pair installs with torch kept, triton-windows 3.8.x, tiny GPT-OSS trains in both arms"


def _fmt(x, nd = 4):
    if isinstance(x, float):
        return f"{x:.{nd}g}"
    return str(x)


def table(obs: dict) -> str:
    rows = ["### Leg 1: released pair + torch 2.14.0+cu130, real host (spoof stripped)", "",
            "| state | pair | pip rc | resolver conflict | torch after | unsloth / zoo | triton-windows | bitsandbytes | xformers | `torch2140` extra in metadata | import triton | import bitsandbytes | import unsloth |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in ("base", "head"):
        o = obs.get(name)
        if not o:
            continue
        l1, i = _l1(o), _insp(o)
        imps = l1.get("imports") or {}

        def imp(m):
            r = imps.get(m) or {}
            if r.get("ok"):
                return f"ok {r.get('version')}"
            return f"{r.get('exc_type')}: {str(r.get('exc', ''))[:120]}".replace("|", "/").replace("\n", " ")
        extras = i.get("unsloth_extras")
        has2140 = None if extras is None else any("torch2140" in e for e in extras)
        rows.append(
            f"| {name} | {o.get('pair')} | {_rc(l1.get('pair_install'))} | {l1.get('resolver_conflict')} "
            f"| {i.get('v_torch')} | {i.get('v_unsloth')} / {i.get('v_unsloth_zoo')} | {i.get('v_triton-windows')} "
            f"| {i.get('v_bitsandbytes')} | {i.get('v_xformers')} | {has2140} | {imp('triton')} "
            f"| {imp('bitsandbytes')} | {imp('unsloth')} |")
    head = obs.get("head") or {}
    l2 = head.get("leg2") or {}
    i2 = l2.get("inspect") or {}
    rows += ["", "### Leg 2 (head only): tiny GPT-OSS q_proj/v_proj LoRA r=4 a=8, 5 steps, NVIDIA spoof over gfx1151, ROCm torch",
             "",
             f"pair install mode `{l2.get('pair_install_mode')}` (rc {_rc(l2.get('pair_install'))}); "
             f"torch {i2.get('v_torch')}, transformers {i2.get('v_transformers')}, trl {i2.get('v_trl')}, "
             f"peft {i2.get('v_peft')}, triton {i2.get('v_triton')}, triton-windows {i2.get('v_triton-windows')}, "
             f"bitsandbytes {i2.get('v_bitsandbytes')} (import ok: {(l2.get('bnb_import') or {}).get('ok')}); "
             f"long paths enabled: {head.get('long_paths_enabled')}",
             "",
             "| arm | ok / stage | experts class | attention forward module | flex sink installed | losses | step-1 lora_B nonzero | LoRA delta max | experts max change | error |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for arm in ("default", "compile_disabled"):
        a = (l2.get("arms") or {}).get(arm) or {}
        nz = a.get("step1_lora_B_nonzero") or {}
        nzs = f"{sum(bool(v) for v in nz.values())}/{len(nz)}" if nz else "-"
        err = f"{a.get('exc_type')}: {str(a.get('exc', ''))[:160]}" if not a.get("ok") else ""
        rows.append(
            f"| {arm} | {a.get('ok')} / {a.get('stage')} | {a.get('experts_class')} | {a.get('attention_forward_module')} "
            f"| {a.get('flex_sink_installed')} | {[round(x, 4) for x in (a.get('losses') or [])]} | {nzs} "
            f"| {_fmt(a.get('lora_delta_max'))} | {_fmt(a.get('experts_max_change'))} | {err.replace('|', '/').replace(chr(10), ' ')} |")
    rows += ["", "Leg 2 is wiring evidence only: the kernels are HIP on a Radeon 8060S, not CUDA. "
             "bitsandbytes NF4 (the real 20B path) cannot run on this hardware, so load_in_4bit=False."]
    return "\n".join(rows)
