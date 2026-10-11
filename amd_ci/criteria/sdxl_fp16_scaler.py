#!/usr/bin/env python3
"""Criteria: SDXL LoRA fp16 gradients underflow without a GradScaler (unslothai/unsloth#13278).

Pairs with probes/sdxl_fp16_probe.py, which runs one state's trainer three times (fp32
reference, fp16, bf16) on hf-internal-testing/tiny-stable-diffusion-xl-pipe, same seed and data.

  defect at a state   fp16 grad_norm departs from the same state's fp32 reference by more
                      than GRAD_RTOL at some step in 2..N (step 1 is pre-update, so equal
                      gradients there are expected with or without the bug), or is missing /
                      non-finite there.
  fixed at a state    fp16 grad_norm within GRAD_RTOL and loss within LOSS_RTOL of fp32 at
                      EVERY step 1..N, AND the bf16 arm (the trainer's own pick on gfx1151)
                      matches the base's bf16 arm within the same tolerances at every step:
                      the default path must not move.

Gates (any failure = INCONCLUSIVE): every arm of every state ran without error, logged N
finite losses, ran on a ROCm (HIP) torch build, on the GPU (UNet output on device `cuda`,
nonzero peak GPU memory), in the dtype the arm asked for. A CPU fallback, a CUDA host, or a
silent bf16 -> fp16 downgrade therefore cannot pass for a gfx1151 result.
"""

from __future__ import annotations

import math

TITLE = "PR 13278: SDXL LoRA fp16 GradScaler on gfx1151, base versus head"
MODE = "differential"
# Every capability the change touches. Only rocm / gpu / linux are answerable on the Linux
# pool; the rest render as declared gaps: the PR also gates DiT training on Apple MPS (mlx
# stands for Apple Silicon here), the CPU-fallback warning fires on XPU / MPS hosts, fp16 is
# the trainer's pick on pre-Ampere NVIDIA (nvidia), and the trainer runs on Windows too.
NEEDS: list[str] = ["linux", "rocm", "gpu", "nvidia", "xpu", "mlx", "windows", "discrete_gpu"]

GRAD_RTOL = 0.05
LOSS_RTOL = 0.01
ARM_DTYPE = {"fp32": "torch.float32", "fp16": "torch.float16", "bf16": "torch.bfloat16"}

# _decide calls base_shows_defect(base) before head_is_fixed(head); head_is_fixed needs the
# base's bf16 arm for the no-regression half, so it is kept here. Fails closed if absent.
_BASE: dict = {}


def _states(obs: dict) -> list[str]:
    return [n for n in ("base", "head", "merge") if n in obs]


def _rel(a, b) -> float:
    if a is None or b is None or not math.isfinite(a) or not math.isfinite(b):
        return math.inf
    return abs(a - b) / max(abs(b), 1e-12)


def _arm(o: dict, name: str) -> dict:
    return ((o or {}).get("arms") or {}).get(name) or {}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for s in _states(obs):
        o = obs[s] or {}
        steps = o.get("steps") or 0
        if o.get("error") or not o.get("arms"):
            out.append((f"{s} probe ran", False, o.get("error") or f"no arms; probe rc {o.get('_probe_rc')}"))
            continue
        for name, want in ARM_DTYPE.items():
            a = _arm(o, name)
            dev = a.get("device") or {}
            unet = a.get("unet") or {}
            loss = a.get("loss") or []
            ok_run = a.get("rc") == 0 and not a.get("error") and a.get("finite") is True and len(loss) == steps
            out.append((f"{s} {name} trained {steps} finite steps", ok_run,
                        f"rc={a.get('rc')} error={a.get('error')} steps={len(loss)} finite={a.get('finite')}"
                        + (f" tail={str(a.get('log_tail'))[-200:]}" if a.get("missing_output") else "")))
            out.append((f"{s} {name} on ROCm GPU", bool(dev.get("hip")) and unet.get("devices") == ["cuda"]
                        and (dev.get("max_mem_allocated") or 0) > 0,
                        f"hip={dev.get('hip')} arch={dev.get('arch')} name={dev.get('name')} "
                        f"unet_devices={unet.get('devices')} peak_alloc={dev.get('max_mem_allocated')}"))
            out.append((f"{s} {name} UNet dtype", unet.get("dtypes") == [want],
                        f"UNet output dtypes {unet.get('dtypes')} (want {want})"))
        ref = _arm(o, "fp32").get("grad_norm") or []
        out.append((f"{s} fp32 reference has every grad norm", len(ref) == steps
                    and all(g is not None and math.isfinite(g) for g in ref), f"{ref}"))
    return out


def _fp16_vs_fp32(o: dict) -> tuple[list[float], list[float]]:
    f16, f32 = _arm(o, "fp16"), _arm(o, "fp32")
    g = [_rel(a, b) for a, b in zip(f16.get("grad_norm") or [], f32.get("grad_norm") or [])]
    l = [_rel(a, b) for a, b in zip(f16.get("loss") or [], f32.get("loss") or [])]
    return g, l


def _fmt(xs: list[float]) -> str:
    return "[" + ", ".join("inf" if not math.isfinite(x) else f"{100 * x:.1f}%" for x in xs) + "]"


def base_shows_defect(base: dict) -> tuple[bool, str]:
    _BASE.clear()
    _BASE.update(base)
    g, _ = _fp16_vs_fp32(base)
    bad = [i + 1 for i, x in enumerate(g) if i >= 1 and x > GRAD_RTOL]
    return bool(bad), f"fp16 vs fp32 grad-norm deviation per step {_fmt(g)}; beyond {GRAD_RTOL:.0%} at steps {bad}"


def head_is_fixed(head: dict) -> tuple[bool, str]:
    steps = head.get("steps") or 0
    g, l = _fp16_vs_fp32(head)
    why = []
    if len(g) != steps or any(x > GRAD_RTOL for x in g):
        why.append(f"fp16 grad-norm deviation {_fmt(g)} (rtol {GRAD_RTOL:.0%})")
    if len(l) != steps or any(x > LOSS_RTOL for x in l):
        why.append(f"fp16 loss deviation {_fmt(l)} (rtol {LOSS_RTOL:.0%})")
    if not _BASE:
        why.append("base observations unavailable for the bf16 no-regression check")
    else:
        hb, bb = _arm(head, "bf16"), _arm(_BASE, "bf16")
        bg = [_rel(a, b) for a, b in zip(hb.get("grad_norm") or [], bb.get("grad_norm") or [])]
        bl = [_rel(a, b) for a, b in zip(hb.get("loss") or [], bb.get("loss") or [])]
        if len(bg) != steps or any(x > GRAD_RTOL for x in bg) or len(bl) != steps or any(x > LOSS_RTOL for x in bl):
            why.append(f"bf16 moved vs base: grad {_fmt(bg)}, loss {_fmt(bl)}")
    return not why, "; ".join(why) or "fp16 tracks fp32 at every step and bf16 matches the base"


def _sig(xs: list) -> str:
    return "[" + ", ".join("None" if x is None else f"{x:.5g}" for x in xs) + "]"


def table(obs: dict) -> str:
    rows = ["| state | arm | UNet dtype | losses | grad norms | grad dev vs fp32 |", "|---|---|---|---|---|---|"]
    for s in _states(obs):
        o = obs[s] or {}
        ref = _arm(o, "fp32").get("grad_norm") or []
        for name in ARM_DTYPE:
            a = _arm(o, name)
            gn = a.get("grad_norm") or []
            loss = a.get("loss") or []
            dev = _fmt([_rel(x, y) for x, y in zip(gn, ref)]) if name != "fp32" else "-"
            rows.append(f"| {s} | {name} | {','.join((a.get('unet') or {}).get('dtypes') or []) or '?'} "
                        f"| {_sig(loss)} | {_sig(gn)} | {dev} |")
    return "\n".join(rows)
