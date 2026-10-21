#!/usr/bin/env python3
"""Criteria: is an unsloth change to bf16 detection, the CUDA_DEVICE_ORDER hint and the
XPU / MLX bf16 gates a strict no-op on THIS AMD host (one GPU, or the same GPU presented
as two by the HIP device multiplier)?

Pairs with probes/lora_gate_probe.py (--package unsloth, a tiny dense model). Regression
mode. NO_REGRESSION requires, base versus head on the same host in the same job, per
repeat:

  * the head child completed with --steps finite losses, within LOSS_TOL (widened only
    to the BASE A/A spread) of the base repeat with the same index;
  * identical bf16 answers and training dtype: SUPPORTS_BFLOAT16, unsloth's
    is_bfloat16_supported(), the patched torch.cuda.is_bf16_supported(), zoo's
    device_is_bf16_supported(), decision._amp_dtype(cuda:0), unsloth DEVICE_TYPE, the
    loaded model dtype and the trainer's bf16 / fp16 choice;
  * no CUDA_DEVICE_ORDER hint in the head child's output, and
    cuda_order_mismatch_names() == [] at the head (HIP must stay silent, single card
    or homogeneous), with CUDA_DEVICE_ORDER unset so silence is not the user's choice;
  * internal consistency at the head: trainer bf16 == SUPPORTS_BFLOAT16.

Gates make it non-vacuous: both probes ran every repeat; every base repeat trained;
each state imported unsloth FROM ITS OWN CHECKOUT and unsloth_zoo from one installed
file; different commits; a HIP gfx1151 base; CUDA_DEVICE_ORDER unset in every child;
the device count matches the device-multiplier marker.
"""

from __future__ import annotations

import math

TITLE = "unsloth bf16 detection / CUDA_DEVICE_ORDER hint: LoRA SFT on real gfx1151, base versus head"
MODE = "regression"
# What the change touches: the HIP bf16 path (measured), the NVIDIA mixed-host hint, the
# Intel XPU native-bf16 gate, the MLX Metal-family gate, multi-GPU name comparison and
# the Windows hint wording.
NEEDS = ["linux", "rocm", "gpu", "nvidia", "xpu", "mlx", "multi_gpu", "windows"]

EXPECT_ARCH = "gfx1151"
LOSS_TOL = 1e-3
FLAGS = ("SUPPORTS_BFLOAT16", "unsloth.is_bfloat16_supported()", "torch.cuda.is_bf16_supported()",
         "zoo device_is_bf16_supported()", "decision._amp_dtype(cuda:0)", "unsloth DEVICE_TYPE")
RUN_FIELDS = ("model_dtype", "trainer_bf16", "trainer_fp16", "trainable_tensors")


def _runs(o: dict) -> list[dict]:
    return list((o or {}).get("runs") or [])


def _finite(xs) -> bool:
    return bool(xs) and all(isinstance(x, (int, float)) and math.isfinite(x) for x in xs)


def _completed(r: dict, steps) -> bool:
    ls = r.get("losses") or []
    return (r.get("rc") == 0 and not r.get("error") and not r.get("missing_output")
            and not r.get("parse_error") and _finite(ls) and len(ls) == steps)


def _ident(r: dict) -> dict:
    fl = r.get("flags") or {}
    d = {k: repr(fl.get(k)) for k in FLAGS}
    d.update({k: repr(r.get(k)) for k in RUN_FIELDS})
    return d


def _aa_loss_spread(o: dict) -> float:
    rs = [r.get("losses") or [] for r in _runs(o)]
    worst = 0.0
    for i in range(len(rs)):
        for j in range(i + 1, len(rs)):
            for a, b in zip(rs[i], rs[j]):
                worst = max(worst, abs(a - b))
    return worst


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    zoo_files = set()
    for name in ("base", "head"):
        o = obs.get(name) or {}
        runs = _runs(o)
        cfg = o.get("config") or {}
        want, steps = cfg.get("repeats"), cfg.get("steps")
        b = o.get("bootstrap") or {}
        out.append((f"{name} probe wrote observations for every repeat",
                    o.get("_probe_rc") == 0 and not o.get("_missing_output") and not o.get("_parse_error")
                    and bool(want) and len(runs) == want and bool(o.get("python")),
                    f"probe rc={o.get('_probe_rc')}; repeats {len(runs)}/{want}; python `{o.get('python')}`"
                    + (f"; bootstrap rc={b.get('install_rc')} reused={b.get('reused', False)}" if b else "")))
        out.append((f"{name} checkout is an unsloth tree", o.get("package") == "unsloth"
                    and bool(o.get("package_dir_exists")), f"package={o.get('package')}"))
        for r in runs:
            i = r.get("repeat")
            if name == "base":
                out.append((f"base r{i} trained {steps} finite steps", _completed(r, steps),
                            f"rc={r.get('rc')}; losses={r.get('losses')}"
                            + (f"; {r.get('error')}" if r.get("error") else "")))
            if r.get("overlay") is not None or name == "base":
                ov = r.get("overlay") or {}
                out.append((f"{name} r{i} imported unsloth from its own checkout",
                            bool(ov.get("from_checkout")), f"`{ov.get('file')}`"))
            if (r.get("unsloth_zoo") or {}).get("file"):
                zoo_files.add(r["unsloth_zoo"]["file"])
            env = r.get("env")
            if env is not None:
                out.append((f"{name} r{i} ran with CUDA_DEVICE_ORDER unset",
                            not (env.get("CUDA_DEVICE_ORDER") or "").strip(),
                            f"CUDA_DEVICE_ORDER={env.get('CUDA_DEVICE_ORDER')!r}"))
            dev = r.get("device") or {}
            if dev.get("count") is not None:
                extra = int(o.get("spoofed_devices") or 0)
                out.append((f"{name} r{i} device count matches the device-multiplier marker",
                            dev.get("count") == 1 + extra,
                            f"device_count={dev.get('count')}; AMD_CI_SPOOFED_DEVICES={o.get('spoofed_devices')}"))
    out.append(("unsloth_zoo (not under test) is one installed file at every state and repeat",
                len(zoo_files) == 1, ", ".join(f"`{f}`" for f in sorted(zoo_files)) or "none recorded"))
    b, h = obs.get("base") or {}, obs.get("head") or {}
    out.append(("base and head are different commits",
                bool(b.get("commit")) and b.get("commit") != h.get("commit"),
                f"{str(b.get('commit'))[:9]} vs {str(h.get('commit'))[:9]}"))
    want = (h.get("config") or {}).get("expect_head")
    if want:
        out.append(("head is the expected merge commit (not a later commit naming the PR, e.g. a revert)",
                    str(h.get("commit") or "").startswith(want), f"head {h.get('commit')} vs expected {want}"))
    br = _runs(b)
    bd = (br[0].get("device") if br else None) or {}
    bv = (br[0].get("versions") if br else None) or {}
    out.append((f"base is a HIP {EXPECT_ARCH} host",
                bool(bv.get("hip")) and str(bd.get("gcn_arch") or "").startswith(EXPECT_ARCH),
                f"hip={bv.get('hip')}; arch={bd.get('gcn_arch')}; torch={bv.get('torch')}"))
    return out


def _f(v, nd = 4) -> str:
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def table(obs: dict) -> str:
    rows = ["| state | repeat | rc | devices | SUPPORTS_BFLOAT16 | model dtype | trainer bf16 | order hint lines "
            "| cuda_order_mismatch_names() | losses |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        for r in _runs(obs.get(name) or {}):
            fl = r.get("flags") or {}
            rows.append(f"| {name} | {r.get('repeat')} | {r.get('rc')} | {(r.get('device') or {}).get('count')} | "
                        f"{fl.get('SUPPORTS_BFLOAT16')} | {r.get('model_dtype')} | {r.get('trainer_bf16')} | "
                        f"{len(r.get('order_hint_lines') or [])} | {fl.get('cuda_order_mismatch_names()')} | "
                        f"{[round(x, 4) for x in (r.get('losses') or [])]} |")
    b, h = obs.get("base") or {}, obs.get("head") or {}
    br, hr = _runs(b), _runs(h)
    if br and hr:
        ib, ih = _ident(br[0]), _ident(hr[0])
        rows += ["", "| field (r1) | base | head | same |", "|---|---|---|---|"]
        for k in ib:
            rows.append(f"| {k} | `{ib[k]}` | `{ih.get(k)}` | {'yes' if ib[k] == ih.get(k) else 'NO'} |")
        rows.append(f"| torch / hip | `{(br[0].get('versions') or {}).get('torch')}` / "
                    f"`{(br[0].get('versions') or {}).get('hip')}` | | |")
    rows.append(f"| A/A max loss diff | {_f(_aa_loss_spread(b), 6)} | {_f(_aa_loss_spread(h), 6)} | |")
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    findings: list[str] = []
    steps = (head.get("config") or {}).get("steps")
    tol = max(LOSS_TOL, _aa_loss_spread(base))   # base A/A only: a head-only wobble must not widen its own tolerance
    worst = 0.0
    hr = _runs(head)
    for rb, rh in zip(_runs(base), hr):
        i = rh.get("repeat")
        if not _completed(rh, steps):
            findings.append(f"head r{i} did not train: rc={rh.get('rc')} "
                            f"{rh.get('error') or ''} losses={rh.get('losses')}")
            continue
        lb, lh = rb.get("losses") or [], rh.get("losses") or []
        d = max((abs(x - y) for x, y in zip(lb, lh)), default = 0.0)
        worst = max(worst, d)
        if d > tol:
            findings.append(f"r{i}: losses differ by up to {d:.2e} (> {tol:.2e})")
        ib, ih = _ident(rb), _ident(rh)
        moved = [k for k in ib if ib[k] != ih.get(k)]
        if moved:
            findings.append(f"r{i} bf16 / dtype moved: " + ", ".join(f"`{k}` {ib[k]} -> {ih.get(k)}" for k in moved))
        if rh.get("order_hint_lines"):
            findings.append(f"r{i}: head printed a CUDA_DEVICE_ORDER hint on HIP: {rh['order_hint_lines'][:2]}")
        names = (rh.get("flags") or {}).get("cuda_order_mismatch_names()")
        if names not in ([], "absent"):
            findings.append(f"r{i}: head cuda_order_mismatch_names() = {names!r}, expected [] on HIP")
        sb = (rh.get("flags") or {}).get("SUPPORTS_BFLOAT16")
        if isinstance(sb, bool) and sb != rh.get("trainer_bf16"):
            findings.append(f"r{i}: head trainer bf16 {rh.get('trainer_bf16')} disagrees with SUPPORTS_BFLOAT16 {sb}")
    if len(hr) != len(_runs(base)):
        findings.append(f"head repeats {len(hr)} vs base {len(_runs(base))}")
    if findings:
        return True, "; ".join(findings)
    fl = hr[0].get("flags") or {}
    return False, (f"SUPPORTS_BFLOAT16={fl.get('SUPPORTS_BFLOAT16')}, model dtype {hr[0].get('model_dtype')}, "
                   f"_amp_dtype {fl.get('decision._amp_dtype(cuda:0)')} identical at base and head; no "
                   f"CUDA_DEVICE_ORDER hint, cuda_order_mismatch_names() {fl.get('cuda_order_mismatch_names()')}; "
                   f"max loss diff {worst:.2e} (tol {tol:.0e})")
