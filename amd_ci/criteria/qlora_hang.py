#!/usr/bin/env python3
"""Criteria for probes/qlora_hang_probe.py (issue unslothai/unsloth#11498).

Defect = the GPU hung (no heartbeat inside the stall window), faulted (a HIP/HSA
fault line), crashed, produced a non-finite loss / output, or was unusable after
the arm. `base` is the arm expected to show it, `head` the arm with one thing
removed; any further state (the PEFT control) must also be clean for CONFIRMED.

A base that trained cleanly is VOID, which here means "did not reproduce on this
host", never "fixed". Gates make sure each arm actually ran what it claims:
- enough micro-steps to cover the reporter's earliest failure (544) unless it failed
- the kill switch / release pin actually changed what ran (engagement)
"""

from __future__ import annotations

TITLE = "Issue 11498: long QLoRA run, GPU hang / fault"
MODE = "differential"
# The reporter's card is a discrete gfx1100 driving a desktop; this pool is a gfx1151 APU.
NEEDS = ["gpu", "rocm", "discrete_gpu"]

# Earliest failure reported: optimizer step 34 at GA 16.
MIN_MICRO = 544
BAD = ("hang", "fault", "crash", "hard_timeout")


def _failed(o: dict) -> bool:
    st = str(o.get("status", ""))
    if any(st.startswith(b) for b in BAD) or o.get("nonfinite"):
        return True
    if not (o.get("health_after") or {}).get("ok", True):
        return True
    fs = o.get("fla_stress") or {}
    return bool(fs) and (any(str(fs.get("status", "")).startswith(b) for b in BAD) or fs.get("nonfinite"))


def _eng(o: dict) -> dict:
    return o.get("engagement") or {}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name, o in obs.items():
        if name.startswith("_"):
            continue
        arm = o.get("arm")
        st = o.get("status")
        out.append((f"{name} ({arm}) ran", st not in ("skipped", "error", None),
                    f"status={st} {o.get('why', '')}".strip()))
        if not _failed(o):
            out.append((f"{name} ({arm}) covered >= {MIN_MICRO} micro-steps",
                        (o.get("micro_steps") or 0) >= MIN_MICRO,
                        f"{o.get('micro_steps')} micro / {o.get('optimizer_steps')} steps, "
                        f"max len {o.get('max_len_seen')}"))
        e = _eng(o)
        if arm in ("main_default", "hist"):
            out.append((f"{name}: vendored fla kernels launched", bool(e.get("fla_kernels")),
                        f"{len(e.get('fla_kernels') or {})} fla kernels, vendored={e.get('fla_vendored')}"))
        if arm == "main_nofla":
            out.append((f"{name}: no fla kernel launched", e != {} and not e.get("fla_kernels"),
                        f"{len(e.get('fla_kernels') or {})} fla kernels"))
        if arm == "peft":
            out.append((f"{name}: unsloth never imported", e != {} and not e.get("unsloth_imported"),
                        f"unsloth_imported={e.get('unsloth_imported')}"))
        if arm == "hist":
            v = ((o.get("versions") or {}).get("unsloth") or {}).get("version")
            out.append((f"{name}: runs unsloth 2026.9.7", v == "2026.9.7", f"unsloth {v}"))
    return out


def table(obs: dict) -> str:
    rows = ["| state | arm | status | micro / steps | first..last step loss | peak GiB | fla kernels | "
            "unsloth kernels | fault lines | GPU ok after |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for name, o in obs.items():
        if name.startswith("_"):
            continue
        e = _eng(o)
        ls = [x for x in (o.get("losses_step") or []) if x != "..."]
        loss = f"{ls[0]} .. {ls[-1]}" if ls else "-"
        rows.append(
            f"| {name} | {o.get('arm')} | {o.get('status')} | {o.get('micro_steps')} / "
            f"{o.get('optimizer_steps')} | {loss} | {o.get('peak_gib')} | {len(e.get('fla_kernels') or {})} | "
            f"{len(e.get('unsloth_kernels') or {})} | {', '.join(o.get('fault_lines') or []) or '-'} | "
            f"{(o.get('health_after') or {}).get('ok')} |")
        fs = o.get("fla_stress")
        if fs:
            rows.append(f"| {name} | fla_stress | {fs.get('status')} | {fs.get('micro_steps')} iters | "
                        f"nonfinite={fs.get('nonfinite')} | - | {len(_eng(fs).get('fla_kernels') or {})} | - | "
                        f"{', '.join(fs.get('fault_lines') or []) or '-'} | - |")
    for name, o in obs.items():
        if name.startswith("_"):
            continue
        kl = o.get("kernel_log_after") or {}
        new = kl.get("new_amdgpu_lines")
        rows.append("")
        rows.append(f"{name}: kernel log readable={kl.get('readable')}, "
                    f"new amdgpu lines={len(new) if new is not None else 'n/a'}; "
                    f"versions={ {k: v for k, v in (o.get('versions') or {}).items() if k in ('torch', 'triton', 'bitsandbytes', 'transformers', 'hip')} }")
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    return _failed(base)


def head_is_fixed(head: dict) -> bool:
    return not _failed(head) and head.get("status") == "ok"
