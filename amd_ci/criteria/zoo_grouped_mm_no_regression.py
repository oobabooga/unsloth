#!/usr/bin/env python3
"""Criteria: is an unsloth-zoo change to the torch._grouped_mm support probe, its ROCm
crash-probe child and the Triton scratch buffer a strict no-op for MoE LoRA training on
THIS host (one GPU, or the same GPU presented as two by the HIP device multiplier)?

Pairs with probes/lora_gate_probe.py (--package unsloth_zoo, a tiny Qwen3-MoE model).
Regression mode. NO_REGRESSION requires, base versus head on the same host in the same
job, per repeat:

  * the head child completed with --steps finite losses, within LOSS_TOL (widened only
    to the BASE A/A spread) of the base repeat with the same index;
  * identical decisions: _TORCH_GROUPED_MM_AVAILABLE / _SUPPORTED, the crash-probe
    survival flag, the selected MoE backend, the Python _grouped_mm fallback flag,
    bf16 / model dtype, trainable and expert-LoRA tensor counts;
  * the crash-probe child ran the same number of times with the same outcome (exit 0 or
    not), and AT MOST ONCE at the head over import, load, training and an explicit
    query of every visible device ("probe once, same answer" on a homogeneous host);
  * every visible device answers grouped_mm support the same way at the head, equal to
    the base's answer, and no per-GPU-model entry was created at the head (the host is
    homogeneous, so every card must use the process-wide answer).

Gates make it non-vacuous: both probes ran every repeat; every base repeat trained;
each state imported unsloth_zoo FROM ITS OWN CHECKOUT and unsloth from one installed
file; different commits; a HIP gfx1151 base; the grouped_mm decision was actually
taken at the base (the flag is not None, so the MoE path ran) and the crash-probe child
actually ran at the base (so the ROCm guard was exercised, not bypassed); the device
count matches the device-multiplier marker.
"""

from __future__ import annotations

import math

TITLE = "unsloth-zoo grouped_mm probe / ROCm crash probe: MoE LoRA SFT on real gfx1151, base versus head"
MODE = "regression"
# What the change touches: the HIP crash-probe path (measured), the CUDA / XPU in-process
# probe, mixed-model multi-GPU hosts (per-model answers, child masks indexing into
# HIP_VISIBLE_DEVICES), the Windows CREATE_NO_WINDOW / SetErrorMode branch.
NEEDS = ["linux", "rocm", "gpu", "nvidia", "xpu", "multi_gpu", "multi_gpu_amd", "discrete_gpu", "windows"]

EXPECT_ARCH = "gfx1151"
LOSS_TOL = 1e-3
MOE_FIELDS = ("_TORCH_GROUPED_MM_AVAILABLE", "_TORCH_GROUPED_MM_SUPPORTED", "_GROUPED_MM_SURVIVES",
              "select_moe_backend()", "python_grouped_mm_fallback")
RUN_FIELDS = ("model_dtype", "trainer_bf16", "trainable_tensors", "expert_lora_tensors")


def _runs(o: dict) -> list[dict]:
    return list((o or {}).get("runs") or [])


def _finite(xs) -> bool:
    return bool(xs) and all(isinstance(x, (int, float)) and math.isfinite(x) for x in xs)


def _completed(r: dict, steps) -> bool:
    ls = r.get("losses") or []
    return (r.get("rc") == 0 and not r.get("error") and not r.get("missing_output")
            and not r.get("parse_error") and _finite(ls) and len(ls) == steps)


def _children(r: dict) -> list[dict]:
    return list(r.get("crash_probe_children") or [])


def _outcome(r: dict) -> list[str]:
    return ["ok" if c.get("rc") == 0 else f"failed({c.get('rc', c.get('exception'))})" for c in _children(r)]


def _ident(r: dict) -> dict:
    moe = r.get("moe") or {}
    d = {k: repr(moe.get(k)) for k in MOE_FIELDS}
    d.update({k: repr(r.get(k)) for k in RUN_FIELDS})
    d["crash child outcomes"] = repr(["ok" if x == "ok" else "failed" for x in _outcome(r)])
    answers = moe.get("per_device_answers") or []
    d["grouped_mm answer (all devices)"] = repr(sorted({repr(a) for a in answers}))
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
    unsloth_files = set()
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
        out.append((f"{name} checkout is an unsloth_zoo tree", o.get("package") == "unsloth_zoo"
                    and bool(o.get("package_dir_exists")), f"package={o.get('package')}"))
        for r in runs:
            i = r.get("repeat")
            if name == "base":
                out.append((f"base r{i} trained {steps} finite steps", _completed(r, steps),
                            f"rc={r.get('rc')}; losses={r.get('losses')}"
                            + (f"; {r.get('error')}" if r.get("error") else "")))
            if r.get("overlay") is not None or name == "base":
                ov = r.get("overlay") or {}
                out.append((f"{name} r{i} imported unsloth_zoo from its own checkout",
                            bool(ov.get("from_checkout")), f"`{ov.get('file')}`"))
            if (r.get("unsloth") or {}).get("file"):
                unsloth_files.add(r["unsloth"]["file"])
            dev = r.get("device") or {}
            if dev.get("count") is not None:
                extra = int(o.get("spoofed_devices") or 0)
                out.append((f"{name} r{i} device count matches the device-multiplier marker",
                            dev.get("count") == 1 + extra,
                            f"device_count={dev.get('count')}; AMD_CI_SPOOFED_DEVICES={o.get('spoofed_devices')}"))
    out.append(("unsloth (not under test) is one installed file at every state and repeat",
                len(unsloth_files) == 1, ", ".join(f"`{f}`" for f in sorted(unsloth_files)) or "none recorded"))
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
    for r in br:
        moe = r.get("moe") or {}
        out.append((f"base r{r.get('repeat')} took the grouped_mm decision (MoE path ran)",
                    moe.get("_TORCH_GROUPED_MM_SUPPORTED") in (True, False),
                    f"_TORCH_GROUPED_MM_SUPPORTED={moe.get('_TORCH_GROUPED_MM_SUPPORTED')}; "
                    f"backend={moe.get('select_moe_backend()')}"))
        out.append((f"base r{r.get('repeat')} ran the ROCm crash-probe child",
                    len(_children(r)) >= 1,
                    f"children={_outcome(r)}; _TORCH_GROUPED_MM_AVAILABLE={moe.get('_TORCH_GROUPED_MM_AVAILABLE')}; "
                    f"python fallback={moe.get('python_grouped_mm_fallback')}"))
    return out


def _f(v, nd = 4) -> str:
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def table(obs: dict) -> str:
    rows = ["| state | repeat | rc | devices | backend | grouped_mm supported | crash children (phase: rc, HIP mask) "
            "| per-device answers | losses |", "|---|---|---|---|---|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        for r in _runs(obs.get(name) or {}):
            moe = r.get("moe") or {}
            ch = "; ".join(f"{c.get('phase')}: {c.get('rc', c.get('exception'))}, {c.get('HIP_VISIBLE_DEVICES')}"
                           for c in _children(r)) or "none"
            rows.append(f"| {name} | {r.get('repeat')} | {r.get('rc')} | {(r.get('device') or {}).get('count')} | "
                        f"{moe.get('select_moe_backend()')} | {moe.get('_TORCH_GROUPED_MM_SUPPORTED')} | {ch} | "
                        f"{moe.get('per_device_answers')} | "
                        f"{[round(x, 4) for x in (r.get('losses') or [])]} |")
    b, h = obs.get("base") or {}, obs.get("head") or {}
    br, hr = _runs(b), _runs(h)
    if br and hr:
        ib, ih = _ident(br[0]), _ident(hr[0])
        rows += ["", "| field (r1) | base | head | same |", "|---|---|---|---|"]
        for k in ib:
            rows.append(f"| {k} | `{ib[k]}` | `{ih.get(k)}` | {'yes' if ib[k] == ih.get(k) else 'NO'} |")
        for k in ("_PERSISTENT_BUFFER", "_TORCH_GROUPED_MM_SUPPORTED_BY_KIND after query",
                  "_GROUPED_MM_SURVIVES_BY_KIND after query", "_GROUPED_MM_FOREIGN after query"):
            rows.append(f"| {k} (informational) | `{(br[0].get('moe') or {}).get(k)}` | "
                        f"`{(hr[0].get('moe') or {}).get(k)}` | |")
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
            findings.append(f"r{i} decisions moved: " + ", ".join(f"`{k}` {ib[k]} -> {ih.get(k)}" for k in moved))
        if len(_children(rh)) != len(_children(rb)):
            findings.append(f"r{i}: crash-probe children base {len(_children(rb))} vs head {len(_children(rh))}")
        moe = rh.get("moe") or {}
        answers = moe.get("per_device_answers") or []
        if len({repr(a) for a in answers}) > 1:
            findings.append(f"r{i}: head devices disagree on grouped_mm support {answers}")
        for k in ("_TORCH_GROUPED_MM_SUPPORTED_BY_KIND after query", "_GROUPED_MM_SURVIVES_BY_KIND after query"):
            if moe.get(k) not in ({}, None, "absent"):
                findings.append(f"r{i}: head created per-GPU-model entries on a homogeneous host: `{k}` {moe.get(k)}")
    if len(hr) != len(_runs(base)):
        findings.append(f"head repeats {len(hr)} vs base {len(_runs(base))}")
    if findings:
        return True, "; ".join(findings)
    r0 = hr[0]
    moe = r0.get("moe") or {}
    return False, (f"grouped_mm supported={moe.get('_TORCH_GROUPED_MM_SUPPORTED')}, crash child "
                   f"{_outcome(r0)}, backend {moe.get('select_moe_backend()')}, per-device answers "
                   f"{moe.get('per_device_answers')} identical at base and head; max loss diff {worst:.2e} "
                   f"(tol {tol:.0e}); one crash-probe child per process at the head")
