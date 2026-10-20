#!/usr/bin/env python3
"""Criteria: Triton kernel performance audit on gfx1151 ROCm (merged PR 14351 = audited unsloth SHA e9127c9).

This is a measurement run, not a fix validation. Regression mode is used as an A/A check: the base (parent) and
head (merge commit) carry byte-identical kernel files (gated), so head "worse" means the measurement itself is
noisier than WORSE_FRAC on some entry point, i.e. the numbers are not trustworthy. The audit numbers themselves
(ours vs torch eager / torch.compile / fla latest) live in the uploaded triton_audit_rows_*.jsonl and are judged
off-box against the NVIDIA tables.
Gates: torch is a HIP build on gfx1151; deps cloned; the bench ran and emitted timing rows in both states;
kernel files are identical between the states.
"""
from __future__ import annotations

TITLE = "Triton kernel perf audit on gfx1151 ROCm (A/A across identical-kernel states)"
MODE = "regression"
NEEDS = ["rocm", "gpu", "nvidia", "multi_gpu"]
WORSE_FRAC = 0.25
MIN_MS = 0.05


def gates(obs):
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        e = o.get("env") or {}
        out.append((f"{name} HIP torch on gfx1151", bool(e.get("hip")) and "gfx1151" in str(e.get("arch")),
                    f"{e or o.get('env_error', '')}"[:300]))
        out.append((f"{name} deps cloned", "deps" in o, str(o.get("deps_error", ""))[:200]))
        k = o.get("row_kinds") or {}
        out.append((f"{name} bench emitted timing rows", k.get("time", 0) > 0,
                    f"rc={o.get('bench_rc')} kinds={k} secs={o.get('bench_secs')}"))
    b, h = (obs.get("base") or {}).get("kernel_sha1"), (obs.get("head") or {}).get("kernel_sha1")
    out.append(("kernel files identical between states", bool(b) and b == h, ""))
    return out


def table(obs):
    rows = ["| state | torch | triton | timing rows | errors | bench s |", "|---|---|---|---|---|---|"]
    for name in ("base", "head"):
        o = obs.get(name) or {}
        e = o.get("env") or {}
        k = o.get("row_kinds") or {}
        rows.append(f"| {name} | {e.get('torch')} | {e.get('triton')} | {k.get('time', 0)} | {k.get('error', 0)} | {o.get('bench_secs', 0):.0f} |")
    return "\n".join(rows)


def head_is_worse(base, head):
    b, h = base.get("ours_ms") or {}, head.get("ours_ms") or {}
    bad = []
    for k, hv in h.items():
        bv = b.get(k)
        if bv and hv and max(bv, hv) >= MIN_MS and hv > bv * (1 + WORSE_FRAC):
            bad.append(f"{k}: {bv:.3g} -> {hv:.3g} ms")
    if bad:
        return True, "A/A spread above %d%% on %d entries (measurement noise, not a code change): %s" % (
            WORSE_FRAC * 100, len(bad), "; ".join(bad[:8]))
    return False, "A/A within %d%% on all %d shared entries" % (WORSE_FRAC * 100, len(set(b) & set(h)))
