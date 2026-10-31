#!/usr/bin/env python3
"""Criteria (T7k.1): DPO + GRPO on gfx1151, BASE (pre-#16071) vs HEAD (main). Pairs with probes/t7k1_rl_probe.py.

Gates (INCONCLUSIVE when any fails): every arm ran and finished; each arm imported unsloth from its own worktree
(revisions.unsloth == that worktree's SHA); the device is a HIP gfx1151 GPU; every job has >= 2 rounds per arm.
REGRESSION when, for a job: any head run failed its job checks; head median steady step ms > base median + base spread
(max - min of the base rounds); head peak memory > base peak * 1.02; head final loss is not deterministic across rounds
while base's is, or (base deterministic) differs from base by more than LOSS_TOL relative (#16071 changes bf16 rounding on purpose).
"""

from __future__ import annotations

import statistics

TITLE = "T7k.1: DPO + GRPO on gfx1151, pre-#16071 base vs main (fused LoRA order)"
MODE = "regression"
NEEDS = ["rocm", "gpu", "discrete_gpu", "nvidia", "multi_gpu"]
LOSS_TOL = 0.05
MEM_TOL = 1.02


def _runs(obs):
    return ((obs.get("head") or {}).get("runs") or [])


def _s(r):
    return (r.get("metrics") or {}).get("summary") or {}


def _by(obs):
    out: dict = {}
    for r in _runs(obs):
        out.setdefault(r["job"], {}).setdefault(r["arm"], []).append(r)
    return out


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    h = obs.get("head") or {}
    runs = _runs(obs)
    shas = h.get("arm_sha") or {}
    out = [("head state ran both arms", bool(runs), f"{len(runs)} runs")]
    fin = [r for r in runs if r.get("rc") != 0 or not (r.get("metrics") or {}).get("finished")]
    out.append(("every arm finished", bool(runs) and not fin,
                "; ".join(f"{r['job']} r{r['round']} {r['arm']} rc={r.get('rc')} {str(r.get('tail'))[-300:]}" for r in fin)))
    own = [r for r in runs if not str(((r.get("metrics") or {}).get("revisions") or {}).get("unsloth") or "").startswith(
        (shas.get(r["arm"]) or "?")[:9])]
    out.append(("each arm imported its own unsloth worktree", bool(runs) and not own,
                f"arm_sha {shas}; mismatches {[(r['job'], r['arm'], ((r.get('metrics') or {}).get('revisions') or {}).get('unsloth')) for r in own][:4]}"))
    dev = {str((r.get("metrics") or {}).get("device")) for r in runs}
    vers = {str(((r.get("metrics") or {}).get("versions") or {}).get("torch")) for r in runs}
    out.append(("HIP GPU device", bool(runs) and all("cuda" in d or "rocm" in d or "hip" in d for d in dev), f"{dev} {vers}"))
    by = _by(obs)
    rounds = {j: {a: len(v) for a, v in arms.items()} for j, arms in by.items()}
    out.append((">= 2 rounds per arm per job", bool(by) and all(min(a.values()) >= 2 and len(a) == 2 for a in rounds.values()),
                f"{rounds}"))
    return out


def _stats(rs, key):
    xs = [_s(r).get(key) for r in rs if _s(r).get(key) is not None]
    return xs


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    bad, notes = [], []
    by = _by({"head": head})
    for job, arms in sorted(by.items()):
        b, h = arms.get("base") or [], arms.get("head") or []
        for r in h:
            if not (r.get("metrics") or {}).get("passed"):
                bad.append(f"{job} head r{r['round']}: job checks failed {(r.get('metrics') or {}).get('checks')}")
        bs, hs = _stats(b, "steady_step_ms"), _stats(h, "steady_step_ms")
        if bs and hs:
            bm, hm, spread = statistics.median(bs), statistics.median(hs), max(bs) - min(bs)
            notes.append(f"{job}: steady ms base {bm:.1f} (spread {spread:.1f}) head {hm:.1f} ({(hm / bm - 1) * 100:+.2f}%)")
            if hm > bm + spread:
                bad.append(f"{job}: head {hm:.1f} ms > base {bm:.1f} + spread {spread:.1f}")
        bp, hp = _stats(b, "peak_mem_gb"), _stats(h, "peak_mem_gb")
        if bp and hp:
            notes.append(f"{job}: peak GB base {max(bp):.4f} head {max(hp):.4f}")
            if max(hp) > max(bp) * MEM_TOL:
                bad.append(f"{job}: peak {max(hp):.4f} > base {max(bp):.4f} * {MEM_TOL}")
        bl, hl = _stats(b, "final_loss"), _stats(h, "final_loss")
        if bl and hl:
            notes.append(f"{job}: final loss base {bl} head {hl}")
            bdet, hdet = len(set(bl)) == 1, len(set(hl)) == 1
            if bdet and not hdet:
                bad.append(f"{job}: head final loss not deterministic across rounds ({hl}) while base is ({bl})")
            ref = statistics.median(bl)
            # Sampled-loss jobs (GRPO) are not deterministic even at the base: reported, not gated.
            if bdet and abs(statistics.median(hl) - ref) > LOSS_TOL * max(abs(ref), 1e-6):
                bad.append(f"{job}: head final loss {statistics.median(hl)} vs base {ref} beyond {LOSS_TOL:.0%}")
    return bool(bad), "; ".join(bad + notes)


def table(obs: dict) -> str:
    rows = ["| job | round | arm | rc | unsloth | steady step ms | first step ms | peak GB | final loss | passed |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for r in _runs(obs):
        m = r.get("metrics") or {}
        s = m.get("summary") or {}
        rows.append(f"| {r['job']} | {r['round']} | {r['arm']} | {r.get('rc')} | {str((m.get('revisions') or {}).get('unsloth'))[:9]} | "
                    f"{s.get('steady_step_ms')} | {s.get('first_step_ms')} | {s.get('peak_mem_gb')} | {s.get('final_loss')} | "
                    f"{m.get('passed')} |")
    return "\n".join(rows)
