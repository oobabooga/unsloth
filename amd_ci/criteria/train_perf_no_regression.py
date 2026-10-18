#!/usr/bin/env python3
"""Criteria: did a merged unsloth / unsloth-zoo change regress single-GPU LoRA training
speed, peak memory, loss or compile behaviour on THIS host?

Pairs with probes/train_perf_probe.py. Regression mode. Each state is measured
REPEATS times (each a fresh process with its own compile cache), so the A/A spread
of the median step time is observed rather than assumed.

NO_REGRESSION requires, base versus head on the same host in the same job:

  * losses: every head repeat within LOSS_TOL (1e-3 abs, widened only to the observed
    A/A loss spread if that is larger) of the base repeat with the same index, per step;
  * speed: head median step time (median over repeats of the per-repeat median, first
    WARMUP steps excluded) not above base by more than max(5%, observed A/A spread);
  * memory: head peak max_memory_allocated (max over repeats) not above base by more
    than max(2% of base, 0.05 GB);
  * compile: head graph breaks (max over repeats) <= base (max over repeats);
  * identity: model dtype, SUPPORTS_BFLOAT16, is_bf16_supported(), _detect_gfx_arch(),
    get_amd_attention_implementation() identical.

Gates make it non-vacuous: both states ran every repeat and wrote observations, each
imported the package under test FROM ITS OWN CHECKOUT, the other package from the
same installed file at both states, base and head are different commits, the base
is a gfx1151 HIP host, and every repeat produced the requested number of finite
losses and step times.
"""

from __future__ import annotations

import math
import statistics

TITLE = "Single-GPU LoRA training speed / peak memory / loss on real gfx1151, base versus head"
MODE = "regression"
# What the changes touch: the HIP path (measured), the NVIDIA capability / TMA gates,
# mixed and multi-GPU hosts (per-device caches), Windows (llama_cpp target, install).
NEEDS = ["linux", "rocm", "gpu", "nvidia", "multi_gpu", "windows"]

EXPECT_ARCH = "gfx1151"
WARMUP = 5
LOSS_TOL = 1e-3
TIME_TOL = 0.05
MEM_REL = 0.02
MEM_ABS_GB = 0.05
IDENTITY = ("model_dtype", "SUPPORTS_BFLOAT16", "is_bf16_supported()", "_detect_gfx_arch",
            "get_amd_attention_implementation", "zoo DEVICE_TYPE", "attn_implementation")


def _runs(o: dict) -> list[dict]:
    return list((o or {}).get("runs") or [])


def _finite(xs) -> bool:
    return bool(xs) and all(isinstance(x, (int, float)) and math.isfinite(x) for x in xs)


def _run_median(r: dict) -> float | None:
    st = r.get("step_times") or []
    st = st[WARMUP:] if len(st) > WARMUP else st
    return statistics.median(st) if st else None


def _state_median(o: dict) -> float | None:
    ms = [m for m in (_run_median(r) for r in _runs(o)) if m is not None]
    return statistics.median(ms) if ms else None


def _aa_time_spread(o: dict) -> float:
    ms = [m for m in (_run_median(r) for r in _runs(o)) if m is not None]
    return (max(ms) - min(ms)) / min(ms) if len(ms) >= 2 and min(ms) > 0 else 0.0


def _aa_loss_spread(o: dict) -> float:
    rs = [r.get("losses") or [] for r in _runs(o)]
    worst = 0.0
    for i in range(len(rs)):
        for j in range(i + 1, len(rs)):
            for a, b in zip(rs[i], rs[j]):
                worst = max(worst, abs(a - b))
    return worst


def _peak(o: dict, key: str = "peak_allocated_gb") -> float | None:
    vs = [r.get(key) for r in _runs(o) if isinstance(r.get(key), (int, float))]
    return max(vs) if vs else None


def _breaks(o: dict) -> int | None:
    vs = [r.get("graph_breaks") for r in _runs(o) if isinstance(r.get("graph_breaks"), int)]
    return max(vs) if vs else None


def _identity(o: dict) -> dict:
    """Identity fields per repeat; a state whose repeats disagree reports them all."""
    out = {}
    for k in IDENTITY:
        vals = []
        for r in _runs(o):
            v = r.get(k) if k in ("model_dtype", "attn_implementation") else (r.get("flags") or {}).get(k)
            vals.append(repr(v))
        out[k] = vals[0] if len(set(vals)) == 1 else "repeats disagree: " + " / ".join(vals)
    return out


def _loss_tol(base: dict, head: dict) -> float:
    return max(LOSS_TOL, _aa_loss_spread(base), _aa_loss_spread(head))


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    other_files = {}
    for name in ("base", "head"):
        o = obs.get(name) or {}
        runs = _runs(o)
        want = (o.get("config") or {}).get("repeats")
        steps = (o.get("config") or {}).get("steps")
        out.append((f"{name} probe wrote observations for every repeat",
                    o.get("_probe_rc") == 0 and not o.get("_missing_output") and not o.get("_parse_error")
                    and not o.get("dry_run") and bool(want) and len(runs) == want,
                    f"probe rc={o.get('_probe_rc')}; repeats {len(runs)}/{want}"
                    + ("; DRY RUN" if o.get("dry_run") else "")))
        pkg = o.get("package")
        other = "unsloth_zoo" if pkg == "unsloth" else "unsloth"
        for r in runs:
            i = r.get("repeat")
            ok = r.get("rc") == 0 and not r.get("error") and not r.get("missing_output")
            out.append((f"{name} r{i} training child completed", ok,
                        f"rc={r.get('rc')}" + (f"; {r.get('error')}" if r.get("error") else "")
                        + ("; missing output" if r.get("missing_output") else "")))
            ov = r.get("overlay") or {}
            out.append((f"{name} r{i} imported {pkg} from its own checkout", bool(ov.get("from_checkout")),
                        f"`{ov.get('file')}`"))
            losses, st = r.get("losses") or [], r.get("step_times") or []
            out.append((f"{name} r{i} has {steps} finite losses and step times",
                        _finite(losses) and _finite(st) and len(losses) == steps and len(st) == steps,
                        f"{len(losses)} losses, {len(st)} step times"))
            other_files.setdefault(other, set()).add((r.get(other) or {}).get("file"))
            if not isinstance(r.get("peak_allocated_gb"), (int, float)):
                out.append((f"{name} r{i} recorded peak memory", False, "no peak_allocated_gb"))
    for other, files in other_files.items():
        out.append((f"{other} (not under test) is the same installed file at every state and repeat",
                    len(files) == 1 and None not in files, ", ".join(f"`{f}`" for f in sorted(map(str, files)))))
    b, h = obs.get("base") or {}, obs.get("head") or {}
    out.append(("base and head are different commits",
                bool(b.get("commit")) and b.get("commit") != h.get("commit"),
                f"{str(b.get('commit'))[:9]} vs {str(h.get('commit'))[:9]}"))
    bd = (_runs(b)[0].get("device") if _runs(b) else None) or {}
    bv = (_runs(b)[0].get("versions") if _runs(b) else None) or {}
    out.append((f"base is a HIP {EXPECT_ARCH} host",
                bool(bv.get("hip")) and str(bd.get("gcn_arch") or "").startswith(EXPECT_ARCH),
                f"hip={bv.get('hip')}; arch={bd.get('gcn_arch')}; name={bd.get('name')}"))
    return out


def _f(v, nd = 4) -> str:
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def table(obs: dict) -> str:
    rows = ["| state | repeat | median step s (excl. first %d) | mean step s | peak alloc GB | peak reserved GB | "
            "graph breaks | unique graphs | train s | first / last loss |" % WARMUP,
            "|---|---|---|---|---|---|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        if name not in obs:
            continue
        for r in _runs(obs[name]):
            st = r.get("step_times") or []
            ls = r.get("losses") or []
            rows.append(f"| {name} | {r.get('repeat')} | {_f(_run_median(r))} | "
                        f"{_f(statistics.mean(st[WARMUP:]) if len(st) > WARMUP else None)} | "
                        f"{_f(r.get('peak_allocated_gb'), 3)} | {_f(r.get('peak_reserved_gb'), 3)} | "
                        f"{_f(r.get('graph_breaks'))} | {_f(r.get('unique_graphs'))} | "
                        f"{_f(r.get('train_seconds'), 1)} | "
                        f"{_f(ls[0] if ls else None)} / {_f(ls[-1] if ls else None)} |")
    b, h = obs.get("base") or {}, obs.get("head") or {}
    rows += ["", "| field | base | head | same |", "|---|---|---|---|"]
    ib, ih = _identity(b), _identity(h)
    for k in IDENTITY:
        rows.append(f"| {k} | `{ib[k]}` | `{ih[k]}` | {'yes' if ib[k] == ih[k] else 'NO'} |")
    for k in ("capability", "capability_per_device_cache"):
        vb = [(r.get("device") or {}).get(k) for r in _runs(b)]
        vh = [(r.get("device") or {}).get(k) for r in _runs(h)]
        rows.append(f"| torch.cuda {k} (informational) | `{vb}` | `{vh}` | |")
    rows.append(f"| A/A median step spread | {_f(_aa_time_spread(b) * 100, 2)}% | "
                f"{_f(_aa_time_spread(h) * 100, 2)}% | |")
    rows.append(f"| A/A max loss diff | {_f(_aa_loss_spread(b), 6)} | {_f(_aa_loss_spread(h), 6)} | |")
    br, hr = _runs(b), _runs(h)
    for i, (rb, rh) in enumerate(zip(br, hr), 1):
        lb, lh = rb.get("losses") or [], rh.get("losses") or []
        d = max((abs(x - y) for x, y in zip(lb, lh)), default = None)
        rows.append(f"| r{i} max abs loss diff base vs head | | | {_f(d, 6)} |")
    lb = (br[0].get("losses") if br else None) or []
    lh = (hr[0].get("losses") if hr else None) or []
    rows.append(f"| r1 losses | `{[round(x, 4) for x in lb]}` | `{[round(x, 4) for x in lh]}` | |")
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    findings: list[str] = []
    notes: list[str] = []

    tol = _loss_tol(base, head)
    if tol > LOSS_TOL:
        notes.append(f"loss tolerance widened from {LOSS_TOL} to the observed A/A loss spread {tol:.2e}")
    worst = 0.0
    for rb, rh in zip(_runs(base), _runs(head)):
        lb, lh = rb.get("losses") or [], rh.get("losses") or []
        if len(lb) != len(lh):
            findings.append(f"r{rh.get('repeat')}: head has {len(lh)} losses vs base {len(lb)}")
            continue
        d = max((abs(x - y) for x, y in zip(lb, lh)), default = 0.0)
        worst = max(worst, d)
        if d > tol:
            findings.append(f"r{rh.get('repeat')}: losses differ by up to {d:.2e} (> {tol:.2e})")

    mb, mh = _state_median(base), _state_median(head)
    spread = max(_aa_time_spread(base), _aa_time_spread(head))
    allow = max(TIME_TOL, spread)
    if mb is None or mh is None:
        findings.append("median step time missing")
    elif mh > mb * (1 + allow):
        findings.append(f"head median step {mh:.4f}s vs base {mb:.4f}s (+{(mh / mb - 1) * 100:.1f}%, "
                        f"allowed +{allow * 100:.1f}%)")

    pb, ph = _peak(base), _peak(head)
    mem_allow = max(MEM_REL * (pb or 0.0), MEM_ABS_GB)
    if pb is None or ph is None:
        findings.append("peak memory missing")
    elif ph > pb + mem_allow:
        findings.append(f"head peak allocated {ph:.3f} GB vs base {pb:.3f} GB (allowed +{mem_allow:.3f} GB)")

    gb, gh = _breaks(base), _breaks(head)
    if gb is None or gh is None:
        findings.append("graph break count missing")
    elif gh > gb:
        findings.append(f"head graph breaks {gh} > base {gb}")

    ib, ih = _identity(base), _identity(head)
    moved = [k for k in IDENTITY if ib[k] != ih[k]]
    if moved:
        findings.append("identity moved at the head: " + ", ".join(f"`{k}` {ib[k]} -> {ih[k]}" for k in moved))

    if findings:
        return True, "; ".join(findings + notes)
    pct = (mh / mb - 1) * 100
    return False, (f"median step base {mb:.4f}s head {mh:.4f}s ({pct:+.1f}%, allowed +{allow * 100:.1f}%, "
                   f"A/A spread {spread * 100:.1f}%); peak allocated base {pb:.3f} GB head {ph:.3f} GB; "
                   f"graph breaks base {gb} head {gh}; max loss diff {worst:.2e} (tol {tol:.0e}); "
                   f"dtype {ib['model_dtype']}, bf16 {ib['SUPPORTS_BFLOAT16']}, gfx {ib['_detect_gfx_arch']}, "
                   f"attention {ib['get_amd_attention_implementation']} identical"
                   + ("; " + "; ".join(notes) if notes else ""))
