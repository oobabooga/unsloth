#!/usr/bin/env python3
"""Criteria (K3-03, block swap lossless bf16 fetch codec). REGRESSION mode, base vs head on gfx1151.
head_is_worse: a test id failing at head that did not fail at base (by id; the new test file is absent at base, so
any failure in it counts), or the new file ran nothing at head. Gates: both states ran pytest (exit 0 / 1) on a HIP
gfx1151 device, the head collected the new file and at least one GPU round trip test passed there (the decode ran on
the card bitwise, torch path at minimum)."""
from __future__ import annotations

TITLE = "K3-03 block swap bf16 fetch codec, base vs head on gfx1151"
MODE = "regression"
NEEDS = ["rocm", "gpu", "nvidia"]
NEW = "test_block_swap_compress"
GPU_IDS = ("test_device_decode_is_bitwise[False]", "test_device_decode_is_bitwise[True]", "test_self_check_passes")


def _ids(o, k):
    v = (o or {}).get(k) or []
    return set(v if isinstance(v, list) else list(v))


def gates(obs):
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        d = o.get("device") or {}
        out.append((f"{name}: torch sees a HIP gfx1151 device", bool(d.get("hip")) and "gfx1151" in str(d.get("arch")),
                    str(d)[:300]))
        out.append((f"{name}: pytest ran (exit 0 / 1)", o.get("rc") in (0, 1),
                    f"rc={o.get('rc')} {str((o.get('pytest') or {}).get('stderr_tail'))[:300]}"))
    h = obs.get("head") or {}
    new_ran = [t for t in _ids(h, "passed") | _ids(h, "failed") if NEW in t]
    out.append(("head: new codec test file collected", len(new_ran) > 0, f"{len(new_ran)} ran"))
    gpu_pass = [t for t in _ids(h, "passed") if NEW in t and any(t.endswith(g) for g in GPU_IDS)]
    out.append(("head: a GPU decode round trip passed on the card", len(gpu_pass) > 0, ", ".join(gpu_pass) or "none"))
    return out


def table(obs):
    rows = ["| state | passed | failed | skipped | triton | failing tests |", "|---|---|---|---|---|---|"]
    for name in ("base", "head"):
        o = obs.get(name)
        if not o:
            continue
        f = sorted(_ids(o, "failed") | _ids(o, "errors"))
        rows.append(f"| {name} | {o.get('n_passed')} | {o.get('n_failed')} | {o.get('n_skipped')} | "
                    f"{(o.get('device') or {}).get('triton')} | {', '.join('`' + x.split('::')[-1] + '`' for x in f) or 'none'} |")
    h = obs.get("head") or {}
    sk = {k: v for k, v in ((h.get("skipped") or {}).items() if isinstance(h.get("skipped"), dict) else []) if NEW in k}
    if sk:
        rows.append("\nHead skips in the new file: " + "; ".join(f"`{k.split('::')[-1]}` ({v})" for k, v in sk.items()))
    return "\n".join(rows)


def head_is_worse(base, head):
    b = _ids(base, "failed") | _ids(base, "errors")
    h = _ids(head, "failed") | _ids(head, "errors")
    new = sorted(h - b)
    if new:
        return True, "newly failing at the head: " + ", ".join(f"`{n.split('::')[-1]}`" for n in new)
    if not [t for t in _ids(head, "passed") if NEW in t]:
        return True, "the new codec tests passed nothing at the head"
    return False, f"no test that passed at the base fails at the head; fixed at head: {sorted(b - h)}"
