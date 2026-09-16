#!/usr/bin/env python3
"""Criteria: does the base dequantise against a stale stream, and does the head stop?

unsloth#10563. The defect is only observable if this host can run bitsandbytes 4-bit
kernels at all. If it cannot, the gate fails and the run is INCONCLUSIVE, which is the
correct answer: a comparison whose base condition was never established is no result
rather than a passing one.

Pairs with probes/bnb_stream_probe.py.
"""

from __future__ import annotations

TITLE = "bitsandbytes dequantisation under a non-default stream"
MODE = "differential"
NEEDS = ["gpu", "rocm", "bitsandbytes_cuda", "nvidia", "multi_gpu", "mig", "xpu", "mlx",
         "windows"]


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name, v in obs.items():
        if name.startswith("_"):
            continue
        out.append((f"{name}: bitsandbytes imported", v.get("bnb_error") is None,
                    v.get("bnb_error") or f"bitsandbytes {v.get('bnb_version')}"))
        out.append((f"{name}: the 4-bit measurement ran",
                    v.get("side_stream_iters") is not None,
                    v.get("measure_error") or f"{v.get('side_stream_iters')} iterations"))
    return out


def table(obs: dict) -> str:
    rows = ["| state | CUDA_STREAMS built at import | `CUDA_STREAMS[` uses | bnb | "
            "side-stream mismatches | max abs diff |", "|---|---|---|---|---|---|"]
    for name, v in obs.items():
        if name.startswith("_"):
            continue
        s = v.get("source") or {}
        rows.append(f"| {name} | {s.get('builds_CUDA_STREAMS_at_import')} | "
                    f"{s.get('n_CUDA_STREAM_uses')} | {v.get('bnb_version') or v.get('bnb_error', 'absent')} | "
                    f"{v.get('side_stream_mismatches', '-')}/{v.get('side_stream_iters', '-')} | "
                    f"{v.get('side_stream_max_abs_diff', '-')} |")
    return "\n".join(rows)


def base_shows_defect(base: dict) -> tuple[bool, str]:
    n = base.get("side_stream_mismatches")
    if n is None:
        return False, ("the base measurement never ran, so the race was not established here: "
                       + str(base.get("measure_error") or base.get("bnb_error"))[:200])
    if n == 0:
        return False, (f"0 of {base.get('side_stream_iters')} side-stream dequantisations "
                       f"differed from the default-stream reference on this chip, so the base "
                       f"does not exhibit the defect and the differential is VOID")
    return True, (f"{n} of {base.get('side_stream_iters')} side-stream dequantisations differed "
                  f"(max abs {base.get('side_stream_max_abs_diff')})")


def head_is_fixed(head: dict) -> tuple[bool, str]:
    n = head.get("side_stream_mismatches")
    if n is None:
        return False, "the head measurement never ran"
    return n == 0, (f"{n} of {head.get('side_stream_iters')} side-stream dequantisations differed "
                    f"(max abs {head.get('side_stream_max_abs_diff')})")
