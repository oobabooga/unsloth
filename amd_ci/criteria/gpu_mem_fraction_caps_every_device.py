#!/usr/bin/env python3
"""Criteria: does unsloth#11077's memory cap reach every GPU, on real cards?

The defect at the base is that `UNSLOTH_GPU_MEM_FRACTION` caps nothing on NVIDIA:
the only cap the worker had was behind `IS_ROCM`. The head is fixed when, with the
variable set, EVERY visible device reports the fraction and the allocator really
refuses an allocation above it.

Two properties are checked rather than one, because they fail differently. A
fraction that is recorded but not enforced would pass a bookkeeping check; an
allocation that fails on a busy card would pass an enforcement check for the wrong
reason. So the refusal must name the cap ("allowed memory"), and the base must have
allocated the very same size on the very same card moments earlier.

`cuda:0` alone is not enough, and that is the specific regression this run exists
to catch: `set_per_process_memory_fraction(f)` with no device argument caps
`current_device()` only, and a sharded run would have left every later card free.

Pairs with probes/gpu_mem_fraction_probe.py.
"""

from __future__ import annotations

TITLE = "UNSLOTH_GPU_MEM_FRACTION on 8 real NVIDIA cards, base versus head"
MODE = "differential"

NEEDS = ["gpu", "nvidia", "multi_gpu", "discrete_gpu",
         "rocm", "multi_gpu_amd", "windows", "xpu", "mlx", "mig"]


def _alloc(state: dict) -> dict:
    return {a["device"]: a for a in (state.get("allocations") or [])}


def _capped_devices(state: dict, want: float) -> list:
    out = []
    for i, f in enumerate(state.get("fractions_after") or []):
        if isinstance(f, (int, float)) and abs(float(f) - want) < 1e-6:
            out.append(i)
    return out


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    states = {n: v for n, v in obs.items() if not n.startswith("_")}
    for name, v in states.items():
        out.append((f"{name}: torch saw the cards", bool(v.get("cuda_available")),
                    f"torch={v.get('torch_version')} count={v.get('device_count')} "
                    f"{v.get('torch_error') or ''}"))
    base = obs.get("base") or {}
    head = obs.get("head") or {}
    out.append(("more than one real GPU is visible, so 'every device' is a question with "
                "content", (base.get("device_count") or 0) > 1,
                f"device_count={base.get('device_count')}: "
                f"{[d.get('name') for d in (base.get('devices') or [])][:2]}..."))
    out.append(("the fraction getter exists on this torch, so the bookkeeping half is "
                "readable", bool(head.get("has_fraction_getter")),
                f"torch={head.get('torch_version')}"))
    out.append(("the head really executed the shipped block",
                bool(head.get("executed")) and bool(head.get("section_1h_present")),
                f"present={head.get('section_1h_present')} executed={head.get('executed')} "
                f"{str(head.get('exec_error') or '')[:300]}"))
    # Non-vacuity: the probe allocation has to be one the cards can actually serve
    # when uncapped, or "refused" says nothing.
    ba = _alloc(base)
    served = [i for i, a in ba.items() if a.get("allocated")]
    out.append(("uncapped, every card served the probe allocation, so a later refusal is "
                "the cap and not a full card", len(served) == (base.get("device_count") or 0)
                and bool(served),
                f"served {served} of {base.get('device_count')}; sizes "
                f"{sorted({a.get('bytes') for a in ba.values()})}"))
    return out


def table(obs: dict) -> str:
    rows = ["| state | block present | executed | fractions after | log lines |",
            "|---|---|---|---|---|"]
    for name, v in obs.items():
        if name.startswith("_"):
            continue
        rows.append(f"| {name} | {v.get('section_1h_present')} | {v.get('executed')} | "
                    f"{v.get('fractions_after')} | {len(v.get('log_info') or [])} |")
    rows.append("")
    rows.append("| device | card | total GiB | base: allocated | head: allocated | "
                "head refusal |")
    rows.append("|---|---|---|---|---|---|")
    base, head = obs.get("base") or {}, obs.get("head") or {}
    ba, ha = _alloc(base), _alloc(head)
    for d in base.get("devices") or []:
        i = d["index"]
        h = ha.get(i, {})
        why = ("cap" if h.get("refused_by_cap") else
               "OOM" if h.get("refused_as_oom") else
               "-" if h.get("allocated") else str(h.get("error_type")))
        rows.append(f"| cuda:{i} | {d.get('name')} | {d.get('total_bytes', 0) / 1024 ** 3:.0f} | "
                    f"{ba.get(i, {}).get('allocated')} | {h.get('allocated')} | {why} |")
    rows.append("")
    for line in (head.get("log_info") or [])[:12]:
        rows.append(f"    {line}")
    return "\n".join(rows)


def base_shows_defect(base: dict) -> tuple[bool, str]:
    want = base.get("fraction_requested")
    capped = _capped_devices(base, want)
    served = [i for i, a in _alloc(base).items() if a.get("allocated")]
    if capped:
        return False, (f"the base already caps devices {capped} at {want}, so there is no "
                       f"uncapped state to compare against")
    if not served:
        return False, ("the base allocated nothing on any card, so the probe size is not one "
                       "this box can serve and a refusal at the head would prove nothing")
    return True, (f"with {base.get('env_name')}={want} set, the base caps no device "
                  f"(fractions {base.get('fractions_after')}) and every card served an "
                  f"allocation of {base.get('probe_fraction')} of its total: {served}")


def head_is_fixed(head: dict) -> tuple[bool, str]:
    want = head.get("fraction_requested")
    n = head.get("device_count") or 0
    capped = _capped_devices(head, want)
    if len(capped) != n or n == 0:
        return False, (f"the head caps {capped} of {n} devices "
                       f"(fractions {head.get('fractions_after')})")
    allocs = _alloc(head)
    enforced = [i for i, a in allocs.items() if a.get("refused_by_cap")]
    if len(enforced) != n:
        leaked = [i for i, a in allocs.items() if a.get("allocated")]
        return False, (f"the fraction is recorded on all {n} devices but the allocator still "
                       f"served an over-cap allocation on {leaked}; enforced on {enforced}")
    return True, (f"all {n} devices report {want} and all {n} refused an allocation of "
                  f"{head.get('probe_fraction')} of their own total with a cap error, not an "
                  f"OOM. This is one process on one box; it does not speak to ROCm, where the "
                  f"same block is served by section 1g instead")
