#!/usr/bin/env python3
"""Criteria: does the checkout give Studio an INT8 attention path on this AMD GPU that is at least as fast as
comfy-kitchen's HIP INT8 kernel at every shape and inside the accuracy gate? Judges probes/int8_attn_amd_probe.py.

Accuracy gate (vs fp64 on 3 heads x 256 query rows): q / k gain 1: cos >= 0.999 and rel L1 <= 0.05; gain 5:
cos >= 0.99; every output finite at gains 1 / 5 / 8. Gain 8 is reported, not gated.

Noise per shape: max(A/A |median(sdpa_aa) / median(sdpa) - 1|, spread of sdpa's per-process medians, 2%).
"On par" = the candidate's median is at most comfy-kitchen's median times (1 + noise).

The candidate is the checkout's ``studio`` op when it ran the INT8 kernel (its call counter moved), else the best of
the checkout's Triton arms (tri_i8i8, tri_i8h) that passes the gate, reported as such: only the studio op counts
for head_is_fixed.

base_shows_defect: on the base the Studio op does not run INT8 attention on this GPU (no op, or 0 INT8 calls).
head_is_fixed: the Studio op ran INT8 (calls > 0), passes the accuracy gate at every shape and is on par with or
faster than ck_int8 at every shape.
"""

from __future__ import annotations

TITLE = "gfx1151: INT8 attention vs comfy-kitchen HIP INT8 and torch SDPA (timing, exclusive GPU)"
MODE = "differential"
NEEDS = ["gpu", "rocm", "exclusive_gpu", "nvidia", "discrete_gpu", "windows", "windows_rocm_wddm", "multi_gpu"]
CANDIDATES = ("tri_i8h_skt", "tri_i8ht", "tri_i8i8t")
SHOWN = ("sdpa", "sdpa_aa", "sdpa_contig", "sdpa_kv", "sdpa_flash", "sdpa_eff", "ck_int8", "ck_sol", "sage_tri",
         "aiter_fa", "aiter_sage", "aiter_sage_rot", "tri_half", *CANDIDATES, "studio")


def _states(obs: dict) -> dict:
    return {n: v for n, v in obs.items() if not n.startswith("_")}


def _summary(st: dict) -> dict:
    return st.get("summary") or {}


def _acc(st: dict) -> dict:
    return {s: c.get("accuracy", {}) for s, c in ((st.get("accuracy") or {}).get("cells") or {}).items()}


def _aa(cell: dict) -> float:
    s, aa = cell.get("sdpa"), cell.get("sdpa_aa")
    if not s or not aa:
        return 1.0
    return abs(aa["median"] / s["median"] - 1)


def _noise(cell: dict, arm: str = "") -> float:
    """Noise of a candidate vs ck_int8 comparison: the process spread of either arm, at least 2%. SDPA's own spread
    (up to ~30% at 17k tokens on this APU) is reported as the A/A column, not charged to the INT8 arms."""
    spreads = [(cell.get(a) or {}).get("spread") or 0.0 for a in ("ck_int8", arm) if a]
    return max(*spreads, 0.02)


def gate_ok(acc: dict) -> tuple[bool, str]:
    g1, g5, g8 = acc.get("g1", {}), acc.get("g5", {}), acc.get("g8", {})
    if any("error" in g for g in (g1, g5, g8)) or not g1:
        return False, "error"
    finite = all(g.get("nonfinite", 1) == 0 for g in (g1, g5, g8))
    ok = finite and g1["cos"] >= 0.999 and g1["rel_l1"] <= 0.05 and g5["cos"] >= 0.99
    return ok, f"cos {g1['cos']:.5f} relL1 {g1['rel_l1']:.4f} sharp5 {g5['cos']:.4f} sharp8 {g8.get('cos', 0):.4f}"


def _studio_int8_calls(st: dict) -> int:
    calls = 0
    for t in st.get("timing") or []:
        calls += int((t.get("studio_calls_rocm") or t.get("studio_calls") or {}).get("int8", 0) or 0)
    return calls


def _on_par(st: dict, arm: str) -> list[tuple[str, float | None, bool]]:
    rows = []
    acc = _acc(st)
    for shape, cell in _summary(st).items():
        if arm not in cell or "ck_int8" not in cell:
            rows.append((shape, None, False))
            continue
        ratio = cell[arm]["median"] / cell["ck_int8"]["median"]
        ok_acc = gate_ok(acc.get(shape, {}).get(arm, {}))[0]
        rows.append((shape, ratio, ratio <= 1 + _noise(cell, arm) and ok_acc))
    return rows


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    st = _states(obs)
    out = []
    archs = {k: (v.get("accuracy") or {}).get("arch") for k, v in st.items()}
    out.append(("runner is gfx1151 in every state", all(a == "gfx1151" for a in archs.values()), str(archs)))
    measured = {k: sum(1 for c in _summary(v).values() if "sdpa" in c) for k, v in st.items()}
    out.append(("every state timed all shapes with SDPA", all(n >= 6 for n in measured.values()), str(measured)))
    ck = {k: sum(1 for c in _summary(v).values() if "ck_int8" in c) for k, v in st.items()}
    out.append(("comfy-kitchen HIP INT8 timed at every shape in every state", all(n >= 6 for n in ck.values()),
                str(ck) + " " + str({k: (v.get("accuracy") or {}).get("ck_error") for k, v in st.items()})))
    procs = {k: len(v.get("timing") or []) for k, v in st.items()}
    out.append(("3 fresh timing processes per state", all(n >= 3 for n in procs.values()), str(procs)))
    return out


def table(obs: dict) -> str:
    st = _states(obs)
    rows = []
    for name, v in st.items():
        acc_info = v.get("accuracy") or {}
        rows.append(f"### {name}: torch {acc_info.get('torch')} hip {acc_info.get('hip')} triton {acc_info.get('triton')} "
                    f"ck {acc_info.get('ck_version')} aiter {v.get('aiter_commit', '')[:10]} configs {v.get('configs')}")
        rows.append("")
        summ, acc = _summary(v), _acc(v)
        arms = [a for a in SHOWN if any(a in c for c in summ.values())]
        rows.append("| shape | SDPA A/A | " + " | ".join(arms) + " |")
        rows.append("|---|---|" + "---|" * len(arms))
        for shape, cell in summ.items():
            vals = []
            for a in arms:
                if a not in cell:
                    vals.append("-")
                    continue
                g = acc.get(shape, {}).get(a, {})
                ok, _ = gate_ok(g)
                c1 = g.get("g1", {}).get("cos")
                c5 = g.get("g5", {}).get("cos")
                vals.append(f"{cell[a]['median']:.2f} ({cell[a]['spread'] * 100:.0f}%) cos {c1 if c1 is None else round(c1, 5)}"
                            f" s5 {c5 if c5 is None else round(c5, 4)}{'' if ok else ' FAILGATE'}")
            rows.append(f"| {shape} | {_aa(cell) * 100:.1f}% | " + " | ".join(vals) + " |")
        rows.append("")
        for a in CANDIDATES + ("studio",):
            par = _on_par(v, a)
            if any(r is not None for _, r, _ in par):
                rows.append(f"{a} / ck_int8: " + ", ".join(
                    f"{s}={'n/a' if r is None else f'{r:.3f}'}{'' if ok else ' (not on par)'}" for s, r, ok in par))
        rt = v.get("rocm_tests") or {}
        rows.append(f"ROCm tests: present={rt.get('present')} rc={rt.get('rc')} gpu passed/failed(all)/skipped = "
                    f"{_gpu_tests(v)}; failing: {[k for k, o in (rt.get('cases') or {}).items() if o == 'failed']}")
        rows.append(f"studio INT8 calls: {_studio_int8_calls(v)}; studio verdict: "
                    f"{(v.get('accuracy') or {}).get('studio_verdict')}; errors: "
                    f"{ {k: (v.get('accuracy') or {}).get(k) for k in ('studio_error', 'triton_port_error', 'aiter_fa_error', 'aiter_sage_error', 'sage_error', 'ck_error')} }")
        rows.append("")
    rows.append("Cells: median ms over 3 fresh processes (spread across process medians), accuracy vs fp64 "
                "(cos at gain 1, s5 = cos at q / k gain 5). FAILGATE = outside cos >= 0.999, rel L1 <= 0.05, "
                "sharp cos >= 0.99, finite.")
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    return _studio_int8_calls(base) == 0


def _gpu_tests(st: dict) -> tuple[int, int, int]:
    cases = (st.get("rocm_tests") or {}).get("cases") or {}
    gpu = {k: v for k, v in cases.items() if "card" in k or "strided_views" in k}
    return (sum(v == "passed" for v in gpu.values()), sum(v == "failed" for v in cases.values()),
            sum(v == "skipped" for v in gpu.values()))


def head_is_fixed(head: dict) -> bool:
    if _studio_int8_calls(head) == 0:
        return False
    passed, failed, skipped = _gpu_tests(head)
    if failed or skipped or passed < 10:
        return False
    par = _on_par(head, "studio")
    return bool(par) and all(ok for _, _, ok in par)
