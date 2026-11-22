#!/usr/bin/env python3
"""Criteria for probes/issues_npu_apu_probe.py on Windows (unsloth#18805, with #18763 riding along).

Defect: a full-offload model the real carve-out verifier credits in full still gets a host-shortfall
message at the launch-time preflight. Fixed: those cells are clear at head, while partial-offload
and over-carve-out controls and the NPU model list are unchanged from base."""

from __future__ import annotations

TITLE = "unsloth#18805 Windows APU launch-time host shortfall (+ #18763 NPU list), base versus head"
MODE = "differential"
NEEDS = ["windows", "windows_rocm_wddm", "exclusive_gpu", "discrete_gpu", "multi_gpu"]
GiB = 1024**3
_OBS: dict = {}


def _states(obs: dict) -> dict:
    return {k: v for k, v in obs.items() if not k.startswith("_")}


def _fits(cell: dict) -> bool:
    return cell.get("case") == "full" and (cell.get("credit") or 0) >= 0.95 * cell["gib"] * GiB


def _cells(o: dict) -> dict:
    return {(c["case"], c["gib"]): c for c in ((o.get("apu") or {}).get("cells") or [])}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    _OBS.clear()
    _OBS.update(obs)
    out = []
    for name, o in _states(obs).items():
        apu = o.get("apu") or {}
        t = apu.get("torch") or {}
        out.append((f"{name}: apu probe ran", "fatal" not in apu and bool(apu.get("cells")),
                    str(apu.get("fatal") or f"{len(apu.get('cells') or [])} cells")[:300]))
        out.append((f"{name}: ROCm torch sees the GPU", bool(t.get("hip")) and bool(t.get("available")), str(t)[:200]))
        out.append((f"{name}: APU classified unified", bool(apu.get("unified_ids")), str(apu.get("unified_ids"))))
        out.append((f"{name}: registry carve-out read", (apu.get("carve_out_bytes") or 0) > 0,
                    f"{(apu.get('carve_out_bytes') or 0) / GiB:.1f} GiB"))
        out.append((f"{name}: WDDM used counters read", apu.get("wddm_used_bytes") is not None,
                    str(apu.get("wddm_used_bytes"))))
        b = apu.get("ballast") or {}
        avail = b.get("avail_after_mib") or 0
        out.append((f"{name}: Windows RAM held below the carve-out", 0 < avail <= 12 * 1024,
                    f"{b.get('avail_before_mib')} -> {avail} MiB, {b.get('held_gib')} GiB held"))
        ff = apu.get("forces_full") or {}
        out.append((f"{name}: -ngl 99 is full offload, -ngl 2 is not", ff.get("full") is True and ff.get("partial") is False, str(ff)))
        npu = (o.get("npu") or {}).get("catalog")
        out.append((f"{name}: NPU catalog() ran", npu is not None, str(npu)[:200]))
    return out


def _msg(c: dict | None) -> str:
    if not c:
        return "-"
    if c.get("error"):
        return "ERROR " + c["error"][:60]
    return "warn" if c.get("message") else "none"


def table(obs: dict) -> str:
    base, head = obs.get("base") or {}, obs.get("head") or {}
    ba = base.get("apu") or {}
    rows = [f"carve-out {(ba.get('carve_out_bytes') or 0) / GiB:.1f} GiB, WDDM used "
            f"{(ba.get('wddm_used_bytes') or 0) / GiB:.1f} GiB, gpu rows {ba.get('get_gpu_memory')}, "
            f"avail {ba.get('avail_mib')} MiB, torch {ba.get('torch')}", ""]
    rows += ["| GiB | case | verifier credit GiB | base | head | base ms | head ms |", "|---|---|---|---|---|---|---|"]
    bc, hc = _cells(base), _cells(head)
    for key in sorted(set(bc) | set(hc), key = lambda k: (k[0], k[1])):
        b, h = bc.get(key), hc.get(key)
        rows.append(f"| {key[1]} | {key[0]} | {((b or {}).get('credit') or 0) / GiB:.1f} | {_msg(b)} | {_msg(h)} "
                    f"| {(b or {}).get('ms')} | {(h or {}).get('ms')} |")
    sample = next((c.get("message") for c in bc.values() if c.get("message")), None)
    if sample:
        rows += ["", f"base message sample: `{sample[:300]}`"]
    rows += ["", "| state | NPU catalog() | flm list stderr | after JSON |", "|---|---|---|---|"]
    for name, o in _states(obs).items():
        n = o.get("npu") or {}
        lst = (n.get("raw") or {}).get("list") or {}
        rows.append(f"| {name} | {n.get('catalog')} | `{(lst.get('stderr') or '').strip()[:200]!r}` "
                    f"| `{(lst.get('after') or '').strip()[:200]!r}` |")
    rows += ["", f"NPU host: `{(base.get('npu') or {}).get('host')}`"]
    return "\n".join(rows)


def base_shows_defect(base: dict) -> tuple[bool, str]:
    hits = [c["gib"] for c in _cells(base).values() if _fits(c) and c.get("message")]
    if hits:
        return True, f"base warns for fully credited full-offload sizes {hits} GiB"
    return False, "no fully credited full-offload size warned at base"


def head_is_fixed(head: dict) -> tuple[bool, str]:
    base = _OBS.get("base") or {}
    bc, hc = _cells(base), _cells(head)
    bad = []
    for key, h in hc.items():
        b = bc.get(key) or {}
        if h.get("error") and not b.get("error"):
            bad.append(f"{key} errors at head: {h['error'][:80]}")
        elif _fits(h):
            if h.get("message"):
                bad.append(f"{key} still warns")
        elif bool(h.get("message")) != bool(b.get("message")):
            bad.append(f"control {key}: base {_msg(b)} head {_msg(h)}")
    bn = (base.get("npu") or {}).get("catalog") or {}
    hn = (head.get("npu") or {}).get("catalog") or {}
    if bn.get("ok") and not hn.get("ok"):
        bad.append(f"NPU list fails at head only: {hn.get('error')}")
    if bad:
        return False, "; ".join(bad)
    return True, "fully credited sizes are clear at head; partial and over-carve-out controls match base"
