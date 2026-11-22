#!/usr/bin/env python3
"""Criteria for probes/issues_npu_apu_probe.py --no-ballast on a Windows Strix Halo whose carve-out
is small (the pool's boxes): the launch-time preflight must give base's answer for every size and
offload case, cost base's time on loads that do not warn, and the NPU list must still work."""

from __future__ import annotations

TITLE = "unsloth#18805 head on a small-carve-out Windows APU: same answers, no added cost (+ #18763)"
MODE = "regression"
NEEDS = ["windows", "windows_rocm_wddm", "exclusive_gpu", "discrete_gpu", "multi_gpu"]
GiB = 1024**3
EXTRA_MS = 50


def _states(obs: dict) -> dict:
    return {k: v for k, v in obs.items() if not k.startswith("_")}


def _cells(o: dict) -> dict:
    return {(c["case"], c["gib"]): c for c in ((o.get("apu") or {}).get("cells") or [])}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name, o in _states(obs).items():
        apu = o.get("apu") or {}
        t = apu.get("torch") or {}
        out.append((f"{name}: apu probe ran", "fatal" not in apu and bool(apu.get("cells")),
                    str(apu.get("fatal") or f"{len(apu.get('cells') or [])} cells")[:300]))
        out.append((f"{name}: ROCm torch sees the GPU", bool(t.get("hip")) and bool(t.get("available")), str(t)[:160]))
        out.append((f"{name}: APU classified unified", bool(apu.get("unified_ids")), str(apu.get("unified_ids"))))
        out.append((f"{name}: host RAM readable", bool(apu.get("avail_mib")), str(apu.get("avail_mib"))))
        warned = sum(1 for c in _cells(o).values() if c.get("message"))
        out.append((f"{name}: some sizes warn (the verifier path is reached)", warned > 0, f"{warned} warning cells"))
        out.append((f"{name}: NPU catalog() ran", (o.get("npu") or {}).get("catalog") is not None,
                    str((o.get("npu") or {}).get("catalog"))[:160]))
    return out


def _msg(c):
    if not c:
        return "-"
    return "ERROR" if c.get("error") else ("warn" if c.get("message") else "none")


def table(obs: dict) -> str:
    base, head = obs.get("base") or {}, obs.get("head") or {}
    ba = base.get("apu") or {}
    rows = [f"carve-out {(ba.get('carve_out_bytes') or 0) / GiB:.2f} GiB, gpu rows {ba.get('get_gpu_memory')}, "
            f"avail {ba.get('avail_mib')} MiB", "",
            "| GiB | case | credit GiB | base | head | base ms | head ms |", "|---|---|---|---|---|---|---|"]
    bc, hc = _cells(base), _cells(head)
    for key in sorted(set(bc) | set(hc)):
        b, h = bc.get(key) or {}, hc.get(key) or {}
        rows.append(f"| {key[1]} | {key[0]} | {(b.get('credit') or 0) / GiB:.2f} | {_msg(b)} | {_msg(h)} | {b.get('ms')} | {h.get('ms')} |")
    rows += ["", "| state | NPU catalog() |", "|---|---|"]
    for name, o in _states(obs).items():
        rows.append(f"| {name} | {(o.get('npu') or {}).get('catalog')} |")
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    bc, hc = _cells(base), _cells(head)
    bad = []
    for key, b in bc.items():
        h = hc.get(key)
        if h is None:
            bad.append(f"{key} missing at head")
            continue
        if _msg(h) != _msg(b):
            bad.append(f"{key}: base {_msg(b)} head {_msg(h)}")
        elif not b.get("message") and (h.get("ms") or 0) > (b.get("ms") or 0) + EXTRA_MS:
            bad.append(f"{key}: non-warning load costs {h.get('ms')} ms at head vs {b.get('ms')} ms")
    bn = (base.get("npu") or {}).get("catalog") or {}
    hn = (head.get("npu") or {}).get("catalog") or {}
    if bn.get("ok") and not hn.get("ok"):
        bad.append(f"NPU list fails at head: {hn.get('error')}")
    if bad:
        return True, "; ".join(bad)
    warn_ms = [(k, b.get("ms"), hc[k].get("ms")) for k, b in bc.items() if b.get("message")]
    return False, ("every cell gives base's answer; non-warning loads cost the same; warning loads "
                   f"(base ms, head ms): {[(k[1], k[0], b, h) for k, b, h in warn_ms][:6]}")
