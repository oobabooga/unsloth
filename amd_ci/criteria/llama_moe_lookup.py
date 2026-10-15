#!/usr/bin/env python3
"""Criteria: does the MoE cache keep its device lookup under --load-mode mmap?

Defect (base, unslothai/llama.cpp #268 head): with mmap the host experts sit in a
read-only file mapping that hipHostRegister refuses, so init_device_lookup logs
"cannot map the host experts" and decode falls back to host lookups (slow).
Fix (head = #268 + materialize patch): the mapped experts are copied into
anonymous memory, the lookup runs on the device, decode tracks --load-mode none.
The kill switch arm (LLAMA_MOE_CACHE_MATERIALIZE=0) must reproduce the defect.
"""

from __future__ import annotations

import statistics

TITLE = "llama.cpp #268 MoE cache device lookup under mmap (gfx1151)"
MODE = "differential"
NEEDS = ["rocm", "gpu", "integrated_gpu", "windows", "nvidia", "multi_gpu"]


def _runs(state: dict, arm: str) -> list:
    return ((state.get("arms") or {}).get(arm) or {}).get("runs") or []


def _med(state: dict, arm: str):
    v = [r["decode_tps"] for r in _runs(state, arm) if r.get("decode_tps")]
    return statistics.median(v) if v else None


def _all(state: dict, arm: str, key: str, pred) -> bool:
    rs = _runs(state, arm)
    return bool(rs) and all(pred(r.get(key) or 0) for r in rs)


def gates(obs: dict) -> list:
    out = []
    for name, st in obs.items():
        if name.startswith("_"):
            continue
        runs = [r for a in (st.get("arms") or {}).values() for r in a["runs"]]
        ok = bool(runs) and all(r.get("rc") == 0 and r.get("decode_tps") for r in runs)
        out.append((f"{name}: every run exited 0 with a timing line", ok,
                    f"{sum(1 for r in runs if r.get('rc') == 0)}/{len(runs)} rc=0"))
        gpu = any("ROCm0" in ln or "gfx1151" in ln for r in runs for ln in r.get("lines", []))
        out.append((f"{name}: ROCm device in the logs", gpu, "ROCm0/gfx1151 seen" if gpu else "no ROCm device line"))
    return out


def base_shows_defect(base: dict) -> bool:
    mm, no = _med(base, "mmap"), _med(base, "none")
    return (_all(base, "mmap", "n_fallback", lambda x: x > 0)
            and _all(base, "mmap", "n_device_lookup", lambda x: x == 0)
            and mm is not None and no is not None and mm < 0.8 * no)


def head_is_fixed(head: dict) -> bool:
    mm, no = _med(head, "mmap"), _med(head, "none")
    return (_all(head, "mmap", "n_materialize", lambda x: x > 0)
            and _all(head, "mmap", "n_fallback", lambda x: x == 0)
            and _all(head, "mmap", "n_device_lookup", lambda x: x > 0)
            and mm is not None and no is not None and mm >= 0.85 * no
            # kill switch reproduces the defect
            and _all(head, "mmap_mat0", "n_fallback", lambda x: x > 0)
            and _all(head, "mmap_mat0", "n_materialize", lambda x: x == 0))


def table(obs: dict) -> str:
    rows = ["| state | arm | load mode | env | tok/s per rep | median | spread | md5 | fallback / materialize / device-lookup | load_mode logged |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for name, st in obs.items():
        if name.startswith("_"):
            continue
        for arm, a in (st.get("arms") or {}).items():
            rs = a["runs"]
            v = [r.get("decode_tps") for r in rs]
            good = [x for x in v if x]
            med = statistics.median(good) if good else None
            spread = (max(good) - min(good)) if good else None
            md5 = sorted({r["md5"][:10] for r in rs})
            flags = "; ".join(f"{r.get('n_fallback')}/{r.get('n_materialize')}/{r.get('n_device_lookup')}" for r in rs)
            lm = sorted({str(r.get("load_mode_logged")) for r in rs})
            env = ",".join(f"{k}={v}" for k, v in a.get("env", {}).items()) or "-"
            rows.append(f"| {name} | {arm} | {a['mode']} | {env} | {', '.join(str(x) for x in v)} | "
                        f"{med if med is None else round(med, 2)} | {spread if spread is None else round(spread, 2)} | "
                        f"{' '.join(md5)} | {flags} | {' '.join(lm)} |")
    return "\n".join(rows)
