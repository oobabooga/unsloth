#!/usr/bin/env python3
"""Criteria: does the Windows ROCm GPU monitor see a HIP load and the adapter temperature? (#18005)

Pairs with probes/win_gpu_monitor_probe.py. Defect: the utilization reading counts only 3D
engines, which a HIP matmul leaves idle, and no temperature source exists. Base shows it when,
under a load the independent per-LUID counter sees at >= 50 %, the checkout's utilization reads
below 20 % (or unknown) and no temperature is reported. Fixed when the checkout's reading is
>= 50 % and within 25 points of the independent busiest engine type, a call fits the 5 s
spawn timeout, and the temperature is a plausible 20-110 C reading.
"""

from __future__ import annotations

TITLE = "Windows ROCm GPU monitor: utilization under a HIP load, adapter temperature"
MODE = "differential"
NEEDS = ["windows", "rocm", "gpu", "integrated_gpu", "discrete_gpu", "multi_gpu", "linux"]


def _states(obs: dict):
    return {n: v for n, v in obs.items() if not n.startswith("_")}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    st = _states(obs)
    out = [("every state probed without error", all(not v.get("error") and v.get("luid") for v in st.values()),
            "; ".join(f"{n}: {(v.get('error') or '').splitlines()[-1:] or 'ok'}" for n, v in st.items()))]
    out.append(("HIP torch on gfx1151", all(v.get("hip") and str(v.get("arch", "")).startswith("gfx1151")
                                            for v in st.values()),
                ", ".join(f"{n}: torch={v.get('torch')} hip={v.get('hip')} arch={v.get('arch')}" for n, v in st.items())))
    out.append(("the HIP load ran through every sample",
                all((v.get("load_ops") or 0) > 0 and v.get("load_alive_after_sampling") for v in st.values()),
                ", ".join(f"{n}: ops={v.get('load_ops')} alive={v.get('load_alive_after_sampling')}"
                          for n, v in st.items())))
    out.append(("independent counter saw the load (busiest type >= 50 %)",
                all((v.get("load_raw_busiest_median") or 0) >= 50 for v in st.values()),
                ", ".join(f"{n}: busiest={v.get('load_raw_busiest_median')} 3d={v.get('load_raw_3d_median')}"
                          for n, v in st.items())))
    return out


def table(obs: dict) -> str:
    rows = ["| state | phase | util fn | payload util | raw busiest type | raw 3D | temp fn | payload temp | util call max s |",
            "|---|---|---|---|---|---|---|---|---|"]
    for n, v in _states(obs).items():
        for ph in ("idle", "load"):
            rows.append(f"| {n} | {ph} | {v.get(ph + '_util_fn_median')} | {v.get(ph + '_payload_util_median')} | "
                        f"{v.get(ph + '_raw_busiest_median')} | {v.get(ph + '_raw_3d_median')} | "
                        f"{v.get(ph + '_temp_fn_median')} | {v.get(ph + '_payload_temp_median')} | "
                        f"{v.get(ph + '_util_fn_s_max')} |")
    rows.append("")
    for n, v in _states(obs).items():
        load = (v.get("load") or [{}])[0]
        rows.append(f"- {n}: all-engine Get-Counter {v.get('all_engines_query_s')} s; per-LUID "
                    f"{(load.get('raw') or {}).get('seconds')} s; types {(load.get('raw') or {}).get('types')}; "
                    f"payload backend {(load.get('payload') or {}).get('backend')}")
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    util = base.get("load_util_fn_median")
    return (util is None or util < 20) and base.get("load_payload_temp_median") is None


def head_is_fixed(head: dict) -> bool:
    util, raw = head.get("load_util_fn_median"), head.get("load_raw_busiest_median")
    temp = head.get("load_temp_fn_median")
    return (util is not None and raw is not None and util >= 50 and abs(util - min(raw, 100.0)) <= 25
            and (head.get("load_util_fn_s_max") or 99) < 5.0
            and temp is not None and 20 <= temp <= 110)
