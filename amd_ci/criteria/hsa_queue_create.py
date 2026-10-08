#!/usr/bin/env python3
"""Criteria: does the bundle's ROCm runtime create a HIP queue when libhsakmt's
topology-derived wave count disagrees with amdkfd's (unslothai/unsloth#12205)?

amdkfd validates the CWSR+debug area two ways (kfd_queue.c):
  * BO path (HSA_USE_SVM=0, or no SVM support): the mapping must EQUAL the size
    KFD computes from 32 waves/CU, so any disagreement is rejected.
  * SVM path (default where supported): the range must only COVER that size, so
    an oversized area passes and only an undersized one is rejected.
max_waves_per_simd is spoofed through an fopen shim (KFD keeps its real value):
20 gives 40 waves/CU (oversized), 8 gives 16 (undersized).

Defect (base): real topology runs on the GPU on both paths, while spoof20 on the
BO path and spoof8 on the SVM path abort with "ROCm error: out of memory" at
hipStreamCreateWithFlags.
Fixed (head): every mode runs on the GPU, and the real-topology generation is
identical to the base's (greedy, same seed): no change where it already works.
spoof20 on the SVM path is expected to pass on BOTH sides and is reported, not judged.
"""

from __future__ import annotations

TITLE = "HIP queue creation with spoofed KFD max_waves_per_simd (#12205)"
MODE = "differential"
# RDNA1/RDNA2 silicon whose firmware reports != 16 waves/SIMD, and NixOS's kernel
# config, are what the issue needs and are not here.
NEEDS = ["gpu", "rocm", "linux", "discrete_gpu"]

ALL_MODES = ("real", "real_nosvm", "spoof20", "spoof20_nosvm", "spoof8")
DEFECT_MODES = ("spoof20_nosvm", "spoof8")
CONTROL_MODES = ("real", "real_nosvm")

_CTX: dict = {"base_real_text": None}


def _states(obs: dict) -> dict:
    return {n: v for n, v in obs.items() if not n.startswith("_")}


def _ran_on_gpu(run: dict) -> bool:
    # Not the "offloaded N/N" line: llama.cpp prints it with every layer on the CPU.
    return (run.get("rc") == 0 and bool(run.get("generated"))
            and not run.get("rocm_init_failed")
            and (run.get("layers_rocm") or 0) > 0 and (run.get("layers_cpu") or 0) == 0
            and (run.get("rocm_model_buffer_mib") or 0) > 0)


def _queue_fail(run: dict) -> bool:
    return run.get("rc") != 0 and bool(run.get("queue_create_oom"))


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    st = _states(obs)
    out = []
    nodes = {n: v.get("kfd_gpu_nodes") or [] for n, v in st.items()}
    first = next(iter(nodes.values()), [])
    out.append(("KFD topology readable, one GPU node",
                all(len(x) == 1 for x in nodes.values()),
                "; ".join(f"{n}: {x}" for n, x in nodes.items())))
    real_waves = first[0].get("max_waves_per_simd") if first else None
    out.append(("real max_waves_per_simd is 16 (so 20 and 8 are genuine disagreements)",
                real_waves == 16, f"real={real_waves}"))
    eng_ok, eng = True, []
    for n, v in st.items():
        for m in ALL_MODES:
            r = v.get(m) or {}
            want = r.get("spoof_waves") is not None
            got = len(r.get("spoof_rewrites") or [])
            eng_ok &= (got > 0) == want
            eng.append(f"{n}/{m}={got}")
    out.append(("shim rewrote topology in exactly the spoof runs", eng_ok, ", ".join(eng)))
    prov_ok, prov = True, []
    for n, v in st.items():
        b = v.get("bundle", "?")
        for m in ALL_MODES:
            got = (v.get(m) or {}).get("hsa_loaded") or []
            ok = len(got) == 1 and got[0] == f"{b}/libhsa-runtime64.so.1"
            prov_ok &= ok
            if not ok:
                prov.append(f"{n}/{m}: {got or 'none'}")
    out.append(("every run initialised its own state's libhsa-runtime64", prov_ok,
                "; ".join(prov) or "all runs"))
    shas = {n: v.get("hsa_sha256") for n, v in st.items()}
    out.append(("base and head runtimes differ", len(set(shas.values())) == len(shas),
                ", ".join(f"{n}={str(s)[:12]}" for n, s in shas.items())))
    msha = {v.get("model_sha256") for v in st.values()}
    out.append(("same model in every state", len(msha) == 1 and None not in msha,
                str(sorted(str(m)[:12] for m in msha))))
    _CTX["base_real_text"] = (st.get("base", {}).get("real") or {}).get("generated")
    return out


def base_shows_defect(base: dict):
    ctrl = {m: _ran_on_gpu(base.get(m) or {}) for m in CONTROL_MODES}
    if not all(ctrl.values()):
        return False, f"a real-topology control failed on the base: {ctrl}"
    fails = {m: _queue_fail(base.get(m) or {}) for m in DEFECT_MODES}
    return all(fails.values()), f"queue-create failure per defect mode: {fails}"


def head_is_fixed(head: dict):
    gpu = {m: _ran_on_gpu(head.get(m) or {}) for m in ALL_MODES}
    text = (head.get("real") or {}).get("generated")
    same = bool(text) and text == _CTX["base_real_text"]
    return all(gpu.values()) and same, f"on GPU per mode: {gpu}; real text identical to base: {same}"


def table(obs: dict) -> str:
    rows = ["| state | mode | spoofed waves/SIMD | HSA_USE_SVM | rc | placement | queue-create OOM | generated (first 50 chars) |",
            "|---|---|---|---|---|---|---|---|"]
    for n, v in _states(obs).items():
        for m in ALL_MODES:
            r = v.get(m) or {}
            gen = (r.get("generated") or "").replace("|", "/").replace("\n", " ")[:50]
            placed = f"{r.get('layers_rocm', 0)} ROCm / {r.get('layers_cpu', 0)} CPU, {r.get('rocm_model_buffer_mib', 0):.2f} MiB"
            rows.append(f"| {n} | {m} | {r.get('spoof_waves') or '-'} | {r.get('hsa_use_svm') or 'default'} | "
                        f"{r.get('rc')} | {placed} | "
                        f"{'yes' if r.get('queue_create_oom') else 'no'} | {gen} |")
    return "\n".join(rows)
