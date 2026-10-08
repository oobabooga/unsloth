#!/usr/bin/env python3
"""Criteria: does the bundle's ROCm runtime create a HIP queue when libhsakmt's
topology-derived wave count disagrees with amdkfd's (unslothai/unsloth#12205)?

Defect shape: with max_waves_per_simd spoofed to 20 (2 SIMDs/CU -> 40 waves/CU,
against KFD's fixed 32), the unpatched thunk sizes the CWSR debug area larger
than amdkfd expects, AMDKFD_IOC_CREATE_QUEUE fails, and llama.cpp aborts with
"ROCm error: out of memory" at hipStreamCreateWithFlags. The same base state must
run normally on the real topology, or the spoof broke something else.

Fixed: the head runs on BOTH topologies, offloads every layer to the GPU, and on
the real topology generates exactly what the base generated (greedy, same seed):
the patch must not change anything on a GPU that already works.
"""

from __future__ import annotations

TITLE = "HIP queue creation with spoofed KFD max_waves_per_simd (#12205)"
MODE = "differential"
# The change touches the ROCm runtime on Linux AMD GPUs. RDNA1/RDNA2 silicon whose
# firmware really reports 20 waves/SIMD is what the issue needs and is not here.
NEEDS = ["gpu", "rocm", "linux", "discrete_gpu"]

_CTX: dict = {"base_real_text": None}


def _states(obs: dict) -> dict:
    return {n: v for n, v in obs.items() if not n.startswith("_")}


def _ran_on_gpu(run: dict) -> bool:
    off = run.get("offloaded")
    return (run.get("rc") == 0 and bool(off) and off[0] == off[1] and off[1] > 0
            and bool(run.get("generated")))


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    st = _states(obs)
    out = []
    nodes = {n: v.get("kfd_gpu_nodes") or [] for n, v in st.items()}
    first = next(iter(nodes.values()), [])
    out.append(("KFD topology readable, one GPU node",
                all(len(x) == 1 for x in nodes.values()),
                "; ".join(f"{n}: {x}" for n, x in nodes.items())))
    real_waves = first[0].get("max_waves_per_simd") if first else None
    spoof_vals = {v["spoof"].get("spoof_waves") for v in st.values() if v.get("spoof")}
    out.append(("spoofed value differs from the real one",
                real_waves is not None and all(s and int(s) != real_waves for s in spoof_vals),
                f"real={real_waves} spoof={sorted(spoof_vals)}"))
    eng = {n: (len(v.get("spoof", {}).get("spoof_rewrites") or []),
               len(v.get("real", {}).get("spoof_rewrites") or [])) for n, v in st.items()}
    out.append(("shim rewrote topology in every spoof run and in no real run",
                all(s > 0 and r == 0 for s, r in eng.values()),
                ", ".join(f"{n}: spoof={s} real={r}" for n, (s, r) in eng.items())))
    prov_ok, prov = True, []
    for n, v in st.items():
        want = v.get("bundle", "") + "/libhsa-runtime64.so.1"
        for mode in ("real", "spoof"):
            got = (v.get(mode) or {}).get("hsa_loaded") or []
            ok = len(got) == 1 and got[0].endswith("libhsa-runtime64.so.1") and \
                got[0].startswith(v.get("bundle", "?"))
            prov_ok &= ok
            prov.append(f"{n}/{mode}: {'ok' if ok else got or 'none'}")
    out.append(("every run initialised its own state's libhsa-runtime64", prov_ok,
                "; ".join(prov)))
    shas = {n: v.get("hsa_sha256") for n, v in st.items()}
    out.append(("base and head runtimes differ", len(set(shas.values())) == len(shas),
                ", ".join(f"{n}={str(s)[:12]}" for n, s in shas.items())))
    msha = {v.get("model_sha256") for v in st.values()}
    out.append(("same model in every state", len(msha) == 1 and None not in msha,
                str(sorted(str(m)[:12] for m in msha))))
    _CTX["base_real_text"] = (st.get("base", {}).get("real") or {}).get("generated")
    return out


def base_shows_defect(base: dict):
    real, spoof = base.get("real") or {}, base.get("spoof") or {}
    if not _ran_on_gpu(real):
        return False, "base failed on the REAL topology too, so the spoof is not the cause"
    if spoof.get("rc") == 0:
        return False, "base created its queue despite the spoofed wave count"
    return bool(spoof.get("queue_create_oom")), \
        f"spoof rc={spoof.get('rc')} errors={spoof.get('any_rocm_error')}"


def head_is_fixed(head: dict):
    real, spoof = head.get("real") or {}, head.get("spoof") or {}
    same = real.get("generated") == _CTX["base_real_text"] and bool(real.get("generated"))
    ok = _ran_on_gpu(real) and _ran_on_gpu(spoof) and same
    return ok, (f"real gpu={_ran_on_gpu(real)} spoof gpu={_ran_on_gpu(spoof)} "
                f"real text identical to base={same}")


def table(obs: dict) -> str:
    rows = ["| state | topology | rc | layers on GPU | queue-create OOM | ROCm error | generated (first 60 chars) |",
            "|---|---|---|---|---|---|---|"]
    for n, v in _states(obs).items():
        for mode in ("real", "spoof"):
            r = v.get(mode) or {}
            topo = "real" if mode == "real" else f"spoofed {r.get('spoof_waves')} waves/SIMD"
            off = r.get("offloaded")
            err = (r.get("any_rocm_error") or [""])[0]
            gen = (r.get("generated") or "").replace("|", "/").replace("\n", " ")[:60]
            rows.append(f"| {n} | {topo} | {r.get('rc')} | {f'{off[0]}/{off[1]}' if off else '-'} | "
                        f"{'yes' if r.get('queue_create_oom') else 'no'} | {err} | {gen} |")
    return "\n".join(rows)
