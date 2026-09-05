#!/usr/bin/env python3
"""Criteria: does the head prebuilt clear a corruption the base prebuilt shows?

Judges only. Pairs with probes/qwen4exp_backend_probe.py. Differential mode:
the base must show the defect or the run is VOID. A "defect" is any corruption
signature in the multi-segment (c2) or four-slot unified-cache (c3) cells, a
decode error in the server log, a slot that produced no output, or cross-slot
text sharing. The single-segment cell (c1) is the negative control and must be
clean in every state, or the harness itself is what is broken.
"""
from __future__ import annotations

TITLE = "qwen4exp UD-IQ1_S on gfx1151: base vs head prebuilt"
MODE = "differential"
# What the reports touch, not what this host has: Windows + WDDM, discrete RDNA,
# NVIDIA (the CUDA controls ran elsewhere), multiple GPUs.
NEEDS = ["gpu", "rocm", "vulkan", "integrated_gpu", "discrete_gpu", "windows",
         "windows_rocm_wddm", "nvidia", "multi_gpu", "multi_gpu_amd"]

BAD_LOG = ("inconsistent sequence positions", "GGML_ASSERT", "failed to decode", "nan", "NaN")

def _states(obs: dict) -> dict:
    return {k: v for k, v in obs.items() if not k.startswith("_")}

def _cell(st: dict, name: str) -> dict:
    return ((st.get("cells") or {}).get(name)) or {}

def _defects(st: dict) -> list[str]:
    out: list[str] = []
    for name in ("c2", "c3"):
        c = _cell(st, name)
        for i, p in enumerate(c.get("prompts") or []):
            if p.get("signatures"):
                out.append(f"{name}[{i}] {','.join(p['signatures'])}")
            if not p.get("ok"):
                out.append(f"{name}[{i}] request failed: {p.get('error')}")
            elif (p.get("n_tokens") or 0) < 8:
                out.append(f"{name}[{i}] produced {p.get('n_tokens')} tokens")
        bad = {k: v for k, v in (c.get("log_sig_counts") or {}).items() if k in BAD_LOG}
        if bad:
            out.append(f"{name} server log: {bad}")
        if c.get("cross_slot_shared_60char"):
            out.append(f"{name} cross-slot shared text between slots "
                       + ", ".join(f"{i}/{j}" for i, j, _ in c["cross_slot_shared_60char"]))
    return out

def _c1_clean(st: dict) -> tuple[bool, str]:
    c = _cell(st, "c1"); ps = c.get("prompts") or []
    if not ps:
        return False, "c1 never ran"
    p = ps[0]
    ok = bool(p.get("ok")) and (p.get("n_tokens") or 0) >= 8 and not p.get("signatures")
    return ok, f"n_tokens={p.get('n_tokens')} sigs={p.get('signatures')} err={p.get('error')}"

def gates(obs: dict) -> list[tuple[str, bool, str]]:
    states = _states(obs); out = []
    launched = {n: bool(v.get("version")) and not v.get("setup_error") for n, v in states.items()}
    out.append(("every state ran its binary", all(launched.values()),
                "; ".join(f"{n}: {v.get('setup_error') or v.get('error') or (v.get('version') or ['?'])[0]}" for n, v in states.items())))
    builds = {n: (v.get("version") or ["?"])[0] for n, v in states.items()}
    out.append(("base and head are different builds", len(set(builds.values())) == len(builds),
                "; ".join(f"{n}={b}" for n, b in builds.items())))
    c1 = {n: _c1_clean(v) for n, v in states.items()}
    out.append(("single-segment control (c1) clean in every state", all(ok for ok, _ in c1.values()),
                "; ".join(f"{n}: {d}" for n, (_, d) in c1.items())))
    sent = {n: (v.get("sentinel") or {}) for n, v in states.items()}
    pre_ok = all((s.get("pre") or {}).get("clean", True) for s in sent.values())
    out.append(("known-good sentinel clean before each state (GPU not poisoned)", pre_ok,
                "; ".join(f"{n}: pre={(s.get('pre') or {}).get('clean','n/a')} post={(s.get('post') or {}).get('clean','n/a')}" for n, s in sent.items())))
    return out

def table(obs: dict) -> str:
    rows = ["| state | build | UMA env | c1 | c2 | c3 | defects | sentinel pre/post |", "|---|---|---|---|---|---|---|---|"]
    for n, v in _states(obs).items():
        def cs(name):
            ps = (_cell(v, name).get("prompts") or [])
            return "/".join(str(p.get("n_tokens")) for p in ps) if ps else "-"
        env = [e for e in (v.get("env") or []) if "UNIFIED" in e] or (["unset"] if "GGML_CUDA_ENABLE_UNIFIED_MEMORY" in (v.get("unset_env") or []) else ["absent"])
        s = v.get("sentinel") or {}
        rows.append(f"| {n} | {(v.get('version') or ['?'])[0][:40]} | {env[0]} | {cs('c1')} | {cs('c2')} | {cs('c3')} | "
                    f"{'; '.join(_defects(v)) or 'none'} | {(s.get('pre') or {}).get('clean','-')}/{(s.get('post') or {}).get('clean','-')} |")
    return "\n".join(rows)

def base_shows_defect(base: dict) -> tuple[bool, str]:
    d = _defects(base)
    return bool(d), "; ".join(d) or "no corruption signature, no decode error, every slot produced output"

def head_is_fixed(head: dict) -> tuple[bool, str]:
    d = _defects(head)
    return not d, "; ".join(d) or "clean on every cell"
