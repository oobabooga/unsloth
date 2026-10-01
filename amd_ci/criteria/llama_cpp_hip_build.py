#!/usr/bin/env python3
"""Criteria: does install_llama_cpp(gpu_support=True) build a HIP llama.cpp on ROCm?

unsloth-zoo PR 512. Base passes -DGGML_CUDA=ON to cmake on a ROCm host, which
cannot configure without nvcc, so the source build fails. Head passes
-DGGML_HIP=ON (+ HIP compiler / GPU_TARGETS) and should configure, build and
link the ggml HIP backend.

Gates (non-vacuity): both arms really ran on ROCm torch with a visible GPU, both
reached the cmake configure step (else something earlier, e.g. a missing system
package, decided the outcome for both), and the host has a ROCm HIP dev toolkit
(else a head failure says nothing about the change).
"""

from __future__ import annotations

TITLE = "unsloth-zoo #512: llama.cpp source build with gpu_support=True on ROCm (gfx1151)"
MODE = "differential"
NEEDS = ["rocm", "gpu", "nvidia", "windows"]


def _states(obs: dict):
    return {n: v for n, v in obs.items() if not n.startswith("_")}


def _hip_toolkit(v: dict) -> tuple[bool, str]:
    tk = v.get("toolkit") or {}
    ok = bool(tk.get("which_hipconfig") or tk.get("which_hipcc") or tk.get("rocm_cmake_configs"))
    return ok, (f"root={tk.get('rocm_root_checked')} exists={tk.get('rocm_root_exists')} "
                f"hipconfig={tk.get('which_hipconfig')} hipcc={tk.get('which_hipcc')} "
                f"cmake_cfgs={len(tk.get('rocm_cmake_configs') or [])}")


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    st = _states(obs)
    out = []
    rocm = {n: (v.get("torch") or {}) for n, v in st.items()}
    out.append(("ROCm torch with a visible gfx GPU in every state",
                bool(st) and all(t.get("hip") and t.get("cuda_available") and t.get("detect_gpu_target")
                                 for t in rocm.values()),
                "; ".join(f"{n}: hip={t.get('hip')} target={t.get('detect_gpu_target')}" for n, t in rocm.items())))
    out.append(("every state reached cmake configure",
                bool(st) and all(v.get("configure_line") for v in st.values()),
                "; ".join(f"{n}: {'yes' if v.get('configure_line') else 'NO: ' + str(v.get('exception_head') or v.get('load_error'))[:300]}"
                          for n, v in st.items())))
    tk = [(n, *_hip_toolkit(v)) for n, v in st.items()]
    out.append(("host has a ROCm HIP dev toolkit", bool(tk) and all(ok for _, ok, _ in tk),
                "; ".join(f"{n}: {ev}" for n, _, ev in tk)))
    return out


def table(obs: dict) -> str:
    rows = ["| state | configure GPU flags | configure ok | build ok | install ok | llama-quantize links | CMakeCache GGML_HIP | ggml-hip built | --list-devices |",
            "|---|---|---|---|---|---|---|---|---|"]
    for n, v in _states(obs).items():
        cl = v.get("configure_line") or ""
        flags = " ".join(t for t in cl.split() if t.startswith(("-DGGML_CUDA", "-DGGML_HIP", "-DCMAKE_HIP", "-DGPU_TARGETS")))
        links = v.get("links") or {}
        linked = ",".join(k for k, b in links.items() if b) or "-"
        cache = (v.get("cmake_cache_before_rm") or v.get("cmake_cache_after_fail") or {}).get("GGML_HIP", "-")
        ld = (v.get("list_devices") or {}).get("out", "")
        dev = " / ".join(l.strip() for l in ld.splitlines() if "ROCm" in l or "CUDA" in l)[:160] or "-"
        rows.append(f"| {n} | `{flags or '-'}` | {v.get('configure_ok')} | {v.get('build_cmd_ok')} | "
                    f"{v.get('install_ok')} | {linked} | `{cache}` | {v.get('ggml_hip_dir_built')} | {dev} |")
    rows.append("")
    for n, v in _states(obs).items():
        if not v.get("install_ok"):
            rows.append(f"- {n} exception: `{str(v.get('exception_head'))[:600]!s}`")
    return "\n".join(rows)


def base_shows_defect(base: dict):
    cl = base.get("configure_line") or ""
    shown = "-DGGML_CUDA=ON" in cl and "GGML_HIP" not in cl and not base.get("install_ok")
    return shown, f"configure={cl[:200]!r} install_ok={base.get('install_ok')}"


def head_is_fixed(head: dict):
    cl = head.get("configure_line") or ""
    links = head.get("links") or {}
    cache = (head.get("cmake_cache_before_rm") or {}).get("GGML_HIP", "")
    hip_evidence = bool(links.get("libamdhip64")) or "GGML_HIP:BOOL=ON" in cache
    fixed = ("-DGGML_HIP=ON" in cl and "-DGGML_CUDA=ON" not in cl and bool(head.get("configure_ok"))
             and bool(head.get("build_cmd_ok")) and bool(head.get("install_ok")) and hip_evidence)
    return fixed, f"configure={cl[:200]!r} build={head.get('build_cmd_ok')} hip_evidence={hip_evidence}"
