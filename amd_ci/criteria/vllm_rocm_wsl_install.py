#!/usr/bin/env python3
"""Criteria (Windows): does Studio run vLLM on the AMD GPU through its private WSL distro (PR #12785)?

Base: Studio refuses vLLM on this Windows + AMD host. Head: the managed install (WSL distro,
ROCm + librocdxg, vLLM lock) succeeds and a greedy chat completion answers "Paris".
"""

from __future__ import annotations

TITLE = "Managed vLLM on Windows + AMD through Studio's WSL distro"
MODE = "differential"
NEEDS = ["gpu", "windows", "windows_rocm_wddm", "multi_gpu", "nvidia", "discrete_gpu", "mig", "xpu", "mlx"]


def _states(obs):
    return {n: v for n, v in obs.items() if not n.startswith("_")}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    states = _states(obs)
    imported = all("import_error" not in v for v in states.values())
    video = {n: ((v.get("host") or {}).get("video") or {}).get("out", "") for n, v in states.items()}
    return [
        ("every state imported its engine installer", imported,
         "; ".join(f"{n}: {v.get('import_error')}" for n, v in states.items() if "import_error" in v)),
        ("an AMD Radeon GPU is present", all("Radeon" in t for t in video.values()),
         " | ".join(t[:120] for t in video.values())),
    ]


def _gen(state: dict) -> dict:
    return (state.get("generations") or [{}])[0]


def table(obs: dict) -> str:
    rows = ["| state | WSL active | gpu_platform | vLLM verdict | SGLang verdict | install | answer |",
            "|---|---|---|---|---|---|---|"]
    for n, v in _states(obs).items():
        g = _gen(v)
        answer = repr((g.get("text") or "")[:40]) if g.get("text") is not None else (g.get("error") or g.get("body") or "-")[:160]
        rows.append(
            f"| {n} | {v.get('wsl_active')} | {v.get('gpu_platform')} | {str(v.get('vllm_reason'))[:160]} | "
            f"{str(v.get('sglang_reason'))[:80]} | {v.get('install_ok', '-')} {v.get('install_s', '')}s | {answer} |"
        )
    for n, v in _states(obs).items():
        h = v.get("host") or {}
        rows += ["", f"{n} host: admin={h.get('admin')} hypervisor={(h.get('virtualization') or {}).get('out', '').strip()}",
                 "```", ((h.get("wsl_status") or {}).get("out") or str(h.get("wsl_status")))[:800], "```"]
    head = _states(obs).get("head") or {}
    if head.get("install_error"):
        rows += ["", "head install error:", "```", head["install_error"][-2000:],
                 "\n".join(head.get("install_log_tail") or [])[-2500:], "```"]
    g = _gen(head)
    if g.get("error"):
        rows += ["", "head load error:", "```", (g.get("traceback") or g["error"])[-1500:],
                 "\n".join(g.get("engine_tail") or [])[-2500:], "```"]
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    return bool(base.get("vllm_reason")) and not base.get("install_ok")


def head_is_fixed(head: dict) -> bool:
    return (
        head.get("vllm_reason") is None
        and head.get("install_ok") is True
        and (head.get("installed_info") or {}).get("platform") == "rocm"
        and "paris" in (_gen(head).get("text") or "").lower()
    )
