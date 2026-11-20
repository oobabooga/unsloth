#!/usr/bin/env python3
"""Criteria for probes/ps1_rocm_install_probe.py.

Defect (base): on a Windows gfx1151 box whose ROCm torch cannot be installed (AMD multi-arch index unreachable),
install.ps1 leaves a CPU torch in the Studio venv without an [ERROR] that names ROCm, i.e. the install reads as a
normal one while every GPU arm runs on CPU.

Fixed (head): the same broken-index install ends with the [ERROR] naming gfx1151 and the reason, exits non-zero,
and setup never installs CPU torch; the plain install still gives ROCm torch that sees the gfx1151 GPU (same as base),
and the setup.ps1 unit test passes under Windows PowerShell 5.1.
"""

from __future__ import annotations

TITLE = "install.ps1 on Windows gfx1151: ROCm torch, and a loud failure instead of a silent CPU torch"
MODE = "differential"
NEEDS = ["windows", "gpu", "rocm"]


def _t(arm: dict) -> dict:
    return (arm or {}).get("torch") or {}


def _is_rocm_gpu(arm: dict) -> bool:
    t = _t(arm)
    return bool(t.get("hip")) and "+rocm" in str(t.get("version")) and bool(t.get("is_available")) \
        and str(t.get("arch") or "").startswith("gfx1151") and bool(t.get("matmul_on_gpu"))


def _is_cpu(arm: dict) -> bool:
    t = _t(arm)
    return bool(t.get("version")) and not t.get("hip") and not t.get("is_available")


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        out.append((f"{name}: probe ran to the end", not o.get("probe_error") and "broken" in o,
                    o.get("probe_error") or "ok"))
        out.append((f"{name}: runs on Windows", o.get("platform") == "win32", str(o.get("platform"))))
        n = o.get("normal") or {}
        out.append((f"{name}: plain install.ps1 gives ROCm torch on the gfx1151 GPU", _is_rocm_gpu(n),
                    f"rc {n.get('rc')}, torch {_t(n).get('version')}, hip {_t(n).get('hip')}, "
                    f"avail {_t(n).get('is_available')}, arch {_t(n).get('arch')}"))
        b = o.get("broken") or {}
        out.append((f"{name}: broken-index install really could not reach the ROCm index",
                    not b.get("says_gpu_rocm_installed") and "+rocm" not in str(_t(b).get("version")),
                    f"rc {b.get('rc')}, torch {_t(b).get('version')}"))
        u = o.get("unit") or {}
        out.append((f"{name}: unit test ran under 5.1", (u.get("pass", 0) + u.get("fail", 0)) > 0,
                    f"{u.get('pass')} pass / {u.get('fail')} fail, rc {u.get('rc')}"))
    return out


def table(obs: dict) -> str:
    rows = ["| state | arm | install.ps1 exit | torch | hip | cuda.is_available | arch | ROCm [ERROR] | "
            "'GPU ROCm PyTorch installed' | setup installed CPU torch |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for name in ("base", "head"):
        o = obs.get(name) or {}
        for arm in ("normal", "broken"):
            a = o.get(arm) or {}
            t = _t(a)
            rows.append(f"| {name} | {arm} | {a.get('rc')} | `{t.get('version')}` | {t.get('hip')} | "
                        f"{t.get('is_available')} | {t.get('arch')} | {a.get('says_rocm_error')} | "
                        f"{a.get('says_gpu_rocm_installed')} | {a.get('setup_installed_cpu')} |")
    rows.append("")
    for name in ("base", "head"):
        u = (obs.get(name) or {}).get("unit") or {}
        rows.append(f"- {name} unit test (Windows PowerShell 5.1): {u.get('pass')} pass, {u.get('fail')} fail, "
                    f"rc {u.get('rc')}")
    lp = (obs.get("base") or {}).get("lpips") or {}
    if lp:
        rows.append(f"- run 44279815226 shape (bare venv + `pip install lpips`, no install.ps1): torch "
                    f"`{_t(lp).get('version')}`, hip {_t(lp).get('hip')}, avail {_t(lp).get('is_available')}")
    for name in ("base", "head"):
        for arm in ("normal", "broken"):
            lines = ((obs.get(name) or {}).get(arm) or {}).get("lines") or []
            rows.append("")
            rows.append(f"<details><summary>{name} {arm}: detection / torch lines</summary>\n\n```")
            rows.extend(ln[:220] for ln in lines[:45])
            rows.append("```\n</details>")
    return "\n".join(rows)


def base_shows_defect(base: dict) -> tuple[bool, str]:
    b = base.get("broken") or {}
    cpu = _is_cpu(b)
    loud = bool(b.get("says_rocm_error"))
    ok = cpu and not loud
    return ok, (f"broken index: install.ps1 exit {b.get('rc')}, torch {_t(b).get('version')} (cpu={cpu}), "
                f"ROCm [ERROR] printed={loud}, setup installed CPU torch={b.get('setup_installed_cpu')}")


def head_is_fixed(head: dict) -> tuple[bool, str]:
    b = head.get("broken") or {}
    loud = bool(b.get("says_rocm_error"))
    failed = b.get("rc") not in (0, None)
    no_cpu = not b.get("setup_installed_cpu")
    normal = _is_rocm_gpu(head.get("normal") or {})
    u = head.get("unit") or {}
    unit = u.get("rc") == 0 and u.get("fail") == 0 and u.get("pass", 0) > 0
    ok = loud and failed and no_cpu and normal and unit
    return ok, (f"broken index: [ERROR] printed={loud}, install.ps1 exit {b.get('rc')} (non-zero={failed}), "
                f"setup did not install CPU torch={no_cpu}; plain install ROCm on gfx1151={normal}; "
                f"5.1 unit test {u.get('pass')} pass / {u.get('fail')} fail")
