#!/usr/bin/env python3
"""Criteria: unsloth-zoo fused-linear unit tests on a WINDOWS Strix Halo box, zoo base vs head, GPU run.

Pairs with probes/zoo_pytest_probe.py --diag (zoo_pytest_plugin.py inside pytest, zoo_env_diag.py outside it).
On Windows, AMD's torch wheels are expected to carry no Triton, so the head should take the torch fallback path of the
fused linear engine: that is REPORTED (CALLS torch vs triton, _HAS_TRITON, _use_triton), not required either way.
Gates (INCONCLUSIVE when any fails):
  - pytest exited 0/1 and collected tests at base and head; each state imported unsloth_zoo from its own checkout
  - head: a GPU is visible inside pytest (torch.cuda.is_available(); HIP or otherwise, named in the evidence)
  - head: >= 1 cuda-parametrised test RAN (passed or failed; a skip is not a run)
  - head: the fused linear engine ran inside pytest (CALLS torch + triton > 0)
REGRESSION = a test id failing / erroring at the head and not at the base (tests added by the PR are absent at the base,
so any head failure in them counts). For a host whose torch has no GPU, the workflow re-decides the same observations
with criteria/zoo_win_tests_cpu.py, a clearly labelled CPU run.
"""

from __future__ import annotations

import os

TITLE = "unsloth-zoo fused-linear unit tests on Windows (Strix Halo), zoo base vs head: GPU run"
MODE = "regression"
NEEDS = ["windows", "rocm", "gpu", "discrete_gpu", "nvidia", "multi_gpu", "mig", "xpu", "mlx"]
_RAN = (0, 1)


def _states(obs):
    return {n: v for n, v in obs.items() if not n.startswith("_") and not (v or {}).get("skipped")}


def _norm(p: str) -> str:
    return os.path.normcase(os.path.abspath(p)) if p else ""


def _cuda_ran(o: dict) -> list:
    return [k for k in (o.get("cuda_passed") or [])] + [k for k in (o.get("failed") or []) + (o.get("errors") or [])
                                                        if "cuda" in k.split("[", 1)[-1]]


def common_gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name, o in _states(obs).items():
        rc = o.get("rc")
        n = sum((o.get(k) or 0) for k in ("n_passed", "n_failed", "n_skipped", "n_errors"))
        out.append((f"{name}: pytest ran and collected", rc in _RAN and n > 0,
                    f"rc={rc}, {n} outcomes; {o.get('error') or o.get('note') or ''} {str(o.get('stderr_tail'))[-200:]}"))
        f, co = o.get("unsloth_zoo_file") or "", o.get("checkout") or ""
        out.append((f"{name}: unsloth_zoo imported from its own checkout", bool(f) and _norm(f).startswith(_norm(co)),
                    f"{f} (checkout {co}, zoo {str(o.get('zoo_sha'))[:9]})"))
    head = obs.get("head") or {}
    calls = head.get("calls") or {}
    out.append(("head: fused linear engine ran inside pytest (CALLS torch + triton > 0)",
                (calls.get("torch", 0) or 0) + (calls.get("triton", 0) or 0) > 0,
                f"CALLS {calls}; tests that launched Triton: {len(head.get('triton_tests') or [])}"))
    return out


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    head = obs.get("head") or {}
    t = head.get("torch") or {}
    d = (head.get("diag") or {})
    out = common_gates(obs)
    out.insert(len(out) - 1, ("head: GPU visible in the pytest process", bool(t.get("is_available")),
                              f"is_available={t.get('is_available')} hip={t.get('hip')} cuda={t.get('cuda')} "
                              f"name={t.get('device_name')} arch={t.get('arch')} torch={t.get('version')} "
                              f"directml={d.get('torch_directml')} {t.get('error') or ''}"))
    ran = _cuda_ran(head)
    out.insert(len(out) - 1, ("head: >= 1 cuda-parametrised test ran", len(ran) > 0,
                              f"{len(ran)} ran, {len(head.get('cuda_skipped') or [])} skipped"))
    return out


def _diag_lines(o: dict) -> list:
    d = o.get("diag") or {}
    if not d:
        return [f"(no diag: rc={o.get('diag_rc')} {str(o.get('diag_stderr_tail'))[-300:]})"]
    t, tr, fl = d.get("torch") or {}, d.get("triton") or {}, d.get("fused_linear") or {}
    lines = [f"torch {t.get('version')} hip={t.get('hip')} cuda={t.get('cuda')} gpu={t.get('is_available')} "
             f"{t.get('device_name') or ''} {t.get('arch') or ''}; triton={tr.get('version')} {tr.get('error') or ''}; "
             f"torch_directml={d.get('torch_directml')}",
             f"fused_linear: _HAS_TRITON={fl.get('_HAS_TRITON')} _TRITON={fl.get('_TRITON')} "
             f"_use_triton={fl.get('use_triton')} {d.get('fused_linear_error') or ''}"]
    if d.get("zoo_import_error_without_allow_cpu"):
        lines.append(f"zoo import needed UNSLOTH_ALLOW_CPU=1: {d['zoo_import_error_without_allow_cpu']}")
    for k, v in (d.get("compile") or {}).items():
        lines.append(f"torch.compile {k}: {'ok' if v.get('ok') else 'FAILED'} ({v.get('s')} s) {v.get('error') or ''}")
    for k, v in (d.get("fused_ce_smoke") or {}).items():
        lines.append(f"fused CE smoke {k}: abs_err={v.get('abs_err')} CALLS delta={v.get('calls_delta')} "
                     f"{v.get('error') or ''}")
    return lines


def table(obs: dict) -> str:
    rows = ["| state | zoo | passed | failed | errors | skipped | cuda ran | CALLS torch | CALLS triton | failing |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for name, o in _states(obs).items():
        bad = ", ".join(f"`{x.split('::')[-1]}`" for x in (o.get("failed") or []) + (o.get("errors") or [])) or "none"
        c = o.get("calls") or {}
        rows.append(f"| {name} | {str(o.get('zoo_sha'))[:9]} | {o.get('n_passed', 0)} | {o.get('n_failed', 0)} | "
                    f"{o.get('n_errors', 0)} | {o.get('n_skipped', 0)} | {len(_cuda_ran(o))} | {c.get('torch')} | "
                    f"{c.get('triton')} | {bad} |")
    head = obs.get("head") or {}
    rows += ["", "Head host facts (zoo_env_diag.py, outside pytest):", ""] + [f"- {x}" for x in _diag_lines(head)]
    skips: dict = {}
    for k, why in (head.get("skip_reasons") or {}).items():
        skips.setdefault(why, []).append(k)
    if skips:
        rows += ["", "Head skips by reason:", ""]
        rows += [f"- {len(v)} x {why}: " + ", ".join(f"`{x.split('::')[-1]}`" for x in v[:8]) + (" ..." if len(v) > 8 else "")
                 for why, v in sorted(skips.items(), key = lambda kv: -len(kv[1]))]
    absent = (obs.get("base") or {}).get("absent_at_this_state") or []
    if absent:
        rows += ["", "Absent at the base (added by the PR): " + ", ".join(f"`{a}`" for a in absent)]
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    b = set(base.get("failed") or []) | set(base.get("errors") or [])
    h = set(head.get("failed") or []) | set(head.get("errors") or [])
    new = sorted(h - b)
    if new:
        return True, "failing at the head and not at the base: " + ", ".join(f"`{n}`" for n in new)
    detail = f"head {head.get('n_passed')} passed, {head.get('n_skipped')} skipped, 0 new failures"
    if h:
        detail += "; failing at BOTH states (pre-existing): " + ", ".join(f"`{n}`" for n in sorted(h))
    return False, detail
