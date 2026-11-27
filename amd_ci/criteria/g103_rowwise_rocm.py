#!/usr/bin/env python3
"""Criteria for probes/g103_rowwise_probe.py (#19900). Defect: on ROCm rowwise_supported() answers without running
its probe (no _scaled_mm call) and the CPU probe test fails. Fixed: the probe calls the real _scaled_mm on the device,
its answer agrees with the direct observation of the same calls, the context still works, and the test passes."""
from __future__ import annotations

TITLE = "fp8_lora.rowwise_supported() on ROCm gfx1151: shortcut vs real probe"
MODE = "differential"
NEEDS = ["gpu", "rocm"]


def _states(obs):
    return {k: v for k, v in obs.items() if not k.startswith("_")}


def _expected(cell):
    d = (cell.get("direct") or {}).get("float8_e4m3fn") or {}
    tw, rw = d.get("tensorwise") or {}, d.get("rowwise") or {}
    if not (tw.get("ok") and tw.get("correct")):
        return True, "e4m3fn tensorwise fails here: no verdict, True, not cached"
    if not (rw.get("ok") and rw.get("correct")):
        return False, "tensorwise runs, row-wise fails: False"
    return True, "both run: True"


def gates(obs):
    out = []
    for name, o in _states(obs).items():
        c = o.get("cell") or {}
        out.append((f"{name}: torch is a ROCm build", bool(c.get("hip")), str(c.get("hip"))))
        out.append((f"{name}: a GPU is visible", bool(c.get("cuda_available")), f"{c.get('device')} {c.get('arch')}"))
        out.append((f"{name}: direct fp8 calls observed", bool(c.get("direct")), "" if c.get("direct") else str(c.get("error"))))
        out.append((f"{name}: unsloth imported from the state's checkout", "rowwise_supported" in c,
                    str(c.get("unsloth_file") or c.get("error"))))
        out.append((f"{name}: the probe test ran", bool(o.get("pytest_passed") or o.get("pytest_failed")),
                    f"rc {o.get('pytest_rc')}"))
    return out


def table(obs):
    rows = ["| | " + " | ".join(_states(obs)) + " |", "|---|" + "---|" * len(_states(obs))]
    def cell(o, f):
        try:
            return str(f(o)).replace("|", "/")[:160]
        except Exception as e:
            return f"n/a ({e})"
    fields = [
        ("arch / torch", lambda o: f"{o['cell'].get('arch')} / {o['cell'].get('torch')}"),
        ("e4m3fn tensorwise", lambda o: o["cell"]["direct"]["float8_e4m3fn"].get("tensorwise")),
        ("e4m3fn row-wise", lambda o: o["cell"]["direct"]["float8_e4m3fn"].get("rowwise")),
        ("e4m3fnuz tensorwise", lambda o: o["cell"]["direct"]["float8_e4m3fnuz"].get("tensorwise")),
        ("e4m3fnuz row-wise", lambda o: o["cell"]["direct"]["float8_e4m3fnuz"].get("rowwise")),
        ("rowwise_supported()", lambda o: o["cell"].get("rowwise_supported")),
        ("_scaled_mm calls by the probe", lambda o: o["cell"].get("probe_scaled_mm_calls")),
        ("cached verdict", lambda o: o["cell"].get("probe_cached")),
        ("context ok after", lambda o: o["cell"].get("context_ok_after")),
        ("test failed", lambda o: o.get("pytest_failed")),
        ("test passed", lambda o: o.get("pytest_passed")),
    ]
    for label, f in fields:
        rows.append(f"| {label} | " + " | ".join(cell(o, f) for o in _states(obs).values()) + " |")
    return "\n".join(rows)


def base_shows_defect(base):
    c = base.get("cell") or {}
    no_call = c.get("probe_scaled_mm_calls") == 0
    failed = bool(base.get("pytest_failed"))
    if no_call and failed:
        return True, "rowwise_supported() made 0 _scaled_mm calls and the probe test failed"
    return False, f"calls {c.get('probe_scaled_mm_calls')}, test failures {base.get('pytest_failed')}"


def head_is_fixed(head):
    c = head.get("cell") or {}
    exp, why = _expected(c)
    bad = []
    if not (c.get("probe_scaled_mm_calls") or 0) >= 1:
        bad.append("probe made no _scaled_mm call")
    if c.get("rowwise_supported") is not exp:
        bad.append(f"answer {c.get('rowwise_supported')} != expected {exp} ({why})")
    if c.get("context_ok_after") is not True:
        bad.append(f"context after probe: {c.get('context_ok_after')}")
    if head.get("pytest_failed") or not head.get("pytest_passed"):
        bad.append(f"test failed {head.get('pytest_failed')}")
    if bad:
        return False, "; ".join(bad)
    return True, f"real probe ran ({c.get('probe_scaled_mm_calls')} calls), answer {exp}: {why}; test passed"
