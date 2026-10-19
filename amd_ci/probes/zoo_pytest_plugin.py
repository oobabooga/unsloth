"""pytest plugin loaded by zoo_pytest_probe.py (`-p zoo_pytest_plugin`). Observes only.

Writes $ZOO_PLUGIN_OUT (JSON) at session end:
  torch       is_available / version.hip / version.cuda / device name / gcnArchName, read in this pytest process
  outcomes    nodeid -> passed / failed / skipped / error (call phase, or setup/teardown when that phase failed)
  triton_tests  nodeids during whose call fused_linear CALLS["triton"] increased (the Triton kernel launched)
  calls       fused_linear CALLS at session end, _TRITON_BROKEN left behind, and the unsloth_zoo file imported
"""

from __future__ import annotations

import json
import os
import sys

import pytest

_OUT: dict = {"outcomes": {}, "triton_tests": [], "torch": {}}


def _fl():
    return sys.modules.get("unsloth_zoo.fused_losses.fused_linear")


def _triton_count():
    fl = _fl()
    calls = getattr(fl, "CALLS", None) if fl is not None else None
    return int(calls.get("triton", 0)) if isinstance(calls, dict) else None


def pytest_sessionstart(session):
    t = _OUT["torch"]
    try:
        import torch

        t["version"] = torch.__version__
        t["hip"] = getattr(torch.version, "hip", None)
        t["cuda"] = getattr(torch.version, "cuda", None)
        t["is_available"] = bool(torch.cuda.is_available())
        if t["is_available"]:
            p = torch.cuda.get_device_properties(0)
            t["device_name"] = torch.cuda.get_device_name(0)
            t["arch"] = getattr(p, "gcnArchName", None)
            t["capability"] = list(torch.cuda.get_device_capability(0))
        import triton

        t["triton"] = triton.__version__
    except Exception as e:  # noqa: BLE001
        t["error"] = f"{type(e).__name__}: {e}"


@pytest.hookimpl(hookwrapper = True)
def pytest_runtest_call(item):
    before = _triton_count()
    yield
    after = _triton_count()
    if after is not None and (before or 0) < after:
        _OUT["triton_tests"].append(item.nodeid)


def pytest_runtest_logreport(report):
    o = _OUT["outcomes"]
    if report.when == "call":
        o[report.nodeid] = report.outcome
    elif report.failed:
        o[report.nodeid] = "error"
    elif report.when == "setup" and report.skipped:
        o[report.nodeid] = "skipped"


def pytest_sessionfinish(session, exitstatus):
    fl = _fl()
    _OUT["exitstatus"] = int(exitstatus)
    if fl is not None:
        _OUT["calls"] = dict(getattr(fl, "CALLS", {}) or {})
        _OUT["triton_broken"] = [str(d) for d in getattr(fl, "_TRITON_BROKEN", ())]
    zoo = sys.modules.get("unsloth_zoo")
    _OUT["unsloth_zoo_file"] = getattr(zoo, "__file__", None)
    path = os.environ.get("ZOO_PLUGIN_OUT")
    if path:
        with open(path, "w", encoding = "utf-8") as fh:
            json.dump(_OUT, fh, indent = 1)
