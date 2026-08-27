#!/usr/bin/env python3
"""Observe the dense torchao quant ladder's view of a ROCm GPU, for PR 9828.

Observes only. Every judgement lives in criteria/rocm_dense_quant_gate.py.

Two things shape this probe.

The first is that the BASE leg deliberately runs the code the PR exists because it
can fault: on the reporter's gfx1103 the int8 smoke probe takes the process down
with SIGSEGV. So each observation runs in its own child, re-execing this file with
``--worker``; a child that dies is recorded as a crash rather than taking the
differential's own JSON with it.

The second is that this must never actually launch a torchao kernel to prove the
base is wrong. The question is whether the ladder is ENTERED on an AMD card, and
that is answered by a recording tripwire standing in for ``_scheme_supported``.
Running the real probe would add no evidence and could kill the child on a card
where it faults.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

WORKERS = ("env", "gate", "select", "te", "cache")


# --------------------------------------------------------------------------- workers


def _import(checkout: str):
    """Put this state's backend first on the path and import the modules under test."""
    backend = str(Path(checkout) / "studio" / "backend")
    sys.path.insert(0, backend)
    os.chdir(backend)
    import core._torchao_stub as stub
    import core.inference.diffusion_transformer_quant as tq
    import core.inference.diffusion_precision as dp
    return stub, tq, dp


def _target(dtype_bf16: bool = True):
    import torch
    from types import SimpleNamespace
    return SimpleNamespace(device = "cuda", dtype = torch.bfloat16 if dtype_bf16 else torch.float16)


def w_env(checkout: str) -> dict:
    """Runtime facts, plus the health check that separates "torchao is wrong here" from
    "every kernel on this box faults". The PR's own description documents a mismatched
    ROCm wheel under which `torch.ones(1024, device="cuda")` SIGSEGVs, and a run in that
    state says nothing about torchao."""
    out: dict = {}
    import torch
    out["torch"] = torch.__version__
    out["hip"] = torch.version.hip
    out["cuda"] = torch.version.cuda
    out["cuda_available"] = bool(torch.cuda.is_available())
    if out["cuda_available"]:
        out["device_count"] = torch.cuda.device_count()
        props = torch.cuda.get_device_properties(0)
        out["device_name"] = props.name
        out["arch"] = getattr(props, "gcnArchName", "") or ""
        out["is_integrated"] = getattr(props, "is_integrated", None)
        out["capability"] = list(torch.cuda.get_device_capability())
        out["current_device"] = int(torch.cuda.current_device())
    try:
        x = torch.ones(1024, device = "cuda")
        out["alloc_ok"] = bool(float(x.sum()) == 1024.0)
    except Exception as e:  # noqa: BLE001
        out["alloc_ok"] = False
        out["alloc_error"] = f"{type(e).__name__}: {e}"
    try:
        a = torch.randn(64, 64, device = "cuda", dtype = torch.bfloat16)
        b = (a @ a).float()
        out["bf16_matmul_ok"] = bool(b.isfinite().all())
    except Exception as e:  # noqa: BLE001
        out["bf16_matmul_ok"] = False
        out["bf16_matmul_error"] = f"{type(e).__name__}: {e}"

    stub, _tq, _dp = _import(checkout)
    out["torchao_stubbed"] = bool(stub.is_stubbed("torchao"))
    try:
        import torchao
        out["torchao"] = getattr(torchao, "__version__", "unknown")
    except Exception as e:  # noqa: BLE001
        out["torchao"] = None
        out["torchao_import_error"] = f"{type(e).__name__}: {e}"
    return out


def w_gate(checkout: str) -> dict:
    """The gate itself. `torch_is_rocm` does not exist at the base, and its absence is the
    marker rather than a probe failure, so it is feature-detected."""
    stub, tq, _dp = _import(checkout)
    out: dict = {}
    fn = getattr(stub, "torch_is_rocm", None)
    out["has_torch_is_rocm"] = fn is not None
    out["torch_is_rocm"] = bool(fn()) if fn is not None else None
    out["is_windows_rocm"] = bool(stub._is_windows_rocm())

    target = _target()
    out["dense_transformer_supported"] = bool(tq.dense_transformer_supported(target))
    reason_fn = getattr(tq, "dense_transformer_unsupported_reason", None)
    out["has_unsupported_reason"] = reason_fn is not None
    out["unsupported_reason"] = reason_fn(target) if reason_fn is not None else None
    out["capability_seen_by_module"] = list(tq._capability() or ())
    return out


def w_select(checkout: str) -> dict:
    """Scheme selection with a recording tripwire in place of `_scheme_supported`.

    The tripwire answers True, so a base that reaches it selects a scheme and the count
    is non-zero; a head that refuses on ROCm never calls it at all. No kernel runs."""
    _stub, tq, _dp = _import(checkout)
    calls: list[list] = []

    def tripwire(scheme, device, *, unproven_ok = False):
        calls.append([scheme, str(device), bool(unproven_ok)])
        return True

    tq._scheme_supported = tripwire
    target = _target()
    out: dict = {"selected": {}, "errors": {}}
    for mode in ("auto",) + tuple(tq.TQ_SCHEMES):
        try:
            out["selected"][mode] = tq.select_transformer_quant_scheme(target, mode)
        except Exception as e:  # noqa: BLE001
            out["errors"][mode] = f"{type(e).__name__}: {e}"
    try:
        out["auto_candidates"] = list(tq.auto_scheme_candidates(target))
    except Exception as e:  # noqa: BLE001
        out["auto_candidates"] = None
        out["errors"]["auto_candidates"] = f"{type(e).__name__}: {e}"
    out["tripwire_calls"] = calls
    out["tripwire_call_count"] = len(calls)
    return out


def w_te(checkout: str) -> dict:
    """Text-encoder modes. Plain layerwise fp8 is a dtype cast and is meant to survive on
    ROCm; the three torchao-backed modes are not."""
    _stub, _tq, dp = _import(checkout)
    target = _target()
    out: dict = {"supported": {}, "errors": {}}
    for mode in ("fp8", "int8", "fp8_dynamic", "nvfp4"):
        try:
            out["supported"][mode] = bool(dp.te_quant_supported(target, mode))
        except Exception as e:  # noqa: BLE001
            out["errors"][mode] = f"{type(e).__name__}: {e}"
    return out


def w_cache(checkout: str) -> dict:
    """The cache-key leg: does a completed child verdict get consumed, or re-probed?

    `_child_probe_table` is stubbed with a finished verdict and `_run_smoke_probe` with a
    counter, so nothing spawns and nothing allocates. At the base the child's answer is
    filed under ("int8", "cuda") while `_smoke_probe` looks under ("int8", "cuda:0"), so
    the in-process probe runs anyway. At the head both use the qualified key."""
    _stub, tq, _dp = _import(checkout)
    asked: list[str] = []
    ran: list[list] = []

    def fake_child_table(device):
        asked.append(str(device))
        return {"int8": True, "fp8": False}

    def fake_run(scheme, device):
        ran.append([scheme, str(device)])
        return True

    tq._SMOKE_CACHE.clear()
    tq._child_probe_table = fake_child_table
    tq._run_smoke_probe = fake_run

    out: dict = {}
    try:
        out["int8_supported"] = bool(tq._scheme_supported("int8", "cuda"))
    except Exception as e:  # noqa: BLE001
        out["int8_supported"] = None
        out["error"] = f"{type(e).__name__}: {e}"
    out["child_asked_about"] = asked
    out["in_process_probe_calls"] = ran
    out["in_process_probe_count"] = len(ran)
    # "@" and not "|": these strings are rendered into a markdown table cell.
    out["cache_keys"] = sorted(f"{s}@{d}" for s, d in tq._SMOKE_CACHE)
    out["cache_device_key_for_cuda"] = tq._smoke_cache_device_key("cuda")
    out["has_crashed_child_verdict"] = hasattr(tq, "_crashed_child_verdict")
    out["has_select_probe_card"] = hasattr(tq, "_select_probe_card")
    out["probe_crash_signals"] = sorted(getattr(tq, "_PROBE_CRASH_SIGNALS", ()) or ())
    return out


# --------------------------------------------------------------------------- driver


def _run_worker(name: str, checkout: str, out_path: Path, python: str) -> dict:
    """One observation, in its own process. A child killed by a signal is DATA here."""
    log = out_path.with_suffix(".log")
    cmd = [python, os.path.abspath(__file__), "--worker", name,
           "--checkout", checkout, "--out", str(out_path)]
    with open(log, "w") as fh:
        rc = subprocess.run(cmd, stdout = fh, stderr = subprocess.STDOUT).returncode
    rec: dict = {"rc": rc}
    if rc < 0:
        rec["killed_by_signal"] = -rc
    if out_path.is_file():
        try:
            rec.update(json.loads(out_path.read_text()))
        except Exception as e:  # noqa: BLE001
            rec["parse_error"] = f"{type(e).__name__}: {e}"
    else:
        rec["no_output"] = True
        rec["log_tail"] = log.read_text(errors = "replace")[-1500:] if log.is_file() else ""
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default = "")
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--worker", default = None, choices = WORKERS)
    ap.add_argument("--python", default = sys.executable)
    args = ap.parse_args()
    # Before anything else: `_import` chdirs into the checkout's backend, so a relative
    # --out would be written somewhere other than where the runner looks for it.
    args.out = args.out.resolve()
    args.checkout = os.path.abspath(args.checkout)
    args.out.parent.mkdir(parents = True, exist_ok = True)

    if args.worker:
        try:
            payload = globals()[f"w_{args.worker}"](args.checkout)
        except Exception as e:  # noqa: BLE001
            import traceback
            payload = {"worker_error": f"{type(e).__name__}: {e}",
                       "traceback": traceback.format_exc()[-2000:]}
        args.out.write_text(json.dumps(payload, indent = 2))
        return 0

    obs: dict = {"state": args.state, "checkout": args.checkout}
    for name in WORKERS:
        print(f"-- worker {name}", flush = True)
        obs[name] = _run_worker(
            name, args.checkout, args.out.parent / f"{args.out.stem}.{name}.json", args.python)
    args.out.write_text(json.dumps(obs, indent = 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
