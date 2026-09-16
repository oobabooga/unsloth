#!/usr/bin/env python3
"""Probe: with a memory cap asked for, which GPUs actually end up capped?

unsloth#11077 makes the training worker's memory cap backend-neutral, and a review
round on that branch widened it from `set_per_process_memory_fraction(f)` -- which
torch applies to `current_device()` alone -- to one explicit call per visible
device. Both claims are about what the ALLOCATOR ends up enforcing, and a fake
`torch.cuda` can only show that the shipped lines were executed.

So this executes the same shipped block (section 1h of `run_training_process`,
sliced out of `worker.py` exactly as the PR's own test does) against the REAL
torch on the real cards, and then asks the allocator two independent questions per
device:

  * what fraction does `get_per_process_memory_fraction(i)` report, and
  * does an allocation larger than that fraction actually get refused?

The second matters because the first is bookkeeping: a fraction recorded but not
enforced would read as a pass. A state whose checkout has no such block records
that, rather than being given one.

Observes only. criteria/gpu_mem_fraction_caps_every_device.py judges.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import traceback
from pathlib import Path
from types import SimpleNamespace

SECTION_START = "    # ── 1h. Explicit GPU memory cap ──"
SECTION_END = "    # ── 2. Now import ML libraries"
ENV_NAME = "UNSLOTH_GPU_MEM_FRACTION"


def _section_source(worker_py: Path) -> str | None:
    text = worker_py.read_text(encoding = "utf-8")
    try:
        start = text.index(SECTION_START)
        end = text.index(SECTION_END, start)
    except ValueError:
        return None
    return textwrap.dedent(text[start:end])


def _logger():
    rec = SimpleNamespace(info = [], warning = [], debug = [])

    def make(level):
        def log(message, *args):
            getattr(rec, level).append(message % args if args else message)
        return log

    return SimpleNamespace(info = make("info"), warning = make("warning"),
                           debug = make("debug")), rec


def _allocation_probe(torch, index: int, nbytes: int) -> dict:
    """Try to allocate `nbytes` on cuda:index. The classification is the point:
    a cap refusal says "allowed memory", a genuinely full card says something else,
    and calling the second one a cap would be the easy way to fake this."""
    out: dict = {"device": index, "bytes": nbytes}
    try:
        buf = torch.empty(nbytes, dtype = torch.uint8, device = f"cuda:{index}")
        out["allocated"] = True
        del buf
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        out["allocated"] = False
        out["error_type"] = type(e).__name__
        out["error"] = msg[:1200]
        low = msg.lower()
        # torch words a fraction refusal as "... 44.59 GiB allowed", inside an
        # otherwise ordinary OutOfMemoryError, and it prints the free memory in the
        # same sentence. Reading both numbers is what separates "the cap refused an
        # allocation the card could have served" from "the card is genuinely full",
        # which a substring match on "out of memory" cannot do.
        import re
        allowed = re.search(r"([\d.]+)\s*GiB allowed", msg)
        free = re.search(r"([\d.]+)\s*GiB is free", msg)
        gib = 1024 ** 3
        out["allowed_gib"] = float(allowed.group(1)) if allowed else None
        out["free_gib"] = float(free.group(1)) if free else None
        want = nbytes / gib
        out["requested_gib"] = round(want, 2)
        out["refused_by_cap"] = bool(
            allowed and out["allowed_gib"] < want
            and (out["free_gib"] is None or out["free_gib"] >= want))
        out["refused_as_oom"] = ("out of memory" in low) and not out["refused_by_cap"]
    finally:
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--fraction", type = float, default = 0.25,
                    help = "the cap to ask for, as the user's env variable would")
    ap.add_argument("--probe-fraction", type = float, default = 0.35,
                    help = "allocation size as a fraction of each card's TOTAL; must sit "
                           "above --fraction so an enforced cap refuses it")
    args = ap.parse_args()

    obs: dict = {"state": args.state, "fraction_requested": args.fraction,
                 "probe_fraction": args.probe_fraction, "env_name": ENV_NAME}

    backend_dir = args.checkout / "studio" / "backend"
    worker_py = backend_dir / "core" / "training" / "worker.py"
    obs["worker_py"] = str(worker_py)
    source = _section_source(worker_py) if worker_py.is_file() else None
    obs["section_1h_present"] = source is not None
    obs["section_1h_lines"] = len(source.splitlines()) if source else 0

    try:
        import torch
    except Exception as e:  # noqa: BLE001
        obs["torch_error"] = f"{type(e).__name__}: {e}"
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0

    obs["torch_version"] = torch.__version__
    obs["cuda_available"] = bool(torch.cuda.is_available())
    obs["device_count"] = torch.cuda.device_count() if obs["cuda_available"] else 0
    obs["devices"] = []
    for i in range(obs["device_count"]):
        p = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        obs["devices"].append({"index": i, "name": p.name,
                               "total_bytes": int(p.total_memory),
                               "driver_free_bytes": int(free),
                               "driver_total_bytes": int(total)})

    getter = getattr(torch.cuda, "get_per_process_memory_fraction", None)
    obs["has_fraction_getter"] = getter is not None

    def fractions() -> list:
        if getter is None:
            return []
        out = []
        for i in range(obs["device_count"]):
            try:
                out.append(getter(i))
            except Exception as e:  # noqa: BLE001
                out.append(f"error: {type(e).__name__}: {e}")
        return out

    obs["fractions_before"] = fractions()

    if source is None:
        # Not an error: this checkout genuinely has no such block, and that is the
        # reading. It is also the control for the allocation probe below.
        obs["executed"] = False
    else:
        sys.path.insert(0, str(backend_dir))
        logger, recorded = _logger()
        try:
            from core.training import worker as worker_module
            environ = {ENV_NAME: str(args.fraction)}

            def resolve_env(backend, environ_ = None):
                return worker_module._mem_fraction_env_value(
                    backend, environ if environ_ is None else environ_)

            ns = {
                "_hw": SimpleNamespace(IS_ROCM = False),
                "os": SimpleNamespace(environ = environ),
                "sys": sys,
                "logger": logger,
                "_mem_fraction_env_value": resolve_env,
                "_parse_mem_fraction_env": worker_module._parse_mem_fraction_env,
                "_gpu_memory_fraction": worker_module._gpu_memory_fraction,
                "torch": torch,
            }
            exec(compile(source, str(worker_py), "exec"), ns)  # noqa: S102
            obs["executed"] = True
            obs["log_info"] = list(recorded.info)
            obs["log_warning"] = list(recorded.warning)
        except Exception as e:  # noqa: BLE001
            obs["executed"] = False
            obs["exec_error"] = f"{type(e).__name__}: {e}"
            obs["exec_traceback"] = traceback.format_exc()[-3000:]

    obs["fractions_after"] = fractions()

    obs["allocations"] = []
    for d in obs["devices"]:
        nbytes = int(d["total_bytes"] * args.probe_fraction)
        obs["allocations"].append(_allocation_probe(torch, d["index"], nbytes))

    args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
