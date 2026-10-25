#!/usr/bin/env python3
"""Probe: on this GPU, does the checkout's keep-cpu offload path page-lock host weights?

Observes only. Records the driver's integrated flag and arch, what the checkout's
host_pinning helper answers (absent on base), and whether ``_pin_host_weights``
pinned a small module's kept host weights. Pairs with criteria/integrated_no_pin.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()

    obs: dict = {"state": args.state}
    backend = args.checkout / "studio" / "backend"
    sys.path.insert(0, str(backend))
    try:
        import torch

        props = torch.cuda.get_device_properties(0)
        obs["is_integrated"] = getattr(props, "is_integrated", None)
        obs["arch"] = getattr(props, "gcnArchName", None)
        obs["hip"] = getattr(torch.version, "hip", None)
    except Exception as e:  # noqa: BLE001
        obs["torch_error"] = f"{type(e).__name__}: {e}"

    try:
        from core.inference import host_pinning

        obs["helper_present"] = True
        obs["device_is_integrated"] = bool(host_pinning.device_is_integrated())
        obs["host_pinning_allowed"] = bool(host_pinning.host_pinning_allowed())
    except ImportError:
        obs["helper_present"] = False
    except Exception as e:  # noqa: BLE001
        obs["helper_error"] = f"{type(e).__name__}: {e}"

    try:
        import torch

        from core.inference import diffusion_memory as dm

        module = torch.nn.Sequential(torch.nn.Linear(256, 256), torch.nn.Linear(256, 256))
        host = {name: True for name, _ in module.named_parameters()}
        obs["pinned_bytes"] = int(dm._pin_host_weights(module, host))
        obs["any_param_pinned"] = any(p.data.is_pinned() for p in module.parameters())
    except Exception as e:  # noqa: BLE001
        obs["pin_error"] = f"{type(e).__name__}: {e}"

    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
