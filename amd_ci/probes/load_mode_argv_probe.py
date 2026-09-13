#!/usr/bin/env python3
"""Probe: which ``--load-mode`` tokens can a real launch actually carry?

Observes only. Two separate producers can put a ``--load-mode`` on a
llama-server argv, and the question is whether they can ever collide:

* the offload planner computes ``Plan.load_mode_none`` (offload_planner.py:931)
  when the HOST side of a spill fits in host RAM, and ``plan_to_args`` turns that
  into ``--load-mode none``;
* PR #10618's no-reserve policy appends ``--load-mode dio`` for a confirmed FULL
  offload on Windows.

Statically the first never reaches an argv: the launch seam is
``_spill_plan_flags_for``, not ``plan_to_args``. That is the claim this probe
exists to test against a real interpreter on real hardware rather than against a
grep, because a grep cannot see a dynamic dispatch and cannot see which of the
two helpers the installed revision actually calls.

It records the tokens each helper returns for the SAME plan -- one built to want
``none`` as hard as a plan can -- and the load-mode the no-reserve policy would
emit on this host. Whether any of that is acceptable is for the criteria.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import sys
from pathlib import Path

#: What a HIP build prints for this host under `--list-devices`. The stub is the
#: INPUT to the decision, not the decision. Same value the dio probe assumes.
ASSUMED_DEVICE_IDS = ["ROCm0"]

#: A spill big enough to be real, and a host side small enough that
#: `load_mode_none` is unambiguously True. If the planner will ever ask for
#: `none`, it asks for it here.
_SPILL_PATTERN = r"blk\.3\.ffn.*"


def _build_wanting_plan(planner) -> tuple[object | None, dict]:
    """A Plan that both spills and wants ``load_mode_none``.

    Built field-by-field off the dataclass rather than positionally: the base and
    head revisions need not agree on Plan's shape, and a constructor call that
    guesses would fail on one leg and be read as "the helper is absent".
    """
    note: dict = {}
    Plan = getattr(planner, "Plan", None)
    if Plan is None or not dataclasses.is_dataclass(Plan):
        return None, {"error": "no Plan dataclass in this revision"}

    wanted = {
        "changed": True,
        "n_ctx": 8192,
        "ot_patterns": (_SPILL_PATTERN,),
        "load_mode_none": True,
        "cache_type_k": None,
        "cache_type_v": None,
        "spilled_blocks": (3,),
        "spilled_lm_head": False,
        "vram_bytes": 1,
        "host_bytes": 1,
        "predicted_gen_penalty_ms": 0.0,
        "reason": "probe",
    }
    names = {f.name for f in dataclasses.fields(Plan)}
    note["plan_fields"] = sorted(names)
    note["unknown_fields_skipped"] = sorted(set(wanted) - names)
    kwargs = {k: v for k, v in wanted.items() if k in names}
    try:
        return Plan(**kwargs), note
    except Exception as e:  # noqa: BLE001
        return None, {**note, "construct_error": repr(e)}


def _load_mode_values(tokens) -> list[str]:
    """Every value following a ``--load-mode`` / ``-lm`` in an argv."""
    out = []
    toks = list(tokens or [])
    for i, t in enumerate(toks):
        if t in ("--load-mode", "-lm") and i + 1 < len(toks):
            out.append(toks[i + 1])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--subdir", default = "studio/backend")
    args = ap.parse_args()
    # Resolved BEFORE the chdir, or a relative --out lands inside the checkout.
    args.out = args.out.resolve()

    obs: dict = {
        "state": args.state,
        "platform": sys.platform,
        "machine": platform.machine(),
        "node": platform.node(),
    }

    def emit() -> int:
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0

    # Resolved, like --out: it goes on sys.path and is then chdir'd into, so a
    # relative one silently stops importing once the cwd moves.
    backend_root = (Path(args.checkout).resolve() / args.subdir)
    obs["backend_root"] = str(backend_root)
    if not backend_root.is_dir():
        obs["error"] = f"no such directory: {backend_root}"
        return emit()

    sys.path.insert(0, str(backend_root))
    os.chdir(backend_root)

    try:
        from core.inference import offload_planner as planner
    except Exception as e:  # noqa: BLE001
        obs["planner_import_error"] = repr(e)
        return emit()

    obs["smart_offload_enabled"] = bool(planner.smart_offload_enabled(os.environ))
    obs["plan_to_args_present"] = hasattr(planner, "plan_to_args")

    plan, note = _build_wanting_plan(planner)
    obs["plan_note"] = note
    if plan is None:
        return emit()

    obs["plan_load_mode_none"] = bool(getattr(plan, "load_mode_none", False))
    obs["plan_spills_anything"] = bool(getattr(plan, "spills_anything", False))

    # The DEAD helper: what would reach an argv if the launch called it.
    if obs["plan_to_args_present"]:
        try:
            toks = list(planner.plan_to_args(plan))
            obs["plan_to_args_tokens"] = toks
            obs["plan_to_args_load_modes"] = _load_mode_values(toks)
        except Exception as e:  # noqa: BLE001
            obs["plan_to_args_error"] = repr(e)

    # The PRODUCTION seam: what the launch path really appends for that plan.
    try:
        from core.inference.llama_cpp import LlamaCppBackend as B
    except Exception as e:  # noqa: BLE001
        obs["backend_import_error"] = repr(e)
        return emit()

    obs["spill_seam_present"] = hasattr(B, "_spill_plan_flags_for")
    if obs["spill_seam_present"]:
        try:
            toks = list(B._spill_plan_flags_for(plan))
            obs["spill_seam_tokens"] = toks
            obs["spill_seam_load_modes"] = _load_mode_values(toks)
        except Exception as e:  # noqa: BLE001
            obs["spill_seam_error"] = repr(e)

    # And the other producer, on this host. Same shape the dio probe reads, so
    # the two runs cannot disagree about the same machine.
    obs["feature_present"] = hasattr(B, "_gpu_offload_confirmed")
    try:
        obs["amd_apu_wants_unified_memory"] = bool(B._amd_apu_wants_unified_memory([0]))
    except Exception as e:  # noqa: BLE001
        obs["amd_apu_wants_unified_memory_error"] = repr(e)
    host_resident = bool(obs.get("amd_apu_wants_unified_memory", False))
    obs["host_resident_fed_to_the_gate"] = host_resident

    if obs["feature_present"]:
        original = B._enumerated_gpu_devices
        try:
            B._enumerated_gpu_devices = classmethod(  # type: ignore[assignment]
                lambda cls, binary = None, env = None: list(ASSUMED_DEVICE_IDS)
            )
            obs["gpu_offload_confirmed"] = bool(
                B._gpu_offload_confirmed("llama-server", {}, [0], host_resident, True, None)
            )
        except Exception as e:  # noqa: BLE001
            obs["gpu_offload_confirmed_error"] = repr(e)
        finally:
            B._enumerated_gpu_devices = original  # type: ignore[assignment]

        try:
            from core.inference.llama_server_args import (
                MANAGED_DIO_FLAGS,
                resolve_launch_load_mode,
            )

            real_platform = sys.platform
            sys.platform = "win32"      # native on the Windows leg
            try:
                emitted, effective = resolve_launch_load_mode(
                    None,
                    supports_load_mode = True,
                    weights_in_host_memory = host_resident,
                    gpu_offload_confirmed = bool(obs.get("gpu_offload_confirmed", False)),
                    requested_load_mode = None,
                    env = {},
                    settings = (False, True),   # keep_resident off, no-reserve ON
                )
            finally:
                sys.platform = real_platform
            obs["policy_emitted_dio"] = bool(emitted)
            obs["effective_dio"] = bool(effective)
            obs["decision_platform_is_native"] = real_platform == "win32"

            # The union, which is the whole question: the tokens a single argv
            # could carry if BOTH producers fired on one launch.
            combined = list(obs.get("spill_seam_tokens") or [])
            if emitted:
                combined += list(MANAGED_DIO_FLAGS)
            obs["combined_argv_tokens"] = combined
            obs["combined_load_modes"] = _load_mode_values(combined)
        except Exception as e:  # noqa: BLE001
            obs["policy_error"] = repr(e)
    else:
        obs["note"] = "this revision has no _gpu_offload_confirmed; nothing can emit dio"
        obs["combined_argv_tokens"] = list(obs.get("spill_seam_tokens") or [])
        obs["combined_load_modes"] = _load_mode_values(obs["combined_argv_tokens"])

    return emit()


if __name__ == "__main__":
    raise SystemExit(main())
