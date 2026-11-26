#!/usr/bin/env python3
"""Wrapper around wintiers/arm.py (one Studio render arm) that records which bf16 Linear GEMMs the denoisers run and
whether Studio's TunableOp table is active, for the TunableOp table e2e check (A7 'tune').

  tune_arm.py --hits FILE [--record-keys] <arm.py> <arm.py args...>

Timed arms run WITHOUT the Linear hooks (they would sit inside the compiled denoiser); --record-keys is for a separate
untimed arm.

Records, per denoiser nn.Linear call: the TunableOp key the call maps to (F.linear(x [M, K], W [N, K]) ->
``tn_N_M_K_ld_K_ldx_N``), dtype and count; plus Studio's activation result, torch.cuda.tunable.is_enabled() after the
load, and (record_untuned on while the table is active) the untuned-GEMM file torch writes for keys it did not find.
Written after every denoiser forward, since arm.py ends with os._exit."""
from __future__ import annotations

import json
import os
import runpy
import sys


def main() -> None:
    if sys.argv[1] != "--hits":
        raise SystemExit("usage: tune_arm.py --hits FILE arm.py args...")
    hits_path = sys.argv[2]
    rest = sys.argv[3:]
    record_keys = rest[0] == "--record-keys"
    if record_keys:
        rest = rest[1:]
    arm = rest[0]
    argv = rest
    root = argv[argv.index("--root") + 1]
    sys.path.insert(0, root)
    state = {"install": [], "keys": {}, "linear_classes": {}, "forwards": 0, "tunable_enabled": None,
             "record_untuned": None, "cwd": os.getcwd()}

    def dump():
        try:
            with open(hits_path, "w", encoding = "utf-8") as fh:
                json.dump(state, fh)
        except Exception:  # noqa: BLE001
            pass

    import torch
    from core.inference import diffusion_rocm_tunableop as RT

    real_install = RT.install_for_pipe

    def lin_pre(mod, args):
        x = args[0] if args else None
        if not isinstance(x, torch.Tensor) or x.dim() < 2:
            return
        w = mod.weight
        if not isinstance(w, torch.Tensor) or w.dim() != 2 or type(w) is not torch.nn.Parameter and type(w) is not torch.Tensor:
            state["linear_classes"]["nonplain_weight:" + type(w).__name__] = \
                state["linear_classes"].get("nonplain_weight:" + type(w).__name__, 0) + 1
            return
        n, k = int(w.shape[0]), int(w.shape[1])
        m = x.numel() // max(k, 1)
        ldx = k if (x.stride(-1) == 1 and (x.is_contiguous() or (x.dim() == 2 and x.stride(0) == k))) else \
            (int(x.stride(-2)) if x.stride(-1) == 1 else -1)
        key = f"{str(x.dtype).replace('torch.', '')}|{str(w.dtype).replace('torch.', '')}|tn_{n}_{m}_{k}_ld_{k}_{ldx}_{n}"
        state["keys"][key] = state["keys"].get(key, 0) + 1

    def dit_post(_mod, _args, _out):
        state["forwards"] += 1
        if state["forwards"] <= 3 or state["forwards"] % 10 == 0:
            dump()

    def install_for_pipe(pipe, *a, **k):
        res = real_install(pipe, *a, **k)
        rec = {"result": bool(res), "active": bool(RT.is_active())}
        try:
            state["tunable_enabled"] = bool(torch.cuda.tunable.is_enabled())
        except Exception as exc:  # noqa: BLE001
            state["tunable_enabled"] = repr(exc)[:200]
        if res and hasattr(torch.cuda.tunable, "record_untuned_enable"):
            try:
                torch.cuda.tunable.record_untuned_enable(True)
                state["record_untuned"] = bool(torch.cuda.tunable.record_untuned_is_enabled())
            except Exception as exc:  # noqa: BLE001
                state["record_untuned"] = repr(exc)[:200]
        n_lin = 0
        for name in (("transformer", "transformer_2", "unet") if record_keys else ()):
            dit = getattr(pipe, name, None)
            if dit is None or not hasattr(dit, "modules"):
                continue
            dit.register_forward_hook(dit_post)
            for mod in dit.modules():
                cls = type(mod).__name__
                state["linear_classes"][cls] = state["linear_classes"].get(cls, 0) + (1 if "Linear" in cls else 0)
                if isinstance(mod, torch.nn.Linear):
                    mod.register_forward_pre_hook(lin_pre)
                    n_lin += 1
        rec["hooked_linears"] = n_lin
        state["install"].append(rec)
        dump()
        return res

    RT.install_for_pipe = install_for_pipe
    # Studio imports the function by name inside the load path in some trees: patch the module attribute only, and
    # also any already-imported alias.
    for mod in list(sys.modules.values()):
        if getattr(mod, "install_for_pipe", None) is real_install and mod is not RT:
            setattr(mod, "install_for_pipe", install_for_pipe)
    state["linear_classes"] = {}
    # torch writes TunableOp results only at interpreter exit, which arm.py's os._exit skips: write them first, in
    # torch's CSV layout (validators + results), and record how many results the process held.
    real_exit = os._exit

    def _exit(code):
        try:
            tun = torch.cuda.tunable
            state["tunable_enabled_end"] = bool(tun.is_enabled())
            results = list(tun.get_results() or ())
            state["results_end"] = len(results)
            out = os.environ.get("A7_TUNABLEOP_WRITE")
            if out and results:
                lines = [f"Validator,{k},{v}" for k, v in (tun.get_validators() or ())]
                lines += [",".join(str(x) for x in row) for row in results]
                with open(out, "w", encoding = "utf-8") as fh:
                    fh.write("\n".join(lines) + "\n")
                state["written"] = out
        except Exception as exc:  # noqa: BLE001
            state["exit_error"] = repr(exc)[:300]
        dump()
        real_exit(code)
    os._exit = _exit
    dump()
    sys.argv = argv
    runpy.run_path(arm, run_name = "__main__")


if __name__ == "__main__":
    main()
