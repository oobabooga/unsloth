#!/usr/bin/env python3
"""One isolated cell for the gfx1151 gated-delta (Qwen3.5 linear attention) illegal-access hunt.

Observes only; JSON via --out, rewritten after every step, so a HIP abort leaves the last step on disk.
Every Triton launch is appended (unbuffered os.write) to --launch-log before it happens: with
AMD_SERIALIZE_KERNEL=3 the last line before an abort names the faulting kernel and its launch config.

Cells:
  env                     versions, device props, every autotuned fla kernel and its config list
  model:<dtype>           tiny Qwen3.5-MoE (trl-internal-testing, K=V=128, H=16/HV=32) forward + backward
  op:<dtype>:B:T:H:K:V    fla chunk_gated_delta_rule fwd + bwd vs an fp64 recurrent reference
Env:
  ZLA_PIN="kernel_name=idx[,kernel_name=idx]"  restrict that autotuner to its config #idx
  ZLA_UPSTREAM_FLA=<dir>  import fla from that directory instead (no unsloth import)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

RES: dict = {"steps": {}}
OUT: Path | None = None
LAUNCH_FD = None
ASM_DIR: Path | None = None


def save():
    if OUT is not None:
        OUT.write_text(json.dumps(RES, indent = 2, default = str), encoding = "utf-8")


def step(name, fn):
    RES["current_step"] = name
    save()
    t0 = time.time()
    try:
        val = fn()
        import torch
        torch.cuda.synchronize()
        RES["steps"][name] = "ok"
        RES.setdefault("step_s", {})[name] = round(time.time() - t0, 2)
        return val
    except Exception as e:  # noqa: BLE001
        RES["steps"][name] = f"ERR {type(e).__name__}: {str(e)[:600]}"
        RES.setdefault("tracebacks", {})[name] = traceback.format_exc()[-5000:]
        raise
    finally:
        save()


def _log_launch(line: str):
    if LAUNCH_FD is not None:
        os.write(LAUNCH_FD, (line + "\n").encode())


def install_launch_log():
    import triton
    from triton.runtime import jit as tjit
    from triton.runtime import autotuner as tat
    orig = tjit.JITFunction.run

    def run(self, *args, grid, warmup, **kwargs):
        if not warmup:
            cfg = {k: v for k, v in kwargs.items() if k in ("num_warps", "num_stages", "BT", "BK", "BV", "BC", "BS", "BD", "BM", "BN", "BH")}
            shapes = [tuple(a.shape) + (str(a.dtype).replace("torch.", ""),) for a in args if hasattr(a, "shape")][:6]
            _log_launch(json.dumps({"k": self.fn.__name__, "cfg": cfg, "shapes": shapes}))
        ck = orig(self, *args, grid = grid, warmup = warmup, **kwargs)
        if not warmup:
            try:
                meta = {"post": self.fn.__name__, "shared": getattr(ck.metadata, "shared", None),
                        "n_regs": getattr(ck, "n_regs", None), "n_spills": getattr(ck, "n_spills", None),
                        "warp_size": getattr(ck.metadata, "warp_size", None), "num_warps": getattr(ck.metadata, "num_warps", None)}
                asm = getattr(ck, "asm", {}) or {}
                gcn = asm.get("amdgcn") if hasattr(asm, "get") else None
                if gcn and ASM_DIR is not None:
                    key = f"{self.fn.__name__}_w{meta['num_warps']}_" + "_".join(f"{k}{v}" for k, v in sorted(cfg.items()) if k != "num_warps")
                    p = ASM_DIR / f"{key}.amdgcn"
                    if not p.exists():
                        p.write_text(gcn, encoding = "utf-8")
                    for line in gcn.splitlines():
                        s = line.strip()
                        if s.startswith((".private_segment_fixed_size", ".group_segment_fixed_size", ".sgpr_count", ".vgpr_count",
                                         ".wavefront_size", ".sgpr_spill_count", ".vgpr_spill_count")):
                            kv = s.split(":", 1)
                            if len(kv) == 2:
                                meta[kv[0].lstrip(".")] = kv[1].strip()
                _log_launch(json.dumps(meta))
            except Exception as e:  # noqa: BLE001
                _log_launch(json.dumps({"post": self.fn.__name__, "meta_err": repr(e)}))
        return ck

    tjit.JITFunction.run = run
    orig_bench = tat.Autotuner._bench

    def _bench(self, *args, config, **meta):
        _log_launch(json.dumps({"bench": getattr(self, "kernel_name", None) or self.base_fn.__name__, "config": str(config)}))
        return orig_bench(self, *args, config = config, **meta)

    tat.Autotuner._bench = _bench
    RES["triton"] = triton.__version__


def autotuners():
    """{kernel_name: Autotuner} for every autotuned kernel in the imported fla modules."""
    from triton.runtime.autotuner import Autotuner
    found = {}
    for name, mod in list(sys.modules.items()):
        if not (name == "fla" or name.startswith("fla.")) or mod is None:
            continue
        for attr in list(vars(mod).values()):
            obj, depth = attr, 0
            while obj is not None and depth < 5:
                if isinstance(obj, Autotuner):
                    kname = getattr(obj, "kernel_name", None) or obj.base_fn.__name__
                    found[kname] = obj
                    break
                obj, depth = getattr(obj, "fn", None), depth + 1
    return found


def apply_pins():
    pins = os.environ.get("ZLA_PIN", "").strip()
    if not pins:
        return
    tuners = autotuners()
    applied = {}
    for item in pins.split(","):
        kname, idx = item.split("=")
        at = tuners[kname]
        cfg = at.configs[int(idx)]
        at.configs = [cfg]
        applied[kname] = str(cfg)
    RES["pins"] = applied


def import_fla():
    up = os.environ.get("ZLA_UPSTREAM_FLA")
    if up:
        sys.path.insert(0, up)
        import fla  # noqa: F401
    else:
        import unsloth  # noqa: F401  (injects the vendored fla)
        import fla  # noqa: F401
    import fla.ops.gated_delta_rule as gdr
    import fla.ops.common.chunk_delta_h  # noqa: F401
    import fla.ops.common.chunk_o  # noqa: F401
    import fla.ops.gated_delta_rule.chunk_fwd  # noqa: F401
    import fla.ops.gated_delta_rule.wy_fast  # noqa: F401
    import fla.ops.utils.cumsum  # noqa: F401
    import fla.modules.l2norm  # noqa: F401
    RES["fla_file"] = fla.__file__
    RES["fla_vendored"] = bool(getattr(fla, "_UNSLOTH_VENDORED_FLA", False))
    return gdr.chunk_gated_delta_rule


def versions():
    import torch
    RES["versions"] = {"torch": torch.__version__, "hip": getattr(torch.version, "hip", None)}
    p = torch.cuda.get_device_properties(0)
    RES["device"] = {k: str(getattr(p, k, None)) for k in ("name", "gcnArchName", "warp_size", "multi_processor_count",
                                                          "total_memory", "shared_memory_per_block",
                                                          "shared_memory_per_multiprocessor", "regs_per_multiprocessor")}
    try:
        import triton
        RES["device"]["triton_props"] = {k: v for k, v in triton.runtime.driver.active.utils.get_device_properties(0).items()}
        RES["device"]["triton_target"] = str(triton.runtime.driver.active.get_current_target())
    except Exception as e:  # noqa: BLE001
        RES["device"]["triton_props_err"] = repr(e)


def ref_gdr(q, k, v, g, beta, scale):
    """fp64 recurrent gated delta rule with qk l2norm: (B,T,H,K) layout, g in natural log space."""
    import torch
    q, k, v, g, beta = (x.double() for x in (q, k, v, g, beta))
    q = q / q.norm(dim = -1, keepdim = True).clamp_min(1e-6) * 1.0
    k = k / k.norm(dim = -1, keepdim = True).clamp_min(1e-6)
    B, T, H, K = q.shape
    V = v.shape[-1]
    S = torch.zeros(B, H, K, V, dtype = torch.float64, device = q.device)
    outs = []
    for t in range(T):
        S = S * g[:, t].exp()[..., None, None]
        kt = k[:, t]
        u = beta[:, t][..., None] * (v[:, t] - torch.einsum("bhk,bhkv->bhv", kt, S))
        S = S + kt[..., :, None] * u[..., None, :]
        outs.append(scale * torch.einsum("bhk,bhkv->bhv", q[:, t], S))
    return torch.stack(outs, 1)


def cell_op(spec):
    import torch
    _, dt, B, T, H, K, V = spec.split(":")
    B, T, H, K, V = int(B), int(T), int(H), int(K), int(V)
    dtype = getattr(torch, dt)
    fn = import_fla()
    apply_pins()
    versions()
    gen = torch.Generator().manual_seed(0)
    mk = lambda *s: torch.randn(*s, generator = gen)  # noqa: E731
    q, k, v = mk(B, T, H, K), mk(B, T, H, K), mk(B, T, H, V)
    g = torch.nn.functional.logsigmoid(mk(B, T, H)) * 0.5
    beta = torch.rand(B, T, H, generator = gen)
    do = mk(B, T, H, V) * 0.1
    scale = K ** -0.5
    leaves = [x.to("cuda", dtype).requires_grad_() for x in (q, k, v, g, beta)]
    holder = {}

    def fwd():
        o, _ = fn(*leaves, scale = scale, use_qk_l2norm_in_kernel = True, output_final_state = False)
        holder["o"] = o

    step("fwd", fwd)
    step("bwd", lambda: holder["o"].backward(do.to("cuda", dtype)))
    ref_leaves = [x.double().cuda().requires_grad_() for x in (q, k, v, g, beta)]
    o_ref = ref_gdr(*ref_leaves, scale)
    o_ref.backward(do.double().cuda())
    err = {}
    o = holder["o"].double()
    err["o"] = {"max_abs": float((o - o_ref).abs().max()), "ref_max": float(o_ref.abs().max()), "finite": bool(torch.isfinite(o).all())}
    for name, a, b in zip(("dq", "dk", "dv", "dg", "dbeta"), leaves, ref_leaves):
        ga, gb = a.grad.double(), b.grad
        err[name] = {"max_abs": float((ga - gb).abs().max()), "ref_max": float(gb.abs().max()), "finite": bool(torch.isfinite(ga).all())}
    RES["err_vs_fp64"] = err


def cell_model_real(spec):
    import torch
    dt = spec.split(":")[1]
    dtype = getattr(torch, dt)
    import unsloth  # noqa: F401
    import transformers
    versions()
    RES["fla_loaded"] = "fla" in sys.modules and sys.modules["fla"] is not None
    if RES["fla_loaded"]:
        RES["fla_file"] = getattr(sys.modules["fla"], "__file__", None)
        try:
            import fla.ops.gated_delta_rule  # noqa: F401
            import fla.ops.common.chunk_delta_h  # noqa: F401
            import fla.ops.gated_delta_rule.chunk_fwd  # noqa: F401
            import fla.ops.common.chunk_o  # noqa: F401
            apply_pins()
        except Exception as e:  # noqa: BLE001
            RES["fla_import_err"] = repr(e)
    from huggingface_hub import snapshot_download
    cls = transformers.Qwen3_5MoeForConditionalGeneration
    path = snapshot_download("trl-internal-testing/tiny-Qwen3_5MoeForConditionalGeneration-3.6")
    holder = {}
    step("load", lambda: holder.__setitem__("m", cls.from_pretrained(path, torch_dtype = dtype).to("cuda")))
    mod = sys.modules[holder["m"].model.language_model.layers[0].linear_attn.__class__.__module__]
    RES["chunk_fn"] = repr(getattr(holder["m"].model.language_model.layers[0].linear_attn, "chunk_gated_delta_rule", None))
    RES["modeling_module"] = mod.__name__
    ids = torch.randint(0, 100, (3, 6), generator = torch.Generator().manual_seed(1)).to("cuda")

    def fb():
        out = holder["m"](input_ids = ids, use_cache = False).logits
        holder["logits"] = out.detach().float()
        out.float().pow(2).mean().backward()

    step("forward_backward", fb)
    RES["logits_finite"] = bool(torch.isfinite(holder["logits"]).all())
    RES["logits_absmax"] = float(holder["logits"].abs().max())


def cell_env():
    import_fla()
    versions()
    RES["autotuners"] = {k: [str(c) for c in at.configs] for k, at in sorted(autotuners().items())}


def main():
    global OUT, LAUNCH_FD, ASM_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--launch-log", type = Path)
    ap.add_argument("--asm-dir", type = Path)
    a = ap.parse_args()
    OUT = a.out
    ASM_DIR = a.asm_dir
    if ASM_DIR is not None:
        ASM_DIR.mkdir(parents = True, exist_ok = True)
    RES["cell"] = a.cell
    RES["env"] = {k: os.environ.get(k) for k in ("AMD_SERIALIZE_KERNEL", "ZLA_PIN", "ZLA_UPSTREAM_FLA",
                                                 "AMDGCN_USE_BUFFER_OPS", "UNSLOTH_DISABLE_VENDORED_FLA", "TRITON_CACHE_DIR")}
    if a.launch_log:
        LAUNCH_FD = os.open(str(a.launch_log), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    save()
    try:
        install_launch_log()
        if a.cell == "env":
            cell_env()
        elif a.cell.startswith("op:"):
            cell_op(a.cell)
        elif a.cell.startswith("model:"):
            cell_model_real(a.cell)
        RES["stage"] = "done"
    except Exception as e:  # noqa: BLE001
        RES["error"] = f"{type(e).__name__}: {e}"[:1500]
        RES["traceback"] = traceback.format_exc()[-5000:]
    RES["current_step"] = None if RES.get("stage") == "done" else RES.get("current_step")
    save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
