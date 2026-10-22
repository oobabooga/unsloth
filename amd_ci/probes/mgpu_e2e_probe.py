#!/usr/bin/env python3
"""Probe: Studio's multi-GPU paths end to end on tiny pipelines, under the device multiplier.

Observes only; criteria/mgpu_e2e.py judges. Runs in one state's checkout (its own studio/backend
on sys.path). Phantom cuda:1 is the real gfx1151 wearing another number: these cells test wiring
(selection, decline reasons, device bookkeeping, a replica / split actually engaging and staying
bit-identical to one card), never speed or sharding.

Cells, each in its own process (a crash or hang in one cannot hide the others):
  cfg_flux        tiny FLUX.1, true CFG: selection (0, 1) -> install_cfg_parallel -> render x2 (2nd prompt new)
  cfg_qwen        tiny Qwen-Image, true CFG, same path
  cfg_flux_compiled  cfg_flux with regional block compile recorded (the replica re-applies it on cuda:1)
  zimage_pair     tiny Z-Image built in code: Z-Image CFG split + CFG-parallel pair
  video_wan_cfg   tiny Wan 2.1 text-to-video built in code: CFG-parallel on the video denoiser
  split_flux      tiny FLUX.1 with UNSLOTH_DIFFUSION_LAYER_SPLIT=force, planner told the card is discrete
  split_wan       tiny Wan transformer, layer split forced the same way
  component_flux  tiny FLUX.1 with UNSLOTH_DIFFUSION_MULTIGPU=force (companion split: encoders to cuda:1)
Per cell: the single-card reference hash, the multi-card hash, engagement counters, the decline
reason when not armed, and any exception with its traceback tail. Text I/O names utf-8.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

CELLS = (
    "cfg_flux",
    "cfg_qwen",
    "cfg_flux_compiled",
    "zimage_pair",
    "video_wan_cfg",
    "split_flux",
    "split_wan",
    "component_flux",
)


def _hash(t) -> str:
    import torch

    if isinstance(t, (list, tuple)):
        t = t[0]
    for attr in ("images", "frames"):
        if hasattr(t, attr):
            t = getattr(t, attr)
    if isinstance(t, (list, tuple)):
        t = t[0]
    if not isinstance(t, torch.Tensor):
        import numpy as np

        t = torch.from_numpy(np.ascontiguousarray(t))
    t = t.detach().float().cpu().contiguous()
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()[:16]


def _sync():
    import torch

    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)


def _dtype():
    import torch

    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _tiny(repo: str) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo)


def _flux():
    import diffusers

    pipe = diffusers.FluxPipeline.from_pretrained(_tiny("hf-internal-testing/tiny-flux-pipe"), torch_dtype = _dtype())
    kw = dict(true_cfg_scale = 4.0, negative_prompt = "blurry", guidance_scale = 3.5, height = 64, width = 64)
    return pipe, kw


def _qwen():
    import diffusers

    pipe = diffusers.QwenImagePipeline.from_pretrained(
        _tiny("hf-internal-testing/tiny-qwenimage-pipe"), torch_dtype = _dtype()
    )
    kw = dict(true_cfg_scale = 4.0, negative_prompt = "blurry", height = 64, width = 64)
    return pipe, kw


def _zimage():
    import diffusers
    import torch
    from transformers import AutoTokenizer, Qwen3ForCausalLM

    te_dir = _tiny("trl-internal-testing/tiny-Qwen3ForCausalLM")
    torch.manual_seed(0)
    transformer = diffusers.ZImageTransformer2DModel(
        all_patch_size = (2,), all_f_patch_size = (1,), in_channels = 16, dim = 64, n_layers = 2,
        n_refiner_layers = 1, n_heads = 2, n_kv_heads = 2, norm_eps = 1e-5, qk_norm = True, cap_feat_dim = 8,
        rope_theta = 256.0, t_scale = 1000.0, axes_dims = [16, 8, 8], axes_lens = [512, 64, 64],
    )
    vae = diffusers.AutoencoderKL(
        in_channels = 3, out_channels = 3, latent_channels = 16, block_out_channels = (16, 16),
        down_block_types = ("DownEncoderBlock2D",) * 2, up_block_types = ("UpDecoderBlock2D",) * 2,
        layers_per_block = 1, norm_num_groups = 8, shift_factor = 0.0, scaling_factor = 1.0,
    )
    pipe = diffusers.ZImagePipeline(
        scheduler = diffusers.FlowMatchEulerDiscreteScheduler(), vae = vae,
        text_encoder = Qwen3ForCausalLM.from_pretrained(te_dir), tokenizer = AutoTokenizer.from_pretrained(te_dir),
        transformer = transformer,
    ).to(dtype = _dtype())
    kw = dict(guidance_scale = 4.0, negative_prompt = "blurry", height = 64, width = 64)
    return pipe, kw


def _wan_transformer():
    import diffusers
    import torch

    torch.manual_seed(0)
    return diffusers.WanTransformer3DModel(
        patch_size = (1, 2, 2), num_attention_heads = 2, attention_head_dim = 12, in_channels = 16,
        out_channels = 16, text_dim = 32, freq_dim = 256, ffn_dim = 32, num_layers = 2, cross_attn_norm = True,
        qk_norm = "rms_norm_across_heads", rope_max_seq_len = 32,
    )


def _wan():
    import diffusers
    import torch
    from transformers import AutoTokenizer, T5EncoderModel

    t5 = _tiny("hf-internal-testing/tiny-random-t5")
    torch.manual_seed(0)
    vae = diffusers.AutoencoderKLWan(
        base_dim = 3, z_dim = 16, dim_mult = [1, 1, 1, 1], num_res_blocks = 1, temperal_downsample = [False, True, True]
    )
    pipe = diffusers.WanPipeline(
        transformer = _wan_transformer(), vae = vae, scheduler = diffusers.FlowMatchEulerDiscreteScheduler(shift = 7.0),
        text_encoder = T5EncoderModel.from_pretrained(t5), tokenizer = AutoTokenizer.from_pretrained(t5),
    ).to(dtype = _dtype())
    # the tiny VAE decodes in fp32 as Studio's video path does by default
    pipe.vae.to(torch.float32)
    kw = dict(guidance_scale = 5.0, negative_prompt = "blurry", height = 16, width = 16, num_frames = 9)
    return pipe, kw


BUILD = {"flux": _flux, "qwen": _qwen, "zimage": _zimage, "wan": _wan}


def _call(pipe, kw, prompt, steps, family, zs):
    import contextlib

    import torch

    g = torch.Generator("cpu").manual_seed(1234)
    out_kind = "pt" if family != "wan" else "pt"
    ctx = zs.render(pipe, 1) if (zs is not None and family == "zimage") else contextlib.nullcontext()
    with torch.no_grad(), ctx:
        out = pipe(prompt = prompt, num_inference_steps = steps, generator = g, output_type = out_kind, **kw)
    _sync()
    return out


def _modules():
    from core.inference import diffusion_multigpu as mg

    mods = {"mg": mg}
    try:
        from core.inference import diffusion_zimage_cfg_split as zs

        mods["zs"] = zs
    except Exception:  # noqa: BLE001 - older trees
        mods["zs"] = None
    return mods


def _compile_blocks(transformer) -> int:
    import torch

    n = 0
    for _name, mod in transformer.named_children():
        if isinstance(mod, torch.nn.ModuleList):
            for block in mod:
                block.compile(dynamic = False)
                n += 1
    return n


def _dynamo() -> dict:
    try:
        from torch._dynamo.utils import counters

        return {
            "unique_graphs": int(counters["stats"].get("unique_graphs", 0)),
            "graph_breaks": int(sum(counters["graph_break"].values())),
            "recompiles": int(sum(counters["recompiles"].values())) if "recompiles" in counters else 0,
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def _devices_of(module) -> list[str]:
    try:
        return sorted({str(p.device) for p in module.parameters()})
    except Exception:  # noqa: BLE001
        return []


def cell_cfg(name: str, family: str, compiled: bool) -> dict:
    import torch

    mods = _modules()
    mg, zs = mods["mg"], mods["zs"]
    rec: dict = {"cell": name, "family": family, "compiled": compiled}
    steps = 3
    prompts = ("a photo of a red fox", "a photo of a blue car")

    # single-card reference, same tier
    pipe, kw = BUILD[family]()
    pipe.set_progress_bar_config(disable = True)
    pipe.to("cuda:0")
    if zs is not None and family == "zimage":
        rec["zimage_ref_decline"] = zs.install_zimage_cfg_split(pipe, offload_policy = "none")
    if compiled:
        mg.install_compile_recorder()
        rec["compiled_blocks"] = _compile_blocks(pipe.transformer)
    ref = [_hash(_call(pipe, kw, p, steps, family, zs)) for p in prompts]
    rec["ref_hashes"] = ref
    if zs is not None and family == "zimage":
        zs.uninstall_zimage_cfg_split(pipe)
    del pipe
    torch.cuda.empty_cache()
    if compiled:
        torch._dynamo.reset()

    pipe, kw = BUILD[family]()
    pipe.set_progress_bar_config(disable = True)
    pipe.to("cuda:0")
    if zs is not None and family == "zimage":
        rec["zimage_decline"] = zs.install_zimage_cfg_split(pipe, offload_policy = "none")
    selection = mg.make_selection(0, [0, 1])
    rec["selection"] = None if selection is None else [selection.primary, *selection.secondaries]
    with mg.selection_scope(selection):
        if compiled:
            rec["compiled_blocks"] = _compile_blocks(pipe.transformer)
        rec["decline"] = mg.install_cfg_parallel(pipe, offload_policy = "none")
    router = getattr(pipe.transformer, "_unsloth_cfg_router", None)
    rec["armed"] = router is not None
    rec["secondary"] = None if router is None else str(getattr(router, "secondary", None))
    hashes = []
    d0 = None
    for i, p in enumerate(prompts):
        if i == 1:
            d0 = _dynamo()
        hashes.append(_hash(_call(pipe, kw, p, steps, family, zs)))
        mg.end_cfg_render(pipe)
    rec["hashes"] = hashes
    rec["bit_identical"] = hashes == ref
    if compiled:
        d1 = _dynamo()
        rec["prompt2_new"] = {k: d1.get(k, 0) - (d0 or {}).get(k, 0) for k in ("unique_graphs", "graph_breaks", "recompiles")}
    if router is not None:
        rec["router_stats"] = {k: v for k, v in router.stats.items() if not isinstance(v, (list, dict))}
        rec["router_disabled"] = getattr(router, "disabled_reason", None)
        replica = getattr(router, "replica", None)
        rec["replica_devices"] = _devices_of(replica) if isinstance(replica, torch.nn.Module) else None
    if zs is not None and family == "zimage":
        rec["zimage_status"] = zs.zimage_cfg_split_status(pipe)
    rec["status"] = mg.placement_status(pipe)
    mg.uninstall_cfg_parallel(pipe)
    if zs is not None and family == "zimage":
        zs.uninstall_zimage_cfg_split(pipe)
    return rec


class _Target:
    supports_model_cpu_offload = True
    ordinal = 0
    device = "cuda"
    backend = "rocm"
    is_cuda_torch_device = True


def _plan(pipe, memory_kind: str = "discrete"):
    import torch

    from core.inference import diffusion_memory as dm

    target = _Target()
    try:
        from core.inference.diffusion_device import resolve_diffusion_device_target

        target = resolve_diffusion_device_target()
    except Exception:  # noqa: BLE001
        pass
    sizes = dm._loaded_component_mib(pipe) or {}
    free, total = torch.cuda.mem_get_info(0)
    mem = dm.DeviceMemory(
        backend = getattr(target, "backend", "cuda"), device = "cuda", memory_kind = memory_kind,
        free_mib = int(free) >> 20, total_mib = int(total) >> 20,
    )
    dense = sum(v[0] for v in sizes.values())
    te = sum(v[0] for v in sizes.values() if v[1] == "text_encoder")
    comp = sum(v[0] for v in sizes.values() if v[1] != "dit")
    return dm.plan_diffusion_memory(
        target = target, device_memory = mem, model_dense_mib = dense, companion_dense_mib = comp,
        text_encoder_dense_mib = te, runtime_headroom_mib = 1024, requested_mode = "auto",
    )


class _BarePipe:
    def __init__(self, transformer):
        self.transformer = transformer
        self.components = {"transformer": transformer}

    def to(self, *a, **k):
        self.transformer.to(*a, **k)
        return self


def _wan_inputs():
    import torch

    g = torch.Generator("cpu").manual_seed(7)
    dt = _dtype()
    return dict(
        hidden_states = torch.randn(1, 16, 3, 8, 8, generator = g).to("cuda:0", dt),
        timestep = torch.tensor([500.0], device = "cuda:0"),
        encoder_hidden_states = torch.randn(1, 8, 32, generator = g).to("cuda:0", dt),
        return_dict = False,
    )


def cell_split(name: str, family: str) -> dict:
    """Layer split forced, the planner told the card is discrete (this host's APU declines a split by design)."""
    import torch

    os.environ["UNSLOTH_DIFFUSION_LAYER_SPLIT"] = "force"
    mods = _modules()
    mg = mods["mg"]
    from core.inference import diffusion_layer_split_plan as lsp

    rec: dict = {"cell": name, "family": family}
    if family == "wan":
        model = _wan_transformer().to("cuda:0", _dtype()).eval()
        with torch.no_grad():
            ref = _hash(model(**_wan_inputs())[0])
        pipe = _BarePipe(model)
    else:
        pipe, kw = BUILD[family]()
        pipe.set_progress_bar_config(disable = True)
        pipe.to("cuda:0")
        ref = _hash(_call(pipe, kw, "a photo of a red fox", 2, family, None))
    rec["ref_hash"] = ref
    plan = _plan(pipe)
    rec["single_plan_offload"] = getattr(plan, "offload_policy", None)
    if hasattr(lsp, "_rocm_card_unified"):
        lsp._rocm_card_unified = lambda o: False
    selection = mg.make_selection(0, [0, 1])
    rec["selection"] = None if selection is None else [selection.primary, *selection.secondaries]
    with mg.selection_scope(selection):
        refined = mg.refine_plan_for_multigpu(pipe, plan)
        split = getattr(refined, "layer_split", None)
        rec["engaged_plan"] = split is not None
        rec["declined"] = (lsp.layer_split_status(pipe) or {}).get("declined")
        if split is not None:
            rec["cards"] = list(split.cards)
            rec["ranges"] = [[r.denoiser, r.start, r.stop, r.card] for r in split.ranges]
            mg.apply_split_plan(pipe, refined)
            rec["devices_after_place"] = _devices_of(pipe.transformer)
            if family == "wan":
                with torch.no_grad():
                    rec["hash"] = _hash(pipe.transformer(**_wan_inputs())[0])
            else:
                rec["hash"] = _hash(_call(pipe, kw, "a photo of a red fox", 2, family, None))
            rec["bit_identical"] = rec["hash"] == ref
            rec["split_status"] = lsp.layer_split_status(pipe)
    return rec


def cell_component(name: str, family: str) -> dict:
    """The companion split: MULTIGPU=force on a resident plan moves the encoders / VAE to cuda:1."""
    import torch

    os.environ["UNSLOTH_DIFFUSION_MULTIGPU"] = "force"
    mods = _modules()
    mg = mods["mg"]
    from core.inference import diffusion_layer_split_plan as lsp

    rec: dict = {"cell": name, "family": family}
    pipe, kw = BUILD[family]()
    pipe.set_progress_bar_config(disable = True)
    pipe.to("cuda:0")
    ref = _hash(_call(pipe, kw, "a photo of a red fox", 2, family, None))
    rec["ref_hash"] = ref
    plan = _plan(pipe)
    if hasattr(lsp, "_rocm_card_unified"):
        lsp._rocm_card_unified = lambda o: False
    selection = mg.make_selection(0, [0, 1])
    rec["selection"] = None if selection is None else [selection.primary, *selection.secondaries]
    with mg.selection_scope(selection):
        refined = mg.refine_plan_for_multigpu(pipe, plan)
        rec["secondary_ordinal"] = getattr(refined, "secondary_ordinal", None)
        rec["secondary_components"] = list(getattr(refined, "secondary_components", None) or ())
        if getattr(refined, "secondary_ordinal", None) is not None:
            mg.apply_split_plan(pipe, refined)
            rec["placement"] = (mg.placement_status(pipe) or {}).get("placement")
            rec["hash"] = _hash(_call(pipe, kw, "a photo of a red fox", 2, family, None))
            rec["bit_identical"] = rec["hash"] == ref
    return rec


def child(cell: str) -> dict:
    if cell == "cfg_flux":
        return cell_cfg(cell, "flux", False)
    if cell == "cfg_qwen":
        return cell_cfg(cell, "qwen", False)
    if cell == "cfg_flux_compiled":
        return cell_cfg(cell, "flux", True)
    if cell == "zimage_pair":
        return cell_cfg(cell, "zimage", False)
    if cell == "video_wan_cfg":
        return cell_cfg(cell, "wan", False)
    if cell == "split_flux":
        return cell_split(cell, "flux")
    if cell == "split_wan":
        return cell_split(cell, "wan")
    if cell == "component_flux":
        return cell_component(cell, "flux")
    raise SystemExit(f"unknown cell {cell}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--cells", default = ",".join(CELLS))
    ap.add_argument("--timeout", type = int, default = 900)
    ap.add_argument("--child", default = None)
    args = ap.parse_args()
    args.out = args.out.resolve()
    args.checkout = str(Path(args.checkout).resolve())
    backend = str(Path(args.checkout) / "studio" / "backend")

    if args.child:
        sys.path.insert(0, backend)
        os.chdir(backend)
        try:
            rec = child(args.child)
            rec["ok"] = True
        except BaseException as exc:  # noqa: BLE001 - the observation is the failure
            rec = {"cell": args.child, "ok": False, "error": f"{type(exc).__name__}: {str(exc)[:600]}",
                   "traceback": traceback.format_exc()[-4000:]}
        args.out.write_text(json.dumps(rec, indent = 2, default = str), encoding = "utf-8")
        return 0

    obs: dict = {"state": args.state, "checkout": args.checkout, "cells": {}}
    try:
        info = subprocess.run(
            [args.python, "-c", "import json,torch;print(json.dumps({'device_count':torch.cuda.device_count(),"
             "'names':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],"
             "'hip':torch.version.hip,'torch':torch.__version__}))"],
            capture_output = True, text = True, timeout = 300, encoding = "utf-8",
        )
        obs["host"] = json.loads(info.stdout.strip().splitlines()[-1])
    except Exception as exc:  # noqa: BLE001
        obs["host"] = {"error": f"{type(exc).__name__}: {exc}"}
    tmp = args.out.parent / f"_cells_{args.state}"
    tmp.mkdir(parents = True, exist_ok = True)
    for cell in [c for c in args.cells.split(",") if c]:
        out = tmp / f"{cell}.json"
        t0 = time.time()
        try:
            p = subprocess.run(
                [args.python, os.path.abspath(__file__), "--state", args.state, "--checkout", args.checkout,
                 "--out", str(out), "--child", cell],
                capture_output = True, text = True, timeout = args.timeout, encoding = "utf-8", errors = "replace",
            )
            rec = json.loads(out.read_text(encoding = "utf-8")) if out.exists() else {
                "cell": cell, "ok": False, "error": f"child rc={p.returncode}, no output"}
            rec["rc"] = p.returncode
            if p.returncode != 0 or not rec.get("ok"):
                rec["stderr_tail"] = (p.stderr or "")[-3000:]
        except subprocess.TimeoutExpired as exc:
            rec = {"cell": cell, "ok": False, "error": f"timeout after {args.timeout}s",
                   "stderr_tail": str(exc.stderr or "")[-3000:]}
        rec["wall_s"] = round(time.time() - t0, 1)
        obs["cells"][cell] = rec
        args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
