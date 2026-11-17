#!/usr/bin/env python3
"""Probe: MiniMax-H3 conditioner encode time through Studio's own ``stream_h3_text_encoder`` on this GPU.

Observes only (judged by criteria/h3_te_encode_timing.py). The conditioner is the real H3 text encoder config
(Qwen3-VL-32B, 50 of 64 layers kept, ConvRot INT8 projections, terminator slot, bf16 embedding and vision tower) built
the way the hosted loader builds it, with deterministic synthetic int8 weights (no download). Each mode runs in its own
subprocess so host / device memory starts clean:

  lazy       pin=False: the pageable host copy (the L4 / 16 GiB plan, where the conditioner yields the pin budget)
  lazy_off   the same with UNSLOTH_H3_TE_LAYER_STREAM=0 (the head's kill switch; the base ignores it: an A/A pair)
  pinned     pin=True: the conditioner pinned in place (the A100 plan)
  pinned_off the same with the kill switch

The runner is an integrated GPU, where Studio streams synchronously and never takes these paths; the probe treats the
device as discrete (``host_pinning.device_is_integrated`` -> False) in every state, which is what an L4 / A100 runs.
Between encodes the pinned host cache is emptied, as Studio's post-render ``reclaim_offload_host_memory`` does.
Per encode: wall time (synchronised), peak allocated device bytes, sha256 of hidden_states[50]. One encode per mode is
profiled for runtime API calls (memcpy / synchronize / host alloc counts)."""

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

os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

MODES = {"lazy": (False, None), "lazy_off": (False, "0"), "pinned": (True, None), "pinned_off": (True, "0")}


def _write(path: Path, obs: dict) -> None:
    path.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")


def build(torch, te, config_dir: Path, layers_kept: int):
    import transformers
    from accelerate import init_empty_weights

    config = transformers.AutoConfig.from_pretrained(str(config_dir))
    config.text_config.num_hidden_layers = layers_kept + 1
    with init_empty_weights(include_buffers = False):
        enc = transformers.Qwen3VLForConditionalGeneration(config)
    lm = enc.model.language_model
    lm.layers[layers_kept] = te._terminator_layer_class()()
    lm.norm = torch.nn.Identity()
    enc.lm_head = torch.nn.Identity()
    cls = te._int8_convrot_linear_class()
    for layer in lm.layers[:layers_kept]:
        for parent in (layer.self_attn, layer.mlp):
            for name, child in list(parent.named_children()):
                if isinstance(child, torch.nn.Linear):
                    setattr(parent, name, cls(child.in_features, child.out_features, child.bias is not None, 256))
    gen = torch.Generator().manual_seed(1234)
    pool = torch.randint(-127, 128, (1 << 28,), dtype = torch.int8, generator = gen)  # 256 MiB, tiled
    cursor = [0]

    def fill_int8(n):
        out = torch.empty(n, dtype = torch.int8)
        done = 0
        while done < n:
            start = cursor[0] % pool.numel()
            take = min(n - done, pool.numel() - start)
            out[done : done + take].copy_(pool[start : start + take])
            done += take
            cursor[0] += take + 4099
        return out

    for mname, module in enc.named_modules():
        for table, is_param in ((module._parameters, True), (module._buffers, False)):
            for key, value in list(table.items()):
                if value is None or not value.is_meta:
                    continue
                shape = value.shape
                if value.dtype == torch.int8:
                    real = fill_int8(value.numel()).view(shape)
                elif key == "weight_scale":
                    fan_in = int(getattr(module, "in_features", 1))
                    real = torch.full(shape, 0.5 / (73.0 * fan_in ** 0.5), dtype = torch.float32)
                elif key == "weight" and value.dim() == 1:
                    real = torch.ones(shape, dtype = torch.bfloat16)
                else:
                    real = (torch.randn(shape, generator = gen) * 0.02).to(torch.bfloat16)
                table[key] = torch.nn.Parameter(real, requires_grad = False) if is_param else real
    enc.requires_grad_(False)
    enc.eval()
    return enc


class _Hook:
    def __init__(self, model):
        self.model = model
        self.hook = type("H", (), {"other_hooks": []})()

    def remove(self):
        return None

    def offload(self):
        return None


def run_mode(args) -> dict:
    out: dict = {"mode": args.mode}
    sys.path.insert(0, str(args.checkout / "studio" / "backend"))
    import torch

    import core.inference.host_pinning as hp
    import core.inference.video_minimax_h3_te as te

    # Studio's discrete-GPU route on this UMA device, identically in every state.
    hp.device_is_integrated = lambda device = None: False
    hp.host_pinning_allowed = lambda device = None: True
    props = torch.cuda.get_device_properties(0)
    out["arch"] = (getattr(props, "gcnArchName", "") or "").split(":")[0]
    out["device_name"] = torch.cuda.get_device_name(0)
    out["torch"] = torch.__version__
    pin, switch = MODES[args.mode]
    if switch is not None:
        os.environ["UNSLOTH_H3_TE_LAYER_STREAM"] = switch
    t0 = time.perf_counter()
    enc = build(torch, te, args.config, args.layers)
    out["build_s"] = round(time.perf_counter() - t0, 1)
    out["host_bytes"] = sum(t.numel() * t.element_size() for t in list(enc.parameters()) + list(enc.buffers()))

    class _Log:
        def __init__(self):
            self.lines = []

        def info(self, msg, *a):
            self.lines.append(msg % a if a else msg)

        warning = info

    log = _Log()
    manager = type("M", (), {})()
    manager.model_hooks = [_Hook(enc)]
    t0 = time.perf_counter()
    mode = te.stream_h3_text_encoder(manager, enc, "cuda:0", pin = pin, logger = log)
    out["stream_s"] = round(time.perf_counter() - t0, 1)
    out["te_streamed"] = mode
    out["log"] = log.lines
    streamer = getattr(enc.model, "_unsloth_h3_te_layer_stream", None)
    out["layer_stream_installed"] = streamer is not None
    gen = torch.Generator().manual_seed(7)
    ids = torch.randint(0, 151000, (1, args.tokens), generator = gen).to("cuda:0")
    mask = torch.ones_like(ids)

    def encode():
        with torch.no_grad():
            return enc.model(input_ids = ids, attention_mask = mask, output_hidden_states = True).hidden_states[
                te.H3_TE_READ_LAYER
            ]

    walls, peaks, shas = [], [], []
    for i in range(args.repeats + 1):
        torch._C._host_emptyCache()  # Studio's post-render reclaim
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base_alloc = torch.cuda.memory_allocated()
        t0 = time.perf_counter()
        h = encode()
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        sha = hashlib.sha256(h.detach().float().cpu().numpy().tobytes()).hexdigest()
        finite = bool(torch.isfinite(h).all())
        del h
        if i == 0:
            out["first_encode_s"] = round(wall, 3)  # diffusers records the layer order on the first pass
            continue
        walls.append(round(wall, 3))
        peaks.append(torch.cuda.max_memory_allocated() - base_alloc)
        shas.append(sha)
        out["finite"] = finite
    out["encode_s"] = walls
    out["peak_alloc_bytes"] = peaks
    out["sha256"] = shas
    if streamer is not None:
        out["streamer"] = {k: getattr(streamer, k) for k in ("sessions", "layers_streamed", "bytes_staged", "bytes_direct", "fallbacks")}
    # One profiled encode: runtime API calls on the host.
    try:
        from torch.profiler import ProfilerActivity, profile

        torch._C._host_emptyCache()
        with profile(activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            encode()
            torch.cuda.synchronize()
        calls: dict = {}
        for ev in prof.key_averages():
            k = ev.key
            if any(s in k for s in ("Memcpy", "memcpy", "Synchronize", "HostAlloc", "HostMalloc", "HostRegister",
                                    "hostMalloc", "EventSynchronize", "StreamWait")):
                calls[k[:80]] = {"count": int(ev.count), "cpu_ms": round(ev.cpu_time_total / 1000, 2)}
        out["profile_calls"] = calls
    except Exception as exc:  # noqa: BLE001
        out["profile_error"] = repr(exc)[:300]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--config", type = Path, default = Path(__file__).resolve().parent / "h3_te_config")
    ap.add_argument("--layers", type = int, default = 50)
    ap.add_argument("--tokens", type = int, default = 256)
    ap.add_argument("--repeats", type = int, default = 3)
    ap.add_argument("--modes", default = "lazy,lazy_off,pinned,pinned_off")
    ap.add_argument("--mode", default = None, help = "internal: run one mode in this process")
    ap.add_argument("--mode-timeout-s", type = int, default = 1500)
    args = ap.parse_args()

    if args.mode:
        try:
            res = run_mode(args)
        except Exception as exc:  # noqa: BLE001
            res = {"mode": args.mode, "error": repr(exc)[:500], "trace": traceback.format_exc()[-3000:]}
        _write(args.out, res)
        return 0

    obs: dict = {"state": args.state, "modes": {}}
    for mode in [m for m in args.modes.split(",") if m]:
        part = args.out.with_name(args.out.stem + f".{mode}.json")
        cmd = [sys.executable, __file__, "--state", args.state, "--checkout", str(args.checkout), "--out", str(part),
               "--config", str(args.config), "--layers", str(args.layers), "--tokens", str(args.tokens),
               "--repeats", str(args.repeats), "--mode", mode]
        env = {k: v for k, v in os.environ.items() if k != "UNSLOTH_H3_TE_LAYER_STREAM"}
        t0 = time.time()
        try:
            rc = subprocess.run(cmd, env = env, timeout = args.mode_timeout_s).returncode
            res = json.loads(part.read_text(encoding = "utf-8")) if part.exists() else {"error": f"rc={rc}, no output"}
        except subprocess.TimeoutExpired:
            res = {"error": f"timeout after {args.mode_timeout_s} s"}
        res["mode_wall_s"] = round(time.time() - t0, 1)
        obs["modes"][mode] = res
        _write(args.out, obs)
    _write(args.out, obs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
