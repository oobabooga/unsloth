"""Measure what a real diffusers image pipeline costs on MPS above its weight bytes.

Run once per model in a fresh process: python mps_measure.py klein|zimage <size>
Prints one JSON line prefixed with RESULT.
"""
import ctypes
import json
import os
import sys
import threading
import time

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")  # what Studio sets

import psutil
import torch

GiB = 1024**3


class RusageV4(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"f{i}", ctypes.c_uint64) for i in range(35)]


_libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")


def footprint():
    """(phys_footprint, lifetime_max_phys_footprint) in bytes: what jetsam and Activity Monitor charge."""
    info = RusageV4()
    rc = _libc.proc_pid_rusage(os.getpid(), 4, ctypes.byref(info))
    if rc != 0:
        return None, None
    return info.f7, info.f28


def mps():
    return torch.mps.current_allocated_memory(), torch.mps.driver_allocated_memory()


class Poller(threading.Thread):
    def __init__(self):
        super().__init__(daemon = True)
        self.peak_driver = 0
        self.peak_fp = 0
        self.stop = False

    def run(self):
        while not self.stop:
            self.peak_driver = max(self.peak_driver, torch.mps.driver_allocated_memory())
            fp, _ = footprint()
            self.peak_fp = max(self.peak_fp, fp or 0)
            time.sleep(0.02)


def param_bytes(*modules):
    total = 0
    for m in modules:
        for t in list(m.parameters()) + list(m.buffers()):
            total += t.numel() * t.element_size()
    return total


def load_only(repo, fname, cls_name, base):
    import diffusers
    from diffusers import GGUFQuantizationConfig
    from huggingface_hub import hf_hub_download
    torch.ones(1, device = "mps")
    torch.mps.synchronize()
    fp0, _ = footprint()
    gguf = hf_hub_download(repo, fname)
    cls = getattr(diffusers, cls_name)
    tr = cls.from_single_file(gguf, quantization_config = GGUFQuantizationConfig(compute_dtype = torch.bfloat16),
                              config = base, subfolder = "transformer", torch_dtype = torch.bfloat16).to("mps")
    torch.mps.synchronize()
    fp1, fpmax = footprint()
    w = param_bytes(tr)
    cur, drv = mps()
    g = lambda b: round(b / GiB, 3)
    print("RESULT " + json.dumps({"mode": "load", "file": fname, "gguf_file": os.path.getsize(gguf), "weights": w,
                                  "fp_delta": fp1 - fp0, "fp_max_delta": fpmax - fp0, "mps": [cur, drv]}), flush = True)
    print(f"load {fname}: file {g(os.path.getsize(gguf))} GiB, params {g(w)}, footprint +{g(fp1 - fp0)} "
          f"(over params {g(fp1 - fp0 - w)}), load max +{g(fpmax - fp0)}, metal {g(cur)}/{g(drv)}")


def text_encoder(base, tokens):
    from transformers import AutoModel
    torch.ones(1, device = "mps")
    torch.mps.synchronize()
    fp0, _ = footprint()
    te = AutoModel.from_pretrained(base, subfolder = "text_encoder", torch_dtype = torch.bfloat16).to("mps")
    torch.mps.synchronize()
    fp1, fpmax_load = footprint()
    w = param_bytes(te)
    cur, drv = mps()
    poller = Poller()
    poller.start()
    ids = torch.randint(0, 150000, (1, tokens), device = "mps")
    with torch.no_grad():
        out = te(input_ids = ids, attention_mask = torch.ones_like(ids), output_hidden_states = True)
        h = out.hidden_states[-2]
    torch.mps.synchronize()
    poller.stop = True
    poller.join()
    fp2, fpmax = footprint()
    g = lambda b: round(b / GiB, 3)
    print("RESULT " + json.dumps({"mode": "te", "tokens": tokens, "weights": w, "fp_delta": fp1 - fp0,
                                  "fp_load_max_delta": fpmax_load - fp0, "mps": [cur, drv],
                                  "peak_driver": poller.peak_driver, "fp_max_delta": fpmax - fp0}), flush = True)
    print(f"text encoder {tokens} tokens: params {g(w)}, footprint +{g(fp1 - fp0)} (over params {g(fp1 - fp0 - w)}), "
          f"load max +{g(fpmax_load - fp0)}, metal {g(cur)}/{g(drv)} loaded, encode peak metal {g(poller.peak_driver)} "
          f"(+{g(poller.peak_driver - drv)}), lifetime max footprint +{g(fpmax - fp0)}")


def main():
    which = sys.argv[1]
    if which == "load":
        return load_only(*sys.argv[2:6])
    if which == "te":
        return text_encoder(sys.argv[2], int(sys.argv[3]))
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    from diffusers import GGUFQuantizationConfig
    from huggingface_hub import hf_hub_download

    torch.ones(1, device = "mps")
    torch.mps.synchronize()
    out = {"model": which, "size": size, "torch": torch.__version__,
           "ws": torch.mps.recommended_max_memory(), "ram": psutil.virtual_memory().total}
    out["avail0"] = psutil.virtual_memory().available
    out["fp0"], _ = footprint()
    qc = GGUFQuantizationConfig(compute_dtype = torch.bfloat16)
    if which == "klein":
        from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
        base = "unsloth/FLUX.2-klein-4B"
        gguf = hf_hub_download("unsloth/FLUX.2-klein-4B-GGUF", "flux-2-klein-4b-Q2_K.gguf")
        tr = Flux2Transformer2DModel.from_single_file(
            gguf, quantization_config = qc, config = base, subfolder = "transformer", torch_dtype = torch.bfloat16)
        pipe = Flux2KleinPipeline.from_pretrained(
            base, transformer = tr, text_encoder = None, tokenizer = None, torch_dtype = torch.bfloat16)
        embeds = torch.randn(1, 512, 3 * 2560, dtype = torch.bfloat16)
        kwargs = dict(prompt_embeds = embeds, guidance_scale = 1.0, num_inference_steps = 2)
    else:
        from diffusers import ZImagePipeline, ZImageTransformer2DModel
        base = "unsloth/Z-Image-Turbo"
        gguf = hf_hub_download("unsloth/Z-Image-Turbo-GGUF", "z-image-turbo-Q2_K.gguf")
        tr = ZImageTransformer2DModel.from_single_file(
            gguf, quantization_config = qc, config = base, subfolder = "transformer", torch_dtype = torch.bfloat16)
        pipe = ZImagePipeline.from_pretrained(
            base, transformer = tr, text_encoder = None, tokenizer = None, torch_dtype = torch.bfloat16)
        embeds = [torch.randn(256, 2560, dtype = torch.bfloat16)]
        kwargs = dict(prompt_embeds = embeds, guidance_scale = 0.0, num_inference_steps = 2)
    out["gguf_file"] = os.path.getsize(gguf)
    pipe.to("mps")
    # Studio turns these on for mps
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    torch.mps.synchronize()
    out["weights"] = param_bytes(pipe.transformer, pipe.vae)
    out["fp_loaded"], _ = footprint()
    out["mps_loaded"] = mps()
    out["avail_loaded"] = psutil.virtual_memory().available
    if isinstance(embeds, list):
        kwargs["prompt_embeds"] = [e.to("mps") for e in embeds]
    else:
        kwargs["prompt_embeds"] = embeds.to("mps")
    poller = Poller()
    poller.start()
    t0 = time.time()
    image = pipe(height = size, width = size, generator = torch.Generator("mps").manual_seed(0), **kwargs).images[0]
    torch.mps.synchronize()
    poller.stop = True
    poller.join()
    out["gen_s"] = round(time.time() - t0, 1)
    out["fp_after"], out["fp_lifetime_max"] = footprint()
    out["peak_driver"] = poller.peak_driver
    out["peak_fp_polled"] = poller.peak_fp
    out["image"] = list(image.size)
    print("RESULT " + json.dumps(out), flush = True)
    g = lambda b: round(b / GiB, 3)
    print(f"{which} {size}px: weights {g(out['weights'])} GiB (gguf file {g(out['gguf_file'])}), "
          f"footprint +{g(out['fp_loaded'] - out['fp0'])} at load, lifetime max +{g(out['fp_lifetime_max'] - out['fp0'])}, "
          f"metal driver {g(out['mps_loaded'][1])} loaded / {g(out['peak_driver'])} peak, "
          f"ws {g(out['ws'])}, gen {out['gen_s']} s")


if __name__ == "__main__":
    main()
