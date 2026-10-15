#!/usr/bin/env python3
"""Probe: MiniMax-H3 GGUF on the Vulkan sd.cpp build, driven through THIS checkout's own Studio code.

Observes only; criteria/h3_uma_*.py judge. Per checkout it runs, in Studio's order:
  1. the device listing Studio caches (sd_cpp_listed_device, probe=True), then every read the load makes off that cache
     (class, description, AMD check, and the head's sd_cpp_vulkan_device_is_uma(probe=False)),
  2. the load's Auto flags (offload_flags("group") + GRAPH_CUT_VRAM_FLAGS [+ --stream-layers]) and its load-time fit,
  3. per render size, the resident-vs-offloaded choice (h3_native_render_flags on video._h3_card_free_bytes), then
     video._h3_offloaded_render_flags(..., reuse_live_server=False) exactly as the sd-cli render path calls it,
  4. the render itself through sd_cpp_engine.SdCppEngine.generate_video, and on failure the checkout's
     video._native_memory_failure (the advice a user sees) and output_shows_accelerator_failure.
Explicit modes (--fit-modes) run steps 2-3 only. Observations go to --out as JSON, never stdout.

Env: VK (sd-cli path), MODELS (dir holding the MiniMax-H3 GGUF files), OUT (artefact dir).
"""

from __future__ import annotations

import argparse
import glob
import inspect
import json
import os
import sys
import threading
import time
import types
from pathlib import Path

G = 1024**3
DIT = "minimax_h3_fl2va_pruned-Q4_K.gguf"
TE = "qwen3vl_32b_minimax_h3-Q4_K_M.gguf"
VAE = "vae/minimax_h3_video_vae_fp16.safetensors"
AVAE = "vae/minimax_h3_audio_vae_fp32.safetensors"
POLICY = {"auto": "group", "fast": "none", "balanced": "group", "low_vram": "model"}


def sysfs_sum(name: str) -> int:
    tot = 0
    for p in glob.glob(f"/sys/class/drm/card*/device/{name}"):
        try:
            with open(p, encoding = "utf-8") as fh:
                tot += int(fh.read().strip())
        except (OSError, ValueError):
            pass
    return tot


def meminfo(key: str) -> int:
    with open("/proc/meminfo", encoding = "utf-8") as fh:
        for line in fh:
            if line.startswith(key + ":"):
                return int(line.split()[1]) * 1024
    return 0


def readings() -> dict:
    if not Path("/proc/meminfo").exists():
        return {"note": "no /proc or amdgpu sysfs on this OS"}
    return {
        "gtt_total_gib": round(sysfs_sum("mem_info_gtt_total") / G, 2),
        "gtt_free_gib": round((sysfs_sum("mem_info_gtt_total") - sysfs_sum("mem_info_gtt_used")) / G, 2),
        "vram_total_gib": round(sysfs_sum("mem_info_vram_total") / G, 2),
        "vram_free_gib": round((sysfs_sum("mem_info_vram_total") - sysfs_sum("mem_info_vram_used")) / G, 2),
        "mem_available_gib": round(meminfo("MemAvailable") / G, 2),
    }


def err(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--case", required = True)
    ap.add_argument("--render", default = "864x480", help = "comma list WxH rendered under Auto")
    ap.add_argument("--fit-modes", default = "", help = "comma list of explicit memory modes, fit only")
    ap.add_argument("--frames", type = int, default = 124)
    ap.add_argument("--steps", type = int, default = 2)
    ap.add_argument("--repeats", type = int, default = 1)
    args = ap.parse_args()

    obs: dict = {"state": args.state, "case": args.case, "readings_start": readings()}
    vk_bin, models, out_root = os.environ["VK"], Path(os.environ["MODELS"]), Path(os.environ["OUT"])
    out_dir = out_root / f"{args.case}_{args.state}"
    out_dir.mkdir(parents = True, exist_ok = True)

    def save() -> None:
        args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")

    backend = args.checkout / "studio" / "backend"
    sys.path.insert(0, str(backend))
    try:
        from core.inference import sd_cpp_args as A  # noqa: PLC0415
        from core.inference import sd_cpp_backend as B  # noqa: PLC0415
        from core.inference import video as V  # noqa: PLC0415
        from core.inference import video_minimax_h3 as H  # noqa: PLC0415
        from core.inference.sd_cpp_engine import SdCppEngine  # noqa: PLC0415
        from core.inference.video_families import detect_video_family  # noqa: PLC0415
    except Exception as e:  # noqa: BLE001
        obs["import_error"] = err(e)
        save()
        return 0
    obs["imported_from"] = str(Path(H.__file__).resolve())
    obs["has_uma_api"] = hasattr(B, "sd_cpp_vulkan_device_is_uma")
    obs["install_tag"] = None
    try:
        sys.path.insert(0, str(args.checkout / "studio"))
        import install_sd_cpp_prebuilt as I  # noqa: PLC0415

        obs["install_tag"] = I.DEFAULT_TAG
    except Exception as e:  # noqa: BLE001
        obs["install_tag_error"] = err(e)
    try:
        obs["rocm_missing_sonames"] = B.rocm_runtime_missing_sonames()
    except Exception as e:  # noqa: BLE001
        obs["rocm_missing_error"] = err(e)

    # 1. listing, as the load reads it. Vulkan ordinals are not physical indices, so Studio pins by card name; the
    # runner has one card, which is Vulkan0.
    device_name = "Vulkan0"
    # The load's re-check is what fills the listing cache; every later read there is probe=False.
    obs["accelerator_verdict"] = B.sd_cpp_accelerator_device_verdict(vk_bin)
    listed_class, listed_desc = B.sd_cpp_listed_device(vk_bin, device_name, probe = False)
    check_gtt = listed_class == "vulkan" and H.vulkan_device_is_amd(listed_desc)
    uma = None
    if obs["has_uma_api"]:
        uma = B.sd_cpp_vulkan_device_is_uma(vk_bin, device_name, probe = False)
    obs.update(device_name = device_name, listed_class = listed_class, listed_desc = listed_desc,
               check_gtt = check_gtt, uma_reading = uma)

    # Fit-only legs (no render) may run without the 35 GB download: the published Hub sizes stand in.
    published = {DIT: 11420663904, TE: 18218065024, VAE: 5207808496, AVAE: 605254808}
    size = (lambda f: published[f]) if os.environ.get("H3_PROBE_PUBLISHED_SIZES") == "1" else (  # noqa: E731
        lambda f: os.path.getsize(models / f))
    obs["sizes_source"] = "published" if os.environ.get("H3_PROBE_PUBLISHED_SIZES") == "1" else "files"
    fam = detect_video_family("unsloth/MiniMax-H3-GGUF")
    presets = [p for p in (fam.resolution_presets or ()) if len(p) == 2]
    small_w, small_h = min(presets, key = lambda p: p[0] * p[1]) if presets else (864, 480)
    small_frames = fam.min_num_frames or fam.default_num_frames or 1
    obs["small_clip"] = [small_w, small_h, small_frames]
    graph_cut = B.sd_cpp_graph_cut_options(vk_bin)
    obs["graph_cut_options"] = sorted(graph_cut)

    fit_fields = set(H.H3NativeFitInputs.__dataclass_fields__)
    launch_takes_compute = "compute_bytes" in inspect.signature(V._h3_offloaded_render_flags).parameters
    live_takes_compute = "compute_bytes" in inspect.signature(H.h3_native_live_fit).parameters
    obs["launch_takes_compute"] = launch_takes_compute

    def compute_for(w: int, h: int, f: int):
        if hasattr(H, "h3_native_compute_bytes"):
            return H.h3_native_compute_bytes(w, h, f)
        return None

    def runtime_for(mode: str):
        offload = tuple(A.offload_flags(POLICY[mode], vae_tiling = False, diffusion_fa = True, vae_on_cpu = False))
        if "--max-vram" in graph_cut:
            offload += A.GRAPH_CUT_VRAM_FLAGS
            if "--offload-to-cpu" in offload and "--stream-layers" in graph_cut:
                offload += A.GRAPH_CUT_STREAM_FLAGS
        kw = dict(base_flags = tuple(offload), accelerator = listed_class, check_gtt = check_gtt,
                  gtt_pci_slot = None, dit_bytes = size(DIT), te_bytes = size(TE),
                  vae_bytes = size(VAE) + size(AVAE), device_name = device_name)
        if "uma" in fit_fields:
            kw["uma"] = check_gtt and uma is True
        inputs = H.H3NativeFitInputs(**kw)
        small = compute_for(small_w, small_h, small_frames)
        fit = H.h3_native_live_fit(inputs, mode, small) if live_takes_compute else H.h3_native_live_fit(inputs, mode)
        load = {"refusal": fit.refusal, "note": fit.note, "flags": list(fit.flags),
                "max_vram": getattr(fit, "max_vram", None)}
        committed = offload + tuple(A.device_backend_flags(device_name, list(offload)))
        rt = types.SimpleNamespace(fit_inputs = inputs, server_slot = None, offload_flags = committed,
                                   offloaded_flags = None, offload_refusal = None)
        if mode == "auto" and (fit.refusal or fit.note):
            need = H.h3_native_resident_bytes(size(DIT) + size(TE) + size(VAE) + size(AVAE), small_w, small_h,
                                              small_frames)
            _f, could = H.h3_native_render_flags(offload, memory_mode = mode,
                                                  free_bytes = V._h3_card_free_bytes("cuda", 0), need_bytes = need)
            load["load_refused"] = bool(fit.refusal and not could)
            rt.offload_refusal = fit.refusal
            if fit.note:
                rt.offloaded_flags = tuple(fit.flags) + tuple(A.device_backend_flags(device_name, list(fit.flags)))
        else:
            load["load_refused"] = bool(fit.refusal)
        return rt, load

    def launch(rt, mode: str, w: int, h: int, f: int) -> dict:
        files_bytes = size(DIT) + size(TE) + size(VAE) + size(AVAE)
        render_flags, resident = list(rt.offload_flags), False
        card_free = None
        if "--offload-to-cpu" in render_flags and mode == "auto":
            need = H.h3_native_resident_bytes(files_bytes, w, h, f)
            card_free = V._h3_card_free_bytes("cuda", 0)
            render_flags, resident = H.h3_native_render_flags(rt.offload_flags, memory_mode = mode,
                                                              free_bytes = card_free, need_bytes = need)
        kw = {"compute_bytes": compute_for(w, h, f)} if launch_takes_compute else {}
        res = {"card_free_bytes": card_free, "resident": resident}
        try:
            res["flags"] = V._h3_offloaded_render_flags(rt, render_flags, render_resident = resident,
                                                        memory_mode = mode, reuse_live_server = False, **kw)
            res["refusal"] = None
        except H.H3NativeMemoryRefusal as e:
            res["flags"], res["refusal"] = None, str(e)
        return res

    def render(flags: list, w: int, h: int, f: int, label: str) -> dict:
        from core.inference.sd_cpp_args import SdCppVideoGenParams  # noqa: PLC0415
        from core.inference.sd_cpp_args import SdCppModelFiles  # noqa: PLC0415

        files = SdCppModelFiles(diffusion_model = str(models / DIT), llm = str(models / TE),
                                vae = str(models / VAE), audio_vae = str(models / AVAE))
        params = SdCppVideoGenParams(prompt = "a red fox running through snow, cinematic", width = w, height = h,
                                     num_frames = f, fps = 24, steps = args.steps, cfg_scale = 1.0, seed = 1)
        log_path = out_dir / f"{label}.log"
        lines: list[str] = []
        peak = {"gtt": 0}
        stop = threading.Event()

        def sample() -> None:
            while not stop.is_set():
                peak["gtt"] = max(peak["gtt"], sysfs_sum("mem_info_gtt_used"))
                time.sleep(0.5)

        th = threading.Thread(target = sample, daemon = True)
        th.start()
        r: dict = {"label": label, "readings_before": readings()}
        t0 = time.monotonic()
        try:
            SdCppEngine(vk_bin).generate_video(files, params, output_path = str(out_dir / f"{label}.webm"),
                                               offload = list(flags), on_log = lines.append)
            r["ok"] = True
        except RuntimeError as e:
            r["ok"] = False
            r["error_head"] = str(e)[:300]
            r["user_advice"] = V._native_memory_failure(e)
            r["capacity_failure"] = B.output_shows_capacity_failure(str(e))
            r["accelerator_failure"] = B.output_shows_accelerator_failure(str(e))
        r["secs"] = round(time.monotonic() - t0, 1)
        stop.set()
        th.join()
        r["peak_gtt_used_gib"] = round(peak["gtt"] / G, 2)
        out = out_dir / f"{label}.webm"
        r["output_bytes"] = out.stat().st_size if out.is_file() else 0
        log_path.write_text("\n".join(lines), encoding = "utf-8")
        key = ("auto-detected", "cannot make enough memory", "failed during", "segments", "[E]")
        r["log_highlights"] = [ln[:400] for ln in lines if any(k in ln for k in key)][:20]
        # Every recoverable [W] capacity warning, for the marker question (does a SUCCESSFUL run print it?).
        r["capacity_warnings"] = sum("cannot make enough memory available" in ln for ln in lines)
        return r

    try:
        rt, load = runtime_for("auto")
        obs["auto_load_fit"] = load
        obs["auto"] = {}
        for spec in [s for s in args.render.split(",") if s]:
            w, h = (int(x) for x in spec.split("x"))
            entry = {"launch": launch(rt, "auto", w, h, args.frames), "renders": []}
            obs["auto"][spec] = entry
            save()
            flags = entry["launch"]["flags"]
            if flags is not None and args.repeats > 0:
                for i in range(args.repeats):
                    entry["renders"].append(render(flags, w, h, args.frames, f"{spec}_r{i}"))
                    save()
        obs["explicit"] = {}
        for mode in [m for m in args.fit_modes.split(",") if m]:
            rt_m, load_m = runtime_for(mode)
            obs["explicit"][mode] = {"load_fit": load_m, "launch_864x480": launch(rt_m, mode, 864, 480, args.frames)}
    except Exception as e:  # noqa: BLE001
        import traceback  # noqa: PLC0415

        obs["probe_error"] = err(e)
        obs["probe_traceback"] = traceback.format_exc()[-3000:]
    obs["readings_end"] = readings()
    save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
