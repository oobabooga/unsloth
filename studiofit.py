"""Run Studio's patched H3 native fit on this host's live readings; print the flags (or the refusal) as JSON."""
import json, os, re, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pkg"))
from core.inference import video_minimax_h3 as m

vk, w, h = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
M = os.environ["MODELS"]
listing = subprocess.run([vk, "--list-devices"], capture_output=True, text=True).stdout
listing += subprocess.run([vk, "--list-devices"], capture_output=True, text=True).stderr
uma = re.search(r"^ggml_vulkan: 0 = .*?\| uma: ([01]) \|", listing, re.M)
size = lambda f: os.path.getsize(os.path.join(M, f))
inputs = m.H3NativeFitInputs(
    base_flags=("--offload-to-cpu", "--diffusion-fa", "--max-vram", "-1"),
    accelerator="vulkan", check_gtt=True, gtt_pci_slot=None,
    dit_bytes=size("minimax_h3_fl2va_pruned-Q4_K.gguf"), te_bytes=size("qwen3vl_32b_minimax_h3-Q4_K_M.gguf"),
    vae_bytes=size("vae/minimax_h3_video_vae_fp16.safetensors") + size("vae/minimax_h3_audio_vae_fp32.safetensors"),
    device_name="Vulkan0", uma=bool(uma and uma.group(1) == "1"),
)
fit = m.h3_native_live_fit(inputs, "auto", m.h3_native_compute_bytes(w, h, 124))
flags = list(fit.flags)
te = "cpu" if "--clip-on-cpu" in flags else "Vulkan0"
flags += ["--backend", f"diffusion=Vulkan0,te={te},vae=Vulkan0"]
print(json.dumps({"uma": inputs.uma, "refusal": fit.refusal, "note": fit.note, "max_vram": fit.max_vram, "flags": flags}))
