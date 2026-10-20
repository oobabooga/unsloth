"""Footprint of a Studio root: sizes, heavy packages, recorded mode. Usage: measure.py <root> <label> <outdir>"""
import json
import os
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
label = sys.argv[2]
out = Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)


def size(p):
    total = 0
    for dp, dn, fn in os.walk(p, followlinks=False):
        for f in fn:
            try:
                total += os.lstat(os.path.join(dp, f)).st_size
            except OSError:
                pass
    return total


def gb(n):
    return round(n / 1024**3, 3)


res = {"label": label, "root": str(root), "exists": root.exists()}
if root.exists():
    res["total_gb"] = gb(size(root))
    res["children"] = {}
    for c in sorted(root.iterdir()):
        res["children"][c.name] = gb(size(c)) if c.is_dir() else gb(c.stat().st_size)
    venv = root / "unsloth_studio"
    sp = venv / "Lib" / "site-packages"
    if sp.exists():
        pk = []
        for c in sp.iterdir():
            if c.is_dir() and not c.name.endswith((".dist-info", ".egg-info")):
                pk.append((c.name, size(c)))
        pk.sort(key=lambda x: -x[1])
        res["site_packages_gb"] = gb(sum(s for _, s in pk))
        res["top_packages_mb"] = [(n, round(s / 1024**2, 1)) for n, s in pk[:40]]
        dists = sorted(c.name for c in sp.iterdir() if c.name.endswith(".dist-info"))
        res["dist_count"] = len(dists)
        (out / f"dists-{label}.txt").write_text("\n".join(dists), encoding="utf-8")
        heavy = ["torch", "torchvision", "torchaudio", "triton", "bitsandbytes", "xformers", "rocm", "_rocm_sdk",
                 "nvidia", "transformers", "diffusers", "accelerate", "peft", "trl", "datasets", "vllm", "flash_attn",
                 "scipy", "pandas", "pyarrow", "sklearn", "cv2", "sentencepiece", "tokenizers", "mlx", "unsloth_zoo",
                 "llama_cpp", "onnxruntime", "numpy", "safetensors", "huggingface_hub", "PIL", "av", "soundfile", "librosa", "timm"]
        res["heavy_present"] = {h: any(n.lower() == h.lower() or n.lower().startswith(h.lower() + "_") or n.lower().startswith(h.lower()) for n, _ in pk) for h in heavy}
        res["heavy_dists"] = [d for d in dists if any(d.lower().startswith(x) for x in ["torch", "triton", "rocm", "nvidia", "bitsandbytes", "xformers", "_rocm", "transformers", "diffusers", "accelerate", "peft", "trl-", "datasets", "vllm"])]
    man = venv / "unsloth_install_manifest.json"
    if man.exists():
        try:
            m = json.loads(man.read_text(encoding="utf-8"))
            res["manifest_no_torch"] = m.get("no_torch")
            res["manifest_keys"] = sorted(m)[:60]
            (out / f"manifest-{label}.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
        except Exception as e:
            res["manifest_err"] = str(e)
    res["marker_no_torch"] = (venv / ".unsloth-no-torch").exists()
    py = venv / "Scripts" / "python.exe"
    if py.exists():
        r = subprocess.run([str(py), "-c", "import importlib.util as u; print({m: bool(u.find_spec(m)) for m in ['torch','transformers','unsloth','unsloth_zoo','llama_cpp','diffusers','numpy']})"], capture_output=True, text=True)
        res["importable"] = (r.stdout or r.stderr).strip()[-600:]
        r = subprocess.run([str(py), "-c", "import importlib.metadata as m; print(m.version('unsloth'))"], capture_output=True, text=True)
        res["unsloth_version"] = (r.stdout or r.stderr).strip()[-200:]
        r = subprocess.run([str(py), "-c", "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())"], capture_output=True, text=True)
        res["torch_probe"] = (r.stdout or r.stderr).strip()[-300:]
# Other places an install writes.
home = Path(os.environ["USERPROFILE"])
res["home_unsloth_listing"] = sorted(p.name for p in (home / ".unsloth").iterdir()) if (home / ".unsloth").exists() else None
res["home_unsloth_gb"] = gb(size(home / ".unsloth")) if (home / ".unsloth").exists() else 0
res["hf_cache_gb"] = gb(size(home / ".cache" / "huggingface")) if (home / ".cache" / "huggingface").exists() else 0
lad = Path(os.environ.get("LOCALAPPDATA", ""))
for name in ["ai.unsloth.studio", "uv", "unsloth"]:
    p = lad / name
    if p.exists():
        res[f"localappdata_{name}_gb"] = gb(size(p))
(out / f"measure-{label}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
print(json.dumps(res, indent=2))
