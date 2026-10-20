#!/usr/bin/env python3
"""Diagnostic (#14313): can oneDNN fp32 replace rocm-openblas for bge-small's Linear layers on Windows ROCm torch?

Per fresh process (ROCm multi-arch torch 2.11.0+rocm7.14.0): baseline fp32 (Studio's 16/16 pools), and the same
model with every nn.Linear routed through oneDNN (two call styles). Parity vs the baseline vectors, chunks/s, busy
cores, op profile. Also each variant at one torch thread.
"""
import json, os, subprocess, sys, time
from pathlib import Path

MULTIARCH = "https://repo.amd.com/rocm/whl-multi-arch/"
CELL = r'''
import ctypes, json, os, sys, time
variant, threads = sys.argv[1], int(sys.argv[2])
ref_path = sys.argv[3]
import psutil, numpy as np
me = psutil.Process()
import torch, torch.nn as nn
k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]; k32.GetModuleHandleW.restype = ctypes.c_void_p
blas = ctypes.CDLL("rocm-openblas.dll", handle=k32.GetModuleHandleW("rocm-openblas.dll"))
blas.openblas_set_num_threads.argtypes = [ctypes.c_int]; blas.openblas_get_num_threads.restype = ctypes.c_int
torch.set_num_threads(threads); blas.openblas_set_num_threads(threads)
out = {"variant": variant, "torch_threads": torch.get_num_threads(), "openblas": blas.openblas_get_num_threads(),
       "mkldnn_available": torch.backends.mkldnn.is_available()}

class DenseIO(nn.Module):  # mkldnn_linear on mkldnn-layout tensors
    def __init__(self, lin):
        super().__init__()
        self.in_features, self.out_features = lin.in_features, lin.out_features
        self.weight = lin.weight
        self.bias = lin.bias
        self._w = lin.weight.detach().to_mkldnn()
        self._b = None if lin.bias is None else lin.bias.detach().to_mkldnn()
    def forward(self, x):
        shape = x.shape
        y = torch.ops.aten.mkldnn_linear(x.reshape(-1, shape[-1]).contiguous().to_mkldnn(), self._w, self._b).to_dense()
        return y.reshape(*shape[:-1], self.out_features)

class Pointwise(nn.Module):  # inductor's dense-in dense-out oneDNN linear
    def __init__(self, lin):
        super().__init__()
        self.in_features, self.out_features = lin.in_features, lin.out_features
        self.weight = lin.weight
        self.bias = lin.bias
    def forward(self, x):
        return torch.ops.mkldnn._linear_pointwise(x, self.weight, self.bias, "none", [], "")

def swap(module, cls):
    n = 0
    for name, child in list(module.named_children()):
        if type(child) is nn.Linear:
            setattr(module, name, cls(child)); n += 1
        else:
            n += swap(child, cls)
    return n

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("unsloth/bge-small-en-v1.5", device="cpu")
try:
    with torch.inference_mode():
        if variant == "dense_io":
            out["swapped"] = swap(model, DenseIO)
        elif variant == "pointwise":
            out["swapped"] = swap(model, Pointwise)
    words = ("retrieval augmented generation splits documents into chunks and embeds each chunk with a small "
             "sentence transformer so later questions find the closest passages by cosine similarity ").split()
    texts = [" ".join(words[(i * 7 + j) % len(words)] for j in range(20 + (i * 37) % 300)) + f" #{i}" for i in range(256)]
    t = time.perf_counter()
    vec = model.encode(texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    out["first_pass_s"] = round(time.perf_counter() - t, 2)
    if variant == "baseline" and threads > 1:
        np.save(ref_path, vec)
    if os.path.exists(ref_path):
        ref = np.load(ref_path)
        out["max_abs_diff"] = float(np.abs(vec - ref).max())
        out["min_cosine"] = float((vec * ref).sum(1).min())
    c0, w0, done = sum(me.cpu_times()[:2]), time.perf_counter(), 0
    while time.perf_counter() - w0 < 20:
        model.encode(texts[done % 224: done % 224 + 32], batch_size=32); done += 32
    wall = time.perf_counter() - w0
    out["chunks_per_s"] = round(done / wall, 2)
    out["busy_cores"] = round((sum(me.cpu_times()[:2]) - c0) / wall, 2)
    out["openblas_after"] = blas.openblas_get_num_threads()
    if threads > 1:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU]) as p:
            model.encode(texts[:32], batch_size=32)
        out["profile"] = p.key_averages().table(sort_by="self_cpu_time_total", row_limit=12)
except Exception as e:
    import traceback
    out["error"] = traceback.format_exc()[-2500:]
print("CELL " + json.dumps(out), flush=True)
'''

def sh(cmd, label, timeout=3600, **kw):
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, **kw)
    print(f"[{label}] rc={r.returncode} {round(time.time() - t)}s", (r.stdout + r.stderr)[-1500:] if r.returncode else "", flush=True)
    return r

def main():
    out_dir = Path(sys.argv[1]); out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(os.environ.get("AMD_CI_WORK", out_dir)) / "diag2"; work.mkdir(parents=True, exist_ok=True)
    py = work / "v" / "Scripts" / "python.exe"
    base = getattr(sys, "_base_executable", None) or sys.executable
    sh([base, "-m", "venv", str(work / "v")], "venv")
    if sh([str(py), "-m", "pip", "install", "-q", "--index-url", MULTIARCH, "--extra-index-url", "https://pypi.org/simple",
           "torch[device-gfx1151]==2.11.0+rocm7.14.0", "psutil", "numpy", "sentence-transformers"], "install").returncode:
        sys.exit(1)
    (work / "cell.py").write_text(CELL, encoding="utf-8")
    ref = str(work / "ref.npy")
    env = {k: v for k, v in os.environ.items() if k not in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")}
    env.update(HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="", PYTHONUTF8="1")
    rows = []
    for rep in range(2):
        for variant, threads in (("baseline", 16), ("dense_io", 16), ("pointwise", 16), ("baseline", 1), ("dense_io", 1), ("pointwise", 1)):
            r = subprocess.run([str(py), str(work / "cell.py"), variant, str(threads), ref], capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=1500, env=env, cwd=str(work))
            row = next((json.loads(l[5:]) for l in r.stdout.splitlines() if l.startswith("CELL ")),
                       {"variant": variant, "error": (r.stdout + r.stderr)[-2500:]})
            row.update(rep=rep, threads_req=threads)
            rows.append(row)
            print({k: row.get(k) for k in ("variant", "threads_req", "rep", "swapped", "chunks_per_s", "busy_cores", "max_abs_diff", "min_cosine", "first_pass_s", "openblas_after")}, flush=True)
            if row.get("error"):
                print(row["error"], flush=True)
    (out_dir / "diag2.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    for row in rows[:3]:
        print(row.get("variant"), "\n", row.get("profile"), flush=True)

if __name__ == "__main__":
    main()
