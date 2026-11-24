#!/usr/bin/env python3
"""Probe (PRs 19209 / 19211 / 19213, pick with --pr): observes only, on the real card, this checkout's Studio modules.

19209: ROCm flash probe byte estimate over a grid of geometries / shapes / dtypes (refactored at the head).
19211: Studio's whole-step graph gate for this host's target; a tiny random MOSS-Image transformer captured in a
       HIP graph at 12 distinct text lengths, then churned (eager at 12 more lengths + filler allocations) and
       replayed: replay vs the eager output of the same inputs.
19213: torch.cuda.is_current_stream_capturing() outside / inside / after a real graph capture, plus the head's
       H3 `_capturing()` and diffusion_cuda_graph.recording_here() inside the capture.
One child process, killed on timeout (Windows arms that never exit). JSON via --out only.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

COMMON = r'''
import json, os, sys, traceback, types
sys.path.insert(0, sys.argv[1])
import torch
out = {"torch": torch.__version__, "hip": getattr(torch.version, "hip", None), "platform": sys.platform,
       "cuda_available": torch.cuda.is_available(), "errors": {}}
dev = torch.device("cuda:0")
try:
    out["arch"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None)
except Exception:
    out["arch"] = None
'''

P19209 = r'''
from core.inference import diffusion_attention as A
out["module_file"] = A.__file__
out["has_block_activation_bytes"] = hasattr(A, "block_activation_bytes")
geoms = [(3072, 24, 128), (5120, 40, 128), (1536, 12, 128), (3072, 24, 128, 4096), (2048, 16, 128, 1024),
         (1152, 16, 72), (4096, 32, 128, 2560)]
shapes = [(4096, 4096, 1), (4096, 512, 2), (16384, 16384, 1), (65536, 65536, 1), (1024, 77, 2), (333, 1000, 1)]
dts = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
res, fused = {}, {}
for g in geoms:
    for (t, kv, b) in shapes:
        for dn, dt in dts.items():
            key = f"{g}|{t}|{kv}|{b}|{dn}"
            try:
                res[key] = int(A._rocm_flash_probe_bytes(g, t, kv, b, dt, "cuda"))
                fused[key] = bool(A._native_sdpa_fused_serves("cuda", dt, int(g[1]), int(g[2]), t, kv, b))
            except Exception:
                out["errors"][key] = traceback.format_exc()[-800:]
out["bytes"] = res
out["fused_true"] = sum(fused.values())
out["fused_false"] = len(fused) - sum(fused.values())
try:
    from core.inference import diffusion_multigpu as M
    out["has_replica_activation_mib"] = hasattr(M, "replica_activation_mib")
    if hasattr(M, "replica_activation_mib"):
        mod = torch.nn.Linear(8, 8).to(torch.bfloat16)
        mod.config = {"num_attention_heads": 24, "attention_head_dim": 128, "hidden_size": 3072}
        out["replica_activation_mib_flux_like_4096"] = int(M.replica_activation_mib(mod, tokens = 4096))
except Exception:
    out["errors"]["multigpu"] = traceback.format_exc()[-800:]
'''

P19211 = r'''
from core.inference import diffusion_moss_image as moss
out["module_file"] = moss.__file__
out["has_capturing"] = hasattr(moss, "_capturing")
try:
    from core.inference import diffusion_cuda_graph as cg
    from core.inference import diffusion_device as dd
    target = dd.resolve_diffusion_device_target()
    out["target"] = {"device": getattr(target, "device", None), "backend": getattr(target, "backend", None),
                     "vendor": getattr(target, "vendor", None)}
    ok, why = cg.graph_eligible(target, family = None, pipe = None, offload_active = False, cache_active = False,
                                speed_mode = "default")
    out["studio_graph_eligible"] = [bool(ok), str(why)]
except Exception:
    out["errors"]["gate"] = traceback.format_exc()[-1500:]

cap_true = [0]
if hasattr(moss, "_capturing"):
    real_cap = moss._capturing
    def spy(device):
        r = real_cap(device)
        cap_true[0] += int(bool(r))
        return r
    moss._capturing = spy

graph = {}
try:
    torch.manual_seed(0)
    m = moss.MOSSImageTransformer(dim = 256, ffn_dim = 512, num_heads = 2, num_layers = 24, text_dim = 64)
    with torch.no_grad():
        for p in m.parameters():
            torch.nn.init.normal_(p, std = 0.02)
    m = m.to(dev, torch.bfloat16).eval()
    h, w = 4, 6
    g0 = torch.Generator().manual_seed(1)
    lat = torch.randn(1, 128, h, w, generator = g0)
    lengths = [5, 7, 9, 11, 13, 17, 19, 23, 29, 31, 37, 41]
    churn = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
    def emb(n):
        g = torch.Generator().manual_seed(100 + n)
        return torch.randn(1, n, 64, generator = g).to(torch.bfloat16)
    rec = []
    for L in lengths:
        static = (lat.clone().to(dev), torch.tensor(731.0, device = dev), emb(L).to(dev))
        # warm-up on the current stream, as the churn below: the caching allocator reuses a freed block only for
        # allocations on the stream it was made on, so side-stream warm-ups would hide the reuse being tested
        with torch.no_grad():
            for _ in range(2):
                eager = m(*static, return_dict = False)[0]
        torch.cuda.synchronize()
        ref = eager.float().clone()
        g = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(g):
            og = m(*static, return_dict = False)[0]
        g.replay()
        torch.cuda.synchronize()
        first = float((og.float() - ref).abs().max())
        rec.append((L, g, static, og, ref, first))
    graph["cache_entries_after_capture"] = len(m.rope._tables)
    shapes = set()
    with torch.no_grad():
        for L in churn:
            m(lat.to(dev), torch.tensor(500.0, device = dev), emb(L).to(dev), return_dict = False)
        for L in lengths:
            for pair in m.rope._build(h, w, L, dev, torch.bfloat16) if hasattr(m.rope, "_build") else (
                    (lambda r: (moss._rope_tables(r), moss._rope_tables(r[:, h * w:])))(
                        m.rope.forward(h, w, L, dev).to(torch.bfloat16))):
                for t in pair:
                    shapes.add(tuple(t.shape))
    fill = [torch.full(sh, 7.0, device = dev, dtype = torch.bfloat16) for sh in sorted(shapes) for _ in range(6)]
    torch.cuda.synchronize()
    per = {}
    for (L, g, static, og, ref, first) in rec:
        g.replay()
        torch.cuda.synchronize()
        o = og.float()
        amax = float(ref.abs().max())
        per[str(L)] = {"first_replay_max_abs": first, "late_replay_max_abs": float((o - ref).abs().max()),
                       "ref_absmax": amax, "ref_absmean": float(ref.abs().mean()),
                       "late_finite": bool(torch.isfinite(o).all())}
    graph["per_length"] = per
    graph["fillers"] = len(fill)
    graph["capturing_true_calls"] = cap_true[0]
except Exception:
    graph["error"] = traceback.format_exc()[-3000:]
out["graph"] = graph
'''

P19213 = r'''
from core.inference import video_minimax_h3_forward as H3
from core.inference import diffusion_adaln_precompute as AD
from core.inference import diffusion_cuda_graph as cg
out["module_file"] = H3.__file__
out["h3_has_capturing"] = hasattr(H3, "_capturing")
out["adaln_has_drop_graphs"] = hasattr(AD, "_drop_graphs_then_free")
cap = {}
try:
    x = torch.zeros(1024, device = dev)
    cap["outside"] = bool(torch.cuda.is_current_stream_capturing())
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        y = x * 2 + 1
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    seen = {}
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        seen["inside"] = bool(torch.cuda.is_current_stream_capturing())
        with torch.cuda.device(dev):
            seen["inside_device_ctx"] = bool(torch.cuda.is_current_stream_capturing())
        if hasattr(H3, "_capturing"):
            seen["h3_capturing"] = bool(H3._capturing())
        try:
            seen["recording_here"] = bool(cg.recording_here())
        except Exception as e:
            seen["recording_here"] = f"error: {e!r}"
        y = x * 2 + 1
    cap.update(seen)
    x.fill_(3.0)
    g.replay()
    torch.cuda.synchronize()
    cap["replay_value_ok"] = bool(torch.all(y == 7.0).item())
    cap["after"] = bool(torch.cuda.is_current_stream_capturing())
    if hasattr(H3, "_capturing"):
        cap["h3_capturing_outside"] = bool(H3._capturing())
except Exception:
    cap["error"] = traceback.format_exc()[-2000:]
out["capture"] = cap
'''

TAIL = r'''
print("PROBE_JSON " + json.dumps(out, default = str))
'''

BODIES = {"19209": P19209, "19211": P19211, "19213": P19213}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--pr", required = True, choices = sorted(BODIES))
    ap.add_argument("--timeout", type = int, default = 900)
    a = ap.parse_args()
    backend = str(a.checkout / "studio" / "backend")
    obs: dict = {"state": a.state, "pr": a.pr}
    code = COMMON + BODIES[a.pr] + TAIL
    try:
        p = subprocess.run([sys.executable, "-c", code, backend], cwd = backend, capture_output = True,
                           text = True, encoding = "utf-8", errors = "replace", timeout = a.timeout)
        line = next((ln for ln in p.stdout.splitlines() if ln.startswith("PROBE_JSON ")), None)
        if line:
            obs.update(json.loads(line[len("PROBE_JSON "):]))
        else:
            obs["error"] = (p.stderr or p.stdout)[-4000:]
        obs["rc"] = p.returncode
        obs["stderr_tail"] = (p.stderr or "")[-1500:]
    except subprocess.TimeoutExpired as e:
        obs["error"] = f"timeout after {a.timeout}s (killed)"
        obs["stderr_tail"] = str(e.stderr or "")[-1500:]
    a.out.write_text(json.dumps(obs, indent = 1, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
