"""Native upper bound: the same GEMM shapes with ROCm torch on the iGPU (fp16 / bf16), compute + host copies."""
import json, statistics, sys, time
import torch
SHAPES = {"smollm2_qkv_o": (576, 576), "smollm2_kv": (576, 192), "smollm2_gate_up": (576, 1536), "smollm2_down": (1536, 576),
          "qwen3_q": (1024, 2048), "qwen3_o": (2048, 1024), "qwen3_kv": (1024, 1024), "qwen3_gate_up": (1024, 3072), "qwen3_down": (3072, 1024)}
dev = torch.device("cuda")
rows = []
for M in (512, 1024, 2048):
    for name, (K, N) in SHAPES.items():
        X = torch.randn(M, K); G = torch.randn(M, N) * 0.01
        row = {"tokens": M, "shape": name}
        for dt in (torch.float16, torch.bfloat16):
            W = (torch.randn(N, K, device = dev) * 0.02).to(dt)
            for op in ("fwd", "dX"):
                def f():
                    a = (X if op == "fwd" else G).to(dev, dt, non_blocking = False)
                    y = a @ (W.t() if op == "fwd" else W)
                    return y.float().cpu()
                f(); f(); torch.cuda.synchronize()
                ts = []
                for _ in range(7):
                    t = time.perf_counter(); f(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
                row[f"rocm_{str(dt).split('.')[-1]}_{op}_ms"] = round(statistics.median(ts) * 1e3, 3)
        rows.append(row); print("ROW", json.dumps(row), flush = True)
json.dump({"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "rows": rows}, open(sys.argv[1], "w"), indent = 1)
