"""RDNA W8A8 kernel bench on the AMD runner: per-linear and Q/K/V-block timings (stock compiled vs RDNA kernels vs
bf16), kernel launch census (shared quant), graph breaks and recompiles. Writes one JSON.

    python w8a8_rdna_bench.py --backend <tree>/studio/backend --out bench.json
"""

import argparse
import json
import os
import statistics
import sys
import time


def _time(torch, fn, reps = 20, warm = 3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing = True)
        b = torch.cuda.Event(enable_timing = True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        out.append(a.elapsed_time(b))
    return statistics.median(out)


def _aa(torch, fns, rounds = 3):
    """Interleaved rounds: per arm median of round medians + spread (max - min of the round medians)."""
    got = {k: [] for k in fns}
    for _ in range(rounds):
        for k, fn in fns.items():
            got[k].append(_time(torch, fn))
    return {k: {"ms": round(statistics.median(v), 4), "spread": round(max(v) - min(v), 4), "rounds": [round(t, 4) for t in v]}
            for k, v in got.items()}  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required = True)
    ap.add_argument("--out", required = True)
    args = ap.parse_args()
    import torch
    import torch._dynamo
    from torch._dynamo.utils import counters

    sys.path.insert(0, args.backend)
    from core.inference import diffusion_native_quant as nq
    from core.inference import diffusion_native_w8a8_rdna as rdna

    res = {"torch": torch.__version__, "hip": torch.version.hip, "device": torch.cuda.get_device_name(0)}
    try:
        import triton

        res["triton"] = triton.__version__
    except Exception as exc:  # noqa: BLE001
        res["triton"] = repr(exc)
    t0 = time.time()
    res["prepare"] = rdna.prepare()
    res["prepare_s"] = round(time.time() - t0, 2)
    res["state"] = rdna.state()
    if not res["prepare"]:
        json.dump(res, open(args.out, "w"), indent = 1)
        print("PROBE_FAILED", res["state"])
        return 3
    cls = nq.native_linear_class()
    gen = torch.Generator().manual_seed(0)

    # Tile sweep (each config checked against the eager reference; a config that fails to build is recorded).
    def sweep_rotq():
        rows = {}
        for k in (3072, 12288):
            for group in (256, 0):
                x = (torch.randn(4096, k, generator = gen) * 2).to(torch.bfloat16).cuda()
                rq, rs = rdna.reference_rotq(x, group)
                stock_c = torch.compile(lambda t, g=group: rdna.reference_rotq(t, g), fullgraph = True)
                cell = {"inductor_ms": round(_time(torch, lambda: stock_c(x)), 4)}
                for bm in (16, 32, 64, 128):
                    for bk in (32, 64):
                        for warps in (2, 4, 8):
                            if group == 0 and bk == 64:
                                continue
                            rdna._CFG["rotq"] = (bm, bk, warps, 1)
                            rdna._STATE["broken"] = None
                            if rdna._rotq_tile(k, group) is None:
                                continue
                            try:
                                q, sc = rdna._rotq_launch(x, group)
                                torch.cuda.synchronize()
                                broken = rdna._STATE["broken"]
                                d = (q.int() - rq.int()).abs()
                                ms = None if broken else round(_time(torch, lambda: rdna._rotq_launch(x, group)), 4)
                                cell[f"{bm}x{bk}_w{warps}"] = {"ms": ms, "broken": broken, "max_code_diff": int(d.max()),
                                                               "mismatch": round(float((d > 0).float().mean()), 6)}  # fmt: skip
                            except Exception as exc:  # noqa: BLE001
                                cell[f"{bm}x{bk}_w{warps}"] = {"error": repr(exc)[:300]}
                rows[f"K{k}_g{group}"] = cell
                ok = {c: v["ms"] for c, v in cell.items() if isinstance(v, dict) and v.get("ms") and v["max_code_diff"] <= 1}
                print("rotq", k, group, "inductor", cell["inductor_ms"], "best", min(ok.items(), key = lambda kv: kv[1]) if ok else None, flush = True)
        return rows

    def sweep_mm():
        rows = {}
        tiles = [(128, 128, 128, 8, 8, 2), (128, 128, 64, 8, 8, 2), (128, 256, 64, 8, 8, 2), (256, 128, 64, 8, 8, 2),
                 (128, 128, 128, 8, 4, 2), (64, 128, 128, 8, 4, 2), (128, 64, 128, 8, 4, 2), (256, 128, 128, 8, 8, 2)]  # fmt: skip
        for m, k, n in ((4096, 3072, 3072), (4096, 3072, 12288), (4096, 12288, 3072)):
            xq = torch.randint(-127, 128, (m, k), dtype = torch.int8).cuda()
            wq = torch.randint(-127, 128, (n, k), dtype = torch.int8).cuda()
            xs = torch.rand(m, 1).cuda() * 1e-2
            ws = torch.rand(n).cuda() * 1e-2
            b = torch.randn(n).to(torch.bfloat16).cuda()
            ref = rdna.reference_mm_stock(xq, xs, wq, ws, b, torch.bfloat16)
            stock_c = torch.compile(lambda: rdna.reference_mm_stock(xq, xs, wq, ws, b, torch.bfloat16))
            cell = {"stock_compiled_ms": round(_time(torch, stock_c), 4),
                    "int_mm_ms": round(_time(torch, lambda: torch._int_mm(xq, wq.t())), 4)}  # fmt: skip
            for t in tiles:
                rdna._CFG["mm_large"] = t
                rdna._STATE["broken"] = None
                try:
                    out = rdna._mm_launch(xq, xs, wq, ws, b, torch.bfloat16)
                    torch.cuda.synchronize()
                    broken = rdna._STATE["broken"]
                    err = float((out.float() - ref.float()).abs().max() / ref.float().abs().max())
                    ms = None if broken else round(_time(torch, lambda: rdna._mm_launch(xq, xs, wq, ws, b, torch.bfloat16)), 4)
                    cell["x".join(map(str, t[:3])) + f"_w{t[4]}"] = {"ms": ms, "broken": broken, "rel_err": err}
                except Exception as exc:  # noqa: BLE001
                    cell["x".join(map(str, t[:3])) + f"_w{t[4]}"] = {"error": repr(exc)[:300]}
            rows[f"M{m}_K{k}_N{n}"] = cell
            ok = {c: v["ms"] for c, v in cell.items() if isinstance(v, dict) and v.get("ms") and v["rel_err"] < 1e-2}
            print("mm", m, k, n, "stock", cell["stock_compiled_ms"], "best", min(ok.items(), key = lambda kv: kv[1]) if ok else None, flush = True)
        return rows

    default_cfg = dict(rdna._CFG)
    res["sweep_rotq"] = sweep_rotq()
    res["sweep_mm"] = sweep_mm()

    def best(rows, key, default, parse):
        votes = {}
        for cell in rows.values():
            ok = {c: v["ms"] for c, v in cell.items() if isinstance(v, dict) and v.get("ms")
                  and v.get("max_code_diff", 0) <= 1 and v.get("rel_err", 0) < 1e-2}  # fmt: skip
            base = min(ok.values()) if ok else None
            for c, ms in ok.items():
                votes[c] = votes.get(c, 0.0) + ms / base
        if not votes:
            return default
        return parse(min(votes, key = votes.get))

    rdna._CFG["rotq"] = best(res["sweep_rotq"], "rotq", default_cfg["rotq"],
                             lambda c: (int(c.split("x")[0]), int(c.split("x")[1].split("_")[0]), int(c.split("_w")[1]), 1))  # fmt: skip
    rdna._CFG["mm_large"] = best(res["sweep_mm"], "mm", default_cfg["mm_large"],
                                 lambda c: (*map(int, c.split("_w")[0].split("x")), 8, int(c.split("_w")[1]), 2))  # fmt: skip
    rdna._STATE["broken"] = None
    res["chosen_cfg"] = {k2: list(v) for k2, v in rdna._CFG.items()}
    print("chosen", res["chosen_cfg"], flush = True)
    json.dump(res, open(args.out, "w"), indent = 1)

    def mk_lin(k, n, bias):
        lin = torch.nn.Linear(k, n, bias = bias, dtype = torch.bfloat16)
        with torch.no_grad():
            lin.weight.copy_((torch.randn(n, k, generator = gen) * 0.02).to(torch.bfloat16))
        return lin.cuda()

    def wrap(lin, on):
        if not on:
            os.environ["UNSLOTH_NATIVE_INT8_RDNA_TRITON"] = "0"
        layer = cls(lin, "int8", act_int8 = True, rot_group = 256).cuda().eval()
        os.environ.pop("UNSLOTH_NATIVE_INT8_RDNA_TRITON", None)
        assert layer._rdna == on
        return layer


    # Per linear: stock eager (its compiled _w8a8_rows), stock compiled (model path), RDNA eager, RDNA compiled, bf16.
    lin_rows = {}
    for m, k, n in ((4096, 3072, 3072), (4096, 3072, 12288), (4096, 12288, 3072), (4608, 3072, 9216), (1024, 3072, 3072), (256, 3072, 3072)):
        lin = mk_lin(k, n, True)
        stock, ours = wrap(lin, False), wrap(lin, True)
        x = (torch.randn(m, k, generator = gen) * 2).to(torch.bfloat16).cuda()
        torch._dynamo.reset()
        stock_c = torch.compile(stock, fullgraph = True)
        ours_c = torch.compile(ours, fullgraph = True)
        with torch.no_grad():
            fns = {"bf16": lambda: lin(x), "stock_eager": lambda: stock(x), "stock_compiled": lambda: stock_c(x),
                   "rdna_eager": lambda: ours(x), "rdna_compiled": lambda: ours_c(x)}  # fmt: skip
            row = _aa(torch, fns)
            xq, xs = rdna.rotq(x.reshape(-1, k), 256)
            parts = {"rotq": lambda: rdna.rotq(x, 256), "rdna_mm": lambda: rdna.w8a8_mm(xq, xs, ours.weight_q, ours.weight_scale.view(torch.float32), ours.bias, torch.bfloat16),
                     "int_mm": lambda: torch._int_mm(xq, ours.weight_q.t())}  # fmt: skip
            row.update({f"part_{k2}": v for k2, v in _aa(torch, parts, rounds = 2).items()})
            ref = x.double() @ lin.weight.double().t() + lin.bias.double()
            for arm, fn in (("stock_compiled", stock_c), ("rdna_eager", ours), ("rdna_compiled", ours_c)):
                y = fn(x).double()
                row[f"err_{arm}"] = round(float((y - ref).norm() / ref.norm()), 6)
            row["max_abs_rdna_vs_stock_compiled"] = float((ours_c(x).float() - stock_c(x).float()).abs().max())
        row["broken"] = rdna._STATE["broken"]
        lin_rows[f"M{m}_K{k}_N{n}"] = row
        print(f"M{m}_K{k}_N{n}", {a: v["ms"] for a, v in row.items() if isinstance(v, dict) and "ms" in v}, flush = True)
    res["linear"] = lin_rows

    # Q/K/V block on one input + MLP up/down: shared quant census, graph breaks, recompiles over three token counts.
    class Block(torch.nn.Module):
        def __init__(self, on):
            super().__init__()
            self.q = wrap(mk_lin(3072, 3072, True), on)
            self.k = wrap(mk_lin(3072, 3072, True), on)
            self.v = wrap(mk_lin(3072, 3072, True), on)
            self.up = wrap(mk_lin(3072, 12288, True), on)
            self.down = wrap(mk_lin(12288, 3072, True), on)

        def forward(self, x):
            a = self.q(x) + self.k(x) + self.v(x)
            return self.down(torch.nn.functional.gelu(self.up(a), approximate = "tanh"))

    block_rows = {}
    for on in (False, True):
        blk = Block(on).cuda().eval()
        torch._dynamo.reset()
        counters.clear()
        compiles = []
        blk_c = torch.compile(blk)
        with torch.no_grad():
            for m in (4096, 4352, 4608):
                x = (torch.randn(1, m, 3072, generator = gen) * 2).to(torch.bfloat16).cuda()
                t = time.time()
                blk_c(x)
                torch.cuda.synchronize()
                compiles.append(round(time.time() - t, 2))
            x = (torch.randn(1, 4096, 3072, generator = gen) * 2).to(torch.bfloat16).cuda()
            timing = _aa(torch, {"compiled": lambda: blk_c(x), "eager": lambda: blk(x)})
            from torch.profiler import ProfilerActivity, profile

            with profile(activities = [ProfilerActivity.CUDA]) as prof:
                blk_c(x)
                torch.cuda.synchronize()
            names = {}
            for ev in prof.key_averages():
                if ev.device_type is not None and "cuda" in str(ev.device_type).lower():
                    names[ev.key[:90]] = names.get(ev.key[:90], 0) + ev.count
        block_rows["rdna" if on else "stock"] = {
            "timing": timing,
            "first_call_s": compiles,
            "graph_breaks": dict(counters["graph_break"]),
            "frames": dict(counters["stats"]),
            "recompiles": int(counters["stats"].get("unique_graphs", 0)),
            "rotq_launches": sum(c for k2, c in names.items() if "rotq_rows" in k2),
            "mm_launches": sum(c for k2, c in names.items() if "w8a8_mm" in k2),
            "kernels": dict(sorted(names.items(), key = lambda kv: -kv[1])[:25]),
        }
        print("block", "rdna" if on else "stock", timing, block_rows["rdna" if on else "stock"]["rotq_launches"], flush = True)
    res["block"] = block_rows
    res["broken_at_end"] = rdna._STATE["broken"]
    json.dump(res, open(args.out, "w"), indent = 1)
    print("BENCH_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
