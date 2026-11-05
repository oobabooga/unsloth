"""Stdlib-only: read the per-section JSON files of one or two torch arms and print Markdown tables to --out."""
import argparse
import glob
import json
import os


def load(path):
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def g(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def f(v, nd=1):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, (int, float)) else str(v))


def arm_tables(d, arm):
    J = {s: load(os.path.join(d, f"{arm}_{s}.json")) for s in
         ("env", "dense", "intmm", "triton", "compile", "hip", "tunable", "accuracy", "block", "inline")}
    parts = sorted(glob.glob(os.path.join(d, f"{arm}_hip_*.json")))
    if parts:
        hip = {"peak": {}, "tile": {}, "gemm": {}}
        for pth in parts:
            h = load(pth).get("hip", {})
            hip["peak"].update(h.get("peak", {}))
            hip["tile"].update(h.get("tile", {}))
            for k, v in (h.get("gemm") or {}).items():
                hip["gemm"].setdefault(k, {}).update(v)
            if "iu4" in (h.get("tile") or {}):
                hip["iu4_packing_used"] = h.get("iu4_packing_used")
        J["hip"] = {"hip": hip, "status": "merged", "seconds": None}
    lines = [f"## arm {arm}: torch {g(J['env'], 'env', 'torch')} hip {g(J['env'], 'env', 'hip')} "
             f"triton {g(J['env'], 'env', 'triton')} on {g(J['env'], 'env', 'device')} {g(J['env'], 'env', 'arch')}", ""]
    for s, j in J.items():
        if j:
            lines.append(f"- section {s}: status {j.get('status')} in {j.get('seconds')} s"
                         + (f" CRASH {g(j, 'crash', 'error')}" if j.get("status") == "crashed" else ""))
    lines += ["", "TFLOP/s or TOP/s, median, cold weights (MALL flushed between reps). e2e = quant + GEMM + epilogue.", "",
              "| shape | bf16 | fp16 | bf16+bias | _int_mm | _int_mm rowB | eager W8A8 e2e | eager W8A8+rot256 e2e | "
              "Triton bf16 | Triton W8A8 GEMM | Triton W8A8 e2e | Triton W4A16 | HIP bf16 | HIP f16 | HIP iu8 | HIP iu4 | HIP W4A16 | Tunable off | Tunable on |",
              "|" + "---|" * 20]
    dense, intmm, tri, hip, tun = J["dense"].get("dense", {}), J["intmm"].get("intmm", {}), \
        g(J["triton"], "triton", "shapes") or {}, g(J["hip"], "hip", "gemm") or {}, J["tunable"].get("tunable", {})
    keys = list(dense) or list(intmm) or list(hip)
    for k in keys:
        def e2e(ms_key, sec=intmm):
            return g(sec, k, ms_key, "tops")
        lines.append("| " + " | ".join([
            k, f(g(dense, k, "bf16", "tflops")), f(g(dense, k, "fp16", "tflops")),
            f(g(intmm, k, "bf16_linear_bias", "tflops")), f(g(intmm, k, "int_mm_colmajorB", "tops")),
            f(g(intmm, k, "int_mm_rowmajorB", "tops")), f(e2e("eager_w8a8_e2e")), f(e2e("eager_w8a8_convrot256_e2e")),
            f(g(tri, k, "bf16", "tops")), f(g(tri, k, "w8a8", "tops")), f(g(tri, k, "w8a8_e2e_quant_plus_gemm", "tops")),
            f(g(tri, k, "w4a16", "tops")), f(g(hip, k, "bf16", "tops")), f(g(hip, k, "f16", "tops")),
            f(g(hip, k, "iu8", "tops")), f(g(hip, k, "iu4", "tops")), f(g(hip, k, "w4a16", "tops")),
            f(g(tun, k, "tflops_off")), f(g(tun, k, "tflops_on"))]) + " |")
    lines += ["", "Eager W8A8 breakdown (ms): quant / _int_mm / epilogue / rot256, vs bf16+bias", "",
              "| shape | bf16+bias ms | quant | int_mm | epilogue | rot256 | e2e | e2e+rot | int_mm exact |", "|" + "---|" * 9]
    for k in keys:
        r = intmm.get(k, {})
        lines.append("| " + " | ".join([k, f(g(r, "bf16_linear_bias", "ms"), 3), f(r.get("part_quant_ms"), 3),
                                         f(r.get("part_int_mm_ms"), 3), f(r.get("part_epilogue_ms"), 3),
                                         f(r.get("part_rotate256_ms"), 3), f(g(r, "eager_w8a8_e2e", "ms"), 3),
                                         f(g(r, "eager_w8a8_convrot256_e2e", "ms"), 3), f(r.get("int_mm_exact"))]) + " |")
    pk = g(J["hip"], "hip", "peak") or {}
    lines += ["", "HIP WMMA peak issue rate (register-resident fragments): " +
              ", ".join(f"{n} {f(g(pk, n, 'tops'))}" for n in ("f16", "bf16", "iu8", "iu4")) + " T(FL)OP/s", ""]
    tile = g(J["hip"], "hip", "tile") or {}
    lines.append("Tile layout check: " + json.dumps({n: {vn: vv.get("match_assumed_layout") for vn, vv in
                                                         (t.get("variants") or {}).items()} for n, t in tile.items()}))
    lines.append(f"iu4 packing used: {g(J['hip'], 'hip', 'iu4_packing_used')}")
    errs = {}
    for k in keys:
        for n in ("f16", "bf16", "iu8", "iu4", "w4a16"):
            v = g(hip, k, n, "err", "rel_fro")
            if v is not None:
                errs.setdefault(n, []).append(v)
        for n in ("w8a8", "bf16", "w4a16"):
            v = g(tri, k, n, "err", "rel_fro")
            if v is not None:
                errs.setdefault("triton_" + n, []).append(v)
    lines.append("Correctness, max rel_fro over shapes: " + json.dumps({n: max(v) for n, v in errs.items()}))
    lines += ["", "Tunable: tuning seconds / speedup per shape: " + ", ".join(
        f"{k}: {f(g(tun, k, 'tuning_s'))}s x{f(g(tun, k, 'speedup'), 3)}" for k in keys), ""]
    comp = J["compile"].get("compile", {})
    lines.append("torch.compile:")
    for k, v in comp.items():
        if "error" in v and "traceback" in v:
            lines.append(f"- {k}: FAILED {v['error'][:300]}")
        else:
            lines.append(f"- {k}: eager {f(g(v, 'eager', 'ms'), 3)} ms compiled {f(g(v, 'compiled', 'ms') or v.get('ms'), 3)} ms "
                         f"first {f(v.get('first_call_s'))} s bf16 {f(v.get('bf16_ms'), 3)} ms err {f(g(v, 'err_vs_eager', 'rel_fro') or g(v, 'err', 'rel_fro'), 5)}")
    blk = J["block"].get("block", {})
    lines += ["", "Block GEMM share:"]
    for k, v in blk.items():
        if isinstance(v, dict):
            lines.append(f"- {k}: block {f(v.get('block_ms'), 2)} ms gemm {f(v.get('gemm_ms_sum'), 2)} ms "
                         f"attn {f(v.get('attn_ms'), 2)} ms gemm_share {f(v.get('gemm_share'), 3)} attn_share {f(v.get('attn_share'), 3)} "
                         f"{('profiler ' + json.dumps(v.get('share'))) if 'share' in v else ''} {v.get('error', '')}")
    acc = g(J["accuracy"], "accuracy", "mean_rel_fro")
    if acc:
        lines += ["", "Accuracy (mean relative Frobenius output error over FLUX.2-klein-4B weight slices)", "",
                  "| method | gaussian acts | outlier acts |", "|---|---|---|"]
        for m, v in acc.items():
            lines.append(f"| {m} | {f(v.get('gaussian'), 4)} | {f(v.get('outliers'), 4)} |")
    inl = J["inline"].get("inline")
    if inl:
        lines += ["", f"load_inline: ok={inl.get('ok')} build_s={inl.get('build_s')} ROCM_HOME={inl.get('ROCM_HOME')} {inl.get('error', '')}"]
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    arms = sorted({os.path.basename(p).split("_")[0] for p in glob.glob(os.path.join(a.dir, "*_env.json"))})
    lines = ["# gfx1151 low-precision GEMM probe", ""]
    b = load(os.path.join(a.dir, "build_wmma.json"))
    if b:
        lines += [f"HIP build: chosen {json.dumps(b.get('chosen'))}",
                  f"host hipcc on PATH: {b.get('host_hipcc_on_path')}",
                  "candidates: " + json.dumps(b.get("candidates")),
                  "builds rc: " + json.dumps({k: v.get("rc") for k, v in (b.get("builds") or {}).items()}),
                  "asm: " + json.dumps({k: v.get("wmma_ops") for k, v in (b.get("asm") or {}).items()}), ""]
    for arm in arms:
        lines += arm_tables(a.dir, arm) + [""]
    open(a.out, "w", encoding="utf-8").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
