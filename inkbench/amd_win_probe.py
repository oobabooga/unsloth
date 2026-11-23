#!/usr/bin/env python3
"""Why HEAD's llama-bench dies on Windows HIP (access violation in rocblas_sgemm from ggml-hip.dll) while the CUR
prebuilt runs: build CUR's own source (cur_b12631.diff, from the release source tarball) with the same recipe as HEAD,
then run llama-bench -p 0 -n 16 on root UD-Q2_K_XL for CUR (prebuilt), CURsrc, HEAD, HEAD with every decode-pin kill
switch, and HEAD with each switch alone. Full stderr kept. Correctness only, no timing.

  python amd_win_probe.py --kit DIR --work DIR --out DIR
"""
import argparse, json, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import amd_job as A  # noqa: E402
import tg_bench as T  # noqa: E402

SW = ["SHORT_CONV_FUSION", "TANH_MUL_ADDSUB_FUSION", "ROW_NORM_DIV_FUSION", "ADD_RMS_NORM_FUSION", "DEPTH_MIX_FUSION",
      "REL_BIAS_FA_FUSION", "FA_VEC_MASK_SKIP", "Q8_1_REUSE", "MOE_VEC_ROWS4", "CONCAT_FLAT", "BLACKWELL_DC_DECODE_TUNING"]
CUR_TREE = "6e030e1c96c8fb675af1a478de2b1bcbfd49f81e"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    kit, work, out = Path(a.kit).resolve(), Path(a.work).resolve(), Path(a.out).resolve()
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    res = {"errors": []}
    A.FILES = {"q2": A.FILES["q2"]}
    A.get_models(work / "models", res)
    rroot, rver = A.therock(work)
    R = {"therock": rver, "builds": {}, "runs": {}}
    bins = {}
    for name, diff, tree in (("HEAD", "head_b12631.diff", A.HEAD_TREE), ("CURsrc", "cur_b12631.diff", CUR_TREE)):
        try:
            b, note = A.build_head(kit, work, rroot, out, diff=diff, want_tree=tree, name=name)
            bins[name] = b
            R["builds"][name] = note
        except SystemExit as e:
            R["builds"][name] = f"FAILED: {e}"
        (out / "probe.json").write_text(json.dumps(R, indent=1), encoding="utf-8")
    plat = "windows-x64" if A.WIN else "linux-x64"
    ext = "zip" if A.WIN else "tar.gz"
    specs = [f"CUR=gh:unslothai/llama.cpp@{A.CUR_TAG}/app-{A.CUR_TAG}-{plat}-rocm-{A.GFX}.{ext}"]
    specs += [f"{n}={b}" for n, b in bins.items()]
    arms = T.resolve_arms(specs, work / "tgcache")
    cur_dir = str(Path(arms["CUR"]["bench"]).parent)
    rlib = str(rroot / ("bin" if A.WIN else "lib"))
    model = res["models"]["q2"]
    runs = [("CUR", "CUR", {}, [])]
    for n in bins:
        runs += [(f"{n}-therock", n, {}, [rlib]), (f"{n}-curlibs", n, {}, [cur_dir])]
    if "HEAD" in bins:
        runs += [("HEAD-alloff", "HEAD", {f"GGML_CUDA_DISABLE_{s}": "1" for s in SW}, [cur_dir]),
                 ("HEAD-nofusion", "HEAD", {"GGML_CUDA_DISABLE_FUSION": "1"}, [cur_dir])]
        runs += [(f"HEAD-{s}", "HEAD", {f"GGML_CUDA_DISABLE_{s}": "1"}, [cur_dir]) for s in SW]
    for label, arm, extra, libs in runs:
        env = T.arm_env(arms, arm, [], {}, {arm: libs})
        env.update(extra)
        cmd = T.bench_cmd(arms[arm]["bench"], model, 16, "f16", 99, 0, [])
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900, errors="replace")
            rc, txt = r.returncode, (r.stdout or "") + (r.stderr or "")
        except subprocess.TimeoutExpired:
            rc, txt = "timeout", ""
        (out / f"bench_{label}.log").write_text(txt, encoding="utf-8")
        ts = None
        if rc == 0 and "[" in (txt or ""):
            try:
                ts = json.loads(r.stdout[r.stdout.index("["):])[-1].get("avg_ts")
            except (ValueError, json.JSONDecodeError):
                pass
        R["runs"][label] = {"rc": rc, "ts": ts}
        A.log(f"{label}: rc {rc} ts {ts}")
        (out / "probe.json").write_text(json.dumps(R, indent=1), encoding="utf-8")
    print("WIN_PROBE", json.dumps(R["runs"]), flush=True)


if __name__ == "__main__":
    main()
