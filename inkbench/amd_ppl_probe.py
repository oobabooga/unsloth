#!/usr/bin/env python3
"""Attribute the CUR vs HEAD PPL difference on gfx1151 (HIP): llama-perplexity -c 512 --chunks 4 on root
UD-Q4_K_XL for CUR (twice, A/A), HEAD, HEAD with each GGML_CUDA_DISABLE_* kill switch of the decode pins, and HEAD
with all of them. Correctness only (no timing). Reuses amd_job.py (TheRock + HEAD build).

  python amd_ppl_probe.py --kit DIR --work DIR --out DIR
"""
import argparse, json, os, re, subprocess, sys, zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import amd_job as A  # noqa: E402
import tg_bench as T  # noqa: E402

SW = ["SHORT_CONV_FUSION", "TANH_MUL_ADDSUB_FUSION", "ROW_NORM_DIV_FUSION", "ADD_RMS_NORM_FUSION", "DEPTH_MIX_FUSION",
      "REL_BIAS_FA_FUSION", "FA_VEC_MASK_SKIP", "Q8_1_REUSE", "MOE_VEC_ROWS4", "CONCAT_FLAT", "BLACKWELL_DC_DECODE_TUNING"]


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
    A.FILES = {"q4": A.FILES["q4"]}
    A.get_models(work / "models", res)
    z = work / "wiki.zip"
    A.fetch("https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip", z)
    zipfile.ZipFile(z).extractall(work / "wikitext")
    wiki = next((work / "wikitext").rglob("wiki.test.raw"))
    rroot, rver = A.therock(work)
    head_bin, note = A.build_head(kit, work, rroot, out)
    plat = "windows-x64" if A.WIN else "linux-x64"
    ext = "zip" if A.WIN else "tar.gz"
    arms = T.resolve_arms([f"CUR=gh:unslothai/llama.cpp@{A.CUR_TAG}/app-{A.CUR_TAG}-{plat}-rocm-{A.GFX}.{ext}",
                           f"HEAD={head_bin}"], work / "tgcache")
    rlib = rroot / ("bin" if A.WIN else "lib")
    libs = {"HEAD": [str(rlib)] + ([str(rroot / "lib" / "rocm_sysdeps" / "lib")] if not A.WIN else [])}
    model = res["models"]["q4"]
    runs = [("CUR", "CUR", {}), ("CUR_aa", "CUR", {}), ("HEAD", "HEAD", {})]
    runs += [(f"HEAD-{s}", "HEAD", {f"GGML_CUDA_DISABLE_{s}": "1"}) for s in SW]
    runs += [("HEAD-alloff", "HEAD", {f"GGML_CUDA_DISABLE_{s}": "1" for s in SW})]
    runs += [("HEAD-alloff-curlibs", "HEAD", {**{f"GGML_CUDA_DISABLE_{s}": "1" for s in SW}, "_CURLIBS": "1"})]
    R = {"therock": rver, "head_note": note, "runs": {}}
    for label, arm, extra in runs:
        env = T.arm_env(arms, arm, [], {}, libs)
        if extra.pop("_CURLIBS", None):  # HEAD on CUR's bundled ROCm runtime instead of TheRock's: rules the runtime in or out
            key = "PATH" if A.WIN else "LD_LIBRARY_PATH"
            env[key] = os.pathsep.join([str(Path(arms["HEAD"]["bench"]).parent), str(Path(arms["CUR"]["bench"]).parent), env.get(key, "")])
        env.update(extra)
        pbin = T.find_bin(Path(arms[arm]["bench"]).parent, "llama-perplexity")
        cmd = [str(pbin), "-m", model, "-f", str(wiki), "-c", "512", "--chunks", "4", "-ngl", "99", "-fa", "on"]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, errors="replace")
        txt = (r.stdout or "") + (r.stderr or "")
        (out / f"ppl_{label}.log").write_text(txt, encoding="utf-8")
        m = re.search(r"Final estimate: PPL = ([0-9.]+)", txt)
        ch = re.findall(r"\[\d+\]([0-9.]+),?", txt)
        R["runs"][label] = {"rc": r.returncode, "ppl": m.group(1) if m else None, "chunks": ch}
        A.log(f"{label}: {R['runs'][label]}")
        (out / "probe.json").write_text(json.dumps(R, indent=1), encoding="utf-8")
    cur = R["runs"]["CUR"]["chunks"]
    same = [k for k, v in R["runs"].items() if v["chunks"] == cur]
    R["equal_to_CUR"] = same
    (out / "probe.json").write_text(json.dumps(R, indent=1), encoding="utf-8")
    print("PROBE_EQUAL_TO_CUR", same, flush=True)


if __name__ == "__main__":
    main()
