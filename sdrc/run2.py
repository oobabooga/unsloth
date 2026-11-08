"""Second gfx1151 pass: HIP tile-FA FP32 accumulator fix, base vs head, in one process (interleaved per rep).
  run2.py --bin hip_base=DIR --bin hip_fa=DIR --rocm-lib DIR --models DIR --out DIR [--reps 3]
Observes only: fa_probe JSONL, test-backend-ops FLASH_ATTN_EXT, renders (timings + pixels) into results.json."""
import argparse, json, os, re, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", action="append", default=[])
    ap.add_argument("--rocm-lib", default="")
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bins = {k: v for k, v in (b.split("=", 1) for b in a.bin) if Path(v, "sd-cli").exists()}
    arms = {lab: {"label": lab, "bin": d, "env": {}, "ldpath": [a.rocm_lib] if a.rocm_lib else []} for lab, d in bins.items()}
    res = {"host": run.host_info(), "bins": bins, "srcinfo": {}, "probe": {}, "tbo": [], "runs": []}
    for lab, d in bins.items():
        si = Path(d) / "srcinfo.txt"
        res["srcinfo"][lab] = si.read_text(encoding="utf-8").splitlines() if si.exists() else None

    def save():
        (out / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")

    def env_for(arm):
        return {**os.environ, "LD_LIBRARY_PATH": ":".join([arm["bin"]] + arm["ldpath"])}

    # 1. FA overflow probe (exact expected output), base and head
    for lab, arm in arms.items():
        exe = Path(arm["bin"]) / "sdxp-fa-probe"
        f = out / f"fa_probe_{lab}.jsonl"
        if exe.exists():
            t = time.time()
            p = subprocess.run([str(exe), "ROCm0", str(f)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               env=env_for(arm), text=True, errors="replace", timeout=3600)
            (out / f"fa_probe_{lab}.log").write_text(p.stdout, encoding="utf-8")
            rows = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()] if f.exists() else []
            res["probe"][lab] = {"rc": p.returncode, "wall": round(time.time() - t, 1), "rows": rows}
            print(lab, json.dumps(res["probe"][lab]), flush=True)
    save()
    # 2. test-backend-ops FLASH_ATTN_EXT on ROCm0 (correctness of every FA case vs CPU)
    for lab, arm in arms.items():
        exe = Path(arm["bin"]) / "test-backend-ops"
        if not exe.exists():
            continue
        try:
            p = subprocess.run([str(exe), "test", "-b", "ROCm0", "-o", "FLASH_ATTN_EXT"], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, env=env_for(arm), text=True, errors="replace", timeout=5400)
            rc, text = p.returncode, p.stdout
        except subprocess.TimeoutExpired:
            rc, text = "timeout", ""
        (out / f"tbo_{lab}_FLASH_ATTN_EXT.log").write_text(text, encoding="utf-8")
        r = {"arm": lab, "rc": rc}
        m = re.findall(r"(\d+)/(\d+) tests passed", text)
        if m:
            r["passed"], r["total"] = int(m[-1][0]), int(m[-1][1])
        r["fail_lines"] = [ln.strip()[:220] for ln in text.splitlines() if "FAIL" in ln][:30]
        res["tbo"].append(r)
        print(json.dumps({k: v for k, v in r.items() if k != "fail_lines"}), r["fail_lines"][:3], flush=True)
        save()
    # 3. renders: Z-Image 1024 (DiT, unmasked attention = HIP tile FA) and SD1.5 F16 512
    M = run.fetch(a.models, ["sd15", "zdit", "qwen3", "ae"])
    P = run.PROMPT
    cases = {
        "zimg1024": ["--diffusion-model", M["zdit"], "--llm", M["qwen3"], "--vae", M["ae"], "-p", P, "--cfg-scale", "1",
                     "--steps", "8", "-W", "1024", "-H", "1024", "--seed", "42", "--sampling-method", "euler"],
        "sd15f16": ["-m", M["sd15"], "-p", P, "--steps", "20", "-W", "512", "-H", "512", "--seed", "42",
                    "--sampling-method", "euler", "--type", "f16"],
    }
    for rep in range(a.reps):
        for case, argv in cases.items():
            for lab, arm in arms.items():
                res["runs"].append(run.sd(arm, argv, out, f"{case}_{lab}_r{rep}"))
                save()
    ref = {}
    for r in res["runs"]:
        case = r["tag"].split("_")[0]
        if "png" in r and r["arm"] == "hip_base":
            ref.setdefault(case, r["png"])
    for r in res["runs"]:
        case = r["tag"].split("_")[0]
        if "png" in r and case in ref:
            r["vs_hip_base"] = run.diff(ref[case], r["png"])
    save()
    print("done", flush=True)


if __name__ == "__main__":
    main()
