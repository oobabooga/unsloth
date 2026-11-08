"""#51 device-choice fix: HIP+Vulkan gfx1151 builds of master, #51 and the fix, auto-fit decisions, timings, pixels.
  run_fix51.py --bin hv_master=DIR --bin hv_h51=DIR --bin hv_fix=DIR --rocm-lib DIR --models DIR --out DIR [--reps 3]"""
import argparse, json, statistics as st, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--bin", action="append", default=[])
ap.add_argument("--rocm-lib", default="")
ap.add_argument("--models", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--reps", type=int, default=3)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
bins = {k: v for k, v in (b.split("=", 1) for b in a.bin) if Path(v, "sd-cli").exists()}
VK = {"GGML_VK_VISIBLE_DEVICES": "0"}
arms = [{"label": lab, "bin": d, "env": dict(VK), "ldpath": [a.rocm_lib]} for lab, d in bins.items()]
if "hv_fix" in bins:
    for dev in ("Vulkan0", "ROCm0"):
        arms.append({"label": f"hv_fix_{dev}", "bin": bins["hv_fix"], "env": dict(VK), "ldpath": [a.rocm_lib],
                     "argv": ["--backend", f"diffusion={dev},te={dev},vae={dev}"]})
res = {"host": run.host_info(), "bins": bins, "srcinfo": {}, "runs": []}
for lab, d in bins.items():
    si = Path(d) / "srcinfo.txt"
    res["srcinfo"][lab] = si.read_text(encoding="utf-8").splitlines()[0] if si.exists() else None
save = lambda: (out / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
M = run.fetch(a.models, ["sd15", "zdit", "qwen3", "ae"])
P = run.PROMPT
cases = {
    "sd15f16": ["-m", M["sd15"], "-p", P, "--steps", "20", "-W", "512", "-H", "512", "--seed", "42",
                "--sampling-method", "euler", "--type", "f16"],
    "zimg512": ["--diffusion-model", M["zdit"], "--llm", M["qwen3"], "--vae", M["ae"], "-p", P, "--cfg-scale", "1",
                "--steps", "8", "-W", "512", "-H", "512", "--seed", "42", "--sampling-method", "euler"],
}
for rep in range(a.reps):
    for case, argv in cases.items():
        for arm in arms:
            res["runs"].append(run.sd(arm, argv, out, f"{case}_{arm['label']}_r{rep}"))
            save()
ref = {}
for r in res["runs"]:
    if "png" in r and r["arm"] == "hv_master":
        ref.setdefault(r["tag"].split("_")[0], r["png"])
for r in res["runs"]:
    c = r["tag"].split("_")[0]
    if "png" in r and c in ref:
        r["vs_hv_master"] = run.diff(ref[c], r["png"])
save()
print("done", flush=True)
