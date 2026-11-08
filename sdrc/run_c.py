"""gfx1151: (A) HIP bf16 WMMA operand patch 0028 (+ harness arm with ggml 06db896e3b), (B) Vulkan no host import on
AMD, patch 0029, single RADV device and RADV + llvmpipe (two devices: the multi-device import path).
  run_c.py --bin LABEL=DIR ... --rocm-lib DIR --models DIR --out DIR [--reps 3]
Labels: hip_master hip_wmma hip_06db vk_master vk_uma. Observes only; results.json."""
import argparse, json, os, re, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run  # noqa: E402

run.MODELS["qwen3bf16"] = ("unsloth/Qwen3-4B-GGUF", "Qwen3-4B-BF16.gguf")
ap = argparse.ArgumentParser()
ap.add_argument("--bin", action="append", default=[])
ap.add_argument("--rocm-lib", default="")
ap.add_argument("--models", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--reps", type=int, default=3)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
bins = {k: v for k, v in (b.split("=", 1) for b in a.bin) if Path(v, "sd-cli").exists()}
res = {"host": run.host_info(), "bins": bins, "srcinfo": {}, "list_devices": {}, "tbo": [], "runs": []}
for lab, d in bins.items():
    si = Path(d) / "srcinfo.txt"
    res["srcinfo"][lab] = si.read_text(encoding="utf-8").splitlines() if si.exists() else None
save = lambda: (out / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")

hip = [{"label": l, "bin": bins[l], "env": {}, "ldpath": [a.rocm_lib]} for l in ("hip_master", "hip_wmma", "hip_06db") if l in bins]
vk = []
for l in ("vk_master", "vk_uma"):
    if l in bins:
        vk.append({"label": f"{l}_1dev", "bin": bins[l], "env": {"GGML_VK_VISIBLE_DEVICES": "0"}})
        vk.append({"label": f"{l}_2dev", "bin": bins[l], "env": {"GGML_VK_VISIBLE_DEVICES": "0,1"},
                   "argv": ["--backend", "diffusion=Vulkan0,te=Vulkan0,vae=Vulkan0"]})
for arm in hip + vk:
    env = {**os.environ, "LD_LIBRARY_PATH": ":".join([arm["bin"]] + arm.get("ldpath", [])), **arm["env"]}
    p = subprocess.run([str(Path(arm["bin"]) / "sd-cli"), "--list-devices"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, errors="replace", env=env, timeout=300)
    res["list_devices"][arm["label"]] = p.stdout[-2000:]
    print("==", arm["label"], p.stdout[-600:], flush=True)
save()
# (A) test-backend-ops MUL_MAT with a bf16 operand on ROCm0, and the whole MUL_MAT suite count
for arm in hip:
    exe = Path(arm["bin"]) / "test-backend-ops"
    env = {**os.environ, "LD_LIBRARY_PATH": ":".join([arm["bin"]] + arm["ldpath"])}
    for tag, extra in (("bf16", ["-p", "bf16"]), ("all", [])):
        try:
            p = subprocess.run([str(exe), "test", "-b", "ROCm0", "-o", "MUL_MAT", *extra], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, env=env, text=True, errors="replace", timeout=3600)
            rc, text = p.returncode, p.stdout
        except subprocess.TimeoutExpired:
            rc, text = "timeout", ""
        (out / f"tbo_{arm['label']}_MUL_MAT_{tag}.log").write_text(text, encoding="utf-8")
        m = re.findall(r"(\d+)/(\d+) tests passed", text)
        r = {"arm": arm["label"], "filter": tag, "rc": rc, "passed": int(m[-1][0]) if m else None, "total": int(m[-1][1]) if m else None,
             "fail_lines": [ln.strip()[:200] for ln in text.splitlines() if "FAIL" in ln][:20]}
        res["tbo"].append(r); print(json.dumps(r)[:400], flush=True); save()
M = run.fetch(a.models, ["sd15", "zdit", "qwen3", "qwen3bf16", "ae"])
P = run.PROMPT
sd15 = ["-m", M["sd15"], "-p", P, "--steps", "20", "-W", "512", "-H", "512", "--seed", "42", "--sampling-method", "euler", "--type", "f16"]
zq4 = ["--diffusion-model", M["zdit"], "--llm", M["qwen3"], "--vae", M["ae"], "-p", P, "--cfg-scale", "1", "--steps", "8",
       "-W", "512", "-H", "512", "--seed", "42", "--sampling-method", "euler"]
zbf = ["--diffusion-model", M["zdit"], "--llm", M["qwen3bf16"], "--vae", M["ae"], "-p", P, "--cfg-scale", "1", "--steps", "8",
       "-W", "512", "-H", "512", "--seed", "42", "--sampling-method", "euler"]
plan = [("hsd15", arm, sd15) for arm in hip] + [("hzbf16te", arm, zbf) for arm in hip] + \
       [("vsd15", arm, sd15) for arm in vk] + [("vzq4", arm, zq4) for arm in vk]
for rep in range(a.reps):
    for case, arm, argv in plan:
        res["runs"].append(run.sd(arm, argv, out, f"{case}_{arm['label']}_r{rep}"))
        save()
ref = {}
for r in res["runs"]:
    c = r["tag"].split("_")[0]
    if "png" in r and r["arm"] in ("hip_master", "vk_master_1dev"):
        ref.setdefault(c, r["png"])
for r in res["runs"]:
    c = r["tag"].split("_")[0]
    if "png" in r and c in ref:
        r["vs_master"] = run.diff(ref[c], r["png"])
save()
print("done", flush=True)
