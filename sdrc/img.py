"""Vulkan image s/step on RADV: fork master (M), pure upstream sync (U), sync with the fallback fix (F).

img.py --bin-root DIR --models DIR --out DIR
Flux.1-schnell, Qwen-Image, Z-Image at 1024x1024, 4 steps, cfg 1, seed 42; flash attention auto (default) and off.
Interleaved per rep; md5 over decoded pixels; s/step = sampling / steps.
"""
import argparse, hashlib, json, re, subprocess, time
from pathlib import Path

import numpy as np
from PIL import Image

PROMPT = "a photo of a red fox sitting in fresh snow, golden hour, detailed fur"
PAT = {"sampling": r"sampling completed, taking ([\d.]+)\s*s", "total": r"generate_image completed in ([\d.]+)\s*s"}
DEC = r"(flash attention auto: .*|conv[23]d direct auto: .*|.*auto-fit.*|.*->\s*compute .*)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-root", required=True); ap.add_argument("--models", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m = Path(a.models); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cases = {
        "flux": ["--diffusion-model", str(m / "flux1-schnell-Q2_K.gguf"), "--t5xxl", str(m / "t5-v1_1-xxl-encoder-Q3_K_M.gguf"),
                 "--clip_l", str(m / "clip_l.safetensors"), "--vae", str(m / "ae.safetensors")],
        "qwen": ["--diffusion-model", str(m / "qwen-image-Q2_K.gguf"), "--llm", str(m / "Qwen2.5-VL-7B-Instruct-Q2_K.gguf"),
                 "--vae", str(m / "qwen_vae.safetensors")],
        "zimage": ["--diffusion-model", str(m / "z-image-turbo-Q2_K.gguf"), "--llm", str(m / "Qwen3-4B-Q2_K.gguf"),
                   "--vae", str(m / "ae.safetensors")],
    }
    arms = [(ref, reps) for ref, reps in (("M", 2), ("F", 2), ("U", 1))]
    res = {"runs": [], "summary": {}}
    for case, argv in cases.items():
        for fa in ("auto", "off"):
            for rep in range(2):
                for ref, reps in arms:
                    if rep >= reps:
                        continue
                    rid = f"{case}.{fa}.{ref}.r{rep}"
                    png = out / f"{rid}.png"
                    cmd = [str(Path(a.bin_root) / ref / "sd-cli")] + argv + [
                        "-p", PROMPT, "-W", "1024", "-H", "1024", "--steps", "4", "--cfg-scale", "1", "--seed", "42",
                        "--sampling-method", "euler", "--flash-attn", fa, "-o", str(png), "-v"]
                    log = out / f"{rid}.log"
                    print("run", rid, flush=True)
                    t = time.time()
                    with open(log, "w", encoding="utf-8", errors="replace") as fh:
                        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
                        try:
                            rc = p.wait(timeout=2400)
                        except subprocess.TimeoutExpired:
                            p.kill(); rc = -999
                    text = log.read_text(encoding="utf-8", errors="replace")
                    rec = {"id": rid, "case": case, "fa": fa, "ref": ref, "rep": rep, "rc": rc, "wall": round(time.time() - t, 1),
                           "decisions": sorted(set(x.strip()[:200] for x in re.findall(DEC, text)))[:12]}
                    for k, pat in PAT.items():
                        mm = re.findall(pat, text)
                        if mm:
                            rec[k] = float(mm[-1])
                    if rec.get("sampling"):
                        rec["s_per_step"] = round(rec["sampling"] / 4, 3)
                    if rc == 0 and png.exists():
                        rec["md5"] = hashlib.md5(np.asarray(Image.open(png).convert("RGB")).tobytes()).hexdigest()
                    else:
                        rec["tail"] = text[-2500:]
                    print("  ->", json.dumps({k: v for k, v in rec.items() if k not in ("tail", "decisions")}), flush=True)
                    res["runs"].append(rec)
                    (out / "results.json").write_text(json.dumps(res, indent=1))
    for case in cases:
        for fa in ("auto", "off"):
            row = {}
            for ref, _ in arms:
                rs = [r for r in res["runs"] if r["case"] == case and r["fa"] == fa and r["ref"] == ref]
                row[ref] = {"s_per_step": [r.get("s_per_step") for r in rs], "md5": sorted({r.get("md5") for r in rs if r.get("md5")})}
            row["F_eq_M"] = row["F"]["md5"] == row["M"]["md5"] and bool(row["M"]["md5"])
            row["U_eq_M"] = row["U"]["md5"] == row["M"]["md5"] and bool(row["M"]["md5"])
            res["summary"][f"{case}.{fa}"] = row
            print("summary", case, fa, json.dumps(row), flush=True)
    (out / "results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
