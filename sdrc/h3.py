"""MiniMax-H3 on RADV: fork master (M) vs the upstream-9234ad8 sync branch (H), one machine, interleaved.

h3.py --bin-root DIR --models DIR --out DIR
Small config (nothing chunks): M, H, H with SD_H3_QKV_CHUNK_TOKENS forced, A/A repeats.
Large config (H chunks on its own under the 1 GiB Vulkan buffer limit): H, then M.
Frames are written as PNG sequences; md5 over decoded pixels; PSNR vs the reference arm.
"""
import argparse, hashlib, json, math, os, re, subprocess, time
from pathlib import Path

import numpy as np
from PIL import Image

PROMPT = "a red fox trotting through falling snow, cinematic"
SMALL = dict(w=640, h=384, frames=25, steps=4)
LARGE = dict(w=1024, h=576, frames=97, steps=2)
PAT = {"sampling": r"sampling completed, taking ([\d.]+)\s*s", "total": r"generate_video completed in ([\d.]+)\s*s",
       "decode": r"decode_first_stage completed, taking ([\d.]+)\s*s"}


def frames_of(d):
    return sorted(Path(d).glob("f_*.png"))


def pix_md5(d):
    h = hashlib.md5()
    for f in frames_of(d):
        h.update(np.asarray(Image.open(f).convert("RGB")).tobytes())
    return h.hexdigest()


def psnr(da, db):
    fa, fb = frames_of(da), frames_of(db)
    if not fa or len(fa) != len(fb):
        return {"err": f"frames {len(fa)} vs {len(fb)}"}
    se, n, mx = 0.0, 0, 0
    for a, b in zip(fa, fb):
        x = np.asarray(Image.open(a).convert("RGB"), dtype=np.int16)
        y = np.asarray(Image.open(b).convert("RGB"), dtype=np.int16)
        d = np.abs(x - y)
        mx = max(mx, int(d.max())); se += float((d.astype(np.float64) ** 2).sum()); n += d.size
    mse = se / n
    return {"psnr": 99.0 if mse == 0 else round(10 * math.log10(255 * 255 / mse), 2), "max_abs": mx}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-root", required=True); ap.add_argument("--models", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--skip-large", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    m = Path(a.models)
    base = ["--mode", "vid_gen", "--diffusion-model", str(m / "minimax_h3_fl2va_pruned-Q2_K.gguf"),
            "--llm", str(m / "qwen3vl_32b_minimax_h3-Q2_K_M.gguf"), "--vae", str(m / "vae/minimax_h3_video_vae_fp16.safetensors"),
            "--audio-vae", str(m / "vae/minimax_h3_audio_vae_fp32.safetensors"), "--prompt", PROMPT,
            "--seed", "1234", "--cfg-scale", "1.0", "--rng", "cpu", "--diffusion-fa", "-v"]
    res = {"runs": [], "compare": []}

    def run(rid, ref, cfg, env=None, extra=()):
        d = out / rid; d.mkdir(exist_ok=True)
        cli = Path(a.bin_root) / ref / "sd-cli"
        cmd = [str(cli)] + base + ["--width", str(cfg["w"]), "--height", str(cfg["h"]), "--video-frames", str(cfg["frames"]),
                                   "--steps", str(cfg["steps"]), "-o", str(d / "f_%03d.png")] + list(extra)
        e = dict(os.environ); e.update(env or {})
        log = out / f"{rid}.log"
        print("run", rid, " ".join(cmd[1:]), env or {}, flush=True)
        t = time.time()
        with open(log, "w", encoding="utf-8", errors="replace") as fh:
            p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=e)
            try:
                rc = p.wait(timeout=5400)
            except subprocess.TimeoutExpired:
                p.kill(); rc = -999
        text = log.read_text(encoding="utf-8", errors="replace")
        rec = {"id": rid, "ref": ref, "cfg": cfg, "env": env or {}, "rc": rc, "wall": round(time.time() - t, 1),
               "chunk_log": re.findall(r"MiniMax-H3: QKV projection of \d+ tokens runs in chunks of \d+ tokens", text)[:2],
               "vk_device": re.findall(r"ggml_vulkan: \d+ = .*", text)[:2], "frames": len(frames_of(d))}
        for k, pat in PAT.items():
            mm = re.findall(pat, text)
            if mm:
                rec[k] = float(mm[-1])
        if rc == 0 and rec["frames"]:
            rec["md5"] = pix_md5(d)
        else:
            rec["tail"] = text[-3000:]
        print("  ->", json.dumps({k: v for k, v in rec.items() if k != "tail"}), flush=True)
        res["runs"].append(rec)
        (out / "results.json").write_text(json.dumps(res, indent=1))
        return rec

    def cmp(x, y, expect):
        rx = next(r for r in res["runs"] if r["id"] == x); ry = next(r for r in res["runs"] if r["id"] == y)
        c = {"a": x, "b": y, "expect": expect, "identical": rx.get("md5") is not None and rx.get("md5") == ry.get("md5")}
        if not c["identical"] and rx["frames"] and ry["frames"]:
            c.update(psnr(out / x, out / y))
        res["compare"].append(c); print("compare", json.dumps(c), flush=True)
        (out / "results.json").write_text(json.dumps(res, indent=1))

    run("M.small.r0", "M", SMALL)
    run("H.small.r0", "H", SMALL)
    run("H.small.chunk512", "H", SMALL, {"SD_H3_QKV_CHUNK_TOKENS": "512"})
    run("M.small.r1", "M", SMALL)
    run("H.small.r1", "H", SMALL)
    cmp("H.small.r0", "M.small.r0", "identical")
    cmp("M.small.r1", "M.small.r0", "identical")
    cmp("H.small.r1", "H.small.r0", "identical")
    cmp("H.small.chunk512", "H.small.r0", "close")
    if not a.skip_large:
        run("H.large", "H", LARGE, extra=["--vae-tiling"])
        run("M.large", "M", LARGE, extra=["--vae-tiling"])
        cmp("H.large", "M.large", "any")
    (out / "results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
