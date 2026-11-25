"""a12_probe items 6 (bf16 attention race: AOTriton / CK / aiter rdna_attn) and 7 (AITER Triton conv3d vs MIOpen).
Called as ITEMS[name](ctx, obs, run, rounds, HERE); aiter_setup.sh runs once (bounded, never fails)."""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time


def _setup(ctx, obs, HERE):
    w = ctx["work"] / "aiter"
    st = w / "aiter_setup.json"
    if not st.is_file():
        env = {k: v for k, v in os.environ.items() if "TOKEN" not in k}
        env.update({"AMD_CI_PY": sys.executable, "A12_WORK": str(w)})
        t = time.time()
        subprocess.run(["bash", str(HERE / "aiter_setup.sh")], env = env, timeout = 4800,
                       stdout = open(ctx["log"], "ab"), stderr = subprocess.STDOUT)
        obs["aiter_setup_s"] = round(time.time() - t, 1)
    try:
        obs["aiter_setup"] = json.loads(st.read_text(encoding = "utf-8"))
    except Exception as exc:  # noqa: BLE001
        obs["aiter_setup"] = {"error": repr(exc)}
    os.environ["A12_WORK"] = str(w)
    return w


def item_race(ctx, obs, run, rounds, HERE):
    w = _setup(ctx, obs, HERE)
    res = []
    for shape in ("wan5b", "qwen21", "ltx", "ltx_long", "h3"):
        for rnd, arm in rounds(["aotriton", "ck", "rdna"]):
            o = ctx["work"] / f"race_{shape}_{arm}_{rnd}.json"
            r = run([HERE / "aiter_rdna.py", "--aiter-dir", w / "aiter_rdna", "--arm", arm, "--shape", shape,
                     "--out", o], o, ctx["log"], env = {"A12_WORK": str(w)}, timeout = 2400)
            r.update({"round": rnd, "arm": arm, "shape": shape})
            res.append(r)
            obs["race"] = res


def _psnr(a_path, b_path):
    import torch
    a, b = torch.load(a_path).float(), torch.load(b_path).float()
    if a.shape != b.shape:
        return None
    mse = float(((a - b) ** 2).mean())
    return float("inf") if mse == 0 else 10 * math.log10(4.0 / mse)


def item_conv3d(ctx, obs, run, rounds, HERE):
    w = _setup(ctx, obs, HERE)
    res = []
    for rnd, arm in rounds(["miopen", "aiter"]):
        o = ctx["work"] / f"conv3d_micro_{arm}_{rnd}.json"
        r = run([HERE / "aiter_conv3d.py", "--aiter-dir", w / "aiter", "--mode", "micro", "--arm", arm, "--out", o],
                o, ctx["log"], env = {"A12_WORK": str(w)}, timeout = 3600)
        r.update({"round": rnd, "arm": arm, "mode": "micro"})
        res.append(r)
        obs["conv3d"] = res
    est = {}
    for arm in ("miopen", "aiter"):
        xs = sorted(r.get("est_decode_conv_ms") for r in res if r.get("arm") == arm and r.get("est_decode_conv_ms"))
        est[arm] = xs[len(xs) // 2] if xs else None
    ratio = est["miopen"] / est["aiter"] if est["miopen"] and est["aiter"] else None
    obs["conv3d_micro_ratio"] = ratio
    # The decode can only reach the 1.15x kill line if the convs alone do: decode only when the micro sum is close.
    if not ratio or ratio < 1.12:
        obs["conv3d_decode_skip"] = f"micro conv sum ratio {ratio} < 1.12 (decode kill line 1.15x unreachable)"
        return
    vae = ctx["work"] / "models" / "wan5b"
    code = ("import sys\nfrom huggingface_hub import snapshot_download\nsnapshot_download(sys.argv[1], allow_patterns ="
            " ['vae/*', 'model_index.json'], local_dir = sys.argv[2], token = False)\n")
    subprocess.run([sys.executable, "-c", code, "Wan-AI/Wan2.2-TI2V-5B-Diffusers", str(vae)], timeout = 3600,
                   env = {k: v for k, v in os.environ.items() if "TOKEN" not in k})
    names = []
    for rnd, arm in rounds(["miopen", "aiter"]):
        o = ctx["work"] / f"conv3d_decode_{arm}_{rnd}.json"
        save = ctx["work"] / f"conv3d_decode_{arm}_{rnd}.pt"
        r = run([HERE / "aiter_conv3d.py", "--aiter-dir", w / "aiter", "--mode", "decode", "--arm", arm, "--vae-dir",
                 vae, "--out", o, "--save", save], o, ctx["log"], env = {"A12_WORK": str(w)}, timeout = 3600)
        r.update({"round": rnd, "arm": arm, "mode": "decode"})
        res.append(r)
        names.append(f"{arm}_{rnd}")
        obs["conv3d"] = res
    ref = ctx["work"] / "conv3d_decode_miopen_0.pt"
    obs["conv3d_decode_psnr_vs_miopen0"] = {n: _psnr(ref, ctx["work"] / f"conv3d_decode_{n}.pt") for n in names
                                            if (ctx["work"] / f"conv3d_decode_{n}.pt").is_file() and n != "miopen_0"}
    for n in names:
        p = ctx["work"] / f"conv3d_decode_{n}.pt"
        if p.is_file():
            p.unlink()


ITEMS = {"race": item_race, "conv3d": item_conv3d}
