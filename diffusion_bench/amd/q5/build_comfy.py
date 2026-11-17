#!/usr/bin/env python3
"""Build the tester's ComfyUI file set (Comfy-Org template files) from the OFFICIAL model repos, with ComfyUI's own
conversion tool (Comfy-Org/comfy-model-tools quant_int8_convrot.py, GitHub) for every int8_convrot file.
Nothing is downloaded from a Comfy-Org / comfyanonymous Hugging Face repo; the published files are only compared by
their safetensors header (keys / dtypes / shapes, read earlier by ref_headers.py and shipped as JSON).

  python build_comfy.py --group q21|h3 --out DIR --refs REF_DIR --tool quant_int8_convrot.py [--only dit,te,vae,avae]

Run with the ComfyUI venv python (needs comfy_kitchen, safetensors, torch with CUDA) and COMFY_DIR set.

q21  dit  official transformer shards -> img_mlp [gate_layer; proj] row-concat into gate_up (ComfyUI's fused
          SwiGLU layout, gate first: comfy/lora.py splits gate_up as (0, half) gate_layer, (half, half) proj) -> int8_convrot
     te   official text_encoder shards -> prefix model.language_model. -> model. (model.visual. kept), lm_head included -> int8_convrot
     vae  official vae -> ComfyUI Wan 2.2 layout (qwen21_vae_to_comfy.py, diffusers' own convert_vae_22 map) -> bf16
h3   dit  official FL2VA transformer -> curve-form ("pruned") adaLN: S = silu(time_embedder(t)), t = i / 1024 on 1025
          points (fp32, ComfyUI's TimeEmbedder math); rank-8 SVD S = U S V^T; adaln_t_table = U8 * S8 [1025, 8] f32;
          every adaln_proj.linear.weight W [N, 2688] -> W @ V8 [N, 8] f16, bias f16; time_embedder dropped -> int8_convrot
     te   official text_encoder (Qwen3-VL-32B) -> renamed (model.visual. -> visual.), truncated after LM layer 50 (comfy/text_encoders/minimax.py),
          lm_head dropped, metadata minimax_h3_te -> int8_convrot. SUBSTITUTE for the template's nvfp4_awq file (the AWQ
          calibration recipe is not public); it is the int8_convrot variant Comfy-Org publishes for the same TE.
     vae  official FL2VA video_vae/source -> non-decoder conv kernels f16, + latents_mean / latents_std (from FL2VA/video_vae/config.json) -> int8_convrot
     avae official FL2VA audio_vae/model.safetensors -> weight_norm folded (w = g * v / ||v||) + latents tensors, fp32
qe   (Qwen-Image-Edit-2511, ComfyUI's image_qwen_image_edit_2511_int8 template)
     dit  official transformer shards (already ComfyUI's key layout) + ComfyUI's own `__index_timestep_zero__` buffer
          (comfy/ldm/qwen_image/model.py registers torch.tensor([]); model_detection keys the 2511 ref method on it) -> int8_convrot
     te   official text_encoder shards (Qwen2.5-VL-7B, keys already model.* / visual.*), merged -> int8_convrot
          (visual excluded, lm_head included, as for q21). SUBSTITUTE for the template's qwen_2.5_vl_7b_fp8_scaled.
     vae  official vae (AutoencoderKLQwenImage) -> ComfyUI Wan 2.1 layout (diffusers' own convert_vae map), bf16
Every output gets a header diff against the published file's header (exit 1 on any key / dtype / shape mismatch
unless --allow-diff).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file

FORBIDDEN = ("comfy-org/", "comfyanonymous/")
HERE = Path(__file__).resolve().parent

Q21_REPO, Q21_FETCH = "Qwen/Qwen-Image-2.1", None  # AMD CI: the official repo is public; Studio reads the same cache
H3_REPO = "MiniMaxAI/MiniMax-H3"
QE_REPO = "Qwen/Qwen-Image-Edit-2511"
QI_REPO = "Qwen/Qwen-Image-2512"
OUTNAME = {
    ("q21", "dit"): "qwen_image_2.1_int8_convrot.safetensors",
    ("q21", "te"): "qwen3vl_8b_int8_convrot.safetensors",
    ("q21", "vae"): "qwen_image_2.1_vae_bf16.safetensors",
    ("h3", "dit"): "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    ("h3", "te"): "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
    ("h3", "vae"): "minimax_h3_video_vae_int8_convrot.safetensors",
    ("h3", "avae"): "minimax_h3_audio_vae_fp32.safetensors",
    ("qe", "dit"): "qwen_image_edit_2511_int8_convrot.safetensors",
    ("qe", "te"): "qwen_2.5_vl_7b_int8_convrot.safetensors",
    ("qe", "vae"): "qwen_image_vae.safetensors",
    ("qi", "dit"): "qwen_image_2512_int8_convrot.safetensors",
    ("qi", "te"): "qwen_2.5_vl_7b_qi_int8_convrot.safetensors",
    ("qi", "vae"): "qwen_image_vae_qi.safetensors",
}
REF = {("q21", "dit"): "q21_dit", ("q21", "te"): "q21_te", ("q21", "vae"): "q21_vae", ("h3", "dit"): "h3_dit",
       ("h3", "te"): "h3_te_int8", ("h3", "vae"): "h3_vae", ("h3", "avae"): "h3_avae"}
# which linears the published files quantize (read from their headers); passed to the tool as --include / --exclude
TOOL_ARGS = {
    ("q21", "dit"): [],
    ("q21", "te"): ["--exclude", r"^model\.visual\.", "--include", r"^lm_head$"],
    ("h3", "dit"): ["--exclude", r"^token_refiner\."],
    ("h3", "te"): ["--exclude", r"^visual\.", "--no-quant-embeddings"],
    ("h3", "vae"): [],
    ("qe", "dit"): [],
    ("qe", "te"): ["--exclude", r"^visual\.", "--include", r"^lm_head$"],
    ("qi", "dit"): [],
    ("qi", "te"): ["--exclude", r"^visual\.", "--include", r"^lm_head$"],
}


def say(*a):
    print(*a, flush = True)


SOURCES: list = []  # realpaths fetched for the role being built (dropped after it with --drop-sources)


def fetch(repo, path, fetch_repo = None):
    assert not repo.lower().startswith(FORBIDDEN) and not (fetch_repo or "").lower().startswith(FORBIDDEN)
    t = time.time()
    p = hf_hub_download(fetch_repo or repo, path, token = False)
    say("FETCH", fetch_repo or repo, path, round(os.path.getsize(p) / 1e9, 2), "GB", round(time.time() - t, 1), "s")
    SOURCES.append(os.path.realpath(p))
    return os.path.realpath(p)


def official_sha(repo, prefix):
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo, files_metadata = True, token = False)
    return {s.rfilename: (s.lfs.sha256 if s.lfs else None) for s in info.siblings if s.rfilename.startswith(prefix)}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def fetch_role(repo, prefix, fetch_repo = None, verify = True):
    want = official_sha(repo, prefix)
    paths = []
    for f in sorted(n for n in want if n.endswith(".safetensors")):
        p = fetch(repo, f, fetch_repo)
        if verify and fetch_repo and fetch_repo != repo:
            got = sha256(p)
            if got != want[f]:
                raise SystemExit(f"sha256 mismatch {fetch_repo}/{f} vs {repo}: {got} != {want[f]}")
            say("SHA_OK", f, got[:16])
        paths.append(p)
    return paths


def iter_tensors(paths):
    for p in paths:
        with safe_open(p, "pt") as f:
            for k in f.keys():
                yield k, f.get_tensor(k)


def run_tool(tool, src, dst, extra):
    cmd = [sys.executable, tool, src, dst] + extra
    say("TOOL", " ".join(cmd))
    t = time.time()
    r = subprocess.run(cmd, capture_output = True, text = True)
    tail = (r.stdout or "")[-3000:]
    say(tail)
    if r.returncode:
        say((r.stderr or "")[-4000:])
        raise SystemExit(f"quant tool failed rc={r.returncode}")
    say("TOOL_DONE", dst, round(time.time() - t, 1), "s")


def header(path):
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        h = json.loads(f.read(n))
    meta = h.pop("__metadata__", None)
    return meta, {k: {"dtype": v["dtype"], "shape": v["shape"]} for k, v in h.items()}


def header_diff(path, ref_json):
    meta, t = header(path)
    ref = json.load(open(ref_json))
    rt = ref["tensors"]
    only_b, only_r = sorted(set(t) - set(rt)), sorted(set(rt) - set(t))
    dt = sorted(k for k in set(t) & set(rt) if t[k]["dtype"] != rt[k]["dtype"])
    sh = sorted(k for k in set(t) & set(rt) if list(t[k]["shape"]) != list(rt[k]["shape"]))
    rmeta = ref.get("metadata") or {}
    mk = sorted(set(rmeta) ^ set(meta or {}))
    rep = {"file": str(path), "ref": ref["repo"] + "/" + ref["file"], "n_built": len(t), "n_ref": len(rt),
           "only_built": only_b[:40], "n_only_built": len(only_b), "only_ref": only_r[:40], "n_only_ref": len(only_r),
           "dtype_diff": [(k, t[k]["dtype"], rt[k]["dtype"]) for k in dt[:40]], "n_dtype_diff": len(dt),
           "shape_diff": [(k, t[k]["shape"], rt[k]["shape"]) for k in sh[:40]], "n_shape_diff": len(sh),
           "metadata_keys_built": sorted(meta or {}), "metadata_keys_ref": sorted(rmeta), "metadata_key_diff": mk,
           "bytes": os.path.getsize(path)}
    rep["identical_layout"] = not (only_b or only_r or dt or sh or mk)
    return rep


# ------------------------------------------------------------------ Qwen-Image-2.1
def q21_dit(tmp):
    paths = fetch_role(Q21_REPO, "transformer/", Q21_FETCH)
    sd = dict(iter_tensors(paths))
    for k in [k for k in sd if k.endswith(".img_mlp.gate_layer.weight")]:
        base = k[: -len("gate_layer.weight")]
        sd[base + "gate_up.weight"] = torch.cat([sd.pop(k), sd.pop(base + "proj.weight")], 0).contiguous()
    save_file(sd, tmp, metadata = {"format": "pt"})
    return tmp


def q21_te(tmp):
    paths = fetch_role(Q21_REPO, "text_encoder/", Q21_FETCH)
    sd = {}
    for k, v in iter_tensors(paths):
        k2 = k.replace("model.language_model.", "model.", 1) if k.startswith("model.language_model.") else k
        sd[k2] = v  # model.visual.* stays: ComfyUI's loader maps it (comfy/sd.py prefix_replace)
    save_file(sd, tmp, metadata = {"format": "pt"})
    return tmp


def q21_vae(dst):
    p = fetch_role(Q21_REPO, "vae/", Q21_FETCH)[0]
    import urllib.request

    conv = Path(dst).parent / "convert_wan_to_diffusers.py"
    if not conv.exists():
        urllib.request.urlretrieve("https://raw.githubusercontent.com/huggingface/diffusers/"
                                   "aaf57d03a3edfbd9de457d2872417a92f149a91d/scripts/convert_wan_to_diffusers.py", conv)
    tmp = str(dst) + ".f32.safetensors"
    r = subprocess.run([sys.executable, str(next(p for p in (HERE / "qwen21_vae_to_comfy.py", HERE / "img" / "qwen21_vae_to_comfy.py",
                                                     HERE.parent / "vscomfy_img" / "qwen21_vae_to_comfy.py") if p.exists())), os.environ["COMFY_DIR"],
                        str(conv), p, tmp], capture_output = True, text = True)
    say("VAE_CONVERT", r.returncode, r.stdout[-500:], r.stderr[-1500:])
    if r.returncode:
        raise SystemExit("q21 vae convert failed")
    sd = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in iter_tensors([tmp])}
    save_file(sd, str(dst), metadata = {"modelspec.architecture": "qwen_image_2.1_vae", "format": "pt"})
    os.remove(tmp)


# ------------------------------------------------------------------ Qwen-Image-Edit-2511
def qe_dit(tmp):
    sd = dict(iter_tensors(fetch_role(QE_REPO, "transformer/")))
    sd["__index_timestep_zero__"] = torch.tensor([])
    save_file(sd, tmp, metadata = {"format": "pt"})
    return tmp


def qe_te(tmp):
    sd = dict(iter_tensors(fetch_role(QE_REPO, "text_encoder/")))
    save_file(sd, tmp, metadata = {"format": "pt"})
    return tmp


def qe_vae(dst):
    p = fetch_role(QE_REPO, "vae/")[0]
    import urllib.request

    conv = Path(dst).parent / "convert_wan_to_diffusers.py"
    if not conv.exists():
        urllib.request.urlretrieve("https://raw.githubusercontent.com/huggingface/diffusers/"
                                   "aaf57d03a3edfbd9de457d2872417a92f149a91d/scripts/convert_wan_to_diffusers.py", conv)
    tmp = str(dst) + ".f32.safetensors"
    r = subprocess.run([sys.executable, str(HERE / "qwen21_vae_to_comfy.py"), "--wan21", os.environ["COMFY_DIR"],
                        str(conv), p, tmp], capture_output = True, text = True)
    say("VAE_CONVERT", r.returncode, r.stdout[-500:], r.stderr[-1500:])
    if r.returncode:
        raise SystemExit("qe vae convert failed")
    sd = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in iter_tensors([tmp])}
    save_file(sd, str(dst))
    os.remove(tmp)



def qi_dit(tmp):
    sd = dict(iter_tensors(fetch_role(QI_REPO, "transformer/")))
    save_file(sd, tmp, metadata = {"format": "pt"})
    return tmp


def qi_te(tmp):  # the Qwen-Image family ships one Qwen2.5-VL-7B text_encoder: the qe build path (proven in ComfyUI)
    return qe_te(tmp)


def qi_vae(dst):  # the same Qwen-Image VAE in every release
    qe_vae(dst)


# ------------------------------------------------------------------ MiniMax-H3
def h3_curve(time_embedder: dict, grid = 1025, rank = 8):
    """ComfyUI TimeEmbedder (comfy/ldm/minimax/model.py) on t = linspace(0, 1, grid), then silu, rank-r SVD."""
    import math

    t = torch.linspace(0.0, 1.0, grid, dtype = torch.float64)
    half = 128
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype = torch.float64) / half)
    args = t[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], -1)
    W1, b1 = time_embedder["proj_in.weight"].double(), time_embedder["proj_in.bias"].double()
    W2, b2 = time_embedder["proj_out.weight"].double(), time_embedder["proj_out.bias"].double()
    temb = torch.nn.functional.silu(emb @ W1.T + b1) @ W2.T + b2
    S = torch.nn.functional.silu(temb)  # [grid, 2688]
    U, sv, Vh = torch.linalg.svd(S, full_matrices = False)
    table = (U[:, :rank] * sv[:rank]).float()  # [grid, r]
    basis = Vh[:rank].T.contiguous()  # [2688, r], float64
    recon = table.double() @ basis.T
    rel = ((recon - S).norm() / S.norm()).item()
    maxabs = (recon - S).abs().max().item()
    say(f"H3_CURVE rank {rank} rel_err {rel:.3e} max_abs {maxabs:.3e} sv[:10] {[round(x, 4) for x in sv[:10].tolist()]}")
    return table, basis, {"rank": rank, "grid": grid, "rel_err": rel, "max_abs_err": maxabs,
                          "singular_values": sv[:16].tolist()}


def h3_dit(tmp):
    paths = fetch_role(H3_REPO, "FL2VA/transformer/")
    te = {}
    for p in paths:
        with safe_open(p, "pt") as f:
            for k in f.keys():
                if k.startswith("time_embedder."):
                    te[k[len("time_embedder."):]] = f.get_tensor(k)
    table, basis, info = h3_curve(te)
    sd = {"adaln_t_table": table.contiguous()}
    for k, v in iter_tensors(paths):
        if k.startswith("time_embedder."):
            continue
        if k.endswith("adaln_proj.linear.weight"):
            v = (v.double() @ basis).to(torch.float16).contiguous()
        elif k.endswith("adaln_proj.linear.bias"):
            v = v.to(torch.float16)
        sd[k] = v
    save_file(sd, tmp)
    json.dump(info, open(tmp + ".curve.json", "w"), indent = 1)
    return tmp


def h3_te(tmp):
    paths = fetch_role(H3_REPO, "text_encoder/")
    sd = {}
    keep_layers = 50
    for k, v in iter_tensors(paths):
        if k.startswith("lm_head."):
            continue
        k2 = k.replace("model.language_model.", "model.", 1) if k.startswith("model.language_model.") else k
        k2 = k2.replace("model.visual.", "visual.", 1) if k2.startswith("model.visual.") else k2  # as the published file
        m = re.match(r"model\.layers\.(\d+)\.", k2)
        if m and int(m.group(1)) >= keep_layers:
            continue
        if k2 == "model.norm.weight":
            continue  # the conditioning is the unnormalized hidden after layer 50
        sd[k2] = v
    meta = {"minimax_h3_te": json.dumps({"num_hidden_layers": keep_layers, "output": "unnormalized_hidden_after_layer_50"})}
    save_file(sd, tmp, metadata = meta)
    return tmp


def _h3_latents_meta(cfg_name):
    p = hf_hub_download(H3_REPO, cfg_name)
    return json.load(open(p))


def h3_vae(tmp):
    p = fetch_role(H3_REPO, "FL2VA/video_vae/source/")[0]
    cfg = _h3_latents_meta("FL2VA/video_vae/config.json")
    sd = dict(iter_tensors([p]))
    for k, v in sd.items():  # the published file stores the conv kernels outside the decoder (encoder, quant convs) f16
        if v.dim() >= 4 and not k.startswith("decoder.") and v.is_floating_point():
            sd[k] = v.to(torch.float16)
    sd["latents_mean"] = torch.tensor(cfg["latents_mean"], dtype = torch.float32)
    sd["latents_std"] = torch.tensor(cfg["latents_std"], dtype = torch.float32)
    meta = {"minimax_h3_video_vae": json.dumps({k: cfg[k] for k in cfg if not k.startswith("_")})}
    save_file(sd, tmp, metadata = meta)
    return tmp


def h3_avae(dst):
    p = fetch_role(H3_REPO, "FL2VA/audio_vae/model.safetensors")[0]
    md = json.load(open(hf_hub_download(H3_REPO, "FL2VA/audio_vae/metadata.json")))
    cfg = json.load(open(hf_hub_download(H3_REPO, "FL2VA/audio_vae/config.json")))
    sd = dict(iter_tensors([p]))
    for k in [k for k in sd if k.endswith(".weight_g")]:
        base = k[: -len("weight_g")]
        g, v = sd.pop(k), sd.pop(base + "weight_v")
        norm = v.double().flatten(1).norm(dim = 1).view(-1, *([1] * (v.dim() - 1)))
        sd[base + "weight"] = (g.double() * v.double() / norm).to(v.dtype).contiguous()
    meta_obj = {}
    for src in (cfg, md.get("metadata", md)):
        for k, v in src.items():
            if not k.startswith("_"):
                meta_obj[k] = v
    if "latents_mean" in meta_obj:
        sd["latents_mean"] = torch.tensor(meta_obj["latents_mean"], dtype = torch.float32)
        sd["latents_std"] = torch.tensor(meta_obj["latents_std"], dtype = torch.float32)
    save_file(sd, str(dst), metadata = {"minimax_h3_audio_vae": json.dumps(meta_obj)})


BUILD = {("qi", "dit"): qi_dit, ("qi", "te"): qi_te, ("qe", "dit"): qe_dit, ("qe", "te"): qe_te, ("q21", "dit"): q21_dit, ("q21", "te"): q21_te, ("h3", "dit"): h3_dit, ("h3", "te"): h3_te, ("h3", "vae"): h3_vae}
DIRECT = {("qi", "vae"): qi_vae, ("qe", "vae"): qe_vae, ("q21", "vae"): q21_vae, ("h3", "avae"): h3_avae}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required = True, choices = ["q21", "h3", "qe", "qi"])
    ap.add_argument("--out", required = True)
    ap.add_argument("--tmp", default = None)
    ap.add_argument("--refs", required = True)
    ap.add_argument("--tool", required = True)
    ap.add_argument("--only", default = None)
    ap.add_argument("--allow-diff", action = "store_true")
    ap.add_argument("--keep-tmp", action = "store_true")
    ap.add_argument("--drop-sources", action = "store_true", help = "delete each role's downloaded source files after it")
    ap.add_argument("--tool-extra", default = "{}", help = "JSON role -> extra tool args")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents = True, exist_ok = True)
    tmpd = Path(a.tmp or (out / "_tmp"))
    tmpd.mkdir(parents = True, exist_ok = True)
    extra = json.loads(a.tool_extra)
    roles = [r for (g, r) in OUTNAME if g == a.group]
    if a.only:
        roles = [r for r in roles if r in a.only.split(",")]
    manifest_p = out / f"manifest_{a.group}.json"
    man = json.load(open(manifest_p)) if manifest_p.exists() else {}
    bad = []
    for role in roles:
        key = (a.group, role)
        dst = out / OUTNAME[key]
        t0 = time.time()
        if not dst.exists():
            if key in DIRECT:
                DIRECT[key](dst)
            else:
                tmp = str(tmpd / (OUTNAME[key] + ".src.safetensors"))
                BUILD[key](tmp)
                run_tool(a.tool, tmp, str(dst), TOOL_ARGS.get(key, []) + extra.get(role, []))
                if os.path.exists(tmp + ".curve.json"):
                    man.setdefault("curve", json.load(open(tmp + ".curve.json")))
                if not a.keep_tmp:
                    os.remove(tmp)
        if a.drop_sources:
            freed = 0
            for sp in SOURCES:
                if os.path.exists(sp) and os.path.getsize(sp) > 50e6:
                    freed += os.path.getsize(sp)
                    os.remove(sp)
            say("DROPPED_SOURCES", role, round(freed / 1e9, 1), "GB")
        SOURCES.clear()
        if key not in REF:  # no published header to compare (qe): ComfyUI's loader is the check (trial arm)
            man[role] = {"file": str(dst), "bytes": os.path.getsize(dst), "build_s": round(time.time() - t0, 1),
                         "header_ref": None, "n_built": len(header(dst)[1])}
            json.dump(man, open(manifest_p, "w"), indent = 1)
            say("BUILT_NO_REF", role, json.dumps(man[role]))
            continue
        rep = header_diff(dst, os.path.join(a.refs, REF[key] + ".json"))
        rep["build_s"] = round(time.time() - t0, 1)
        man[role] = rep
        json.dump(man, open(manifest_p, "w"), indent = 1)
        say("HEADER_DIFF", role, json.dumps({k: rep[k] for k in ("identical_layout", "n_built", "n_ref", "n_only_built",
                                                                 "n_only_ref", "n_dtype_diff", "n_shape_diff",
                                                                 "metadata_key_diff")}))
        if not rep["identical_layout"]:
            say("  only_built", rep["only_built"][:12])
            say("  only_ref", rep["only_ref"][:12])
            say("  dtype_diff", rep["dtype_diff"][:12])
            say("  shape_diff", rep["shape_diff"][:6])
            bad.append(role)
    say("BUILD_DONE", a.group, "mismatch:" if bad else "all identical layout", bad)
    return 1 if bad and not a.allow_diff else 0


if __name__ == "__main__":
    sys.exit(main())
