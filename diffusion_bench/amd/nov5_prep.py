#!/usr/bin/env python3
"""ComfyUI-loadable bf16 files for the gfx1151 nov5 round, built on the runner from the OFFICIAL model repos.

Never reads a Comfy-Org / comfyanonymous Hugging Face repo. Shards are merged by streaming raw bytes into one
safetensors file (no full state dict in RAM: the APU's memory is the GPU's memory), renamed or concatenated where
ComfyUI's layout differs from the diffusers shards:

  klein  dit  black-forest-labs/FLUX.2-klein-4B flux-2-klein-4b.safetensors (BFL single file, linked)
         te   text_encoder shards merged (Qwen3-4B, keys as shipped)
         vae  vae/diffusion_pytorch_model.safetensors (linked; ComfyUI converts the diffusers keys)
  f3     dit  model.safetensors (BFL single file) via the ungated byte-identical unsloth/flux-3-image-klein mirror
         te   text_encoder shards, model.language_model. -> model., vision tower dropped (Qwen3-VL-4B text model)
         vae  ae.safetensors
  zimg   dit  Tongyi-MAI/Z-Image-Turbo transformer shards merged + cast fp32 -> bf16
         te   text_encoder shards merged
         vae  BFL FLUX.1 ae.safetensors via unsloth/FLUX.1-schnell (sha256 checked against the official LFS sha256)
  q21    dit  Qwen/Qwen-Image-2.1 transformer shards; img_mlp gate_layer + proj row-concatenated into gate_up
         te   text_encoder shards, model.language_model. -> model.
         vae  qwen21_vae_to_comfy.py (diffusers' own convert_vae_22 name map) -> bf16
Wan2.2-TI2V-5B is built by convert_comfy.py (official Wan-AI repo). Writes <out>/manifest_nov5.json.

  python nov5_prep.py --groups klein,f3,zimg,q21 --out DIR [--comfy-python P --comfy-dir D]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

FORBIDDEN = ("comfy-org/", "comfyanonymous/")
HERE = Path(__file__).resolve().parent
DIFFUSERS_CONVERT = ("https://raw.githubusercontent.com/huggingface/diffusers/"
                     "aaf57d03a3edfbd9de457d2872417a92f149a91d/scripts/convert_wan_to_diffusers.py")
DT_SIZE = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
           "F8_E4M3": 1, "F8_E5M2": 1}


def say(*a):
    print(*a, flush = True)


SNAP: dict = {}  # repo -> the local snapshot dir Studio reads (one download serves both stacks)
GROUP_ALIAS = {"klein": "flux2_klein_4b", "f3": "flux3_klein", "zimg": "zimage_turbo", "q21": "qwen_image_21",
               "wan": "wan22_5b"}


def snapshot(spec_path, groups, root, env_file):
    """Studio's model dirs, exactly as diffusion_probe.fetch_models would fetch them (same allow / ignore), with the
    probe's completion marker, and DBENCH_MODEL_<ALIAS> lines for $GITHUB_ENV so the probe reuses them."""
    from huggingface_hub import snapshot_download

    models = json.loads(Path(spec_path).read_text(encoding = "utf-8"))["models"]
    lines = []
    for g in groups:
        alias = GROUP_ALIAS[g]
        e = models[alias]
        assert not e["repo"].lower().startswith(FORBIDDEN), e["repo"]
        d = root / alias
        SNAP[e["repo"]] = d
        if not (d / ".dbench_complete").exists():
            t = time.time()
            for attempt in range(6):
                try:
                    snapshot_download(repo_id = e["repo"], revision = e.get("revision"), local_dir = str(d),
                                      ignore_patterns = (e.get("download") or {}).get("ignore"),
                                      allow_patterns = (e.get("download") or {}).get("allow"), token = False)
                    break
                except Exception as exc:  # noqa: BLE001 - mirror 500s happen: retry
                    say("SNAPSHOT_RETRY", alias, attempt, f"{type(exc).__name__}: {str(exc)[:300]}")
                    time.sleep(20)
            else:
                raise SystemExit(f"snapshot {alias} failed")
            (d / ".dbench_complete").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"), encoding = "utf-8")
            say("SNAPSHOT", alias, e["repo"], round(time.time() - t, 1), "s")
        key = "DBENCH_MODEL_" + "".join(ch if ch.isalnum() else "_" for ch in alias).upper()
        lines.append(f"{key}={d}")
    if env_file:
        with open(env_file, "a", encoding = "utf-8") as f:
            f.write("".join(x + "\n" for x in lines))
    return lines


def fetch(repo, path, official = None):
    from huggingface_hub import hf_hub_download

    assert not repo.lower().startswith(FORBIDDEN) and not (official or "").lower().startswith(FORBIDDEN), repo
    t = time.time()
    local = SNAP.get(repo)
    if local is not None and (Path(local) / path).is_file():
        p = os.path.realpath(Path(local) / path)
    elif local is not None:
        p = os.path.realpath(hf_hub_download(repo, path, token = False, local_dir = str(local)))
    else:
        p = os.path.realpath(hf_hub_download(repo, path, token = False))
    say("FETCH", repo, path, round(os.path.getsize(p) / 1e9, 2), "GB", round(time.time() - t, 1), "s")
    if official and official != repo:
        from huggingface_hub import HfApi

        want = None
        for s in HfApi().model_info(official, files_metadata = True, token = False).siblings:
            if s.rfilename == path and s.lfs:
                want = s.lfs.sha256 if hasattr(s.lfs, "sha256") else s.lfs.get("sha256")
        got = sha256(p)
        say("SHA", path, got[:16], "official", (want or "?")[:16], "MATCH" if got == want else "MISMATCH")
        if got != want:
            raise SystemExit(f"{repo}/{path} is not byte-identical to {official}/{path}")
    return p


def shards(repo, prefix):
    from huggingface_hub import HfApi

    names = sorted(s.rfilename for s in HfApi().model_info(repo, token = False).siblings
                   if s.rfilename.startswith(prefix) and s.rfilename.endswith(".safetensors"))
    return [fetch(repo, n) for n in names]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return 8 + n, h


class Src:
    """One output tensor: raw byte ranges from source files (concatenated in order), optionally re-cast."""

    def __init__(self, dtype, shape, parts, cast = None):
        self.dtype, self.shape, self.parts, self.cast = dtype, shape, parts, cast

    @property
    def nbytes(self):
        n = DT_SIZE[self.dtype]
        for s in self.shape:
            n *= s
        return n


def plan_from(paths, rename = None, drop = None):
    """key -> Src for every tensor of the shards, keys renamed / dropped."""
    out = {}
    for p in paths:
        base, h = read_header(p)
        for k, v in h.items():
            k2 = rename(k) if rename else k
            if k2 is None or (drop and drop(k2)):
                continue
            b, e = v["data_offsets"]
            out[k2] = Src(v["dtype"], list(v["shape"]), [(p, base + b, e - b)])
    return out


def write(plan, dst, meta = None):
    """Stream the plan into one safetensors file: header, then each tensor's bytes, one at a time."""
    tmp = str(dst) + ".part"
    hdr, off = {}, 0
    keys = sorted(plan)
    for k in keys:
        s = plan[k]
        dt = "BF16" if s.cast == "bf16" else s.dtype
        n = s.nbytes if s.cast is None else s.nbytes // DT_SIZE[s.dtype] * DT_SIZE[dt]
        hdr[k] = {"dtype": dt, "shape": s.shape, "data_offsets": [off, off + n]}
        off += n
    hdr["__metadata__"] = {"format": "pt", **(meta or {})}
    hb = json.dumps(hdr, separators = (",", ":")).encode("utf-8")
    hb += b" " * ((8 - len(hb) % 8) % 8)
    t = time.time()
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for k in keys:
            s = plan[k]
            if s.cast == "bf16" and s.dtype != "BF16":
                import torch
                from safetensors import safe_open

                (p, _, _), = s.parts
                with safe_open(p, "pt") as sf:
                    t_ = sf.get_tensor(src_key(s)).to(torch.bfloat16).contiguous()
                f.write(t_.view(torch.uint8).numpy().tobytes())
                del t_
                continue
            for p, b, n in s.parts:
                with open(p, "rb") as sf:
                    sf.seek(b)
                    left = n
                    while left:
                        chunk = sf.read(min(left, 1 << 26))
                        if not chunk:
                            raise SystemExit(f"short read {p}")
                        f.write(chunk)
                        left -= len(chunk)
    os.replace(tmp, dst)
    say("WROTE", dst, len(keys), "tensors", round(os.path.getsize(dst) / 1e9, 2), "GB", round(time.time() - t, 1), "s")


def src_key(s):
    return s.key


def cast_plan(paths):
    """Like plan_from, but every float tensor is re-cast to bf16 (needs the source key for safe_open)."""
    out = {}
    for p in paths:
        base, h = read_header(p)
        for k, v in h.items():
            b, e = v["data_offsets"]
            s = Src(v["dtype"], list(v["shape"]), [(p, base + b, e - b)],
                    cast = "bf16" if v["dtype"] in ("F32", "F16", "F64") else None)
            s.key = k
            out[k] = s
    return out


def link(src, dst):
    if os.path.lexists(dst):
        os.unlink(dst)
    os.symlink(src, dst)
    say("LINKED", dst, "->", src)
    return str(dst)


def strip_lm(k):
    return k.replace("model.language_model.", "model.", 1) if k.startswith("model.language_model.") else k


def build(group, out, comfy_py, comfy_dir):
    out.mkdir(parents = True, exist_ok = True)
    files = {}
    if group == "klein":
        r = "black-forest-labs/FLUX.2-klein-4B"
        files["dit"] = link(fetch(r, "flux-2-klein-4b.safetensors"), out / "flux-2-klein-4b.safetensors")
        dst = out / "qwen3_4b_klein_merged.safetensors"
        if not dst.exists():
            write(plan_from(shards(r, "text_encoder/")), dst)
        files["te"] = str(dst)
        files["vae"] = link(fetch(r, "vae/diffusion_pytorch_model.safetensors"),
                            out / "flux2_klein_vae_diffusers.safetensors")
    elif group == "f3":
        r, off = "unsloth/flux-3-image-klein", "black-forest-labs/flux-3-image-klein"
        files["dit"] = link(fetch(r, "model.safetensors", official = off), out / "flux-3-image-klein.safetensors")
        # Shards merged as shipped: ComfyUI detects Qwen3-VL-4B by its vision tower (model.visual.deepstack_merger_list)
        # and does the model.language_model. -> model. rename itself (comfy/sd.py, CLIPType.FLUX3 branch); without the
        # tower it loads as plain Qwen3-4B and the 8-layer tap (20480 wide) never happens.
        dst = out / "qwen3vl_4b_flux3_te_bf16.safetensors"
        if not dst.exists():
            write(plan_from(shards(r, "text_encoder/")), dst)
        files["te"] = str(dst)
        files["vae"] = link(fetch(r, "ae.safetensors", official = off), out / "flux3_ae.safetensors")
    elif group == "zimg":
        r = "Tongyi-MAI/Z-Image-Turbo"
        dst = out / "z_image_turbo_official_dit_bf16.safetensors"
        if not dst.exists():
            write(cast_plan(shards(r, "transformer/")), dst)
        files["dit"] = str(dst)
        dst = out / "qwen3_4b_zimage_official_merged.safetensors"
        if not dst.exists():
            write(plan_from(shards(r, "text_encoder/")), dst)
        files["te"] = str(dst)
        files["vae"] = link(fetch("unsloth/FLUX.1-schnell", "ae.safetensors", official = "black-forest-labs/FLUX.1-schnell"),
                            out / "ae_zimage.safetensors")
    elif group == "q21":
        r = "Qwen/Qwen-Image-2.1"
        dst = out / "qwen_image_2.1_official_bf16.safetensors"
        if not dst.exists():
            plan = plan_from(shards(r, "transformer/"))
            for k in [k for k in plan if k.endswith(".img_mlp.gate_layer.weight")]:
                b = k[: -len("gate_layer.weight")]
                g, p = plan.pop(k), plan.pop(b + "proj.weight")
                assert g.dtype == p.dtype and g.shape[1:] == p.shape[1:], k
                plan[b + "gate_up.weight"] = Src(g.dtype, [g.shape[0] + p.shape[0]] + g.shape[1:], g.parts + p.parts)
            write(plan, dst)
        files["dit"] = str(dst)
        dst = out / "qwen3vl_8b_q21_official_bf16.safetensors"
        if not dst.exists():
            write(plan_from(shards(r, "text_encoder/"), rename = strip_lm), dst)
        files["te"] = str(dst)
        dst = out / "qwen_image_2.1_vae_official_bf16.safetensors"
        if not dst.exists():
            import urllib.request

            conv = out / "convert_wan_to_diffusers.py"
            if not conv.exists():
                urllib.request.urlretrieve(DIFFUSERS_CONVERT, conv)
            tmp = str(dst) + ".f32.safetensors"
            rr = subprocess.run([comfy_py, str(HERE / "qwen21_vae_to_comfy.py"), comfy_dir, str(conv),
                                 fetch(r, "vae/diffusion_pytorch_model.safetensors"), tmp],
                                capture_output = True, text = True)
            say("VAE_CONVERT", rr.returncode, rr.stdout[-500:], rr.stderr[-1500:])
            if rr.returncode:
                raise SystemExit("q21 vae convert failed")
            write(cast_plan([tmp]), dst, meta = {"modelspec.architecture": "qwen_image_2.1_vae"})
            os.remove(tmp)
        files["vae"] = str(dst)
    else:
        raise SystemExit(f"unknown group {group}")
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--comfy-python", default = sys.executable)
    ap.add_argument("--comfy-dir", default = os.environ.get("COMFY_DIR", ""))
    ap.add_argument("--spec", help = "spec whose models Studio reads: snapshot them first and reuse their shards")
    ap.add_argument("--snap-root", type = Path)
    ap.add_argument("--snapshot-only", action = "store_true", help = "Studio snapshots only, no ComfyUI files")
    ap.add_argument("--env-file", help = "append DBENCH_MODEL_<ALIAS>=dir lines here (e.g. $GITHUB_ENV)")
    a = ap.parse_args()
    groups = [x for x in a.groups.split(",") if x]
    if a.spec:
        say("SNAPSHOT_ENV", snapshot(a.spec, groups, a.snap_root, a.env_file))
    if a.snapshot_only:
        return 0
    man_p = a.out / "manifest_nov5.json"
    man = json.loads(man_p.read_text(encoding = "utf-8")) if man_p.exists() else {}
    rc = 0
    for g in groups:
        if g == "wan":
            continue  # convert_comfy.py builds Wan; only its Studio snapshot is taken here
        t = time.time()
        try:
            man[g] = {"files": build(g, a.out, a.comfy_python, a.comfy_dir), "s": round(time.time() - t, 1)}
            say("PREP_OK", g, json.dumps(man[g]))
        except BaseException as e:  # noqa: BLE001 - one family's failure must not stop the others
            man[g] = {"error": f"{type(e).__name__}: {e}"}
            say("PREP_FAIL", g, man[g]["error"])
            rc = 1
        man_p.write_text(json.dumps(man, indent = 1), encoding = "utf-8")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
