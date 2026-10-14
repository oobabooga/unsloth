#!/usr/bin/env python3
"""Probe: does --diffusion-conv-direct pay on the sd.cpp builds Studio installs on THIS AMD card, and which flags
does each checkout's real sd.cpp launch path emit for them?

Observes only; criteria/sdcpp_conv_direct.py judges.

Shared half (done once into --shared, reused by every state; recorded in every observation):
  * Studio's installer (`studio/install_sd_cpp_prebuilt.py --accelerator X`, from the FIRST state's checkout, the
    installer is unchanged by the patch: sha256 recorded per state) into a managed home per accelerator
    (<shared>/home_<accel>/stable-diffusion.cpp), `sd-cli --list-devices`;
  * public models (token=False): Z-Image-Turbo Q4_K_M GGUF + Studio's Z-Image TE/VAE mirror; SDXL-Turbo fp16 as the
    conv-heavy UNet control;
  * the A/B matrix: per (accelerator, model), conv-direct on / off interleaved (on, off, off, on, ...), --reps each,
    sd.cpp's own sampling time + wall per run. Repeats of one arm are the A/A noise.

Per state:
  * the state's own SdCppEngine.generate (sd-cli) and SdCppServer.start (sd-server) argv for native speed
    "default", captured just before launch (the engine's _run and the server's command builder are wrapped, the
    real code that picks the flags runs);
  * that captured sd-cli argv RUN --state-reps times end to end on the card.
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

ACCELS = ("vulkan", "rocm")
MODELS = {
    "zimage": {
        "files": {
            "diffusion_model": ("unsloth/Z-Image-Turbo-GGUF", "z-image-turbo-Q4_K_M.gguf"),
            "llm": ("unsloth/Z-Image-Turbo-ComfyUI", "split_files/text_encoders/qwen_3_4b.safetensors"),
            "vae": ("unsloth/Z-Image-Turbo-ComfyUI", "split_files/vae/ae.safetensors"),
        },
        "w": 1024, "h": 1024, "steps": 8, "cfg": 1.0,
    },
    "sdxlt": {
        "files": {"model": ("stabilityai/sdxl-turbo", "sd_xl_turbo_1.0_fp16.safetensors")},
        "w": 768, "h": 768, "steps": 4, "cfg": 1.0,
    },
}
PROMPT = "a red fox sitting in fresh snow at sunrise, detailed fur, photograph"
SEED = 42
TOKEN_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_HUB_TOKEN")
RX = {
    "sampling_s": re.compile(r"sampling completed, taking ([0-9.]+)\s*s"),
    "decode_s": re.compile(r"(?:decode_first_stage|latent decoded)[^0-9]*completed, taking ([0-9.]+)\s*s"),
    "generate_s": re.compile(r"generate_image completed in ([0-9.]+)\s*s"),
    "load_s": re.compile(r"loading tensors completed, taking ([0-9.]+)\s*s"),
}


def say(msg: str) -> None:
    print(f"[sdcpp_conv_probe {time.strftime('%H:%M:%S')}] {msg}", flush = True)


def wjson(path: Path, obj) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent = 1, default = str), encoding = "utf-8")
    os.replace(tmp, path)


def rjson(path: Path):
    try:
        return json.loads(path.read_text(encoding = "utf-8"))
    except Exception:  # noqa: BLE001
        return None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_bin(root: Path, stem: str):
    for p in sorted(root.rglob(stem)):
        if p.is_file() and os.access(p, os.X_OK):
            return p
    return None


def lib_env(binary: Path) -> dict:
    env = dict(os.environ)
    for v in TOKEN_VARS:
        env.pop(v, None)
    d = str(binary.parent)
    env["LD_LIBRARY_PATH"] = d + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    return env


def install(checkout: Path, shared: Path, accel: str, python: str) -> dict:
    home = shared / f"home_{accel}"
    rec = {"accel": accel, "home": str(home)}
    env = dict(os.environ, UNSLOTH_STUDIO_HOME = str(home))
    t0 = time.time()
    p = subprocess.run([python, str(checkout / "studio" / "install_sd_cpp_prebuilt.py"), "--accelerator", accel],
                       env = env, capture_output = True, text = True, timeout = 1800)
    rec.update(rc = p.returncode, install_s = round(time.time() - t0, 1),
               log_tail = (p.stdout + p.stderr)[-3000:])
    root = home / "stable-diffusion.cpp"
    cli = find_bin(root, "sd-cli") if root.is_dir() else None
    rec["sd_cli"] = str(cli) if cli else None
    rec["install_record"] = rjson(root / ".unsloth-sd-cpp-install.json") if root.is_dir() else None
    for f in (root.glob("*.json") if root.is_dir() else []):
        rec.setdefault("root_json", {})[f.name] = rjson(f)
    if cli:
        q = subprocess.run([str(cli), "--list-devices"], env = lib_env(cli), capture_output = True, text = True,
                           timeout = 120)
        rec["list_devices"] = (q.stdout + q.stderr)[-2000:]
        rec["list_rc"] = q.returncode
        rec["gpu_listed"] = any(ln.split("\t")[0].strip().rstrip("0123456789").upper() in ("VULKAN", "ROCM")
                                for ln in q.stdout.splitlines() if "\t" in ln)
    return rec


def fetch_models(models_dir: Path) -> dict:
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    out = {}
    for name, spec in MODELS.items():
        paths = {}
        for key, (repo, fn) in spec["files"].items():
            for attempt in range(6):
                try:
                    paths[key] = hf_hub_download(repo, fn, local_dir = str(models_dir / repo.replace("/", "__")),
                                                 token = False)
                    break
                except Exception as exc:  # noqa: BLE001
                    say(f"download {repo}/{fn} attempt {attempt}: {exc}")
                    time.sleep(10 * (attempt + 1))
        out[name] = paths
    return out


def model_args(name: str, paths: dict) -> list:
    if name == "zimage":
        return ["--diffusion-model", paths["diffusion_model"], "--llm", paths["llm"], "--vae", paths["vae"]]
    return ["-m", paths["model"]]


def gen_args(name: str, out_png: Path) -> list:
    s = MODELS[name]
    return ["-p", PROMPT, "-W", str(s["w"]), "-H", str(s["h"]), "--steps", str(s["steps"]), "--cfg-scale",
            str(s["cfg"]), "-s", str(SEED), "-o", str(out_png), "-v"]


def run_cli(cmd: list, binary: Path, log: Path, timeout: float = 1800) -> dict:
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, env = lib_env(binary), capture_output = True, text = True, timeout = timeout,
                           errors = "replace")
        text, rc = p.stdout + p.stderr, p.returncode
    except subprocess.TimeoutExpired as exc:
        text, rc = str(exc), "timeout"
    wall = time.perf_counter() - t0
    log.parent.mkdir(parents = True, exist_ok = True)
    log.write_text(" ".join(map(str, cmd)) + "\n\n" + text, encoding = "utf-8")
    r = {"rc": rc, "wall_s": round(wall, 3), "log": str(log)}
    for k, rx in RX.items():
        m = rx.findall(text)
        if m:
            r[k] = float(m[-1])
    dev = [ln.strip() for ln in text.splitlines() if "model buffer size" in ln or "using " in ln.lower()][:12]
    r["device_lines"] = dev
    out_png = Path(cmd[cmd.index("-o") + 1])
    if out_png.is_file():
        r["png_sha"] = sha256(out_png)[:16]
        r["png_bytes"] = out_png.stat().st_size
    return r


def matrix(shared: Path, installs: dict, paths: dict, reps: int) -> dict:
    res: dict = {}
    for accel, rec in installs.items():
        cli = Path(rec["sd_cli"]) if rec.get("sd_cli") else None
        if not cli or not rec.get("gpu_listed"):
            res[accel] = {"skipped": f"no GPU-listing sd-cli (rc={rec.get('rc')}, listed={rec.get('gpu_listed')})"}
            continue
        res[accel] = {}
        for name in MODELS:
            if not paths.get(name) or len(paths[name]) != len(MODELS[name]["files"]):
                res[accel][name] = {"skipped": "model download incomplete"}
                continue
            arms = {"on": [], "off": []}
            # warmup (page cache, shader / kernel caches), not counted
            wp = shared / "runs" / accel / name / "warm.png"
            arms_warm = run_cli([str(cli), *model_args(name, paths[name]), *gen_args(name, wp), "--diffusion-fa"],
                                cli, wp.with_suffix(".log"))
            order = []
            for i in range(reps):
                order += (["on", "off"] if i % 2 == 0 else ["off", "on"])
            for j, arm in enumerate(order):
                png = shared / "runs" / accel / name / f"{j:02d}_{arm}.png"
                flags = ["--diffusion-fa"] + (["--diffusion-conv-direct"] if arm == "on" else [])
                r = run_cli([str(cli), *model_args(name, paths[name]), *gen_args(name, png), *flags], cli,
                            png.with_suffix(".log"))
                say(f"{accel} {name} {arm}: sampling {r.get('sampling_s')} s, wall {r['wall_s']} s rc {r['rc']}")
                arms[arm].append(r)
            res[accel][name] = {"warmup": arms_warm, **arms}
    return res


def capture_state_argv(checkout: Path, installs: dict, paths: dict, out_dir: Path) -> dict:
    """The state's real launch code for native speed "default": sd-cli via SdCppEngine.generate, sd-server via
    SdCppServer.start. Each accelerator's managed home is selected through UNSLOTH_STUDIO_HOME."""
    sys.path.insert(0, str(checkout / "studio" / "backend"))
    res: dict = {}
    for accel, rec in installs.items():
        if not rec.get("sd_cli"):
            res[accel] = {"skipped": "no sd-cli"}
            continue
        os.environ["UNSLOTH_STUDIO_HOME"] = rec["home"]
        for mod in [m for m in list(sys.modules) if m.startswith(("core.inference.sd_cpp", "core.inference"))]:
            sys.modules.pop(mod, None)
        entry: dict = {}
        try:
            import core.inference.sd_cpp_args as A  # noqa: PLC0415
            import core.inference.sd_cpp_engine as E  # noqa: PLC0415
            import core.inference.sd_cpp_server as S  # noqa: PLC0415

            entry["has_accelerator_param"] = "accelerator" in A.native_speed_flags.__code__.co_varnames
            entry["binary_accelerator"] = (E.binary_accelerator(rec["sd_cli"])
                                           if hasattr(E, "binary_accelerator") else "n/a (base)")
            files = A.SdCppModelFiles(diffusion_model = paths["zimage"]["diffusion_model"],
                                      llm = paths["zimage"].get("llm"), vae = paths["zimage"].get("vae"))
            params = A.SdCppGenParams(prompt = PROMPT, width = MODELS["zimage"]["w"], height = MODELS["zimage"]["h"],
                                      steps = MODELS["zimage"]["steps"], cfg_scale = MODELS["zimage"]["cfg"],
                                      seed = SEED)
            eng = E.SdCppEngine(rec["sd_cli"])
            captured = {}

            def fake_run(cmd, output_path, **_kw):
                captured["cmd"] = list(cmd)
                raise RuntimeError("captured")

            eng._run = fake_run  # type: ignore[assignment]
            try:
                eng.generate(files, params, output_path = str(out_dir / f"cap_{accel}.png"), native_speed = "default")
            except RuntimeError as exc:
                if str(exc) != "captured":
                    raise
            entry["cli_cmd"] = captured.get("cmd")
            # sd-server: wrap the builder the server calls, abort right after it returns.
            srv_bin = E.find_sd_server_binary()
            entry["sd_server"] = srv_bin
            if srv_bin:
                orig = S.build_sd_cpp_server_command
                cap2 = {}

                def wrapped(*a, **k):
                    cap2["cmd"] = orig(*a, **k)
                    raise RuntimeError("captured")

                S.build_sd_cpp_server_command = wrapped
                try:
                    S.SdCppServer(srv_bin).start(files, native_speed = "default")
                except RuntimeError as exc:
                    if "captured" not in str(exc):
                        entry["server_error"] = str(exc)[:500]
                except Exception as exc:  # noqa: BLE001
                    entry["server_error"] = f"{type(exc).__name__}: {exc}"[:500]
                finally:
                    S.build_sd_cpp_server_command = orig
                entry["server_cmd"] = cap2.get("cmd")
        except Exception as exc:  # noqa: BLE001
            entry["error"] = f"{type(exc).__name__}: {exc}"[:800]
        for k in ("cli_cmd", "server_cmd"):
            cmd = entry.get(k) or []
            entry[k.replace("_cmd", "_conv_direct")] = "--diffusion-conv-direct" in cmd if cmd else None
            entry[k.replace("_cmd", "_fa")] = "--diffusion-fa" in cmd if cmd else None
        res[accel] = entry
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--shared", required = True, type = Path)
    ap.add_argument("--reps", type = int, default = 3)
    ap.add_argument("--state-reps", type = int, default = 2)
    args = ap.parse_args()
    for v in TOKEN_VARS:
        os.environ.pop(v, None)
    shared = args.shared
    shared.mkdir(parents = True, exist_ok = True)
    obs: dict = {"state": args.state, "checkout": str(args.checkout), "probe": "sdcpp_conv_probe/1"}
    obs["installer_sha256"] = sha256(args.checkout / "studio" / "install_sd_cpp_prebuilt.py")
    done = shared / "shared.json"
    sh = rjson(done)
    if sh is None:
        sh = {"installs": {a: install(args.checkout, shared, a, sys.executable) for a in ACCELS}}
        wjson(done, sh)
        sh["models"] = fetch_models(shared / "models")
        wjson(done, sh)
        sh["matrix"] = matrix(shared, sh["installs"], sh["models"], args.reps)
        sh["installed_by"] = args.state
        wjson(done, sh)
    obs["shared"] = sh
    out_dir = args.out.parent / f"sdcpp_{args.state}"
    out_dir.mkdir(parents = True, exist_ok = True)
    cap = capture_state_argv(args.checkout, sh["installs"], sh["models"], out_dir)
    for accel, entry in cap.items():
        cmd = entry.get("cli_cmd")
        rec = sh["installs"].get(accel, {})
        if not cmd or not rec.get("gpu_listed"):
            continue
        cli = Path(rec["sd_cli"])
        runs = []
        for i in range(args.state_reps):
            png = out_dir / f"{accel}_{i}.png"
            c = list(cmd)
            if "-o" in c:
                c[c.index("-o") + 1] = str(png)
            if "-v" not in c:
                c.append("-v")
            runs.append(run_cli(c, cli, png.with_suffix(".log")))
            say(f"{args.state} {accel} studio argv: sampling {runs[-1].get('sampling_s')} s")
        entry["runs"] = runs
    obs["studio_argv"] = cap
    wjson(args.out, obs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
