#!/usr/bin/env python3
"""Round-9 M3: Vulkan spatial-halo CONV_2D / CONV_3D shader (llama.cpp #32629, head 7ef8a3fd) on gfx1151.
Observes only; criteria/r9_judge_m3.py judges. Same code on Linux (RADV) and Windows (AMD driver).

Why not sd.cpp arms: the PR does not apply to the Studio fork's ggml (leejet/ggml d803c238f6): 4 files reject, and the
fork lacks the PR's upstream prerequisites (conv3d split_k #29929 and the 256x128 conv3d tile chain), plus the fork
carries its own ggml ops so upstream ggml cannot be swapped in. So this item is staged:

  1. ops   upstream llama.cpp test-backend-ops, A = merge base a866b4ad + the PR's own tests commit 3fb7b160 (identical
           cases in both arms), B = PR head 7ef8a3fd (= A + r9/m3_halo.patch). Both built with GGML_VULKAN against the
           LunarG SDK 1.4.363.1 (glslc with KHR coopmat, which the halo path needs). `test -o CONV_2D|CONV_3D -b
           Vulkan0` pass counts on A and B; `perf` in order A B Boff B A Boff A B, where Boff = B with the PR's own kill
           switch GGML_VK_DISABLE_CONV_HALO=1 (engagement proof: B != Boff, Boff ~ A). The fork's conv2d_mm.comp is
           byte-identical to the merge base's, so A stands in for the fork at the 2D op level.
  2. fork  Studio's pinned Vulkan bundle (master-982-0518b34-u9455ee1, sha256 from install_sd_cpp_prebuilt.py),
           Z-Image-Turbo Q4_K_M 1024^2 8 steps with Studio's Vulkan flags (--diffusion-fa --conv-direct off,vae=auto):
           the `conv2d direct auto: Vulkan0 -> on|off` decision (on = direct CONV_2D = the path the halo shader
           replaces; off = im2col = M3 cannot engage), decode seconds, and one GGML_VK_PERF_LOGGER run for the CONV_2D
           share of the VAE graphs. Plus one `--conv-direct off` (im2col) render for context.
  The judge turns 1 x 2 into a decode-gain estimate (CONV_2D share x time saved over the 1024^2 VAE perf cases,
  weighted by layer count): under 15% = KILL; at or over 15% = port warranted (not a measured decode gain).

No token is read or passed (SECRET_HINTS from r9_common strip every child env); only public unsloth/* HF files and public GitHub / LunarG downloads.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from r9_common import R9 as PATCH_DIR  # noqa: E402  (probes/r9: m3_tests.patch, m3_halo.patch)
from r9_common import SECRET_HINTS  # noqa: E402
WIN = sys.platform == "win32"
LLAMA = "https://github.com/ggml-org/llama.cpp"
MB_SHA = "a866b4ade63921826c530307a368fbca3a334a0b"
PR_SHA = "7ef8a3fded16cc6806e36dd561f2f2c6a70a8b5b"
VK_SDK = "1.4.363.1"
VK_URL = (f"https://sdk.lunarg.com/sdk/download/{VK_SDK}/windows/vulkan_sdk.exe" if WIN
          else f"https://sdk.lunarg.com/sdk/download/{VK_SDK}/linux/vulkan_sdk.tar.xz")
W64DEVKIT = "https://github.com/skeeto/w64devkit/releases/download/v2.11.0/w64devkit-x64-2.11.0.7z.exe"
SD_TAG = "master-982-0518b34-u9455ee1"
SD_ASSET = f"sd-{SD_TAG}-bin-win-vulkan-x64.zip" if WIN else f"sd-{SD_TAG}-bin-Linux-Ubuntu-22.04-x86_64-vulkan.zip"
SD_SHA = {"sd-master-982-0518b34-u9455ee1-bin-Linux-Ubuntu-22.04-x86_64-vulkan.zip":
          "b37a96f1b8b1503fd26ade909f215f6321d23eb8508f8d391b8dc3b47f40e280",
          "sd-master-982-0518b34-u9455ee1-bin-win-vulkan-x64.zip":
          "b5656e3dc1001c822f68f23e81a7901995526db45eac6e3cd536f0827b86139e"}
HF = "https://huggingface.co"
MODELS = {"dit": ("unsloth/Z-Image-Turbo-GGUF", "z-image-turbo-Q4_K_M.gguf"),
          "llm": ("unsloth/Qwen3-4B-GGUF", "Qwen3-4B-Q8_0.gguf"),
          "ae": ("unsloth/FLUX.1-schnell", "ae.safetensors")}
PROMPT = "a photo of a red fox sitting in fresh snow, golden hour, detailed fur"
PERF_ORDER = ["A", "B", "Boff", "B", "A", "Boff", "A", "B"]
RX_PERF = re.compile(r"^\s*((?:CONV_2D|CONV_3D)\(.*?\)):\s+(\d+) runs -\s+([0-9.]+) us/run", re.M)
RX_PASSED = re.compile(r"(\d+)/(\d+) tests passed")
RX_DECISION = re.compile(r"conv[23]d direct auto: [^\n]*")
RX_DECODE = re.compile(r"decode_first_stage completed, taking ([0-9.]+)(m?s)")
RX_TIMING = re.compile(r"^(\S[^\n]*?): (\d+) x ([0-9.]+) us = ([0-9.]+) us", re.M)
RX_DEVICE = re.compile(r"ggml_vulkan: 0 = [^\n]*")


def _env(extra: dict | None = None) -> dict:
    env = {k: v for k, v in os.environ.items() if not any(h in k.upper() for h in SECRET_HINTS)}
    env.update(extra or {})
    return env


def _sh(cmd, log: Path, cwd=None, env=None, timeout=3600) -> dict:
    t = time.time()
    with open(log, "ab") as fh:
        fh.write(f"\n=== [m3] {' '.join(map(str, cmd))} (cwd={cwd})\n".encode("utf-8", "replace"))
        fh.flush()
        try:
            rc = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env or _env(), stdout=fh,
                                stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
        except OSError as exc:
            fh.write(f"OSError {exc!r}\n".encode())
            rc = f"oserror {exc!r}"[:200]
    return {"rc": rc, "s": round(time.time() - t, 1)}


def _cap(cmd, env=None, timeout=1800, cwd=None) -> tuple:
    t = time.time()
    try:
        p = subprocess.run([str(c) for c in cmd], env=env or _env(), cwd=cwd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or ""), round(time.time() - t, 1)
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        return "timeout", out, round(time.time() - t, 1)
    except OSError as exc:
        return f"oserror {exc!r}"[:200], "", 0.0


def _tail(log: Path, n=1500) -> str:
    try:
        return log.read_text(encoding="utf-8", errors="replace")[-n:]
    except OSError:
        return ""


def _fetch(url: str, dest: Path, log: Path) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return {"cached": True, "bytes": dest.stat().st_size}
    curl = "curl.exe" if WIN else "curl"
    rcs = []
    for _ in range(4):
        r = _sh([curl, "-sS", "--fail", "-L", "--retry", "5", "--retry-delay", "5", "--retry-all-errors",
                 "-o", str(dest) + ".part", url], log, timeout=3600)
        rcs.append(r["rc"])
        if r["rc"] == 0:
            os.replace(str(dest) + ".part", dest)
            return {"bytes": dest.stat().st_size, "s": r["s"]}
        time.sleep(15)
    return {"error": f"curl rc={rcs}", "url": url}


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ toolchain
def _msvc_env(log: Path) -> dict | None:
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / \
        "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.is_file():
        return None
    rc, out, _ = _cap([vswhere, "-latest", "-products", "*", "-requires",
                       "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"])
    inst = out.strip().splitlines()[0].strip() if rc == 0 and out.strip() else ""
    bat = Path(inst) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat" if inst else None
    if not bat or not bat.is_file():
        return None
    rc, out, _ = _cap(["cmd", "/c", f'call "{bat}" >nul 2>&1 && set'])
    if rc != 0:
        return None
    env = {}
    for line in out.splitlines():
        if "=" in line and not line.startswith("="):
            k, v = line.split("=", 1)
            env[k] = v
    return env if shutil.which("cl", path=env.get("Path") or env.get("PATH") or env.get("path")) else None


def toolchain(ctx: dict, W: Path, res: dict, log: Path) -> dict | None:
    """Returns the build env (PATH with cmake / ninja / glslc / compiler, VULKAN_SDK) or None (reason in res)."""
    tc: dict = {}
    res["toolchain"] = tc
    r = _sh([ctx.get("py") or sys.executable, "-m", "pip", "install", "-q", "cmake>=3.28", "ninja"], log, timeout=900)
    tc["pip_cmake_ninja"] = r
    rc, out, _ = _cap([ctx.get("py") or sys.executable, "-c", "import sysconfig; print(sysconfig.get_path('scripts'))"])
    scripts = out.strip().splitlines()[-1] if rc == 0 and out.strip() else sysconfig.get_path("scripts")
    env = _env()
    sep = os.pathsep
    # Vulkan SDK
    sdk_root = W / "vulkan_sdk"
    if WIN:
        exe = W / "dl" / "vulkan_sdk.exe"
        tc["sdk_fetch"] = _fetch(VK_URL, exe, log)
        if "error" in tc["sdk_fetch"]:
            res["stop"] = "Vulkan SDK download failed"
            return None
        if not (sdk_root / "Bin" / "glslc.exe").is_file():
            tc["sdk_install"] = _sh([exe, "--root", sdk_root, "--accept-licenses", "--default-answer",
                                     "--confirm-command", "install", "copy_only=1"], log, timeout=1800)
        sdk = sdk_root
        glslc = sdk / "Bin" / "glslc.exe"
        bin_dir = sdk / "Bin"
    else:
        tarp = W / "dl" / "vulkan_sdk.tar.xz"
        tc["sdk_fetch"] = _fetch(VK_URL, tarp, log)
        if "error" in tc["sdk_fetch"]:
            res["stop"] = "Vulkan SDK download failed"
            return None
        sdk = sdk_root / VK_SDK / "x86_64"
        if not (sdk / "bin" / "glslc").is_file():
            sdk_root.mkdir(parents=True, exist_ok=True)
            tc["sdk_extract"] = _sh(["tar", "-xJf", tarp, "-C", sdk_root], log, timeout=1800)
        cands = sorted(sdk_root.glob("*/x86_64/bin/glslc"))
        if cands:
            sdk = cands[0].parent.parent
        glslc = sdk / "bin" / "glslc"
        bin_dir = sdk / "bin"
        libs = sorted(sdk.rglob("libvulkan.so"))
        if libs:
            env["R9_VK_LIB"] = str(libs[0])
            # system loader first (RADV ICD lives with it); the SDK's loader only as a fallback
            env["LD_LIBRARY_PATH"] = sep.join(x for x in (env.get("LD_LIBRARY_PATH", ""), str(libs[0].parent)) if x)
    tc["sdk"] = str(sdk)
    tc["glslc"] = glslc.is_file()
    if not glslc.is_file():
        res["stop"] = f"no glslc after the SDK step ({glslc})"
        return None
    env["VULKAN_SDK"] = str(sdk)
    # compiler
    if WIN:
        msvc = _msvc_env(log)
        if msvc:
            env.update({k.upper(): v for k, v in msvc.items() if not any(h in k.upper() for h in SECRET_HINTS)})
            tc["compiler"] = "msvc"
        else:
            dk = W / "w64devkit"
            if not (dk / "bin" / "g++.exe").is_file():
                sfx = W / "dl" / "w64devkit.7z.exe"
                tc["w64devkit_fetch"] = _fetch(W64DEVKIT, sfx, log)
                tc["w64devkit_extract"] = _sh([sfx, f"-o{W}", "-y"], log, timeout=900)
            if (dk / "bin" / "g++.exe").is_file():
                env["PATH"] = str(dk / "bin") + sep + env.get("PATH", "")
                env["CC"], env["CXX"] = "gcc", "g++"
                tc["compiler"] = "w64devkit"
            else:
                res["stop"] = "no C++ compiler (no MSVC via vswhere, w64devkit extract failed)"
                return None
    else:
        cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
        if not cxx:
            res["stop"] = "no C++ compiler on PATH"
            return None
        tc["compiler"] = cxx
    if WIN:  # os.environ keys are upper-case on Windows; keep exactly one PATH key
        env = {k.upper(): v for k, v in env.items()}
    env["PATH"] = sep.join([str(bin_dir), scripts, env.get("PATH", "")])
    rc, out, _ = _cap(["cmake", "--version"], env=env)
    tc["cmake"] = out.splitlines()[0] if rc == 0 and out else f"rc={rc}"
    rc, out, _ = _cap([glslc, "--version"], env=env)
    tc["glslc_version"] = out.splitlines()[0][:120] if out else f"rc={rc}"
    return env


# ------------------------------------------------------------------ sources + builds
def source(W: Path, name: str, patches: list, env: dict, log: Path) -> dict:
    d = W / f"src_{name}"
    rec: dict = {"dir": str(d)}
    if not (d / ".git").is_dir():
        d.mkdir(parents=True, exist_ok=True)
        for cmd in (["git", "init", "-q"], ["git", "config", "core.autocrlf", "false"],
                    ["git", "fetch", "-q", "--depth", "1", LLAMA, MB_SHA], ["git", "checkout", "-q", "FETCH_HEAD"]):
            r = _sh(cmd, log, cwd=d, env=env, timeout=1200)
            if r["rc"] != 0:
                rec["error"] = f"{' '.join(cmd)} rc={r['rc']}"
                return rec
    for p in patches:
        if _sh(["git", "apply", "--reverse", "--check", PATCH_DIR / p], log, cwd=d, env=env, timeout=300)["rc"] == 0:
            rec[f"apply_{p}"] = "already"
            continue
        r = _sh(["git", "apply", "--whitespace=nowarn", PATCH_DIR / p], log, cwd=d, env=env, timeout=300)
        rec[f"apply_{p}"] = r["rc"]
        if r["rc"] != 0:
            rec["error"] = f"git apply {p} rc={r['rc']}"
            return rec
    rc, out, _ = _cap(["git", "status", "--short"], env=env, cwd=d)
    rec["changed_files"] = len(out.splitlines())
    return rec


def build(W: Path, name: str, src: str, env: dict, log: Path) -> dict:
    b = W / f"build_{name}"
    flags = ["-DCMAKE_BUILD_TYPE=Release", "-DGGML_VULKAN=ON", "-DLLAMA_BUILD_TESTS=ON", "-DLLAMA_BUILD_COMMON=ON",
             "-DLLAMA_BUILD_TOOLS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_SERVER=OFF",
             "-DLLAMA_BUILD_APP=OFF", "-DLLAMA_OPENSSL=OFF", "-DLLAMA_CURL=OFF", "-DGGML_CCACHE=OFF",
             "-DGGML_NATIVE=OFF", "-DBUILD_SHARED_LIBS=OFF"]
    if env.get("R9_VK_LIB"):
        flags.append(f"-DVulkan_LIBRARY={env['R9_VK_LIB']}")
    rec = {"configure": _sh(["cmake", "-S", src, "-B", b, "-G", "Ninja", *flags], log, env=env, timeout=900)}
    if rec["configure"]["rc"] != 0:
        rec["error"] = "configure failed"
        rec["tail"] = _tail(log)
        return rec
    rec["build"] = _sh(["cmake", "--build", b, "--config", "Release", "-j", str(os.cpu_count() or 8), "--target",
                        "test-backend-ops"], log, env=env, timeout=2400)
    exe = sorted(b.rglob("test-backend-ops.exe" if WIN else "test-backend-ops"))
    exe = [p for p in exe if p.is_file() and (WIN or os.access(p, os.X_OK))]
    rec["exe"] = str(exe[0]) if exe else None
    if rec["build"]["rc"] != 0 or not exe:
        rec["error"] = "build failed"
        rec["tail"] = _tail(log)
    m = re.findall(r"GL_KHR_cooperative_matrix[^\n]*", _tail(log, 400000))
    rec["glslc_coopmat_lines"] = sorted(set(x[:120] for x in m))[:4]
    rec["coopmat_glslc"] = any("supported by glslc" in x and "not" not in x for x in m)
    return rec


def tbo(exe: str, mode: str, op: str, env: dict, extra_env: dict | None = None, timeout=1800) -> dict:
    e = dict(env)
    e.update(extra_env or {})
    rc, out, s = _cap([exe, mode, "-o", op, "-b", "Vulkan0"], env=e, timeout=timeout)
    rec: dict = {"rc": rc, "s": s, "op": op, "mode": mode}
    dev = RX_DEVICE.findall(out)
    rec["device"] = dev[0][:300] if dev else None
    if mode == "test":
        m = RX_PASSED.findall(out)
        rec["passed"], rec["total"] = (int(m[-1][0]), int(m[-1][1])) if m else (None, None)
        rec["fail_lines"] = [ln.strip()[:240] for ln in out.splitlines() if "FAIL" in ln][:30]
        rec["backend_ok"] = bool(re.search(r"Backend Vulkan0:[^\n]*OK", out))
    else:
        rec["cases"] = {k: float(v) for k, _, v in RX_PERF.findall(out)}
        rec["n_cases"] = len(rec["cases"])
    if rc != 0 or (mode == "test" and rec.get("passed") is None) or (mode == "perf" and not rec.get("n_cases")):
        rec["tail"] = out[-1500:]
    return rec


# ------------------------------------------------------------------ fork engagement (pinned Studio bundle)
def _perf_blocks(text: str) -> list:
    blocks = []
    for chunk in text.split("Vulkan Timings:")[1:]:
        body = chunk.split("Total time:")[0]
        ops: dict = {}
        for name, n, _avg, tot in RX_TIMING.findall(body):
            # keys look like "CONV_2D M=Cout=128, K=Cin*KW*KH=1152, ..." (or "<FUSION> OP ..."): bucket by op token
            toks = name.replace(",", " ").split()
            key = "CONV_2D" if "CONV_2D" in toks[:2] else (toks[0] if toks else name)
            ops[key] = ops.get(key, 0.0) + float(tot)
        blocks.append(ops)
    return blocks


def fork_engage(W: Path, res: dict, log: Path, save) -> None:
    f: dict = {"tag": SD_TAG, "asset": SD_ASSET}
    res["fork"] = f
    z = W / "dl" / SD_ASSET
    f["fetch"] = _fetch(f"https://github.com/unslothai/stable-diffusion.cpp/releases/download/{SD_TAG}/{SD_ASSET}",
                        z, log)
    if "error" in f["fetch"]:
        f["stop"] = "bundle download failed"
        return
    got = _sha256(z)
    f["sha256_match"] = got == SD_SHA.get(SD_ASSET)
    if not f["sha256_match"]:
        f["stop"] = f"bundle sha256 mismatch {got}"
        return
    bdir = W / "sd_bundle"
    if not bdir.is_dir():
        with zipfile.ZipFile(z) as zz:
            zz.extractall(bdir)
    clis = sorted(bdir.rglob("sd-cli.exe" if WIN else "sd-cli"))
    if not clis:
        f["stop"] = "no sd-cli in the bundle"
        return
    cli = clis[0]
    if not WIN:
        cli.chmod(0o755)
    env = _env({"LD_LIBRARY_PATH": str(cli.parent) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")}
               if not WIN else None)
    rc, out, _ = _cap([cli, "--list-devices"], env=env, timeout=300)
    f["list_devices"] = out[-600:]
    models = {}
    for k, (repo, name) in MODELS.items():
        dest = W / "models" / repo.replace("/", "__") / name
        r = _fetch(f"{HF}/{repo}/resolve/main/{name}", dest, log)
        if "error" in r:
            f["stop"] = f"model download failed: {repo}/{name}"
            return
        models[k] = str(dest)
    save()
    base = ["--diffusion-model", models["dit"], "--vae", models["ae"], "--llm", models["llm"], "--steps", "8",
            "--cfg-scale", "1.0", "-W", "1024", "-H", "1024", "--prompt", PROMPT, "--seed", "42", "-v"]
    runs = {"studio_auto": (["--diffusion-fa", "--conv-direct", "off,vae=auto"], {}),
            "studio_auto_perflog": (["--diffusion-fa", "--conv-direct", "off,vae=auto"], {"GGML_VK_PERF_LOGGER": "1"}),
            "im2col": (["--diffusion-fa", "--conv-direct", "off"], {}),
            "studio_auto_r2": (["--diffusion-fa", "--conv-direct", "off,vae=auto"], {})}
    f["runs"] = {}
    for name, (flags, xenv) in runs.items():
        png = W / "renders" / f"{name}.png"
        png.parent.mkdir(parents=True, exist_ok=True)
        e = dict(env)
        e.update(xenv)
        rc, out, s = _cap([cli, *base, "--output", png, *flags], env=e, timeout=900)
        (W / "renders" / f"{name}.log").write_text(out, encoding="utf-8", errors="replace")
        d = [float(v) / (1000.0 if u == "ms" else 1.0) for v, u in RX_DECODE.findall(out)]
        rec = {"rc": rc, "wall_s": s, "flags": flags, "env": xenv, "decode_s": d[-1] if d else None,
               "decisions": sorted(set(x.strip()[:200] for x in RX_DECISION.findall(out))),
               "sha256": _sha256(png) if png.is_file() else None}
        if xenv.get("GGML_VK_PERF_LOGGER"):
            blocks = _perf_blocks(out)
            conv_blocks = [b for b in blocks if "CONV_2D" in b]
            tot = sum(sum(b.values()) for b in conv_blocks)
            conv = sum(b.get("CONV_2D", 0.0) for b in conv_blocks)
            rec["perflog"] = {"blocks": len(blocks), "vae_blocks": len(conv_blocks), "vae_total_us": round(tot, 1),
                              "conv2d_us": round(conv, 1), "conv2d_share": round(conv / tot, 4) if tot else None,
                              "top_ops": sorted(((k, round(v)) for b in conv_blocks for k, v in b.items()),
                                                key=lambda x: -x[1])[:12]}
        if rc != 0:
            rec["tail"] = out[-1200:]
        f["runs"][name] = rec
        save()
    dec = " ".join(" ".join(r.get("decisions") or []) for k, r in f["runs"].items() if k.startswith("studio_auto"))
    f["vae_conv2d_direct"] = ("on" if re.search(r"conv2d direct auto: \S+ -> on", dec)
                              else "off" if re.search(r"conv2d direct auto: \S+ -> off", dec) else None)


# ------------------------------------------------------------------ the item
def item_m3(ctx: dict, obs: dict) -> None:
    res: dict = {"mb": MB_SHA, "pr": PR_SHA, "sdk": VK_SDK, "platform": sys.platform}
    obs["m3"] = res
    save = ctx["save"]
    W = Path(ctx["scratch"]) / "m3"
    W.mkdir(parents=True, exist_ok=True)
    log = Path(ctx["work"]) / "m3_build.log"
    t0 = time.time()
    env = toolchain(ctx, W, res, log)
    save()
    if env is None:
        return
    res["src"] = {"A": source(W, "A", ["m3_tests.patch"], env, log)}
    res["src"]["B"] = source(W, "B", ["m3_tests.patch", "m3_halo.patch"], env, log)
    save()
    if any("error" in v for v in res["src"].values()):
        res["stop"] = "source prep failed"
        return
    res["build"] = {}
    for name in ("B", "A"):
        res["build"][name] = build(W, name, res["src"][name]["dir"], env, log)
        save()
    exe = {k: v.get("exe") for k, v in res["build"].items()}
    if not all(exe.values()):
        res["stop"] = "build failed"
        return
    # correctness: test-backend-ops vs the CPU backend
    res["test"] = {}
    for op in ("CONV_2D", "CONV_3D"):
        for name in ("B", "A"):
            res["test"][f"{op}_{name}"] = tbo(exe[name], "test", op, env)
            save()
    # perf, A B Boff B A Boff A B
    res["perf"] = []
    for i, arm in enumerate(PERF_ORDER):
        x = "A" if arm == "A" else "B"
        xenv = {"GGML_VK_DISABLE_CONV_HALO": "1"} if arm == "Boff" else None
        for op in ("CONV_2D", "CONV_3D"):
            r = tbo(exe[x], "perf", op, env, xenv, timeout=2400)
            r.update(arm=arm, i=i)
            res["perf"].append(r)
            save()
    res["ops_s"] = round(time.time() - t0, 1)
    # fork engagement + CONV_2D share of the VAE decode
    try:
        fork_engage(W, res, log, save)
    except Exception as exc:  # noqa: BLE001
        res.setdefault("fork", {})["error"] = repr(exc)[:500]
    res["total_s"] = round(time.time() - t0, 1)
    save()


ITEMS = {"m3": item_m3}
