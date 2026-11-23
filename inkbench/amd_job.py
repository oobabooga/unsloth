#!/usr/bin/env python3
"""AMD gfx1151 (Strix Halo) leg of the Inkling-2-Mini tg1024 benchmark, Linux ROCm or Windows HIP, one CI job.

  python amd_job.py --kit DIR --work DIR [--rounds 3]

1. TheRock ROCm dist tarball for gfx1151, the exact version the b12631-mix-acc0843 ROCm prebuilt used
   (Linux 10.3.0a20261121, Windows 10.3.0a20261107): toolchain for HEAD, ROCm runtime for HEAD and (Linux) MAIN.
2. HEAD = ggml-org b12631 + head_b12631.diff (tree hash checked), the prebuilt's ROCm cmake flags (RPC off).
3. PREV / CUR = unslothai rocm-gfx1151 assets (self-contained, bundle their ROCm), MAIN = ggml-org b12649 rocm-10.0.
4. The four GGUFs from Hugging Face, then tg_bench.py (3 rounds, F16 + Q8_0 KV) and the CUR vs HEAD PPL sanity.
Stdlib only (huggingface_hub when importable). Writes everything under WORK/out.
"""
import argparse, os, re, shutil, subprocess, sys, threading, time, urllib.request, zipfile
from pathlib import Path

WIN = os.name == "nt"
EXE = ".exe" if WIN else ""
GFX = "gfx1151"
PREV_TAG, CUR_TAG, MAIN_TAG = "b12588-mix-c0c3de8", "b12631-mix-acc0843", "b12649"
HEAD_TAG, HEAD_COMMIT, HEAD_TREE = "b12631", "e300919cc7", "d252588b5e8436d4596cbeae98cad435baa90ffe"
THEROCK = {"linux": "10.3.0a20261121", "windows": "10.3.0a20261107"}
REPO = "unsloth/Inkling-2-Mini-GGUF"
FILES = {"q4": "Inkling-2-Mini-UD-Q4_K_XL.gguf", "q2": "Inkling-2-Mini-UD-Q2_K_XL.gguf",
         "mq4": "Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file", "mq2": "Mainline_Support/Inkling-2-Mini-UD-Q2_K_XL.gguf_file"}
T0 = time.time()


def log(*a):
    print(f"[amd_job {time.time() - T0:6.0f}s]", *a, flush=True)


def run(cmd, env=None, cwd=None, logf=None, check=True):
    log("+", " ".join(map(str, cmd))[:400])
    t = time.time()
    p = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=cwd,
                         text=True, errors="replace")
    tail, last = [], time.time()
    f = open(logf, "a", encoding="utf-8") if logf else None
    for line in p.stdout:
        if f:
            f.write(line)
        tail = (tail + [line])[-80:]
        if time.time() - last > 60:
            print("   ", line.rstrip()[:300], flush=True)
            last = time.time()
    rc = p.wait()
    if f:
        f.close()
    log(f"  rc {rc} in {time.time() - t:.0f}s")
    if rc and check:
        print("".join(tail), flush=True)
        raise SystemExit(f"failed: {cmd[0]} rc {rc}")
    return rc


def fetch(url, dst):
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=600) as r, open(dst, "wb") as w:
                shutil.copyfileobj(r, w, 1 << 24)
            return
        except Exception as e:  # noqa: BLE001
            log(f"download {url} failed ({e!r}), retry {attempt}")
            time.sleep(15 * (attempt + 1))
    raise SystemExit(f"could not download {url}")


def therock(work):
    osn = "windows" if WIN else "linux"
    ver = THEROCK[osn]
    root = work / "rocm"
    if not (root / ".ok").exists():
        root.mkdir(parents=True, exist_ok=True)
        tgz = work / "therock.tar.gz"
        url = f"https://nightly.repo.amd.com/rocm/core/tarball/therock-dist-{osn}-{GFX}-{ver}.tar.gz"
        log("TheRock", url)
        fetch(url, tgz)
        run(["tar", "-xzf", tgz, "-C", root])
        tgz.unlink()
        (root / ".ok").write_text(ver, encoding="utf-8")
    # the archive may or may not have a top-level dir
    cl = sorted(root.rglob("clang++" + EXE), key=lambda p: len(p.parts))
    cl = [c for c in cl if c.parent.name == "bin" and c.parent.parent.name == "llvm"]
    if not cl:
        raise SystemExit("no llvm/bin/clang++ in the TheRock tarball")
    r = cl[0].parents[3] if cl[0].parents[2].name == "lib" else cl[0].parents[2]
    log("ROCm root", r, "version", ver)
    return r, ver


def msvc_env(env):
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
    vs = subprocess.run([str(vswhere), "-latest", "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                         "-property", "installationPath"], capture_output=True, text=True).stdout.strip()
    if not vs:
        raise SystemExit("no Visual Studio C++ build tools")
    bat = Path(vs) / "VC/Auxiliary/Build/vcvars64.bat"
    out = subprocess.run(f'cmd /c ""{bat}" >nul && set"', capture_output=True, text=True, shell=True).stdout
    for line in out.splitlines():
        k, _, v = line.partition("=")
        if k:
            env[k] = v
    log("MSVC env from", bat)
    return env


def git(*a, cwd):
    return subprocess.run(["git", "-c", "core.autocrlf=false", "-c", "core.eol=lf", *a], cwd=cwd, check=True,
                          capture_output=True, text=True).stdout.strip()


def build_head(kit, work, rroot, out):
    src, bdir = work / "src_HEAD", work / "build_HEAD"
    if src.exists():
        shutil.rmtree(src)
    src.mkdir(parents=True)
    git("init", "-q", cwd=src)
    git("fetch", "-q", "--depth", "1", "https://github.com/ggml-org/llama.cpp", f"refs/tags/{HEAD_TAG}:refs/tags/{HEAD_TAG}", cwd=src)
    git("checkout", "-q", "--detach", HEAD_TAG, cwd=src)
    c = git("rev-parse", "HEAD", cwd=src)
    if not c.startswith(HEAD_COMMIT):
        raise SystemExit(f"{HEAD_TAG} = {c}, expected {HEAD_COMMIT}")
    git("apply", "--binary", "--whitespace=nowarn", str(kit / "head_b12631.diff"), cwd=src)
    git("add", "-A", cwd=src)
    tree = git("write-tree", cwd=src)
    log("HEAD tree", tree)
    if tree != HEAD_TREE:
        raise SystemExit(f"HEAD tree {tree} != {HEAD_TREE}")
    env = dict(os.environ)
    llvm = rroot / "lib" / "llvm" / "bin"
    if WIN:
        env = msvc_env(env)
    env.update({"HIP_PATH": str(rroot), "ROCM_PATH": str(rroot), "HIP_PLATFORM": "amd", "HIP_CLANG_PATH": str(llvm),
                "HIP_DEVICE_LIB_PATH": str(rroot / "lib" / "llvm" / "amdgcn" / "bitcode"), "CMAKE_PREFIX_PATH": str(rroot)})
    sep = os.pathsep
    env["PATH"] = sep.join([str(rroot / "bin"), str(llvm), env.get("PATH", "")])
    if not WIN:
        env["LD_LIBRARY_PATH"] = sep.join([str(rroot / "lib"), env.get("LD_LIBRARY_PATH", "")])
    # the prebuilt's ROCm flags (unsloth-prebuilt-rocm.yml, b12631 run 45065498687); RPC off, see build_arm.py
    flags = [f"-DCMAKE_C_COMPILER={llvm / ('clang' + EXE)}", f"-DCMAKE_CXX_COMPILER={llvm / ('clang++' + EXE)}",
             f"-DCMAKE_CXX_FLAGS=-I{rroot / 'include'}", "-DCMAKE_BUILD_TYPE=Release", f"-DGPU_TARGETS={GFX}",
             "-DBUILD_SHARED_LIBS=ON", "-DLLAMA_BUILD_TESTS=OFF", "-DGGML_HIP=ON", "-DGGML_OPENMP=OFF",
             "-DGGML_CUDA_FORCE_CUBLAS=OFF", "-DGGML_RPC=OFF", "-DGGML_HIP_ROCWMMA_FATTN=OFF", "-DGGML_NATIVE=OFF",
             "-DGGML_STATIC=OFF", "-DGGML_BACKEND_DL=ON", "-DGGML_CPU_ALL_VARIANTS=ON", "-DLLAMA_BUILD_SERVER=OFF",
             "-DLLAMA_OPENSSL=OFF"]
    if WIN:
        flags.insert(2, "-DCMAKE_C_FLAGS=-Wno-error=incompatible-pointer-types")
    note = "prebuilt ROCm flags, GGML_RPC=OFF, no server"
    tg = ["llama-bench", "llama-perplexity", "llama-completion"]
    for attempt, extra in enumerate([[], ["-DGGML_CPU_ALL_VARIANTS=OFF"]]):
        if bdir.exists():
            shutil.rmtree(bdir)
        rc = run(["cmake", "-S", src, "-B", bdir, "-G", "Ninja", *flags, *extra], env=env, logf=out / "head_cmake.log", check=False)
        if rc == 0:
            rc = run(["cmake", "--build", bdir, "--config", "Release", "-j", str(os.cpu_count() or 8), "--target", *tg],
                     env=env, logf=out / "head_build.log", check=False)
        if rc == 0:
            if extra:
                note += "; GGML_CPU_ALL_VARIANTS=OFF (the all-variants CPU build failed with ROCm clang here)"
            break
        log(f"HEAD build attempt {attempt} failed")
    else:
        raise SystemExit("HEAD build failed")
    b = bdir / "bin"
    (b / "BUILD_INFO.txt").write_text(f"ggml-org {HEAD_TAG} + head_b12631.diff, tree {tree}; TheRock {rroot}; {note}\n", encoding="utf-8")
    return b, note


def get_models(mdir, res):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        hf_hub_download = None
    paths = {}
    for k, fn in FILES.items():
        p = None
        for attempt in range(12):
            try:
                if hf_hub_download:
                    p = Path(hf_hub_download(REPO, fn, local_dir=str(mdir)))
                else:
                    p = mdir / fn
                    p.parent.mkdir(parents=True, exist_ok=True)
                    if not p.exists():
                        fetch(f"https://huggingface.co/{REPO}/resolve/main/{fn}", p)
                break
            except Exception as e:  # noqa: BLE001
                log(f"download {fn} attempt {attempt}: {e!r}")
                time.sleep(20)
        if p is None:
            res["errors"].append(fn)
            continue
        if p.suffix != ".gguf":  # Mainline_Support ships *.gguf_file
            q = p.with_suffix(".gguf")
            if not q.exists():
                shutil.copyfile(p, q) if WIN else q.symlink_to(p.name)
            p = q
        paths[k] = str(p)
        log(f"model {k}: {p} {p.stat().st_size / 1e9:.2f} GB")
    res["models"] = paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    kit, work = Path(a.kit).resolve(), Path(a.work).resolve()
    out = Path(a.out).resolve() if a.out else work / "out"
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=lambda: [time.sleep(300) or log("heartbeat") for _ in iter(int, 1)], daemon=True).start()
    res = {"errors": []}
    th = threading.Thread(target=get_models, args=(work / "models", res), daemon=True)
    th.start()
    z = work / "wiki.zip"
    fetch("https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip", z)
    zipfile.ZipFile(z).extractall(work / "wikitext")
    wiki = next((work / "wikitext").rglob("wiki.test.raw"))
    rroot, rver = therock(work)
    head_bin, note = build_head(kit, work, rroot, out)
    th.join()
    models = res["models"]
    log("models", models, "errors", res["errors"])
    rlib = rroot / ("bin" if WIN else "lib")
    plat = "windows-x64" if WIN else "linux-x64"
    ext = "zip" if WIN else "tar.gz"
    main_asset = f"llama-{MAIN_TAG}-bin-win-rocm-10.0-x64.zip" if WIN else f"llama-{MAIN_TAG}-bin-ubuntu-rocm-10.0-x64.tar.gz"
    cmd = [sys.executable, "-u", str(kit / "tg_bench.py"), "--out", str(out / "tg"), "--cache", str(work / "tgcache"),
           "--rounds", str(a.rounds), "--kv", "f16,q8_0", "--baseline", "CUR",
           "--ratios", "CUR/PREV,HEAD/CUR,HEAD/MAIN,CUR/MAIN",
           "--arm", f"PREV=gh:unslothai/llama.cpp@{PREV_TAG}/app-{PREV_TAG}-{plat}-rocm-{GFX}.{ext}",
           "--arm", f"CUR=gh:unslothai/llama.cpp@{CUR_TAG}/app-{CUR_TAG}-{plat}-rocm-{GFX}.{ext}",
           "--arm", f"HEAD={head_bin}",
           "--arm", f"MAIN=gh:ggml-org/llama.cpp@{MAIN_TAG}/{main_asset}",
           "--arm-lib", f"HEAD={rlib}", "--arm-models", "MAIN=mq4,mq2",
           "--ppl-arms", "CUR,HEAD", "--ppl-model", "q4", "--ppl-text", str(wiki),
           "--note", f"AMD {GFX} {'Windows HIP' if WIN else 'Linux ROCm'}; PREV/CUR unslothai rocm-{GFX} assets (bundled ROCm); "
                     f"MAIN ggml-org {main_asset}; HEAD {HEAD_TAG} + master 6fa79e0460 pins, TheRock {rver}, {note}"]
    if not WIN:
        cmd += ["--arm-lib", f"MAIN={rlib}"]  # the ggml-org Linux ROCm build expects a system ROCm; TheRock's stands in
        sysdeps = rroot / "lib" / "rocm_sysdeps" / "lib"
        if sysdeps.exists():
            cmd += ["--arm-lib", f"HEAD={sysdeps}", "--arm-lib", f"MAIN={sysdeps}"]
    for k in ("q4", "mq4", "q2", "mq2"):
        if k in models:
            cmd += ["--model", f"{k}={models[k]}"]
    rc = subprocess.call(cmd)
    log("tg_bench rc", rc)
    sys.exit(rc)


if __name__ == "__main__":
    main()
