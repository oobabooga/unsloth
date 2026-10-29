#!/usr/bin/env python3
"""Probe: what a checkout's sd.cpp detection / installer / launch env do on a real Windows AMD host
where an AMD HIP SDK 7.x is installed but its bin dir is NOT on PATH (HIP_PATH set, as the SDK MSI
does), and the controls around it.

SDK source, recorded as `sdk.kind`:
  installed    a real HIP SDK 7.x found under HIP_PATH / %ProgramFiles%\\AMD\\ROCm\\<ver>;
  constructed  none on the host: an SDK-shaped tree <tmp>\\PF\\AMD\\ROCm\\7.2\\{bin,.kpack} built from
               AMD's public TheRock ROCm 10.1.0 Windows wheels (the toolchain the upstream win-rocm
               sd.cpp bundle is built with: hipblas.dll, rocblas.dll, rocblas\\library, amdhip64_7.dll).
Every PATH entry holding hipblas.dll / amdhip64_*.dll is dropped from PATH in every case (recorded).

Each case runs in a fresh child process (module globals and failure records do not leak) with a
fresh Studio home, against that checkout's own modules:
  hip_path        HIP_PATH=<sdk root>\\ (trailing backslash, as the MSI writes it)
  hip_path_force  same + UNSLOTH_SD_CPP_WINDOWS_ROCM=1
  program_files   HIP_PATH unset, %ProgramFiles% = the dir holding AMD\\ROCm\\<ver>
  no_sdk          HIP_PATH / ROCM_PATH unset, %ProgramFiles% = an empty dir
  sdk6_only       HIP_PATH = a 6.x-shaped dir (hipblas.dll + amdhip64_6.dll, no amdhip64_7.dll)
Per case: preferred_accelerator("rocm"), the DLL check, ensure_sd_server_binary(rocm), the install
record, what the installed tree starts as via the state's own runtime_env (--version, --list-devices)
and via a PATH-only env (control), and the stored ROCm failure record.

Observes only; criteria/hip_sdk_not_on_path.py judges. JSON via --out only. Public downloads only
(GitHub release assets, stable.repo.amd.com wheels); no token is used or sent.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
import urllib.request
import zipfile
from pathlib import Path

WHL = "https://stable.repo.amd.com/rocm/whl-next"
WHEELS = [
    ("rocm-sdk-libraries/rocm_sdk_libraries-10.1.0-py3-none-win_amd64.whl", "_rocm_sdk_libraries/", ""),
    ("rocm-sdk-device-gfx1151/rocm_sdk_device_gfx1151-10.1.0-py3-none-win_amd64.whl", "_rocm_sdk_libraries/", ""),
    ("rocm-sdk-core/rocm_sdk_core-10.1.0-py3-none-win_amd64.whl", "_rocm_sdk_core/bin/", "bin/"),
]
CASES = ["hip_path", "hip_path_force", "program_files", "no_sdk", "sdk6_only"]


def _has(d, name) -> bool:
    return os.path.isfile(os.path.join(str(d).strip().strip('"'), name))


def _host_facts() -> dict:
    pf = os.environ.get("ProgramFiles") or r"C:\Program Files"
    rocm = Path(pf) / "AMD" / "ROCm"
    vers = {}
    try:
        for p in sorted(rocm.iterdir()):
            b = p / "bin"
            vers[p.name] = {n: (b / n).is_file() for n in
                            ("hipblas.dll", "rocblas.dll", "amdhip64_7.dll", "amdhip64_6.dll")}
    except OSError as exc:
        vers = {"_error": repr(exc)}
    system32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
    path = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    return {
        "env": {k: v for k, v in os.environ.items()
                if k.upper().startswith(("HIP_", "ROCM", "HSA_")) or k.upper() in ("PROGRAMFILES", "PROGRAMW6432")},
        "program_files_rocm": str(rocm),
        "program_files_rocm_versions": vers,
        "path_entries_with_hip": [p for p in path if _has(p, "hipblas.dll") or any(
            n.lower().startswith("amdhip64") for n in _ls(p.strip('"')))],
        "system32_hip": sorted(n for n in _ls(system32) if n.lower().startswith(("amdhip64", "hipblas", "rocblas"))),
    }


def _ls(d) -> list[str]:
    try:
        return os.listdir(d)
    except OSError:
        return []


def _hide_path() -> list[str]:
    parts = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    hidden = [p for p in parts if _has(p, "hipblas.dll") or any(
        n.lower().startswith("amdhip64") for n in _ls(p.strip('"')))]
    os.environ["PATH"] = os.pathsep.join(p for p in parts if p not in hidden)
    return hidden


def _real_sdk_root() -> str | None:
    cands = [os.environ.get("HIP_PATH"), os.environ.get("ROCM_PATH")]
    for k, v in os.environ.items():
        if k.upper().startswith("HIP_PATH_"):
            cands.append(v)
    rocm = Path(os.environ.get("ProgramFiles") or r"C:\Program Files") / "AMD" / "ROCm"
    try:
        cands += [str(p) for p in sorted(rocm.iterdir(), reverse = True)]
    except OSError:
        pass
    for c in cands:
        c = (c or "").strip().strip('"')
        if c and _has(os.path.join(c, "bin"), "hipblas.dll") and _has(os.path.join(c, "bin"), "amdhip64_7.dll"):
            return c
    return None


def _fetch(url: str, dest: Path) -> None:
    if dest.is_file() and dest.stat().st_size > 0:
        return
    tmp = dest.with_suffix(".part")
    req = urllib.request.Request(url, headers = {"User-Agent": "amd-ci-probe"})
    with urllib.request.urlopen(req, timeout = 900) as r, open(tmp, "wb") as f:  # noqa: S310
        shutil.copyfileobj(r, f, 1 << 20)
    tmp.replace(dest)


def _construct_sdk(base: Path) -> dict:
    root = base / "PF" / "AMD" / "ROCm" / "7.2"
    done = root / ".amd_ci_done"
    info = {"root": str(root), "wheels": [w[0] for w in WHEELS]}
    if done.is_file():
        info["reused"] = True
        return info
    dl = base / "wheels"
    dl.mkdir(parents = True, exist_ok = True)
    for rel, prefix, dest_sub in WHEELS:
        whl = dl / rel.rsplit("/", 1)[-1]
        _fetch(f"{WHL}/{rel}", whl)
        with zipfile.ZipFile(whl) as z:
            for m in z.infolist():
                if m.is_dir() or not m.filename.startswith(prefix):
                    continue
                tail = m.filename[len(prefix):]
                if dest_sub == "bin/" and ("/" in tail or not tail.lower().endswith(".dll")):
                    continue  # core: only its top-level bin DLLs (amdhip64_7, comgr, kpack, hiprtc)
                out = root / dest_sub / tail
                out.parent.mkdir(parents = True, exist_ok = True)
                with z.open(m) as src, open(out, "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
        whl.unlink()
    done.write_text("ok", encoding = "utf-8")
    return info


def _start(binary, env, *args) -> dict:
    if not binary or not os.path.isfile(str(binary)):
        return {"path": str(binary) if binary else None, "exists": False}
    previous = None
    try:
        import ctypes

        previous = ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
    except Exception:  # noqa: BLE001
        pass
    try:
        proc = subprocess.run([str(binary), *args], capture_output = True, text = True, encoding = "utf-8",
                              errors = "replace", timeout = 120, env = env, stdin = subprocess.DEVNULL,
                              creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0))
        rc, out, err = proc.returncode, (proc.stdout or "") + (proc.stderr or ""), None
    except Exception as exc:  # noqa: BLE001
        rc, out, err = None, "", repr(exc)
    finally:
        if previous is not None:
            try:
                import ctypes

                ctypes.windll.kernel32.SetErrorMode(previous)
            except Exception:  # noqa: BLE001
                pass
    code = None if rc is None else (rc + (1 << 32) if rc < 0 else rc)
    return {"path": str(binary), "exists": True, "args": list(args), "rc": rc,
            "rc_hex": None if code is None else f"0x{code:08X}", "dll_not_found": code == 0xC0000135,
            "output_tail": out[-700:], "rocm_device_listed": "ROCm0" in out, "error": err}


def run_case(args) -> int:
    case = args.case
    obs: dict = {"case": case}
    out_path = Path(args.out)
    spec = json.loads(Path(args.spec).read_text(encoding = "utf-8"))
    sdk_root = spec["sdk_root"]
    work = Path(spec["work"]) / case
    shutil.rmtree(work, ignore_errors = True)
    (work / "home" / "studio").mkdir(parents = True)
    empty_pf = work / "emptyPF"
    empty_pf.mkdir()
    os.environ["PATH"] = spec["hidden_path"]
    for k in list(os.environ):
        if k.upper().startswith(("HIP_PATH", "ROCM_PATH")) or k in (
                "UNSLOTH_SD_CPP_WINDOWS_ROCM", "UNSLOTH_SD_CPP_VERIFY_INSTALL",
                "UNSLOTH_DIFFUSION_SD_CPP_VULKAN_FALLBACK", "GH_TOKEN", "GITHUB_TOKEN", "HF_TOKEN"):
            os.environ.pop(k, None)
    os.environ["ProgramFiles"] = str(empty_pf)
    if case in ("hip_path", "hip_path_force"):
        os.environ["HIP_PATH"] = sdk_root.rstrip("\\") + "\\"
        if spec["sdk_kind"] == "installed":
            os.environ["ProgramFiles"] = spec["real_program_files"]
    if case == "hip_path_force":
        os.environ["UNSLOTH_SD_CPP_WINDOWS_ROCM"] = "1"
    if case == "program_files":
        os.environ["ProgramFiles"] = str(Path(sdk_root).parents[2])
    if case == "sdk6_only":
        six = work / "ROCm6" / "6.4"
        (six / "bin").mkdir(parents = True)
        for n in ("hipblas.dll", "rocblas.dll", "amdhip64_6.dll"):
            (six / "bin" / n).write_bytes(b"MZ")
        os.environ["HIP_PATH"] = str(six) + "\\"
    os.environ["UNSLOTH_STUDIO_HOME"] = str(work / "home" / "studio")
    obs["env"] = {k: os.environ.get(k) for k in ("HIP_PATH", "ProgramFiles", "UNSLOTH_SD_CPP_WINDOWS_ROCM")}

    def dump():
        out_path.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")

    checkout = Path(args.checkout)
    sys.path.insert(0, str(checkout / "studio" / "backend"))
    sys.path.insert(0, str(checkout / "studio"))
    cache = Path(spec["cache"])
    try:
        import install_sd_cpp_prebuilt as inst

        real_download = inst._download

        def cached_download(url, dest, *a, **k):
            hit = cache / url.rsplit("/", 1)[-1]
            if not hit.is_file():
                real_download(url, hit, *a, **k)
            shutil.copyfile(hit, dest)

        inst._download = cached_download
        fn = getattr(inst, "windows_hip_sdk_bin_dirs", None)
        obs["installer_sdk_bin_dirs"] = fn() if fn else "absent at this state"
        import core.inference.sd_cpp_backend as bk
        import core.inference.sd_cpp_engine as eng

        obs["backend_file"] = bk.__file__
        obs["hip_missing_dlls"] = bk.windows_hip_runtime_missing_dlls()
        obs["preferred_for_rocm"] = bk.preferred_accelerator("rocm")
        dump()
        if case != "program_files":
            try:
                server = bk.ensure_sd_server_binary(allow_install = True, accelerator = "rocm")
                obs["ensure_error"] = None
            except Exception as exc:  # noqa: BLE001
                server, obs["ensure_error"] = None, repr(exc)
            obs["ensure_server"] = server
            root = inst.default_install_dir()
            rec = inst.read_install_record(root)
            try:
                obs["record_accelerator"] = inst.installed_accelerator(root)
            except Exception as exc:  # noqa: BLE001
                obs["record_accelerator"] = rec.get("accelerator") or repr(exc)
            obs["record"] = {k: v for k, v in rec.items() if isinstance(v, (str, int, float, bool))}
            cli = None
            if server:
                bindir = Path(server).parent
                obs["bundle_dlls"] = sorted(n for n in _ls(bindir) if n.lower().endswith(".dll"))
                for n in ("sd-cli.exe", "sd.exe"):
                    if (bindir / n).is_file():
                        cli = str(bindir / n)
                        break
                renv = eng.runtime_env(cli or server)
                obs["runtime_env_path_tail"] = renv.get("PATH", "").split(os.pathsep)[-3:]
                obs["runtime_env_has_sdk_bin"] = any(
                    os.path.normcase(p.rstrip("\\")) == os.path.normcase(os.path.join(sdk_root, "bin"))
                    for p in renv.get("PATH", "").split(os.pathsep))
                obs["start_runtime_env_version"] = _start(cli or server, renv, "--version")
                obs["start_runtime_env_list_devices"] = _start(cli or server, renv, "--list-devices")
                penv = dict(os.environ)
                penv["PATH"] = str(bindir) + os.pathsep + penv.get("PATH", "")
                obs["start_path_only_version"] = _start(cli or server, penv, "--version")
                try:
                    obs["listed_device"] = list(bk.sd_cpp_listed_device(cli or server, probe = True))
                except Exception as exc:  # noqa: BLE001
                    obs["listed_device"] = repr(exc)
            obs["rocm_failure_recorded"] = bk.accelerator_runtime_failed("rocm")
            obs["preferred_for_rocm_after_ensure"] = bk.preferred_accelerator("rocm")
    except Exception as exc:  # noqa: BLE001
        obs["probe_error"] = repr(exc)
        obs["probe_traceback"] = traceback.format_exc()[-2500:]
    dump()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--case", default = None)
    ap.add_argument("--spec", default = None)
    args = ap.parse_args()
    if args.case:
        return run_case(args)

    obs: dict = {"state": args.state, "platform": sys.platform, "cases": {}}
    out_path = Path(args.out)

    def dump():
        out_path.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")

    try:
        for var in ("GH_TOKEN", "GITHUB_TOKEN", "HF_TOKEN"):
            os.environ.pop(var, None)
        obs["host"] = _host_facts()
        pip = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "httpx", "structlog"],
                             capture_output = True, text = True)
        obs["pip_rc"] = pip.returncode
        temp = Path(os.environ.get("RUNNER_TEMP") or os.environ.get("TEMP") or ".")
        base = temp / "amd-ci-hipsdk"
        base.mkdir(parents = True, exist_ok = True)
        real_pf = os.environ.get("ProgramFiles") or r"C:\Program Files"
        real = _real_sdk_root()
        hidden = _hide_path()
        obs["hidden_path_entries"] = hidden
        if real:
            sdk = {"kind": "installed", "root": real}
        else:
            sdk = {"kind": "constructed", **_construct_sdk(base)}
        b = os.path.join(sdk["root"], "bin")
        sdk["bin_has"] = {n: _has(b, n) for n in
                          ("hipblas.dll", "rocblas.dll", "amdhip64_7.dll", "libhipblaslt.dll", "rocsolver.dll")}
        sdk["rocblas_library"] = sorted(_ls(os.path.join(b, "rocblas", "library")))[:5]
        obs["sdk"] = sdk
        dump()
        cache = base / "sdcpp_cache"
        cache.mkdir(exist_ok = True)
        spec = {"sdk_root": sdk["root"], "sdk_kind": sdk["kind"], "hidden_path": os.environ["PATH"],
                "real_program_files": real_pf, "work": str(base / "work" / args.state), "cache": str(cache)}
        spec_path = base / f"spec_{args.state}.json"
        spec_path.write_text(json.dumps(spec), encoding = "utf-8")
        for case in CASES:
            cout = base / f"case_{args.state}_{case}.json"
            log = base / f"case_{args.state}_{case}.log"
            with open(log, "wb") as fh:
                rc = subprocess.run([sys.executable, __file__, "--state", args.state, "--checkout", args.checkout,
                                     "--out", str(cout), "--case", case, "--spec", str(spec_path)],
                                    stdout = fh, stderr = subprocess.STDOUT, timeout = 2400).returncode
            try:
                c = json.loads(cout.read_text(encoding = "utf-8"))
            except Exception as exc:  # noqa: BLE001
                c = {"parse_error": repr(exc)}
            c["_rc"] = rc
            c["_log_tail"] = log.read_text(encoding = "utf-8", errors = "replace")[-1500:]
            obs["cases"][case] = c
            dump()
    except Exception as exc:  # noqa: BLE001
        obs["probe_error"] = repr(exc)
        obs["probe_traceback"] = traceback.format_exc()[-2500:]
    dump()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
