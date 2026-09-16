#!/usr/bin/env python3
"""Probe: what the real sd.cpp ROCm and Vulkan prebuilts do on THIS card, and what
each checkout's sd.cpp backend reads from them.

Observes only. Every judgement -- "did the ROCm build run", "did the head divert a
working host" -- belongs to criteria/sd_cpp_accelerator_reality.py.

Two halves.

**Shared, state-independent.** The prebuilt archives, the model, `--list-devices`
and the real generations are facts about the HOST, not about the checkout, and
`studio/install_sd_cpp_prebuilt.py` is byte-identical across the states of
unsloth#11068. So that work is done once into `--shared`, and every state records
the sha256 of its own copy of the installer so a reader can check the sameness
claim rather than take it.

**Per state.** Each checkout's own `core.inference.sd_cpp_backend` is imported and
asked what it reads FROM those real binaries and that real error text: the
`--list-devices` verdict, the marker tiers, the fallback rung, the preference, the
fingerprint, and whether a record written on this machine retires when the bundle
changes. Functions absent at the base are recorded as None, never emulated.

Nothing here raises for an unwelcome answer: a ROCm build that cannot run is the
observation, not an error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

# The pinned tag `studio/install_sd_cpp_prebuilt.py` installs is the FIRST bundle;
# the retirement check needs a genuinely DIFFERENT one, so it installs the cheap
# Vulkan asset of another release over the managed tree and re-reads the record.
RETIREMENT_TAG = os.environ.get("AMD_CI_SD_CPP_OTHER_TAG", "master-869-07a85c7")

MODELS = {
    # fp16 single-file SD1.5: the dense hipBLAS matmul path, which is where #9278's
    # `CUBLAS_STATUS_INVALID_VALUE at hipblasSetStream` comes from.
    "sd15_fp16": (
        "https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive/resolve/main/"
        "v1-5-pruned-emaonly-fp16.safetensors"
    ),
    # Q4_0 GGUF: the quantized matmul path (`ggml_cuda_mul_mat_q`), which is where
    # the same report's second shape, `unspecified launch failure`, comes from.
    "sd15_q4_0": (
        "https://huggingface.co/second-state/stable-diffusion-v1-5-GGUF/resolve/main/"
        "stable-diffusion-v1-5-pruned-emaonly-Q4_0.gguf"
    ),
}

GEN_TIMEOUT = int(os.environ.get("AMD_CI_SD_CPP_GEN_TIMEOUT", "1800"))

# Harness smoke switch: skip the model download and the renders, so the plumbing
# (install, --list-devices, every per-state read, the fingerprint and the
# retirement) can be exercised off the runner. It is RECORDED in the observations
# and it makes the criteria's "something rendered" gate fail, so a smoke run lands
# on INCONCLUSIVE and can never be mistaken for a measurement.
SMOKE = os.environ.get("AMD_CI_SD_CPP_SMOKE", "") == "1"


# --------------------------------------------------------------------------- utils

def _sha256(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:  # noqa: BLE001
        return None


def _run(cmd: list[str], *, timeout: int, cwd: Path | None = None, env: dict | None = None) -> dict:
    """Run and record. Output is captured whole; the tail is what the markers read."""
    started = time.time()
    try:
        p = subprocess.run(cmd, capture_output = True, text = True, errors = "replace",
                           timeout = timeout, cwd = str(cwd) if cwd else None, env = env)
        out = (p.stdout or "") + (p.stderr or "")
        return {"cmd": cmd, "rc": p.returncode, "seconds": round(time.time() - started, 2),
                "output": out[-20000:], "stdout": (p.stdout or "")[-20000:],
                "output_chars": len(out), "timed_out": False}
    except subprocess.TimeoutExpired as e:  # noqa: BLE001
        out = ""
        for part in (e.stdout, e.stderr):
            if part:
                out += part.decode("utf-8", "replace") if isinstance(part, bytes) else part
        return {"cmd": cmd, "rc": None, "seconds": round(time.time() - started, 2),
                "output": out[-20000:], "output_chars": len(out), "timed_out": True}
    except Exception as e:  # noqa: BLE001
        return {"cmd": cmd, "rc": None, "seconds": round(time.time() - started, 2),
                "output": f"{type(e).__name__}: {e}", "output_chars": 0, "timed_out": False,
                "spawn_error": True}


def _download(url: str, dest: Path, timeout: int = 3600) -> dict:
    """curl if present, else urllib. Recorded either way; a failed fetch is data."""
    if dest.is_file() and dest.stat().st_size > 0:
        return {"path": str(dest), "bytes": dest.stat().st_size, "cached": True, "ok": True}
    dest.parent.mkdir(parents = True, exist_ok = True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    curl = shutil.which("curl")
    if curl:
        r = _run([curl, "-sSL", "--fail", "--retry", "3", "-o", str(tmp), url], timeout = timeout)
        ok = r["rc"] == 0 and tmp.is_file() and tmp.stat().st_size > 0
        detail = r["output"][-2000:]
    else:
        ok, detail = True, ""
        try:
            import urllib.request
            with urllib.request.urlopen(url, timeout = timeout) as src, open(tmp, "wb") as fh:
                shutil.copyfileobj(src, fh)
        except Exception as e:  # noqa: BLE001
            ok, detail = False, f"{type(e).__name__}: {e}"
    if ok:
        tmp.replace(dest)
        return {"path": str(dest), "bytes": dest.stat().st_size, "cached": False, "ok": True}
    return {"path": str(dest), "ok": False, "error": detail}


def _locate(root: Path, names: tuple[str, ...]) -> str | None:
    for name in names:
        for found in sorted(root.rglob(name)):
            if found.is_file():
                return str(found)
    return None


def _png_bytes(path: Path) -> dict:
    """Is the output a real image, and is it not a flat field? Read without PIL."""
    info: dict = {"path": str(path), "exists": path.is_file()}
    if not info["exists"]:
        return info
    data = path.read_bytes()
    info["bytes"] = len(data)
    info["png_magic"] = data[:8] == b"\x89PNG\r\n\x1a\n"
    # An all-one-colour PNG compresses to almost nothing; a real render does not.
    # This is a crude non-vacuity signal, deliberately not a quality judgement.
    info["distinct_byte_values"] = len(set(data[:200000]))
    return info


# ------------------------------------------------------------------ the shared half

def _install_bundle(installer: Path, python: str, accelerator: str, target: Path,
                    tag: str | None = None) -> dict:
    env = dict(os.environ)
    if tag is not None:
        env["UNSLOTH_SD_CPP_TAG"] = tag
    asset = _run([python, str(installer), "--print-asset", "--accelerator", accelerator],
                 timeout = 300, env = env)
    r = _run([python, str(installer), "--accelerator", accelerator,
              "--install-dir", str(target)], timeout = 3600, env = env)
    cli = _locate(target, ("sd-cli", "sd-cli.exe", "sd", "sd.exe"))
    server = _locate(target, ("sd-server", "sd-server.exe"))
    # The name the installer itself uses; read it rather than guessing, since the
    # bundle TAG in here is what the PR's fingerprint calls `bundle`.
    record = None
    for found in sorted(target.rglob(".unsloth-sd-cpp-install.json")):
        try:
            record = json.loads(found.read_text(encoding = "utf-8"))
        except Exception:  # noqa: BLE001
            record = {"unreadable": str(found)}
        break
    # stdout only: the warning about the mirror lacking the pinned tag goes to
    # stderr, and concatenating the two made the warning look like the asset name.
    stdout_lines = [ln for ln in (asset.get("stdout") or "").splitlines() if ln.strip()]
    # `--print-asset` also narrates its repo fallback on stdout, so take the line
    # that is an asset name when there is one rather than simply the last line.
    zips = [ln for ln in stdout_lines if ln.strip().lower().endswith(".zip")]
    stdout_lines = zips or stdout_lines
    out = {"accelerator": accelerator, "tag_requested": tag,
           "asset_resolved": stdout_lines[-1].strip() if stdout_lines else None,
           "asset_stderr": (asset.get("output") or "")[-1500:],
           "install_rc": r["rc"], "install_tail": r["output"][-4000:],
           "cli": cli, "server": server, "install_record": record,
           "cli_bytes": (Path(cli).stat().st_size if cli else None)}
    if cli and shutil.which("ldd"):
        ldd = _run(["ldd", cli], timeout = 120)
        missing = [ln.strip() for ln in ldd["output"].splitlines() if "not found" in ln]
        out["ldd_missing"] = missing
        out["ldd_tail"] = ldd["output"][-3000:]
    return out


def _generate(cli: str, model: Path, out_png: Path, *, steps: int, size: int,
              offload: bool) -> dict:
    """One real txt2img. Explicit --cfg-scale and --sampling-method because an
    UNPATCHED upstream build (which the ROCm and Vulkan assets are) aborts on some
    defaults; that abort is an argument bug, not the card."""
    # `img_gen`, not `txt2img`: sd-cli's modes are [img_gen, adetailer, vid_gen,
    # upscale, convert, metadata], and an unknown mode fails before any backend is
    # touched, which would have looked exactly like a card that cannot run.
    cmd = [cli, "--mode", "img_gen", "--model", str(model),
           "--prompt", "a red apple on a wooden table",
           "--width", str(size), "--height", str(size),
           "--steps", str(steps), "--cfg-scale", "7.0",
           "--sampling-method", "euler", "--seed", "42",
           "--output", str(out_png)]
    if offload:
        # #9278's second shape needs the offloading path; the flag name differs by
        # build, so it is tried and its rejection recorded rather than assumed.
        cmd += ["--offload-to-cpu"]
    r = _run(cmd, timeout = GEN_TIMEOUT)
    low = (r.get("output") or "").lower()
    # An argument this build does not know is a harness fault, not a card fault, and
    # must be legible as such next to a real backend failure.
    r["arg_rejected"] = any(s in low for s in (
        "unknown argument", "unknown option", "invalid option", "unrecognized",
        "error: invalid mode", "usage:"))
    r["image"] = _png_bytes(out_png)
    r["model"] = str(model)
    r["offload"] = offload
    return r


def _device_lines(text: str) -> list[str]:
    """The lines that say which device actually held the weights. `system_info`
    names ROCm but never Vulkan, so a detector keyed on it calls a working Vulkan
    run cpu-only; `load_tensors: <dev> model buffer size` is the honest one."""
    keys = ("model buffer size", "load_tensors", "ggml_vulkan", "ggml_cuda", "using ",
            "backend", "device")
    out = []
    for line in text.splitlines():
        low = line.lower()
        if any(k in low for k in keys):
            out.append(line.strip()[:300])
    return out[:80]


def shared_hardware_facts(shared: Path, installer: Path, python: str) -> dict:
    """Done ONCE, by whichever state probes first. Everything here is a fact about
    the machine, so repeating it per state would only add download time and the
    chance of two states disagreeing about one card."""
    done = shared / "hardware.json"
    if done.is_file():
        try:
            doc = json.loads(done.read_text(encoding = "utf-8"))
            doc["reused"] = True
            return doc
        except Exception:  # noqa: BLE001
            pass

    facts: dict = {"reused": False, "run_id": f"{os.getpid()}-{time.time():.3f}",
                   "installer_sha256": _sha256(installer),
                   "python": python, "platform": {
                       "system": platform.system(), "machine": platform.machine(),
                       "release": platform.release(), "node": platform.node()}}

    for tool, args in (("amd-smi", ["static"]), ("rocminfo", []), ("vulkaninfo", ["--summary"])):
        exe = shutil.which(tool)
        if exe:
            facts[f"{tool}"] = _run([exe, *args], timeout = 300)["output"][:8000]
        else:
            facts[f"{tool}"] = None

    facts["bundles"] = {}
    for accel in ("rocm", "vulkan"):
        facts["bundles"][accel] = _install_bundle(
            installer, python, accel, shared / f"bundle_{accel}")

    # --list-devices on each real binary, which is the probe Studio itself runs.
    facts["list_devices"] = {}
    for accel, b in facts["bundles"].items():
        if b.get("cli"):
            facts["list_devices"][accel] = _run([b["cli"], "--list-devices"], timeout = 600)
        else:
            facts["list_devices"][accel] = None

    facts["models"] = {}
    facts["smoke"] = SMOKE
    for name, url in ({} if SMOKE else MODELS).items():
        facts["models"][name] = _download(url, shared / "models" / url.rsplit("/", 1)[-1])

    facts["generations"] = {}
    for accel, b in facts["bundles"].items():
        cli = b.get("cli")
        if not cli:
            continue
        for mname, m in facts["models"].items():
            if not m.get("ok"):
                continue
            for offload in (False, True) if accel == "rocm" else (False,):
                key = f"{accel}:{mname}:{'offload' if offload else 'plain'}"
                png = shared / "renders" / f"{key.replace(':', '_')}.png"
                png.parent.mkdir(parents = True, exist_ok = True)
                g = _generate(cli, Path(m["path"]), png,
                              steps = 4, size = 256, offload = offload)
                g["device_lines"] = _device_lines(g["output"])
                facts["generations"][key] = g

    done.parent.mkdir(parents = True, exist_ok = True)
    done.write_text(json.dumps(facts, indent = 2), encoding = "utf-8")
    return facts


# ------------------------------------------------------------------- the state half

def import_backend(checkout: Path):
    backend = checkout / "studio" / "backend"
    if not backend.is_dir():
        raise RuntimeError(f"no studio/backend at {backend}")
    sys.path.insert(0, str(backend))
    for stale in [m for m in sys.modules
                  if m.split(".")[0] in ("core", "utils", "storage", "loggers", "routes")]:
        del sys.modules[stale]
    import core.inference.sd_cpp_backend as mod  # noqa: PLC0415
    return mod


def _call(mod, name, *a, **kw):
    """Call a function if this checkout has it. Absent is None, never emulated:
    the base genuinely has no fallback rung and pretending otherwise would be the
    easiest way to fake this comparison."""
    fn = getattr(mod, name, None)
    if fn is None:
        return {"present": False, "value": None}
    try:
        return {"present": True, "value": fn(*a, **kw)}
    except Exception as e:  # noqa: BLE001
        return {"present": True, "error": f"{type(e).__name__}: {e}"}


def _marker_scan(mod, text: str) -> dict:
    """Which of the checkout's own marker strings the REAL error text contains.
    Reading the constants is observation; whether the list is adequate is the
    criteria's call."""
    low = (text or "").lower()
    out: dict = {}
    for tier in ("_ACCELERATOR_DECISIVE_FAILURE_MARKERS",
                 "_ACCELERATOR_AMBIGUOUS_FAILURE_MARKERS",
                 "_ACCELERATOR_CAPACITY_FAILURE_MARKERS"):
        markers = getattr(mod, tier, None)
        if markers is None:
            out[tier] = None
            continue
        out[tier] = {"markers": list(markers),
                     "matched": [m for m in markers if m.lower() in low]}
    return out


def _fingerprint_observations(mod, shared: Path, installer: Path, python: str,
                              studio_home: Path) -> dict:
    """The fingerprint, on real hardware, plus a real retirement.

    The managed tree is a COPY of the shared ROCm bundle, so the record the
    fingerprint reads is the one a real install wrote. The retirement half then
    installs a different release over that same tree and re-asks, which is the
    claim "a record retires when the bundle changes" measured rather than argued.
    """
    obs: dict = {}
    fn = getattr(mod, "_accelerator_fingerprint", None)
    if fn is None:
        return {"present": False}
    obs["present"] = True

    try:
        import core.inference.sd_cpp_engine as engine  # noqa: PLC0415
        root = Path(engine.managed_install_root())
        obs["managed_install_root"] = str(root)
        src = shared / "bundle_rocm"
        if src.is_dir() and not root.exists():
            shutil.copytree(src, root)
        obs["managed_tree_seeded"] = root.is_dir()
    except Exception as e:  # noqa: BLE001
        obs["managed_root_error"] = f"{type(e).__name__}: {e}"

    # Cold, then warm: D1 in this PR's own review was that the cards component was
    # memoised as the cold sentinel, so both readings belong in the record.
    obs["fingerprint_cold"] = _call(mod, "_accelerator_fingerprint")["value"]
    time.sleep(8)
    obs["fingerprint_warm"] = _call(mod, "_accelerator_fingerprint")["value"]
    obs["host_fingerprint"] = _call(mod, "_host_fingerprint")["value"]

    try:
        import torch  # noqa: PLC0415
        obs["torch_version"] = torch.__version__
        obs["torch_version_hip"] = getattr(torch.version, "hip", None)
        obs["torch_version_cuda"] = getattr(torch.version, "cuda", None)
    except Exception as e:  # noqa: BLE001
        obs["torch_error"] = f"{type(e).__name__}: {e}"

    try:
        from utils.hardware.hardware import get_physical_gpu_inventory  # noqa: PLC0415
        def _read(inv):
            # It answers with a mapping, so getattr would silently record None for
            # every field and the "cards, not the unknown sentinel" claim would be
            # unfalsifiable.
            get = inv.get if isinstance(inv, dict) else (lambda k, d = None: getattr(inv, k, d))
            return {"unknown": get("unknown", None), "available": get("available", None),
                    "names": [d.get("name") for d in (get("devices", None) or [])
                              if isinstance(d, dict)],
                    "vendors": sorted({d.get("vendor") for d in (get("devices", None) or [])
                                       if isinstance(d, dict)}),
                    "sources": get("sources", None), "repr": str(inv)[:2000]}

        obs["inventory_nonblocking"] = _read(get_physical_gpu_inventory(block = False))
        obs["inventory_blocking"] = _read(get_physical_gpu_inventory(block = True))
    except Exception as e:  # noqa: BLE001
        obs["inventory_error"] = f"{type(e).__name__}: {e}"

    # A real record, written on this machine, then a real bundle change.
    obs["note"] = _call(mod, "note_accelerator_runtime_failure", "rocm", proven = True)
    obs["failed_after_note"] = _call(mod, "accelerator_runtime_failed", "rocm")
    # Read the PERSISTED half on its own. Without this, an in-process mirror hit is
    # indistinguishable from a record that would survive a restart, which is the
    # whole point of storing it.
    obs["stored_after_note"] = _call(mod, "_stored_accelerator_runtime_failures")
    obs["state_after_note"] = _call(mod, "accelerator_runtime_failure_state")
    obs["preferred_after_note"] = _call(mod, "preferred_accelerator", "rocm")

    root = obs.get("managed_install_root")
    if root:
        obs["retirement_install"] = _install_bundle(
            installer, python, "vulkan", Path(root), tag = RETIREMENT_TAG)
        obs["fingerprint_after_bundle_change"] = _call(mod, "_accelerator_fingerprint")["value"]
        obs["failed_after_bundle_change"] = _call(mod, "accelerator_runtime_failed", "rocm")
        obs["state_after_bundle_change"] = _call(mod, "accelerator_runtime_failure_state")
        obs["preferred_after_bundle_change"] = _call(mod, "preferred_accelerator", "rocm")

    obs["cleared"] = _call(mod, "clear_accelerator_runtime_failures")
    obs["failed_after_clear"] = _call(mod, "accelerator_runtime_failed", "rocm")
    obs["studio_home"] = str(studio_home)
    return obs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--shared", required = True, type = Path,
                    help = "cache for the host-level facts: bundles, model, renders")
    args = ap.parse_args()

    obs: dict = {"state": args.state, "checkout": str(args.checkout),
                 "platform": {"system": platform.system(), "machine": platform.machine(),
                              "node": platform.node()},
                 "python": sys.executable}
    args.shared.mkdir(parents = True, exist_ok = True)

    installer = args.checkout / "studio" / "install_sd_cpp_prebuilt.py"
    obs["installer_sha256"] = _sha256(installer)
    obs["installer_default_tag"] = None
    try:
        for line in installer.read_text(encoding = "utf-8").splitlines():
            if line.startswith("DEFAULT_TAG"):
                obs["installer_default_tag"] = line.split("=", 1)[1].strip().strip('"')
                break
    except Exception:  # noqa: BLE001
        pass

    try:
        obs["shared"] = shared_hardware_facts(args.shared, installer, sys.executable)
    except Exception as e:  # noqa: BLE001
        obs["shared_error"] = f"{type(e).__name__}: {e}"
        obs["shared"] = {}

    # Each state gets its own Studio home so one state's record cannot be read by
    # the next, and so the managed tree the fingerprint reads is this state's.
    home = args.shared / "homes" / args.state
    home.mkdir(parents = True, exist_ok = True)
    os.environ["UNSLOTH_STUDIO_HOME"] = str(home)

    try:
        mod = import_backend(args.checkout)
        obs["module"] = {"imported": True, "file": getattr(mod, "__file__", None)}
    except Exception as e:  # noqa: BLE001
        obs["module"] = {"imported": False, "error": f"{type(e).__name__}: {e}"}
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0

    bundles = (obs.get("shared") or {}).get("bundles") or {}
    obs["reads"] = {}
    for accel, b in bundles.items():
        cli = b.get("cli")
        obs["reads"][accel] = {
            "cli": cli,
            "verdict": _call(mod, "sd_cpp_accelerator_device_verdict", cli),
            "lists_accelerator": _call(mod, "sd_cpp_lists_accelerator_device", cli),
            "installed_accelerator": _call(mod, "_installed_accelerator_of", cli),
        }

    gens = (obs.get("shared") or {}).get("generations") or {}
    obs["error_reading"] = {}
    for key, g in gens.items():
        if not key.startswith("rocm"):
            continue
        text = g.get("output") or ""
        obs["error_reading"][key] = {
            "rc": g.get("rc"),
            "markers": _marker_scan(mod, text),
            "output_shows_accelerator_failure": _call(
                mod, "output_shows_accelerator_failure", text),
        }

    obs["fallback_for"] = {a: _call(mod, "fallback_accelerator_for", a)
                           for a in ("rocm", "vulkan", "cuda", "auto", "cpu")}
    obs["vulkan_fallback_enabled"] = _call(mod, "sd_cpp_vulkan_fallback_enabled")
    # A CLEAN host: no record written yet. This is what a machine whose ROCm build
    # works must keep selecting.
    obs["preferred_clean"] = {a: _call(mod, "preferred_accelerator", a)
                              for a in ("rocm", "vulkan", "cuda", "auto")}
    obs["failed_clean"] = _call(mod, "accelerator_runtime_failed", "rocm")

    try:
        obs["fingerprint"] = _fingerprint_observations(
            mod, args.shared, installer, sys.executable, home)
    except Exception as e:  # noqa: BLE001
        obs["fingerprint"] = {"error": f"{type(e).__name__}: {e}"}

    args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
