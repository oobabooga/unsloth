#!/usr/bin/env python3
"""Validate open Unsloth Studio PRs on this machine, against the Studio that is already installed.

Every check runs in a fresh process under the installed Studio's own Python and venv. Nothing is installed into
that venv: each PR's code is downloaded as a GitHub zip, unpacked into a work folder, and put first on PYTHONPATH.
The "before" arm is the PR's merge base (downloaded the same way), plus the installed release where it matters.

    python devchecks.py --mode quick            # kernels, probes, unit tests (a few minutes)
    python devchecks.py --mode full             # also renders and timings (asks before any model download)
    python devchecks.py --dry-run               # discovery and plan only, downloads nothing

Results: one folder with summary.json and summary.md (plus logs and per-arm JSON). Nothing is uploaded, and no
token is read. HF_TOKEN is used only if it is already set in the environment and a gated repo needs it.
Standard library only, so any Python 3.9+ can run it; the checks themselves run under the Studio Python.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

TOOL_VERSION = "1"
REPO = "unslothai/unsloth"
HERE = Path(__file__).resolve().parent
PROBES = HERE / "probes"
UA = {"User-Agent": "studio-device-checks", "Accept": "application/vnd.github+json"}

# What each PR still needs from real hardware. `tests` are run on the PR head and on its merge base (head's test
# files over base code); `modules` must import from the head tree before anything else runs.
PRS = {
    19665: {
        "title": "Studio FP8: tensorwise fallback when the row-wise fp8 GEMM is missing",
        "branch": "studio-fp8-rowwise-fallback",
        "tests": ["tests/test_diffusion_fp8_rowwise_fallback.py", "tests/test_diffusion_fp8_unit_scale.py"],
        "modules": ["core.inference.diffusion_fp8_unit_scale", "core.inference.diffusion_transformer_quant"],
        "machines": "Windows x64 (RTX 5090), Windows on Arm (RTX Spark); Linux is a no-regression check",
    },
    19664: {
        "title": "Studio NVFP4: bake LoRAs on NVFP4 transformers and cache the runtime NVFP4 build",
        "branch": "studio-nvfp4-lora-cache",
        "tests": [
            "tests/test_diffusion_nvfp4_prequant_cache.py",
            "tests/test_diffusion_lora.py::test_nvfp4_bakes_loras_like_int8_and_fp8_except_on_a_policy_base",
        ],
        "modules": ["core.inference.diffusion_prequant_cache", "core.inference.diffusion_lora"],
        "machines": "DGX Spark (GB10, sm121), RTX 5090 (sm120)",
    },
    19588: {
        "title": "Studio diffusion: keep the sm120 fused W6A8 tile working on Triton 3.9",
        "branch": "studio-triton39-block-ptr",
        "tests": ["tests/test_diffusion_grouped_int8_gemm.py"],
        "modules": ["core.inference.diffusion_grouped_int8_gemm"],
        "machines": "RTX 5090 (sm120) with triton-windows 3.9",
    },
    19621: {
        "title": "Studio video: count hosted prequant downloads in the load progress",
        "branch": "studio-video-download-progress",
        "tests": ["tests/test_video_download_progress_prequant.py"],
        "modules": ["core.inference.video"],
        "machines": "Windows (symlink-less Hugging Face cache)",
    },
}
KLEIN = "black-forest-labs/FLUX.2-klein-4B"
KLEIN_MIRROR = "unsloth/FLUX.2-klein-4B"
KLEIN_FP8 = "unsloth/FLUX.2-klein-4B-FP8"
KLEIN_LORA = "adirik/pixel-art-lora-flux.2-klein-4B"
KLEIN_PIPE = ["model_index.json", "scheduler/*", "tokenizer/*", "text_encoder/*", "transformer/*", "vae/*"]
PYTEST_WHEELS = [  # pinned, hash checked; pure Python, unpacked into the work cache (never into the Studio venv)
    ("pytest-9.0.3-py3-none-any.whl",
     "https://files.pythonhosted.org/packages/d4/24/a372aaf5c9b7208e7112038812994107bc65a84cd00e0354a88c2c77a617/"
     "pytest-9.0.3-py3-none-any.whl", "2c5efc453d45394fdd706ade797c0a81091eccd1d6e4bccfcd476e2b8e0ab5d9"),
    ("pluggy-1.7.0-py3-none-any.whl",
     "https://files.pythonhosted.org/packages/40/9e/2b38731e0fc536806f16490e1a12d7f0dc2a1235aa8cc07bcc75416a7daa/"
     "pluggy-1.7.0-py3-none-any.whl", "7dd7b0d8832ba3cb632c306926ded123429211b83641b35dc5c41ad2d34f9bec"),
    ("iniconfig-2.3.1-py3-none-any.whl",
     "https://files.pythonhosted.org/packages/56/43/4ca9e49d27a1fcf6bece6f6aec0ea46bb9112489b93d4b688fb415457bdb/"
     "iniconfig-2.3.1-py3-none-any.whl", "9121e2c1fdb355232495be3194c8dfe87ccc2d5dee45947b78e68f499790d7a7"),
    ("pygments-2.21.0-py3-none-any.whl",
     "https://files.pythonhosted.org/packages/71/46/17f022dd3e953bf20a04a028a21ec746d942f8d2af30fa0f124fa0e6a684/"
     "pygments-2.21.0-py3-none-any.whl", "2363c69b61c4a97c838da3b130dcd6468f4848992b21a82f2a63ec34377137d9"),
    ("colorama-0.4.6-py2.py3-none-any.whl",
     "https://files.pythonhosted.org/packages/d1/d6/3965ed04c63042e047cb6a3e6ed1a63a35087b6a609aa3a15ed8ac56c221/"
     "colorama-0.4.6-py2.py3-none-any.whl", "4f1d9991f5acc0ca119f9d443620b77f9d6b33703e51011c16baf57afb285fc6"),
]
LPIPS_WHEEL = ("lpips-0.1.4-py3-none-any.whl",
               "https://files.pythonhosted.org/packages/9b/13/1df50c7925d9d2746702719f40e864f51ed66f307b20ad32392f1ad2bb87/"
               "lpips-0.1.4-py3-none-any.whl", "fd537af5828b69d2e6ffc0a397bd506dbc28ca183543617690844c08e102ec5e")

PASS, FAIL, NOT_RUN, INFO, ERROR = "PASS", "FAIL", "NOT_RUN", "INFO", "ERROR"


def log(msg: str) -> None:
    print(f"[devchecks {time.strftime('%H:%M:%S')}] {msg}", flush = True)


# ---------------------------------------------------------------- discovery


def studio_home_candidates(override):
    """The install dir order of install.ps1 / install.sh: UNSLOTH_STUDIO_HOME > STUDIO_HOME > UNSLOTH_HOME/studio >
    ~/.unsloth/studio (USERPROFILE on Windows)."""
    if override:
        return [("--studio-home", Path(override).expanduser())]
    out = []
    for var in ("UNSLOTH_STUDIO_HOME", "STUDIO_HOME"):
        v = (os.environ.get(var) or "").strip()
        if v:
            out.append((var, Path(v).expanduser()))
    v = (os.environ.get("UNSLOTH_HOME") or "").strip()
    if v:
        out.append(("UNSLOTH_HOME", Path(v).expanduser() / "studio"))
    homes = []
    if os.name == "nt" and os.environ.get("USERPROFILE"):
        homes.append(Path(os.environ["USERPROFILE"]))
    homes.append(Path.home())
    for h in homes:
        cand = ("default", h / ".unsloth" / "studio")
        if cand[1] not in [c[1] for c in out]:
            out.append(cand)
    return out


def venv_pythons(home: Path):
    out = []
    for venv in ("unsloth_studio", ".venv"):  # .venv: the legacy layout install.sh still accepts
        out.append(home / venv / "Scripts" / "python.exe")
        out.append(home / venv / "bin" / "python")
    return out


_VALIDATE = (
    "import importlib.util, json, os, sys\n"
    "s = importlib.util.find_spec('studio')\n"
    "locs = list(s.submodule_search_locations or []) if s else []\n"
    "b = [os.path.join(l, 'backend') for l in locs if os.path.isdir(os.path.join(l, 'backend', 'core', 'inference'))]\n"
    "print(json.dumps({'exe': sys.executable, 'version': sys.version.split()[0], 'prefix': sys.prefix,"
    " 'backend': b[0] if b else None}))\n"
)


def validate_python(py: Path):
    try:
        r = subprocess.run([str(py), "-c", _VALIDATE], capture_output = True, text = True, timeout = 120)
    except Exception as exc:  # noqa: BLE001
        return None, f"could not run: {exc}"
    if r.returncode != 0:
        return None, (r.stderr or r.stdout).strip()[-400:]
    try:
        info = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        return None, f"unexpected output: {r.stdout[-200:]}"
    if not info.get("backend"):
        return None, "this Python has no Studio backend (studio/backend/core/inference) installed"
    return info, None


def discover(args):
    tried = []
    if args.studio_python:
        cands = [("--studio-python", None, Path(args.studio_python))]
    else:
        cands = [(src, home, py) for src, home in studio_home_candidates(args.studio_home) for py in venv_pythons(home)]
    for src, home, py in cands:
        if not py.is_file():
            tried.append(f"{py} (missing)")
            continue
        info, problem = validate_python(py)
        if info:
            info.update({"home": str(home) if home else None, "found_via": src, "python": str(py)})
            return info, tried
        tried.append(f"{py} ({problem})")
    return None, tried


# ---------------------------------------------------------------- GitHub zips (no token)


def http_json(url: str, timeout: float = 30):
    req = urllib.request.Request(url, headers = UA)
    with urllib.request.urlopen(req, timeout = timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def resolve_pr(n: int) -> dict:
    """Head SHA and merge base from the public API (unauthenticated, 2 calls). On failure, fall back to the
    refs/pull/N/head and main zips and say so."""
    out = {"pr": n, "head_ref": f"refs/pull/{n}/head", "base_ref": "refs/heads/main", "resolved": False}
    try:
        pr = http_json(f"https://api.github.com/repos/{REPO}/pulls/{n}")
        out.update({"state": pr.get("state"), "merged": bool(pr.get("merged_at")), "title": pr.get("title"),
                    "head_sha": pr["head"]["sha"], "head_branch": pr["head"].get("ref"), "head_ref": pr["head"]["sha"]})
        cmp = http_json(f"https://api.github.com/repos/{REPO}/compare/{pr['base']['sha']}...{pr['head']['sha']}")
        out["base_sha"] = cmp["merge_base_commit"]["sha"]
        out["base_ref"] = out["base_sha"]
        out["resolved"] = True
    except Exception as exc:  # noqa: BLE001
        out["resolve_error"] = f"{type(exc).__name__}: {exc}"[:300]
        out["note"] = "GitHub API unavailable (rate limit?): using refs/pull/N/head and current main as the base"
    return out


def zip_url(ref: str) -> str:
    return f"https://codeload.github.com/{REPO}/zip/{ref}"


def url_ok(url: str) -> bool:
    try:
        req = urllib.request.Request(url, method = "HEAD", headers = {"User-Agent": UA["User-Agent"]})
        with urllib.request.urlopen(req, timeout = 30) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def download(url: str, dest: Path, sha256 = None) -> Path:
    if dest.is_file() and (sha256 is None or file_sha256(dest) == sha256):
        return dest
    dest.parent.mkdir(parents = True, exist_ok = True)
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers = {"User-Agent": UA["User-Agent"]})
    t = time.time()
    with urllib.request.urlopen(req, timeout = 120) as r, open(tmp, "wb") as fh:
        n = 0
        last = 0.0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            n += len(chunk)
            if time.time() - last > 5:
                last = time.time()
                log(f"  {dest.name}: {n / 1e6:.0f} MB")
    if sha256 is not None and file_sha256(tmp) != sha256:
        tmp.unlink()
        raise RuntimeError(f"sha256 mismatch for {url}")
    os.replace(tmp, dest)
    log(f"  downloaded {dest.name} ({dest.stat().st_size / 1e6:.0f} MB, {time.time() - t:.0f} s)")
    return dest


def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


KEEP_PREFIXES = ("studio/backend/", "tests/_shared/")


def extract_tree(zpath: Path, dest: Path) -> dict:
    """Unpack only studio/backend and tests/_shared. Refuses absolute or parent-escaping member names."""
    marker = dest / ".devchecks_complete"
    with zipfile.ZipFile(zpath) as z:
        commit = z.comment.decode("ascii", "replace").strip() or None
        if marker.is_file():
            return {"tree": str(dest), "commit": commit}
        if dest.exists():
            shutil.rmtree(dest)
        root = dest.resolve()
        for info in z.infolist():
            parts = info.filename.split("/", 1)
            if len(parts) < 2 or not parts[1].startswith(KEEP_PREFIXES):
                continue
            rel = parts[1]
            target = (dest / rel).resolve()
            if not str(target).startswith(str(root) + os.sep) or ".." in rel.split("/"):
                raise RuntimeError(f"unsafe path in zip: {info.filename}")
            if info.is_dir():
                target.mkdir(parents = True, exist_ok = True)
                continue
            target.parent.mkdir(parents = True, exist_ok = True)
            with z.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
    (dest / "devchecks_pytest.ini").write_text(
        "[pytest]\nmarkers =\n    gpu: needs a GPU\n    slow: slow\n    server: live server\n"
        "    native_mxc: native Windows MXC\n    reads_real_pip_config: pip config\n", encoding = "utf-8")
    marker.write_text(commit or "", encoding = "utf-8")
    return {"tree": str(dest), "commit": commit}


# ---------------------------------------------------------------- the run


class Run:
    def __init__(self, args):
        self.args = args
        stamp = time.strftime("%Y%m%d_%H%M%S")
        root = Path(args.work_dir).expanduser() if args.work_dir else Path.home() / "unsloth_device_checks"
        self.cache = root / "_cache"
        kind = "dry" if args.dry_run else args.mode
        self.out = Path(args.out).expanduser() if args.out else root / f"run_{kind}_{stamp}"
        for d in (self.cache, self.out / "logs", self.out / "arms", self.out / "images", self.out / "scratch"):
            d.mkdir(parents = True, exist_ok = True)
        self.results = []
        self.prs = {}
        self.trees = {}
        self.deps_paths = []
        self.studio = None
        self.env = {}
        self.hub_cache = None
        self.started = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.downloads_ok = None

    # ---- helpers

    def add(self, check_id, pr, title, status, summary, details = None):
        r = {"id": check_id, "pr": pr, "title": title, "status": status, "summary": summary,
             "details": details or {}}
        self.results.append(r)
        log(f"{status:7s} {check_id}: {summary}")
        return r

    def ask(self, question: str) -> bool:
        if self.args.yes:
            log(question + " [--yes]")
            return True
        if not sys.stdin or not sys.stdin.isatty():
            log(question + " -> no (not interactive; pass --yes to allow)")
            return False
        try:
            return input(question + " [y/N] ").strip().lower() in ("y", "yes")
        except EOFError:
            return False

    def arm_env(self, backend, home_key, extra = None):
        e = dict(os.environ)
        for k in list(e):
            if k.startswith("PYTHON") and k not in ("PYTHONIOENCODING", "PYTHONUTF8"):
                e.pop(k)
        paths = [backend] if backend else []
        paths += self.deps_paths
        e.update({
            "PYTHONPATH": os.pathsep.join(paths),
            "PYTHONNOUSERSITE": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            "PYTHONUNBUFFERED": "1",
            "DEVCHECKS_BACKEND": backend or "",
            "UNSLOTH_STUDIO_HOME": str(self.out / "scratch" / f"home_{home_key}"),
            "UNSLOTH_DISABLE_AUTO_UPDATES": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "DO_NOT_TRACK": "1",
            "TORCHINDUCTOR_CACHE_DIR": str(self.out / "scratch" / f"inductor_{home_key}"),
            "TRITON_CACHE_DIR": str(self.out / "scratch" / f"triton_{home_key}"),
            "TORCH_HOME": str(self.cache / "torch_home"),
        })
        if self.hub_cache:
            e["HF_HUB_CACHE"] = self.hub_cache
        e.update(extra or {})
        return e

    def probe(self, label, script, args, backend, home_key = None, extra_env = None, timeout = 600):
        """Run one probe in a fresh process under the Studio Python; returns its JSON (or None) and the log path."""
        out = self.out / "arms" / f"{label}.json"
        logp = self.out / "logs" / f"{label}.log"
        if out.exists():
            out.unlink()
        cmd = [self.studio["python"], "-u", str(PROBES / script), "--out", str(out), *[str(a) for a in args]]
        env = self.arm_env(backend, home_key or label, extra_env)
        t = time.time()
        log(f"  arm {label} ({script})")
        kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
        with open(logp, "w", encoding = "utf-8", errors = "replace") as lf:
            lf.write("$ " + " ".join(cmd) + "\n")
            lf.flush()
            p = subprocess.Popen(cmd, stdout = lf, stderr = subprocess.STDOUT, env = env, cwd = str(self.out), **kw)
            try:
                rc = p.wait(timeout = timeout)
            except subprocess.TimeoutExpired:
                kill_tree(p)
                rc = "timeout"
        rec = None
        if out.is_file():
            try:
                rec = json.loads(out.read_text(encoding = "utf-8"))
            except Exception:  # noqa: BLE001
                rec = None
        if rec is not None:
            rec["_rc"] = rc
            rec["_wall_s"] = round(time.time() - t, 1)
            rec["_log"] = str(logp)
        else:
            log(f"  arm {label} wrote no result (rc {rc}); last log lines:")
            for line in tail(logp, 12):
                print("    " + line)
        return rec, logp

    # ---- setup

    def setup(self) -> bool:
        info, tried = discover(self.args)
        if not info:
            self.add("discovery", None, "Find the installed Studio", ERROR,
                     "no Studio Python found. Pass -StudioHome / --studio-home (the folder holding unsloth_studio).",
                     {"tried": tried})
            return False
        self.studio = info
        log(f"Studio Python: {info['python']} (Python {info['version']}, found via {info['found_via']})")
        log(f"Installed backend: {info['backend']}")
        env, _ = self.probe("env_installed", "env_probe.py", ["--no-compile"] if self.args.dry_run else [],
                            info["backend"], home_key = "env",
                            extra_env = {"UNSLOTH_STUDIO_HOME": info["home"]} if info.get("home") else {})
        self.env = env or {}
        self.hub_cache = self.args.hf_hub_cache or self.env.get("studio_hf_hub_cache")
        self.add("discovery", None, "Find the installed Studio", PASS if env else ERROR,
                 f"{info['python']}; {describe_machine(self.env)}", {"studio": info, "tried": tried})
        return env is not None

    def gpu(self):
        e = self.env
        if not e.get("cuda_available"):
            return None
        return {"name": e.get("gpu"), "cap": tuple(e.get("capability") or (0, 0)), "rocm": bool(e.get("torch_hip"))}

    def need_nvidia(self, min_cap = None, caps = None):
        g = self.gpu()
        if g is None:
            return "no CUDA GPU visible to the Studio torch (%s)" % (self.env.get("torch") or "torch missing")
        if g["rocm"]:
            return f"needs an NVIDIA GPU; this is a ROCm build on {g['name']}"
        if min_cap and g["cap"] < tuple(min_cap):
            return f"needs compute capability {min_cap[0]}.{min_cap[1]}+; {g['name']} is {g['cap'][0]}.{g['cap'][1]}"
        if caps and g["cap"] not in [tuple(c) for c in caps]:
            want = ", ".join(f"sm{c[0]}{c[1]}" for c in caps)
            return f"needs {want}; {g['name']} is sm{g['cap'][0]}{g['cap'][1]}"
        return None

    def ensure_pytest(self) -> bool:
        if (self.env.get("dists") or {}).get("pytest"):
            return True
        d = self.cache / "deps" / "pytest"
        if not (d / ".complete").is_file():
            log("pytest is not in the Studio venv: unpacking pinned pytest wheels into the work cache (not the venv)")
            try:
                for name, url, sha in PYTEST_WHEELS:
                    w = download(url, self.cache / "wheels" / name, sha)
                    with zipfile.ZipFile(w) as z:
                        z.extractall(d)
                (d / ".complete").write_text("ok", encoding = "utf-8")
            except Exception as exc:  # noqa: BLE001
                log(f"could not fetch pytest: {exc}")
                return False
        if str(d) not in self.deps_paths:
            self.deps_paths.append(str(d))
        return True

    def ensure_lpips(self) -> bool:
        dists = self.env.get("dists") or {}
        if dists.get("lpips"):
            return True
        if not dists.get("torchvision") or not dists.get("scipy"):
            return False
        d = self.cache / "deps" / "lpips"
        try:
            if not (d / ".complete").is_file():
                name, url, sha = LPIPS_WHEEL
                with zipfile.ZipFile(download(url, self.cache / "wheels" / name, sha)) as z:
                    z.extractall(d)
                (d / ".complete").write_text("ok", encoding = "utf-8")
        except Exception as exc:  # noqa: BLE001
            log(f"could not fetch lpips: {exc}")
            return False
        if str(d) not in self.deps_paths:
            self.deps_paths.append(str(d))
        return True

    def pr_info(self, n):
        if n not in self.prs:
            if self.args.tree and n in self.args.tree:
                self.prs[n] = {"pr": n, "resolved": True, "local": True}
            else:
                self.prs[n] = resolve_pr(n)
        return self.prs[n]

    def trees_for(self, n):
        """(head_backend, base_backend) for PR n, downloading and unpacking as needed. Raises with a reason."""
        if n in self.trees:
            return self.trees[n]
        info = self.pr_info(n)
        if self.args.tree and n in self.args.tree:
            head = Path(self.args.tree[n]).resolve()
            base = Path(self.args.base_tree[n]).resolve() if n in (self.args.base_tree or {}) else None
            info.update({"head_tree": str(head), "base_tree": str(base) if base else None})
            self.trees[n] = (str(head / "studio" / "backend"), str(base / "studio" / "backend") if base else None)
            return self.trees[n]
        zips = self.cache / "zips"
        hz = download(zip_url(info["head_ref"]), zips / f"{safe(info['head_ref'])}.zip")
        bz = download(zip_url(info["base_ref"]), zips / f"{safe(info['base_ref'])}.zip")
        h = extract_tree(hz, self.cache / "trees" / f"head_{safe(info['head_ref'])[:40]}")
        # The base gets its own copy per PR: the head's test files go over the base code there.
        b = extract_tree(bz, self.out / "scratch" / f"base_pr{n}")
        info.update({"head_zip_commit": h["commit"], "base_zip_commit": b["commit"],
                     "head_url": zip_url(info["head_ref"]), "base_url": zip_url(info["base_ref"])})
        if info.get("head_sha") and h["commit"] and h["commit"] != info["head_sha"]:
            raise RuntimeError(f"head zip is {h['commit']}, the API says {info['head_sha']}")
        hb = str(Path(h["tree"]) / "studio" / "backend")
        bb = str(Path(b["tree"]) / "studio" / "backend")
        for t in PRS[n]["tests"]:
            rel = t.split("::")[0]
            src = Path(hb) / rel
            if src.is_file():
                (Path(bb) / rel).parent.mkdir(parents = True, exist_ok = True)
                shutil.copyfile(src, Path(bb) / rel)
        self.trees[n] = (hb, bb)
        return self.trees[n]

    def deps_gate(self, n, backend, label):
        """None when the tree imports under the installed venv, else a NOT_RUN reason naming what is missing."""
        rec, _ = self.probe(f"deps_{label}", "deps_probe.py", ["--modules", ",".join(PRS[n]["modules"])], backend,
                            home_key = f"deps_{label}")
        if rec is None:
            return "the dependency probe did not finish (see logs)", None
        bad = rec.get("import_errors") or {}
        if not bad:
            return None, rec
        missing = sorted({v["missing_module"] for v in bad.values() if v.get("missing_module")})
        first = next(iter(bad.values()))["error"]
        if missing:
            return (f"the PR code needs {', '.join(missing)}, which the installed Studio venv does not have "
                    f"({first}). Nothing was installed: update Studio, then rerun."), rec
        if all(v["error"].startswith("ImportError") for v in bad.values()):
            return (f"the PR code needs a newer or different package version than the installed Studio venv has "
                    f"({first}). Nothing was installed: update Studio, then rerun."), rec
        return f"ERROR: the PR code does not import under the installed Studio venv: {first}", rec


def gate_status(reason: str) -> str:
    return ERROR if reason.startswith("ERROR: ") else NOT_RUN


def safe(ref: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in ref)


def tail(p: Path, n: int):
    try:
        return p.read_text(encoding = "utf-8", errors = "replace").splitlines()[-n:]
    except OSError:
        return []


def kill_tree(p):
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output = True)
        else:
            import signal

            os.killpg(p.pid, signal.SIGKILL)
    except Exception:  # noqa: BLE001
        p.kill()
    try:
        p.wait(timeout = 30)
    except Exception:  # noqa: BLE001
        pass


def describe_machine(e: dict) -> str:
    gpu = e.get("gpu") or "no CUDA GPU"
    cap = e.get("capability")
    capd = f" sm{cap[0]}{cap[1]}" if cap else ""
    d = e.get("dists") or {}
    tri = d.get("triton-windows") or d.get("triton") or d.get("pytorch-triton-rocm") or "none"
    return (f"{e.get('system')} {e.get('machine')}, {gpu}{capd}, torch {e.get('torch')}, triton {tri}, "
            f"torchao {d.get('torchao')}, diffusers {d.get('diffusers')}")


def median_spread(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    med = statistics.median(vals)
    return round(med, 4), (round((max(vals) - min(vals)) / med * 100, 1) if med else None)


# ---------------------------------------------------------------- checks


def check_sanity(run: Run):
    """#19569 (merged): torch + Triton import and a tiny torch.compile on the GPU."""
    e = run.env
    d = e.get("dists") or {}
    tri = d.get("triton-windows") or d.get("triton")
    details = {"torch": e.get("torch"), "triton_dist": tri, "triton_import": e.get("triton_import"),
               "triton_import_error": e.get("triton_import_error"), "compile_s": e.get("compile_s"),
               "compile_error": e.get("compile_error"), "windows_compile_gate": e.get("windows_compile_gate")}
    title = "torch + Triton sanity: import both, torch.compile a tiny function on the GPU (#19569)"
    if run.args.dry_run:
        return run.add("sanity.compile", 19569, title, NOT_RUN, "dry run", details)
    reason = run.need_nvidia()
    if reason:
        return run.add("sanity.compile", 19569, title, NOT_RUN, reason, details)
    tv = str(e.get("torch") or "")
    pair = ""
    if tv.startswith("2.15") and tri and not str(tri).startswith("3.9"):
        pair = f" Note: torch 2.15 wants Triton 3.9, found {tri}."
    if e.get("compile_ok"):
        return run.add("sanity.compile", 19569, title, PASS,
                       f"torch {tv}, Triton {tri}: compiled and matched eager in {e.get('compile_s')} s.{pair}",
                       details)
    return run.add("sanity.compile", 19569, title, FAIL,
                   f"torch {tv}, Triton {tri}: {e.get('compile_error') or e.get('triton_import_error')}{pair}",
                   details)


def run_tests(run: Run, n: int, check_id: str):
    title = f"PR unit tests on this machine, merge base vs head ({len(PRS[n]['tests'])} targets)"
    if run.args.dry_run:
        return run.add(check_id, n, title, NOT_RUN, "dry run: would run " + ", ".join(PRS[n]["tests"]))
    if not run.ensure_pytest():
        return run.add(check_id, n, title, NOT_RUN, "pytest is not in the Studio venv and could not be fetched")
    try:
        head, base = run.trees_for(n)
    except Exception as exc:  # noqa: BLE001
        return run.add(check_id, n, title, ERROR, f"could not fetch the PR code: {exc}")
    reason, deps = run.deps_gate(n, head, f"pr{n}_head")
    if reason:
        return run.add(check_id, n, title, gate_status(reason), reason, {"deps": deps})
    arms = {}
    for label, backend in (("base", base), ("head", head)):
        if not backend:
            continue
        ini = str(Path(backend).parent.parent / "devchecks_pytest.ini")
        rec, _ = run.probe(f"tests_pr{n}_{label}", "pytest_arm.py", ["--ini", ini, "--", *PRS[n]["tests"]], backend,
                           extra_env = {"CUDA_VISIBLE_DEVICES": ""} if not run.args.tests_on_gpu else None,
                           timeout = 1500)
        arms[label] = rec
    h = arms.get("head") or {}
    b = arms.get("base") or {}
    details = {"deps_mismatch": (deps or {}).get("mismatch"), "deps_missing": (deps or {}).get("missing"),
               "head": slim_tests(h), "base": slim_tests(b)}
    if h.get("not_run"):
        return run.add(check_id, n, title, NOT_RUN, h["not_run"], details)
    hc = h.get("counts") or {}
    bc = b.get("counts") or {}
    if not h or not hc:
        return run.add(check_id, n, title, ERROR, "the head test run produced no results (see logs)", details)
    for arm, rec in (("head", h), ("base", b)):
        tree = (rec or {}).get("tree") or {}
        if rec and tree.get("own_tree") is False:
            return run.add(check_id, n, title, ERROR, f"the {arm} arm imported code from another tree", details)
    fixed = sorted(k for k, v in h.get("outcomes", {}).items() if v == "passed"
                   and b.get("outcomes", {}).get(k) not in (None, "passed", "skipped"))
    details["fail_on_base_pass_on_head"] = fixed
    head_bad = {k: v for k, v in h.get("outcomes", {}).items() if v in ("failed", "collect_error")}
    summ = (f"head {fmt_counts(hc)}; base {fmt_counts(bc) if bc else 'not run'}; "
            f"{len(fixed)} fail on base and pass on head")
    if head_bad:
        details["head_failures"] = {k: h.get("errors", {}).get(k, "")[-800:] for k in list(head_bad)[:10]}
        return run.add(check_id, n, title, FAIL, summ, details)
    if not hc.get("passed"):
        return run.add(check_id, n, title, NOT_RUN, "every head test skipped here: " + summ, details)
    return run.add(check_id, n, title, PASS, summ, details)


def slim_tests(rec):
    if not rec:
        return None
    return {"counts": rec.get("counts"), "exit_code": rec.get("exit_code"), "pytest": rec.get("pytest"),
            "own_tree": (rec.get("tree") or {}).get("own_tree"), "not_run": rec.get("not_run"),
            "failed": sorted(k for k, v in (rec.get("outcomes") or {}).items() if v != "passed")[:40]}


def fmt_counts(c):
    return ", ".join(f"{v} {k}" for k, v in sorted(c.items())) or "nothing ran"


def check_fp8_quick(run: Run):
    n = 19665
    t1 = "Row-wise vs tensorwise fp8 torch._scaled_mm on this torch build"
    reason = run.need_nvidia(min_cap = (8, 9))
    if run.args.dry_run:
        reason = reason or "dry run"
    if reason:
        run.add("fp8.kernels", n, t1, NOT_RUN, reason)
        run.add("fp8.smoke_probe", n, "Studio FP8 smoke probe: installed release, merge base, PR head", NOT_RUN,
                reason)
    else:
        k, _ = run.probe("fp8_kernels", "fp8_probe.py", ["kernels"], run.studio["backend"])
        ks = (k or {}).get("kernels") or {}
        if not ks:
            run.add("fp8.kernels", n, t1, ERROR, "probe failed (see logs)", {"probe": k})
            missing = None
        else:
            missing = bool(k.get("rowwise_missing"))
            parts = [f"{name} {'ok' if v.get('ok') else 'FAILS: ' + v.get('error', '?')[:120]}"
                     for name, v in ks.items()]
            run.add("fp8.kernels", n, t1, INFO,
                    ("row-wise kernel MISSING (the #19665 case). " if missing else "row-wise kernel present. ")
                    + "; ".join(parts), {"kernels": ks})
        smoke_quick(run, n, missing)
    run_tests(run, n, "fp8.tests")


def smoke_quick(run: Run, n, missing):
    title = "Studio FP8 smoke probe: installed release, merge base, PR head"
    try:
        head, base = run.trees_for(n)
    except Exception as exc:  # noqa: BLE001
        return run.add("fp8.smoke_probe", n, title, ERROR, f"could not fetch the PR code: {exc}")
    reason, _ = run.deps_gate(n, head, f"pr{n}_head")
    if reason:
        return run.add("fp8.smoke_probe", n, title, gate_status(reason), reason)
    arms = {}
    for label, backend in (("installed", run.studio["backend"]), ("base", base), ("head", head)):
        rec, _ = run.probe(f"fp8_smoke_{label}", "fp8_probe.py", ["smoke"], backend)
        arms[label] = rec or {}
    details = {k: {f: v.get(f) for f in ("smoke_probe", "smoke_error", "linear_ok", "linear_error", "gemm_path",
                                         "scaled_mm_calls", "unit_scale_stats", "fatal", "tree")}
               for k, v in arms.items()}
    h = arms["head"]
    path = (h.get("gemm_path") or {}).get("path")
    s = {k: v.get("smoke_probe") for k, v in arms.items()}
    summ = (f"smoke probe installed={s['installed']} base={s['base']} head={s['head']}; head gemm_path={path}; "
            f"head _scaled_mm calls {h.get('scaled_mm_calls')}")
    if h.get("fatal"):
        return run.add("fp8.smoke_probe", n, title, ERROR, "head probe crashed: " + h["fatal"], details)
    if missing:
        ok = h.get("smoke_probe") is True and h.get("linear_ok") and path == "unit_scale_fallback"
        before = s["installed"] is False or s["base"] is False
        summ += "; defect reproduced on the release / base" if before else "; release / base did NOT fail here"
        return run.add("fp8.smoke_probe", n, title, PASS if ok else FAIL, summ, details)
    if missing is False:
        ok = h.get("smoke_probe") is True and path in ("torchao_rowwise", "unit_scale_or_rowwise")
        summ += "; row-wise kernel present, so this is the no-regression case"
        return run.add("fp8.smoke_probe", n, title, PASS if ok else FAIL, summ, details)
    return run.add("fp8.smoke_probe", n, title, INFO, summ, details)


def check_w6a8(run: Run):
    n = 19588
    title = "sm120 fused W6A8 tile: check_tile and fused calls, merge base vs head"
    reason = run.need_nvidia(caps = [(12, 0)])
    if run.args.dry_run:
        reason = reason or "dry run"
    if reason:
        run.add("w6a8.check_tile", n, title, NOT_RUN, reason)
    else:
        try:
            head, base = run.trees_for(n)
            dreason, _ = run.deps_gate(n, head, f"pr{n}_head")
        except Exception as exc:  # noqa: BLE001
            dreason = f"could not fetch the PR code: {exc}"
        if dreason:
            run.add("w6a8.check_tile", n, title, gate_status(dreason), dreason)
        else:
            arms = {}
            for label, backend in (("base", base), ("head", head)):
                rec, _ = run.probe(f"w6a8_{label}", "w6a8_probe.py", [], backend, timeout = 900)
                arms[label] = rec or {}
            h, b = arms["head"], arms["base"]
            details = {k: {f: v.get(f) for f in ("triton", "block_ptr_used", "block_ptr_removed", "arch_tiles",
                                                 "check_tile", "fused_config", "fused_call_count_delta",
                                                 "packed_calls", "kernel_state_error", "not_run", "fatal")}
                       for k, v in arms.items()}
            summ = (f"Triton {h.get('triton')}; head check_tile ok={h.get('check_tile_ok')} "
                    f"(block pointers {'on' if h.get('block_ptr_used') else 'off'}), fused calls "
                    f"{h.get('fused_call_count_delta')}; base check_tile ok={b.get('check_tile_ok')}")
            if h.get("not_run"):
                run.add("w6a8.check_tile", n, title, NOT_RUN, h["not_run"], details)
            elif h.get("fatal"):
                run.add("w6a8.check_tile", n, title, ERROR, "head probe crashed: " + h["fatal"], details)
            else:
                bit = all(c.get("bit_equal", True) for c in h.get("packed_calls") or [])
                ok = h.get("check_tile_ok") and (h.get("fused_call_count_delta") or 0) > 0 and bit
                run.add("w6a8.check_tile", n, title, PASS if ok else FAIL, summ, details)
    if not (run.env.get("triton_import") or run.args.dry_run):
        run.add("w6a8.tests", n, "PR unit tests (Triton interpreter)", NOT_RUN,
                "Triton is not importable in the Studio venv: " + str(run.env.get("triton_import_error")))
    else:
        run_tests(run, n, "w6a8.tests")


# ---- full mode


def confirm_downloads(run: Run, what: str, spec) -> bool:
    spec_path = run.out / "scratch" / f"sizes_{safe(what)}.json"
    spec_path.write_text(json.dumps(spec), encoding = "utf-8")
    rec, _ = run.probe(f"sizes_{safe(what)}", "hf_sizes.py", ["--spec", str(spec_path)], run.studio["backend"],
                       timeout = 300)
    if not rec or rec.get("fatal"):
        log(f"could not size the downloads for {what}: {(rec or {}).get('fatal')}")
        return run.ask(f"{what}: download size unknown. Download models from the Hugging Face Hub anyway?")
    total, cached = rec["total_bytes"], rec["cached_bytes"]
    for r in rec["repos"]:
        log(f"  {r['repo']}: {r['bytes'] / 1e9:.2f} GB in {r['files']} files, {r['cached'] / 1e9:.2f} GB cached"
            + (f" ({r['error']})" if r.get("error") else ""))
    need = max(0, total - cached)
    if need < 50e6:
        return True
    return run.ask(f"{what}: up to {need / 1e9:.1f} GB to download into {run.hub_cache or 'the Hugging Face cache'}"
                   f" ({cached / 1e9:.1f} GB of {total / 1e9:.1f} GB already there). Continue?")


def render(run: Run, label, backend, tq, speed = None, gens = "base:-", n = 2, warmup = 0, load_lora = None,
           home_key = None, extra_env = None):
    args = ["--label", label, "--imgdir", str(run.out / "images"), "--tq", tq, "--gens", gens, "--n", n,
            "--steps", run.args.steps, "--size", run.args.size, "--warmup", warmup]
    if speed:
        args += ["--speed", speed]
    if load_lora is not None:
        args += ["--load-lora", load_lora]
    rec, _ = run.probe(f"render_{label}", "render_arm.py", args, backend, home_key = home_key or label,
                       extra_env = extra_env, timeout = run.args.arm_timeout)
    return rec or {"fatal": "the arm wrote no result"}


def score(run: Run, name, pairs):
    p = run.out / "scratch" / f"pairs_{name}.json"
    p.write_text(json.dumps(pairs), encoding = "utf-8")
    rec, _ = run.probe(f"score_{name}", "score_images.py", ["--imgdir", str(run.out / "images"), "--pairs", str(p)],
                       run.studio["backend"], timeout = 900)
    return (rec or {}).get("pairs") or {}, (rec or {}).get("lpips_unavailable")


def per_image_s(rec, gen = "base"):
    r = (rec.get("renders") or {}).get(gen) or {}
    s = r.get("s") or []
    return statistics.mean(s) if s else None


def check_fp8_full(run: Run):
    n = 19665
    title = "FLUX.2 klein 4B FP8 renders, eager and compiled, vs BF16 at the same seeds (PR head)"
    reason = run.need_nvidia(min_cap = (8, 9))
    if reason:
        return run.add("fp8.render", n, title, NOT_RUN, reason)
    try:
        head, base = run.trees_for(n)
    except Exception as exc:  # noqa: BLE001
        return run.add("fp8.render", n, title, ERROR, f"could not fetch the PR code: {exc}")
    dreason, _ = run.deps_gate(n, head, f"pr{n}_head")
    if dreason:
        return run.add("fp8.render", n, title, gate_status(dreason), dreason)
    spec = [{"repo": KLEIN_MIRROR, "patterns": KLEIN_PIPE, "cache_aliases": [KLEIN]},
            {"repo": KLEIN_FP8, "patterns": ["FLUX.2-klein-4B-FP8.safetensors",
                                             "FLUX.2-klein-4B-text_encoder-FP8.safetensors"]}]
    if not confirm_downloads(run, "FP8 renders", spec):
        return run.add("fp8.render", n, title, NOT_RUN, "model download not confirmed (rerun with --yes)")
    have_lpips = run.ensure_lpips()
    reps = run.args.repeats
    arms = {}
    plan = []
    modes = [("eager", "eager"), ("comp", "default")]
    modes = [m for m in modes if m[0] in run.args.fp8_modes]
    for r in range(1, reps + 1):  # interleaved: every arm once per round, a fresh process each
        for mode, speed in modes:
            plan += [(f"bf16_{mode}_r{r}", "none", speed), (f"fp8_{mode}_r{r}", "fp8", speed)]
    plan.insert(2, ("base_fp8_eager", "fp8", "eager"))
    for label, tq, speed in plan:
        backend = base if label.startswith("base_") else head
        key = label.rsplit("_r", 1)[0]
        arms[label] = render(run, label, backend, tq, speed, n = run.args.images, warmup = 1, home_key = key,
                             extra_env = {"UNSLOTH_DIFFUSION_BG_COMPILE": "0"})
    pairs = []
    for mode, _speed in modes:
        pairs.append({"name": f"fp8_vs_bf16_{mode}", "a": [f"fp8_{mode}_r1", "base"], "b": [f"bf16_{mode}_r1", "base"]})
        for r in range(2, reps + 1):
            for q in ("fp8", "bf16"):
                pairs.append({"name": f"aa_{q}_{mode}_r{r}", "a": [f"{q}_{mode}_r{r}", "base"],
                              "b": [f"{q}_{mode}_r1", "base"]})
    scores, lp_err = score(run, "fp8", pairs)
    timing = {}
    for key in [f"{q}_{m}" for m, _speed in modes for q in ("bf16", "fp8")]:
        vals = [per_image_s(arms.get(f"{key}_r{r}", {})) for r in range(1, reps + 1)]
        med, spread = median_spread(vals)
        timing[key] = {"s_per_image_median": med, "s_per_step_median": round(med / run.args.steps, 4) if med else None,
                       "repeat_spread_pct": spread, "per_repeat": vals}
    errors = {k: v.get("fatal") or next((g.get("error") for g in (v.get("renders") or {}).values()
                                         if g.get("error")), None) for k, v in arms.items()}
    errors = {k: v for k, v in errors.items() if v}
    gemm = ((arms.get("fp8_eager_r1") or {}).get("status") or {}).get("fp8_gemm")
    gemm_c = ((arms.get("fp8_comp_r1") or {}).get("status") or {}).get("fp8_gemm")
    graphs = (arms.get("fp8_comp_r1") or {}).get("dynamo_unique_graphs")
    details = {"timing": timing, "scores": strip_per_image(scores),
               "lpips_unavailable": lp_err if not have_lpips or lp_err else None,
               "fp8_gemm_eager": gemm, "fp8_gemm_compiled": gemm_c, "compiled_unique_graphs": graphs,
               "errors": errors, "base_fp8_eager": errors.get("base_fp8_eager") or "rendered",
               "load_s": {k: v.get("load_s") for k, v in arms.items()}}
    head_err = {k: v for k, v in errors.items() if not k.startswith("base_")}
    e = scores.get("fp8_vs_bf16_eager") or {}
    c = scores.get("fp8_vs_bf16_comp") or {}
    parts = [f"fp8_gemm={(gemm or {}).get('path')}"]
    for mode, sc in (("eager", e), ("comp", c)):
        if f"fp8_{mode}" not in timing:
            continue
        name = "compiled" if mode == "comp" else "eager"
        parts.append(f"{name}: LPIPS vs BF16 {sc.get('lpips_mean')} mean / {sc.get('lpips_max')} max, PSNR min "
                     f"{sc.get('psnr_min')}; s/step fp8 {timing[f'fp8_{mode}']['s_per_step_median']} vs bf16 "
                     f"{timing[f'bf16_{mode}']['s_per_step_median']} (A/A spread over {reps} repeats "
                     f"{timing[f'fp8_{mode}']['repeat_spread_pct']}%)")
    base_err = errors.get("base_fp8_eager")
    parts.append("merge base fp8: " + ("FAILED: " + base_err[:160] if base_err else "rendered"))
    summ = "; ".join(parts)
    if head_err:
        return run.add("fp8.render", n, title, FAIL, summ + "; head errors: " + "; ".join(
            f"{k}: {v[:200]}" for k, v in head_err.items()), details)
    ok = gemm is not None and (e.get("lpips_mean") is None or e["lpips_mean"] < 0.25)
    return run.add("fp8.render", n, title, PASS if ok else FAIL, summ, details)


def strip_per_image(scores):
    return {k: {f: v for f, v in s.items() if f != "per_image"} for k, s in scores.items()}


def check_nvfp4_full(run: Run):
    n = 19664
    t_cache = "NVFP4 klein 4B: first load vs later loads with UNSLOTH_NVFP4_DIFFUSION=1 (prequant cache)"
    t_lora = "NVFP4 LoRA bake: strength 0 renders bit-identical to the NVFP4 base"
    reason = run.need_nvidia(min_cap = (10, 0))
    if reason:
        run.add("nvfp4.cache_timing", n, t_cache, NOT_RUN, reason)
        return run.add("nvfp4.lora_s0", n, t_lora, NOT_RUN, reason)
    try:
        head, base = run.trees_for(n)
    except Exception as exc:  # noqa: BLE001
        run.add("nvfp4.cache_timing", n, t_cache, ERROR, f"could not fetch the PR code: {exc}")
        return None
    dreason, _ = run.deps_gate(n, head, f"pr{n}_head")
    if dreason:
        run.add("nvfp4.cache_timing", n, t_cache, gate_status(dreason), dreason)
        return run.add("nvfp4.lora_s0", n, t_lora, gate_status(dreason), dreason)
    spec = [{"repo": KLEIN_MIRROR, "patterns": KLEIN_PIPE, "cache_aliases": [KLEIN]},
            {"repo": KLEIN_LORA, "patterns": ["*.safetensors", "*.json"]}]
    if not confirm_downloads(run, "NVFP4 loads", spec):
        run.add("nvfp4.cache_timing", n, t_cache, NOT_RUN, "model download not confirmed (rerun with --yes)")
        return run.add("nvfp4.lora_s0", n, t_lora, NOT_RUN, "model download not confirmed")
    pqc = {t: run.out / "scratch" / f"pqc_{t}" for t in ("head", "base")}
    for d in pqc.values():
        if d.exists():
            shutil.rmtree(d)
    reps = run.args.repeats

    def env(t):
        return {"UNSLOTH_NVFP4_DIFFUSION": "1", "UNSLOTH_DIFFUSION_PREQUANT_CACHE_DIR": str(pqc[t])}

    arms = {}
    plan = [("base_first", "base"), ("head_first", "head")]
    for r in range(1, reps + 1):
        plan += [(f"base_again_r{r}", "base"), (f"head_hit_r{r}", "head")]
    for label, t in plan:
        arms[label] = render(run, label, base if t == "base" else head, "nvfp4", n = run.args.images,
                             home_key = f"nv_{t}", extra_env = env(t))
    lora = render(run, "head_lora", head, "nvfp4", gens = "s0:0,lora1:1.0", n = run.args.images, load_lora = 1.0,
                  home_key = "nv_head", extra_env = env("head"))

    def tsum(rec, key):
        v = (rec.get("timings") or {}).get(key)
        return round(sum(v), 2) if v else None

    rows = {k: {"load_s": v.get("load_s"), "quantize_s": tsum(v, "quantize_transformer"),
                "cache_read_s": tsum(v, "pipeline_quant_from_cache"), "error": v.get("fatal"),
                "transformer_quant": (v.get("status") or {}).get("transformer_quant")} for k, v in arms.items()}
    hit_med, hit_spread = median_spread([rows[f"head_hit_r{r}"]["load_s"] for r in range(1, reps + 1)])
    base_med, base_spread = median_spread([rows[f"base_again_r{r}"]["load_s"] for r in range(1, reps + 1)])
    details = {"arms": rows, "head_later_load_median_s": hit_med, "head_later_spread_pct": hit_spread,
               "base_later_load_median_s": base_med, "base_later_spread_pct": base_spread,
               "nvidia_reported_cached_model_load_s": 59.5}
    errs = {k: r["error"] for k, r in rows.items() if r["error"]}
    summ = (f"first load base {rows['base_first']['load_s']} s, head {rows['head_first']['load_s']} s; later loads "
            f"base {base_med} s (spread {base_spread}%), head {hit_med} s (spread {hit_spread}%); head cache read "
            f"{rows['head_hit_r1']['cache_read_s']} s vs base quantize {rows['base_again_r1']['quantize_s']} s "
            f"(NVIDIA reported 59.5 s for a cached-model NVFP4 load)")
    if errs:
        run.add("nvfp4.cache_timing", n, t_cache, FAIL if any(k.startswith("head") for k in errs) else INFO,
                summ + "; errors: " + "; ".join(f"{k}: {v[:200]}" for k, v in errs.items()), details)
    else:
        hit = all(rows[f"head_hit_r{r}"]["cache_read_s"] for r in range(1, reps + 1))
        faster = hit_med is not None and base_med is not None and hit_med < base_med
        run.add("nvfp4.cache_timing", n, t_cache, PASS if hit and faster else FAIL, summ, details)
    pairs = [{"name": f"s0_vs_hit_r{r}", "a": ["head_lora", "s0"], "b": [f"head_hit_r{r}", "base"]}
             for r in range(1, reps + 1)]
    pairs.append({"name": "s0_vs_head_first", "a": ["head_lora", "s0"], "b": ["head_first", "base"]})
    pairs.append({"name": "lora_effect", "a": ["head_lora", "lora1"], "b": ["head_lora", "s0"]})
    pairs.append({"name": "aa_hit_r2_vs_r1", "a": ["head_hit_r2", "base"], "b": ["head_hit_r1", "base"]})
    scores, _ = score(run, "nvfp4", pairs)
    st = lora.get("status") or {}
    ld = {"scores": strip_per_image(scores), "lora_status": {k: st.get(k) for k in ("lora_unavailable_reason",
                                                                                     "baked_loras", "supports_lora")},
          "error": lora.get("fatal") or next((g.get("error") for g in (lora.get("renders") or {}).values()
                                              if g.get("error")), None)}
    if ld["error"]:
        return run.add("nvfp4.lora_s0", n, t_lora, FAIL, "LoRA arm failed: " + ld["error"][:300], ld)
    s0 = [v for k, v in scores.items() if k.startswith("s0_vs")]
    ident = any(v.get("bit_equal_all") for v in s0)
    eff = (scores.get("lora_effect") or {})
    s0_list = ", ".join("%s=%s" % (k, v.get("bit_equal_all")) for k, v in scores.items() if k.startswith("s0_vs"))
    summ = (f"strength 0 bit-identical to an NVFP4 base render: {ident} ({s0_list});"
            f" LoRA effect at strength 1: mean abs {eff.get('mean_abs_mean')}, LPIPS {eff.get('lpips_mean')}; "
            f"lora_unavailable_reason={st.get('lora_unavailable_reason')}")
    return run.add("nvfp4.lora_s0", n, t_lora, PASS if ident else FAIL, summ, ld)


def check_video_live(run: Run):
    n = 19621
    title = "Hosted video download: Studio's progress counter vs bytes on disk (scratch cache, deleted after)"
    reason = run.need_nvidia()
    if reason:
        return run.add("video_dl.live", n, title, NOT_RUN, reason)
    try:
        head, base = run.trees_for(n)
    except Exception as exc:  # noqa: BLE001
        return run.add("video_dl.live", n, title, ERROR, f"could not fetch the PR code: {exc}")
    secs = run.args.video_seconds
    if not run.ask(f"Video download sample: about {secs} s of real download per arm (2 arms) into a scratch cache "
                   "that is deleted afterwards. Continue?"):
        return run.add("video_dl.live", n, title, NOT_RUN, "not confirmed (rerun with --yes)")
    res = {}
    for label, backend in (("head", head), ("base", base)):
        cache = run.out / "scratch" / f"video_hf_{label}"
        rec, _ = run.probe(f"video_dl_{label}", "video_dl_probe.py", ["--seconds", secs], backend,
                           extra_env = {"HF_HUB_CACHE": str(cache), "HF_HOME": str(cache.parent / f"hf_home_{label}"),
                                        "HF_XET_CACHE": str(cache.parent / f"xet_{label}")},
                           timeout = secs + 600)
        res[label] = rec or {}
        for d in (cache, cache.parent / f"hf_home_{label}", cache.parent / f"xet_{label}"):
            shutil.rmtree(d, ignore_errors = True)

    def last(rec):
        s = [x for x in rec.get("samples") or [] if x.get("on_disk")]
        return s[-1] if s else {}

    h, b = last(res["head"]), last(res["base"])
    details = {k: {"samples": v.get("samples"), "error": v.get("fatal") or v.get("begin_load_error")}
               for k, v in res.items()}
    summ = (f"head: counter {h.get('downloaded')} of {h.get('expected')} with {h.get('on_disk')} bytes on disk; "
            f"base: counter {b.get('downloaded')} of {b.get('expected')} with {b.get('on_disk')} on disk")
    if not h:
        return run.add("video_dl.live", n, title, NOT_RUN, "no bytes arrived during the sample: " +
                       str(details["head"]["error"]), details)
    ok = (h.get("downloaded") or 0) > 0
    return run.add("video_dl.live", n, title, PASS if ok else FAIL, summ, details)


# ---------------------------------------------------------------- report


def write_summary(run: Run):
    order = {FAIL: 0, ERROR: 1, PASS: 2, INFO: 3, NOT_RUN: 4}
    counts = {}
    for r in run.results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    machine = {k: run.env.get(k) for k in ("system", "machine", "platform", "python", "gpu", "capability",
                                           "vram_total_gb", "torch", "torch_cuda", "torch_hip", "triton_import")}
    machine["dists"] = run.env.get("dists")
    doc = {"tool": "studio_device_checks", "tool_version": TOOL_VERSION, "mode": run.args.mode,
           "dry_run": run.args.dry_run, "started": run.started, "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "host_os": f"{platform.system()} {platform.machine()}", "machine": machine,
           "studio": run.studio, "prs": run.prs, "counts": counts, "checks": run.results,
           "results_dir": str(run.out)}
    (run.out / "summary.json").write_text(json.dumps(doc, indent = 1, default = str), encoding = "utf-8")
    lines = [f"# Studio device checks ({run.args.mode}{', dry run' if run.args.dry_run else ''})", "",
             f"- Machine: {describe_machine(run.env) if run.env else 'unknown'}",
             f"- Studio Python: {(run.studio or {}).get('python')}",
             f"- Installed backend: {(run.studio or {}).get('backend')}",
             f"- Started {run.started}, results in `{run.out}`",
             f"- Counts: {', '.join(f'{k} {v}' for k, v in sorted(counts.items()))}", ""]
    for n, p in sorted(run.prs.items()):
        lines.append(f"- PR #{n}: head {p.get('head_sha') or p.get('head_ref')}, "
                     f"base {p.get('base_sha') or p.get('base_ref')}"
                     + (f" ({p['note']})" if p.get("note") else ""))
    lines += ["", "| Status | Check | PR | Result |", "|---|---|---|---|"]
    for r in sorted(run.results, key = lambda r: (order.get(r["status"], 9), r["id"])):
        summ = str(r["summary"]).replace("|", "/").replace("\n", " ")
        lines.append(f"| {r['status']} | {r['id']} | {('#' + str(r['pr'])) if r['pr'] else ''} | {summ} |")
    lines += ["", "Statuses: PASS = the PR behaves as claimed here; FAIL = it does not (details in summary.json); "
              "INFO = an observation with no pass bar; NOT_RUN = this machine cannot answer it (reason given); "
              "ERROR = the harness itself failed.", ""]
    (run.out / "summary.md").write_text("\n".join(lines), encoding = "utf-8")
    return doc


def parse_args(argv):
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices = ("quick", "full"), default = "quick")
    ap.add_argument("--dry-run", action = "store_true", help = "discovery and the plan only; downloads nothing")
    ap.add_argument("--studio-home", help = "the Studio home (holds unsloth_studio/); default as the installers")
    ap.add_argument("--studio-python", help = "use this Python directly instead of discovering it")
    ap.add_argument("--prs", default = ",".join(str(n) for n in PRS) + ",19569",
                    help = "comma list of PR numbers to check (default: all)")
    ap.add_argument("--work-dir", help = "cache and results root (default ~/unsloth_device_checks)")
    ap.add_argument("--out", help = "results folder (default <work-dir>/run_<mode>_<time>)")
    ap.add_argument("--yes", action = "store_true", help = "answer yes to download questions")
    ap.add_argument("--repeats", type = int, default = 3)
    ap.add_argument("--images", type = int, default = 2, help = "timed images per render arm")
    ap.add_argument("--steps", type = int, default = 4)
    ap.add_argument("--size", type = int, default = 1024)
    ap.add_argument("--arm-timeout", type = int, default = 3600)
    ap.add_argument("--fp8-modes", default = "eager,comp", help = "FP8 render arms: eager, comp (compiled)")
    ap.add_argument("--video-seconds", type = int, default = 90)
    ap.add_argument("--skip-video-download", action = "store_true")
    ap.add_argument("--tests-on-gpu", action = "store_true", help = "let unit tests see the GPU (default: CPU only)")
    ap.add_argument("--hf-hub-cache", help = "Hugging Face hub cache for model loads (default: Studio's own)")
    ap.add_argument("--tree", action = "append", default = [], metavar = "PR=DIR",
                    help = "use a local checkout as the PR head instead of downloading (testing)")
    ap.add_argument("--base-tree", action = "append", default = [], metavar = "PR=DIR")
    a = ap.parse_args(argv)
    a.tree = {int(k): v for k, v in (t.split("=", 1) for t in a.tree)}
    a.base_tree = {int(k): v for k, v in (t.split("=", 1) for t in a.base_tree)}
    a.prs = [int(x) for x in a.prs.split(",") if x.strip()]
    a.fp8_modes = [m.strip() for m in a.fp8_modes.split(",") if m.strip()]
    return a


def main(argv = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    run = Run(args)
    log(f"results folder: {run.out}")
    try:
        if not run.setup():
            write_summary(run)
            print((run.out / "summary.md").read_text(encoding = "utf-8"))
            return 2
        log("machine: " + describe_machine(run.env))
        if 19569 in args.prs:
            check_sanity(run)
        if args.dry_run:
            for n in [p for p in args.prs if p in PRS]:
                info = run.pr_info(n)
                if not info.get("local"):
                    for side in ("head_ref", "base_ref"):
                        u = zip_url(info[side])
                        info[side.replace("_ref", "_zip_reachable")] = url_ok(u)
                        log(f"PR #{n} {side}: {u} reachable={info[side.replace('_ref', '_zip_reachable')]}")
        if 19665 in args.prs:
            check_fp8_quick(run)
        if 19664 in args.prs:
            run_tests(run, 19664, "nvfp4.tests")
        if 19588 in args.prs:
            check_w6a8(run)
        if 19621 in args.prs:
            run_tests(run, 19621, "video_dl.tests")
        if args.mode == "full":
            full = [(19665, "fp8.render", check_fp8_full), (19664, "nvfp4.cache_timing", check_nvfp4_full)]
            if not args.skip_video_download:
                full.append((19621, "video_dl.live", check_video_live))
            for n, cid, fn in full:
                if n not in args.prs:
                    continue
                if args.dry_run:
                    run.add(cid, n, PRS[n]["title"], NOT_RUN, "dry run")
                else:
                    fn(run)
    except KeyboardInterrupt:
        run.add("interrupted", None, "Run interrupted", ERROR, "stopped by the user; partial results kept")
    except Exception as exc:  # noqa: BLE001
        import traceback

        run.add("harness", None, "Harness error", ERROR, f"{type(exc).__name__}: {exc}",
                {"tb": traceback.format_exc()[-3000:]})
    write_summary(run)
    print()
    print((run.out / "summary.md").read_text(encoding = "utf-8"))
    print(f"Paste back or upload: {run.out / 'summary.md'} and {run.out / 'summary.json'}")
    return 1 if any(r["status"] in (FAIL, ERROR) for r in run.results) else 0


if __name__ == "__main__":
    sys.exit(main())
