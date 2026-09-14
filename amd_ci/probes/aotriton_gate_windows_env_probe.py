#!/usr/bin/env python3
"""Observe PR 8821's AOTriton gate on WINDOWS: placement, override, inheritance.

PR 8821 puts `os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")`
at the top of `unsloth/__init__.py` and of `studio/backend/main.py`, and deletes the
`/etc/profile.d/unsloth-rocm-wsl.sh` export the WSL installer used to write.

Torch reads that variable lazily, into a function-local `static const bool` inside
`check_flash_attention_hardware_support` / `check_mem_efficient_hardware_support`, and
only inside `#if USE_ROCM`. That half needs a ROCm torch and is NOT what this probe
measures; whether these Windows boxes carry one is unmeasured, so the criteria declares
it as a gap rather than pretending to answer it.

What IS fully answerable on Windows without torch, and is what this probe records:

  * the value of the gate after importing each entry point, from three different
    STARTING states -- unset, an explicit "0", an explicit "1" -- because `setdefault`
    is only correct if an explicit "0" survives it
  * whether that value reaches a SPAWNED worker. Windows has no fork, so a value set in
    the parent's `os.environ` only helps a worker if the process environment block
    CreateProcess copies was updated too. `studio/backend/core/training/training.py`
    runs every job under `mp.get_context("spawn")`, so this probe spawns through exactly
    that mechanism, plus the two `subprocess` shapes Studio uses elsewhere, plus a
    deliberately scrubbed child as a control
  * how far each entry point's module body actually executed, read off the traceback, so
    "the gate was set" is not confused with "the import succeeded"
  * whether anything ELSE on this platform supplies the variable: the machine and user
    registry scopes, `/etc/profile.d` (which does not exist on Windows), and every file
    in the checkout that names the variable at all

It observes and never judges. Nothing here decides whether a reading is good.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

GATE = "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"

# Entry points, and the source file whose line numbers a traceback is measured against.
ENTRIES = {
    "unsloth": "unsloth/__init__.py",
    "studio_main": "studio/backend/main.py",
}
STARTS = ("unset", "0", "1")

# Run as a FILE, never `python -c`: multiprocessing's spawn start method re-imports the
# child's `__main__`, and with `-c` there is no module for it to re-import.
_INNER = r'''#!/usr/bin/env python3
"""Inner half of the Windows AOTriton gate probe. One entry point, one starting value."""

import argparse
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys

GATE = "GATE_NAME_PLACEHOLDER"


def _child_report(queue):
    """Body of the spawned worker. It reads the environment it was handed, nothing else."""
    queue.put({"gate": os.environ.get(GATE), "pid": os.getpid()})


def _deepest_line_in(exc, suffix):
    """Deepest line reached inside the file whose path ends with `suffix`.

    This is what separates "the module body ran past the gate and then hit a missing
    third-party dependency" from "the file never executed".
    """
    tb, deepest = exc.__traceback__, None
    while tb is not None:
        name = tb.tb_frame.f_code.co_filename.replace("\\", "/")
        if name.endswith(suffix):
            deepest = tb.tb_lineno if deepest is None else max(deepest, tb.tb_lineno)
        tb = tb.tb_next
    return deepest


def _run_child(argv, env = None):
    code = ("import os, json, sys\n"
            "sys.stdout.write(json.dumps({'gate': os.environ.get(%r)}))\n" % GATE)
    try:
        r = subprocess.run([sys.executable, "-c", code] + list(argv),
                           capture_output = True, text = True, encoding = "utf-8",
                           env = env, timeout = 300)
    except Exception as e:  # noqa: BLE001
        return {"error": "%s: %s" % (type(e).__name__, e)}
    out = {"rc": r.returncode, "stderr_tail": (r.stderr or "")[-400:]}
    try:
        out.update(json.loads((r.stdout or "").strip().splitlines()[-1]))
    except Exception as e:  # noqa: BLE001
        out["parse_error"] = "%s: %s" % (type(e).__name__, e)
        out["stdout_tail"] = (r.stdout or "")[-400:]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", required = True)
    ap.add_argument("--rel", required = True, help = "source file the traceback is measured against")
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True)
    args = ap.parse_args()

    rec = {
        "entry": args.entry,
        "pid": os.getpid(),
        "executable": sys.executable,
        "python": platform.python_version(),
        "sys_platform": sys.platform,
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "machine": platform.machine(),
        "computername": os.environ.get("COMPUTERNAME"),
        "gate_before_import": os.environ.get(GATE),
    }

    if args.entry == "unsloth":
        sys.path.insert(0, args.checkout)
        module = "unsloth"
    else:
        # main.py imports its siblings top-level (`from utils...`), so studio/backend is
        # the import root, which is also how run.py and `uvicorn main:app` launch it.
        sys.path.insert(0, os.path.join(args.checkout, "studio", "backend"))
        module = "main"

    try:
        __import__(module)
        rec["import_ok"] = True
    except BaseException as e:  # noqa: BLE001  -- SystemExit is a real outcome here
        rec["import_ok"] = False
        rec["import_error"] = "%s: %s" % (type(e).__name__, str(e)[:400])
        rec["import_error_type"] = type(e).__name__
        rec["deepest_line_in_entry_file"] = _deepest_line_in(e, args.rel)

    rec["gate_after_import"] = os.environ.get(GATE)
    # os.environ is a Python-side mapping; os.getenv reads the same mapping, while a
    # child process is handed the C-level block. Recording both makes a divergence
    # visible instead of assumed.
    rec["getenv_after_import"] = os.getenv(GATE)

    # 1. The mechanism Studio's training jobs really use: training.py holds
    #    `_CTX = mp.get_context("spawn")` and runs every job in one of its Processes.
    try:
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        proc = ctx.Process(target = _child_report, args = (queue,))
        proc.start()
        rec["mp_spawn_child"] = queue.get(timeout = 300)
        proc.join(120)
        rec["mp_spawn_child"]["exitcode"] = proc.exitcode
        rec["mp_start_method"] = ctx.get_start_method()
    except Exception as e:  # noqa: BLE001
        rec["mp_spawn_error"] = "%s: %s" % (type(e).__name__, e)

    # 2. subprocess with no env= at all: the shape sd_cpp_engine.py and the llama server
    #    launcher use, which inherits the block CreateProcess copies.
    rec["popen_inherit_child"] = _run_child([])

    # 3. subprocess handed an explicit copy of os.environ, the other Studio shape.
    rec["popen_env_copy_child"] = _run_child([], env = os.environ.copy())

    # 4. CONTROL: the same reader, handed an environment with the variable removed. If
    #    this one reports a value, the reader is not reading what it claims to.
    rec["popen_scrubbed_child"] = _run_child(
        [], env = {k: v for k, v in os.environ.items() if k != GATE})

    with open(args.out, "w", encoding = "utf-8") as fh:
        json.dump(rec, fh, indent = 2, sort_keys = True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def _registry_scope(scope: str):
    """The gate as persisted in a Windows environment scope, or None off Windows."""
    if platform.system() != "Windows":
        return None
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"[Environment]::GetEnvironmentVariable('{GATE}','{scope}')"],
            capture_output = True, text = True, encoding = "utf-8", timeout = 120)
    except Exception as e:  # noqa: BLE001
        return f"error: {type(e).__name__}: {e}"
    value = (r.stdout or "").strip()
    return value or None


def _files_naming_gate(checkout: Path) -> list:
    """Every file on an install or launch path that names the variable at all.

    This is the "was the deleted /etc/profile.d export the only thing setting it here"
    question, asked of the tree rather than of memory.
    """
    named = [
        "unsloth/__init__.py", "studio/backend/main.py", "studio/backend/run.py",
        "install.ps1", "install.sh", "scripts/install_rocm_wsl_strixhalo.sh",
        "scripts/uninstall.ps1", "scripts/uninstall.sh",
        "docker/entrypoint.sh", "docker/studio_launch.sh", "docker/run.sh",
        "studio/backend/core/training/training.py",
        "studio/backend/core/training/worker.py",
    ]
    found = []
    for rel in named:
        path = checkout / rel
        if not path.is_file():
            found.append({"file": rel, "exists": False})
            continue
        try:
            text = path.read_text(encoding = "utf-8", errors = "replace")
        except Exception as e:  # noqa: BLE001
            found.append({"file": rel, "read_error": f"{type(e).__name__}: {e}"})
            continue
        found.append({"file": rel, "exists": True, "names_gate": GATE in text})
    # Every Windows-side script in the tree, since those are the only ones that could
    # persist the variable on this platform.
    scripts = []
    for pattern in ("*.ps1", "*.bat", "*.cmd"):
        for path in sorted(checkout.rglob(pattern)):
            if any(part in {".git", "node_modules", "frontend"} for part in path.parts):
                continue
            try:
                if GATE in path.read_text(encoding = "utf-8", errors = "replace"):
                    scripts.append(str(path.relative_to(checkout)).replace("\\", "/"))
            except Exception:  # noqa: BLE001
                continue
    return [found, scripts]


def _torch_facts(python: str) -> dict:
    """Whether the test environment has a torch at all, and whether it is a ROCm one.

    The kernel-selection half of this PR is only reachable through a ROCm torch. Recording
    the absence explicitly is what lets the verdict say UNMEASURABLE instead of silently
    reporting the half it could measure as the whole answer.
    """
    code = ("import json, sys\n"
            "out = {}\n"
            "try:\n"
            "    import torch\n"
            "    out['version'] = torch.__version__\n"
            "    out['hip'] = getattr(torch.version, 'hip', None)\n"
            "    out['cuda'] = getattr(torch.version, 'cuda', None)\n"
            "    out['cuda_available'] = bool(torch.cuda.is_available())\n"
            "except Exception as e:\n"
            "    out['error'] = '%s: %s' % (type(e).__name__, e)\n"
            "sys.stdout.write(json.dumps(out))\n")
    try:
        r = subprocess.run([python, "-c", code], capture_output = True, text = True,
                           encoding = "utf-8", errors = "replace", timeout = 600)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    try:
        return json.loads((r.stdout or "").strip().splitlines()[-1])
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}", "stdout_tail": (r.stdout or "")[-300:]}


def _run_pr_own_tests(python: str, checkout: Path, out_dir: Path, timeout: int) -> dict:
    """The PR's own guard tests, executed on Windows.

    `tests/python/test_rocm_aotriton_gate.py` is AST-based and needs only pytest; it
    exists at the head and not at the base, so this is an observation about the head
    rather than half of the differential.
    """
    rel = "tests/python/test_rocm_aotriton_gate.py"
    path = checkout / rel
    if not path.is_file():
        return {"file": rel, "exists": False}
    report = out_dir / "pr_tests_report.json"
    cmd = [python, "-m", "pytest", str(path), "-p", "no:cacheprovider", "-q",
           "--timeout", "900", "-rs"]
    try:
        r = subprocess.run(cmd, cwd = str(checkout), capture_output = True, text = True,
                           encoding = "utf-8", errors = "replace", timeout = timeout)
    except subprocess.TimeoutExpired:
        return {"file": rel, "exists": True, "timeout": True}
    tail = (r.stdout or "")[-3000:]
    out = {"file": rel, "exists": True, "rc": r.returncode, "tail": tail,
           "stderr_tail": (r.stderr or "")[-1000:]}
    for line in reversed(tail.splitlines()):
        if " passed" in line or " failed" in line or " error" in line:
            out["summary"] = line.strip()
            break
    if report.exists():
        report.unlink()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    # Per matrix run, not for the whole probe: eighteen runs share one job, and an entry
    # point that hangs rather than raising must not eat the job's budget by itself.
    ap.add_argument("--run-timeout", type = int, default = 420)
    ap.add_argument("--timeout", type = int, default = 1500)
    args = ap.parse_args()

    # Everything below runs with cwd set to the checkout, so a relative --out would be
    # resolved against the wrong directory and the inner script would not be found.
    checkout = args.checkout.resolve()
    args.out = args.out.resolve()
    work = args.out.parent / f"inner_{args.state}"
    work.mkdir(parents = True, exist_ok = True)
    inner = work / "gate_inner.py"
    inner.write_text(_INNER.replace("GATE_NAME_PLACEHOLDER", GATE), encoding = "utf-8")

    obs: dict = {
        "state": args.state,
        "checkout": str(checkout),
        "gate_name": GATE,
        "probe_sys_platform": sys.platform,
        "probe_platform_system": platform.system(),
        "probe_platform_release": platform.release(),
        "probe_computername": os.environ.get("COMPUTERNAME"),
        "probe_python": platform.python_version(),
        # The job's own environment, before anything is imported.
        "gate_in_job_environment": os.environ.get(GATE),
        "gate_in_machine_scope": _registry_scope("Machine"),
        "gate_in_user_scope": _registry_scope("User"),
        # The removed export lived here. On Windows the directory does not exist at all,
        # which is the point: verified, not assumed.
        "etc_profile_d_exists": Path("/etc/profile.d").is_dir(),
        "etc_profile_d_unsloth_file_exists": Path("/etc/profile.d/unsloth-rocm-wsl.sh").is_file(),
    }

    # Which of this state's sources carry the gate statement at all.
    for label, rel in ENTRIES.items():
        path = checkout / rel
        try:
            obs[f"{label}_sets_gate"] = GATE in path.read_text(encoding = "utf-8")
        except Exception as e:  # noqa: BLE001
            obs[f"{label}_read_error"] = f"{type(e).__name__}: {e}"

    named, scripts = _files_naming_gate(checkout)
    obs["files_checked_for_gate"] = named
    obs["windows_scripts_naming_gate"] = scripts

    runs: dict = {}
    for entry, rel in ENTRIES.items():
        for start in STARTS:
            key = f"{entry}|{start}"
            env = {k: v for k, v in os.environ.items() if k != GATE}
            if start != "unset":
                env[GATE] = start
            env["PYTHONPATH"] = os.pathsep.join(
                [str(checkout)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            inner_out = work / f"{entry}_{start}.json"
            cmd = [args.python, str(inner), "--entry", entry, "--rel", rel,
                   "--checkout", str(checkout), "--out", str(inner_out)]
            record: dict = {"requested_start": start}
            try:
                r = subprocess.run(cmd, cwd = str(checkout), capture_output = True,
                                   text = True, encoding = "utf-8", errors = "replace",
                                   env = env, timeout = args.run_timeout)
                record["rc"] = r.returncode
                record["stderr_tail"] = (r.stderr or "")[-1500:]
                record["stdout_tail"] = (r.stdout or "")[-800:]
            except subprocess.TimeoutExpired:
                record["timeout"] = True
            if inner_out.is_file():
                try:
                    record.update(json.loads(inner_out.read_text(encoding = "utf-8")))
                except Exception as e:  # noqa: BLE001
                    record["parse_error"] = f"{type(e).__name__}: {e}"
            else:
                record["no_record"] = True
            runs[key] = record
    obs["runs"] = runs

    obs["torch_in_test_environment"] = _torch_facts(args.python)
    obs["pr_own_tests"] = _run_pr_own_tests(args.python, checkout, work, args.timeout)

    args.out.write_text(json.dumps(obs, indent = 2, sort_keys = True), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
