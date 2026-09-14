"""Exercise unslothai/unsloth#10653 on a real OS: prefetch, isolation, busy lock, old CLI,
offline swap of the installer's core step, and control without pins.

usage: prefetch_matrix.py <repo checkout> <mode: torch|no-torch|auto> [--mlx]
"""
import json, os, platform, shutil, subprocess, sys, tempfile, textwrap, time
from pathlib import Path

REPO = Path(sys.argv[1]).resolve()
MODE = sys.argv[2]
MLX = "--mlx" in sys.argv
WIN = platform.system() == "Windows"
OLD, FLOOR = "2026.9.1", "2026.9.3"
HOME = Path.home()
SH = HOME / ".unsloth" / "studio"
VENV = SH / "unsloth_studio"
PY = VENV / ("Scripts/python.exe" if WIN else "bin/python")
results = []

def log(msg): print(f"[matrix] {msg}", flush=True)
def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    log(f"{'PASS' if ok else 'FAIL'} {name} {detail}")

def run(cmd, env=None, cwd=None, timeout=3600, check_rc=True):
    log("$ " + " ".join(map(str, cmd)))
    r = subprocess.run([str(c) for c in cmd], env=env, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    if check_rc and r.returncode != 0:
        print(r.stdout[-4000:], r.stderr[-4000:])
        raise SystemExit(f"command failed rc={r.returncode}")
    return r

def site_packages(py):
    return Path(run([py, "-c", "import sysconfig;print(sysconfig.get_paths()['purelib'])"]).stdout.strip())

def freeze(py):
    return run(["uv", "pip", "freeze", "--python", py]).stdout

def cli(py, *args, env=None, cwd=None, check_rc=False, timeout=3600):
    return run([py, "-I", "-c", "from unsloth_cli import app; app()", *args], env=env, cwd=cwd, check_rc=check_rc, timeout=timeout)

log(f"platform={platform.system()} machine={platform.machine()} python={platform.python_version()} mode={MODE} mlx={MLX}")
shutil.rmtree(SH, ignore_errors=True)
SH.mkdir(parents=True)
run(["uv", "venv", "--python", "3.12", VENV])
cli_deps = ["typer", "pyyaml", "pydantic", "click"]

if MODE == "torch":
    extra = [] if platform.system() == "Darwin" else ["--torch-backend", "cpu"]
    # Under Studio's constraints, as a real install is: otherwise the core plan carries constraint
    # downgrades and the not-behind guard (correctly) withholds the pins.
    constraints = REPO / "studio" / "backend" / "requirements" / "single-env" / "constraints.txt"
    run(["uv", "pip", "install", "--python", PY, *extra, f"unsloth=={OLD}", f"unsloth-zoo=={OLD}", *cli_deps, "-c", constraints], timeout=5400)
else:
    run(["uv", "pip", "install", "--python", PY, "--no-deps", f"unsloth=={OLD}", f"unsloth-zoo=={OLD}"])
    run(["uv", "pip", "install", "--python", PY, *cli_deps])
    if MODE == "no-torch":
        (VENV / ".unsloth-no-torch").write_text("")

SP = site_packages(PY)
# This PR's CLI and studio tree, as its wheel would install them.
shutil.rmtree(SP / "unsloth_cli", ignore_errors=True)
shutil.copytree(REPO / "unsloth_cli", SP / "unsloth_cli")
shutil.rmtree(SP / "studio", ignore_errors=True)
shutil.copytree(REPO / "studio", SP / "studio", ignore=shutil.ignore_patterns("frontend", "src-tauri", "node_modules", "__pycache__", ".venv*"))
for _ in range(8):
    r = cli(PY, "studio", "prefetch-update", "--help")
    if r.returncode == 0:
        break
    import re
    m = re.search(r"No module named '([^'.]+)", r.stderr + r.stdout)
    if not m:
        print(r.stdout[-3000:], r.stderr[-3000:]); raise SystemExit("CLI does not import")
    run(["uv", "pip", "install", "--python", PY, {"yaml": "pyyaml"}.get(m.group(1), m.group(1))])

if MLX:
    overrides = SP / "studio" / "backend" / "requirements" / "single-env" / "overrides-darwin-arm64.txt"
    env = {**os.environ, "UV_OVERRIDE": str(overrides)}
    run(["uv", "pip", "install", "--python", PY, "mlx==0.32.1", "mlx-metal==0.32.1", "mlx-lm==0.31.3", "mlx-vlm>=0.4.4,<0.7.0",
         "-c", SP / "studio" / "backend" / "requirements" / "single-env" / "constraints.txt"], env=env, timeout=3600)

before = freeze(PY)
env = {**os.environ, "UNSLOTH_TAURI_UPDATE": "1", "UNSLOTH_DESKTOP_BACKEND_VERSION": FLOOR, "UNSLOTH_TAURI_SHELL_VERSION": "0.1.999-ci"}
for k in ("UNSLOTH_STUDIO_HOME", "STUDIO_HOME", "UV_CACHE_DIR", "UNSLOTH_NO_TORCH"):
    env.pop(k, None)
t0 = time.time()
r = cli(PY, "studio", "prefetch-update", env=env, cwd=HOME, timeout=3600)
print(r.stdout[-6000:]); print(r.stderr[-3000:])
check("prefetch exits 0", r.returncode == 0, f"rc={r.returncode} in {time.time()-t0:.1f}s")
marker = json.loads((SH / ".update-prefetch" / "PREFETCHED.json").read_text(encoding="utf-8"))
log("marker " + json.dumps({k: marker.get(k) for k in ("state", "backend_version", "zoo_version", "installed_backend_version", "no_torch", "python", "cache_dir", "floor")}))
log("core_plan " + json.dumps(marker.get("core_plan")))
log("requirements " + json.dumps({k: (len(v.get("pins") or {}), v.get("skipped_reason")) for k, v in marker.get("requirements", {}).items()}))
check("marker ready or partial", marker.get("state") in ("ready", "partial"), marker.get("state"))
check("plan meets floor", marker.get("backend_version") and tuple(map(int, marker["backend_version"].split(".")[:3])) >= (2026, 9, 3), marker.get("backend_version"))
expect_no_torch = MODE != "torch"
check("no-torch routing", marker.get("no_torch") is expect_no_torch, f"no_torch={marker.get('no_torch')} expected={expect_no_torch}")
plan = marker.get("core_plan") or {}
check("core plan leaves torch alone", not any(n == "torch" or n.startswith("nvidia-") for n in plan), sorted(plan))
if MLX:
    check("core plan leaves MLX alone", not any(n.startswith("mlx") for n in plan), sorted(plan))
check("marker python is the managed venv python", Path(marker["python"]) == PY, f"{marker['python']} vs {PY}")
check("fetched scratch tree removed", not (SH / ".update-prefetch" / "site").exists())
check("live venv unchanged by prefetch", freeze(PY) == before)

# Busy: a held lock makes a second prefetch exit 3.
holder = subprocess.Popen([str(PY), "-I", "-c", textwrap.dedent(f"""
    import time; from pathlib import Path; from unsloth_cli import _studio_prefetch as p
    with p.prefetch_lock(Path(r'{SH}')):
        print('held', flush=True); time.sleep(60)""")], stdout=subprocess.PIPE, text=True)
holder.stdout.readline()
r = cli(PY, "studio", "prefetch-update", env=env, cwd=HOME, timeout=600)
holder.kill(); holder.wait()
check("held lock -> exit 3", r.returncode == 3 and "already running" in r.stdout, f"rc={r.returncode}")

# Desktop status reader (prefetch.rs) on this real cache, when the helper was built.
status_bin = os.environ.get("PREFETCH_STATUS_BIN")
def rust_state():
    out = run([status_bin, SH], env={k: v for k, v in os.environ.items() if k != "UV_CACHE_DIR"}).stdout
    return json.loads(out)["state"]
if status_bin:
    check("prefetch.rs status ready on real cache", rust_state() == marker["state"], rust_state())

# Pins exactly as `unsloth studio update` computes them.
pins_script = textwrap.dedent("""
    import platform
    from unsloth_cli.commands import studio
    script = studio._find_setup_script(None)
    cwd = None if platform.system() == "Windows" else script.parent
    env = studio._with_studio_uv_cache(None, cwd=cwd)
    env = studio._with_prefetched_core_pins(env, cwd=cwd)
    print("PINS=" + (env or {}).get("UNSLOTH_PREFETCHED_CORE_PINS", ""))
    print("CACHE=" + (env or {}).get("UV_CACHE_DIR", ""))
""")
out = run([PY, "-I", "-c", pins_script], env=env, cwd=HOME).stdout
pins = next(l[5:] for l in out.splitlines() if l.startswith("PINS="))
cache = next(l[6:] for l in out.splitlines() if l.startswith("CACHE="))
check("update names the prefetched pins", pins.split() and all("==" in x for x in pins.split()), pins)
check("pins recorded for the cache the update reads", os.path.normcase(os.path.normpath(cache)) == os.path.normcase(os.path.normpath(marker["cache_dir"])), f"{cache} vs {marker['cache_dir']}")

swap_script = textwrap.dedent("""
    import os, sys, subprocess
    sys.path.insert(0, os.path.join(sys.argv[1], "studio"))
    import install_python_stack as s
    s.USE_UV = True
    s.VERBOSE = True
    args = ["--no-cache-dir"] + (["--no-deps"] if sys.argv[2] != "torch" else []) + [
        "--upgrade-package", "unsloth", "--upgrade-package", "unsloth-zoo", "unsloth>=%s" % sys.argv[3], "unsloth-zoo"]
    try:
        s.pip_install("core step", *args, offline_pins=s._prefetched_core_pins())
        print("SWAP=ok")
    except SystemExit as e:
        print("SWAP=exit", e.code)
    import importlib.metadata as m
    print("VERSIONS=%s %s" % (m.version("unsloth"), m.version("unsloth-zoo")))
""")
blocked = {**env, "HTTPS_PROXY": "http://127.0.0.1:9", "HTTP_PROXY": "http://127.0.0.1:9", "ALL_PROXY": "http://127.0.0.1:9",
           "https_proxy": "http://127.0.0.1:9", "http_proxy": "http://127.0.0.1:9", "NO_PROXY": "", "no_proxy": "",
           "UV_HTTP_RETRIES": "1", "PIP_RETRIES": "0", "PIP_TIMEOUT": "5", "UV_CACHE_DIR": cache}
venv_copy = Path(tempfile.mkdtemp()) / "venv-copy"
shutil.copytree(VENV, venv_copy, symlinks=True)

# Control first: no pins, index unreachable -> the core step fails as it does today.
ctl = run([PY, "-I", "-c", swap_script, SP, MODE, FLOOR], env={**blocked, "UNSLOTH_PREFETCHED_CORE_PINS": ""}, timeout=1800, check_rc=False)
print(ctl.stdout[-2500:])
check("control: offline core step without pins fails", "SWAP=exit" in ctl.stdout and f"VERSIONS={OLD} {OLD}" in ctl.stdout)
shutil.rmtree(VENV); shutil.copytree(venv_copy, VENV, symlinks=True)

torch_before = run([PY, "-I", "-c", "import importlib.metadata as m\ntry: print(m.version('torch'))\nexcept Exception: print('none')"]).stdout.strip()
sw = run([PY, "-I", "-c", swap_script, SP, MODE, FLOOR], env={**blocked, "UNSLOTH_PREFETCHED_CORE_PINS": pins}, timeout=1800, check_rc=False)
print(sw.stdout[-3000:])
want = {p.split("==")[0]: p.split("==")[1] for p in pins.split()}
check("offline swap succeeds from the cache", "SWAP=ok" in sw.stdout and f"VERSIONS={want.get('unsloth')} {want.get('unsloth-zoo')}" in sw.stdout)
torch_after = run([PY, "-I", "-c", "import importlib.metadata as m\ntry: print(m.version('torch'))\nexcept Exception: print('none')"]).stdout.strip()
check("torch untouched by the swap", torch_before == torch_after, f"{torch_before} -> {torch_after}")
if MODE == "torch":
    t = run([PY, "-I", "-c", "import torch; print(torch.__version__, torch.ones(2).sum().item())"], check_rc=False)
    check("torch imports after swap", t.returncode == 0, t.stdout.strip() or t.stderr[-300:])
if MLX:
    t = run([PY, "-I", "-c", "import mlx.core as mx, importlib.metadata as m; print(m.version('mlx'), m.version('mlx-vlm'), mx.array([1,2]).sum().item())"], check_rc=False)
    check("mlx imports after swap", t.returncode == 0, t.stdout.strip() or t.stderr[-300:])

if status_bin:
    run(["uv", "cache", "clean", "unsloth", "--cache-dir", cache], check_rc=False)
    check("prefetch.rs status stale after uv cache clean unsloth", rust_state() == "stale", rust_state())

# An installed CLI older than the PR: exit 2 with click's "No such command" on stderr.
old = Path(tempfile.mkdtemp()) / "old"
run(["uv", "venv", "--python", "3.12", old])
old_py = old / ("Scripts/python.exe" if WIN else "bin/python")
run(["uv", "pip", "install", "--python", old_py, "--no-deps", f"unsloth=={OLD}", f"unsloth-zoo=={OLD}"])
run(["uv", "pip", "install", "--python", old_py, *cli_deps])
oenv = {**env, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"} if WIN else env
for _ in range(8):
    r = cli(old_py, "studio", "prefetch-update", env=oenv, cwd=HOME, timeout=600)
    import re
    m = re.search(r"No module named '([^'.]+)", r.stderr + r.stdout)
    if not m: break
    run(["uv", "pip", "install", "--python", old_py, {"yaml": "pyyaml"}.get(m.group(1), m.group(1))])
check("old CLI: exit 2 + 'No such command' on stderr", r.returncode == 2 and "No such command" in r.stderr, f"rc={r.returncode} stderr={r.stderr[-200:]!r}")

print("\n==== SUMMARY ====")
for name, ok, detail in results:
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
sys.exit(0 if all(ok for _, ok, _ in results) else 1)
