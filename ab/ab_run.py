"""A/B the NPU runtime version display on an AMD Ryzen AI runner (Linux or Windows).

A = upstream main at BASE_SHA, B = A + branch.patch. One Studio install (install.sh / install.ps1
--local --no-torch) supplies the venv; each checkout runs its own backend and frontend build
against one shared studio home, so B meets the runtime A enabled. B-bumped then moves B's pins
to the latest Lemonade/FastFlowLM with scripts/update_lemonade_pins.py and restarts.
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_SHA = sys.argv[1]
WIN = sys.platform == "win32"
WORK = Path(os.environ["RUNNER_TEMP"]) / "npuv"
HOME = WORK / "studio-home"
OUT = WORK / "out"
for d in (WORK, HOME, OUT, WORK / "tmp"):
    d.mkdir(parents = True, exist_ok = True)
env = dict(os.environ)
env.update(
    UNSLOTH_STUDIO_HOME = str(HOME),
    UNSLOTH_SKIP_AUTOSTART = "1",
    TMPDIR = str(WORK / "tmp"),
    HF_HOME = str(WORK / "hf"),
    UV_CACHE_DIR = str(WORK / "uv-cache"),
    PLAYWRIGHT_BROWSERS_PATH = str(WORK / "ms-playwright"),
    PYTHONUTF8 = "1",
)


def run(cmd, cwd = None, log = None, check = True, shell = False):
    print(f"\n$ {cmd if isinstance(cmd, str) else ' '.join(map(str, cmd))}", flush = True)
    t0 = time.time()
    if log:
        with open(log, "w", encoding = "utf-8", errors = "replace") as fh:
            proc = subprocess.run(cmd, cwd = cwd, env = env, stdout = fh, stderr = subprocess.STDOUT, shell = shell)
    else:
        proc = subprocess.run(cmd, cwd = cwd, env = env, shell = shell)
    print(f"  -> exit {proc.returncode} in {time.time() - t0:.0f}s", flush = True)
    if check and proc.returncode != 0:
        if log:
            print(Path(log).read_text(encoding = "utf-8", errors = "replace")[-6000:])
        raise SystemExit(f"failed: {cmd}")
    return proc.returncode


a, b = WORK / "a", WORK / "b"
if not a.exists():
    run(["git", "init", "-q", str(a)])
    run(["git", "-C", str(a), "fetch", "-q", "--depth", "1", "https://github.com/unslothai/unsloth", BASE_SHA])
    run(["git", "-C", str(a), "checkout", "-q", "FETCH_HEAD"])
    run(["git", "clone", "-q", str(a), str(b)])
    run(["git", "-C", str(b), "checkout", "-q", BASE_SHA])
    run(["git", "-C", str(b), "apply", str(HERE / "branch.patch")])
    run(["git", "-C", str(b), "diff", "--stat"])

# One install from A supplies the venv both checkouts run under.
if WIN:
    child = "$ErrorActionPreference = 'Stop'; $ProgressPreference = 'SilentlyContinue'; & ./install.ps1 --local --no-torch --skip-autostart; exit $LASTEXITCODE"
    run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", child], cwd = a, log = OUT / "install.log")
    py = HOME / "unsloth_studio" / "Scripts" / "python.exe"
else:
    run(["bash", "install.sh", "--local", "--no-torch"], cwd = a, log = OUT / "install.log")
    py = HOME / "unsloth_studio" / "bin" / "python"
assert py.exists(), py

for checkout in (a, b):
    fe = checkout / "studio" / "frontend"
    run("npm ci --no-audit --no-fund", cwd = fe, log = OUT / f"npm-ci-{checkout.name}.log", shell = True)
    run("npx vite build", cwd = fe, log = OUT / f"vite-{checkout.name}.log", shell = True)

pw = WORK / "pw"
run([sys.executable, "-m", "venv", str(pw)])
pw_py = pw / ("Scripts/python.exe" if WIN else "bin/python")
run([str(pw_py), "-m", "pip", "install", "-q", "playwright>=1.45,<2"])
if run([str(pw_py), "-m", "playwright", "install", "--with-deps", "chromium"], check = False) != 0:
    run([str(pw_py), "-m", "playwright", "install", "chromium"])


def boot(checkout, port):
    log = open(OUT / f"studio-{checkout.name}-{port}.log", "w", encoding = "utf-8", errors = "replace")
    kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if WIN else {"start_new_session": True}
    proc = subprocess.Popen(
        [str(py), "run.py", "--host", "127.0.0.1", "--port", str(port), "--frontend", str(checkout / "studio/frontend/dist")],
        cwd = checkout / "studio" / "backend",
        env = env,
        stdout = log,
        stderr = subprocess.STDOUT,
        **kwargs,
    )
    for _ in range(600):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout = 5) as r:
                if r.status == 200:
                    return proc
        except Exception:
            pass
        if proc.poll() is not None:
            break
        time.sleep(1)
    log.flush()
    print((OUT / f"studio-{checkout.name}-{port}.log").read_text(errors = "replace")[-6000:])
    raise SystemExit(f"Studio from {checkout} never became healthy on {port}")


def stop(proc):
    if WIN:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output = True)
    else:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout = 60)
        except Exception:
            os.killpg(proc.pid, signal.SIGKILL)
    proc.wait()
    time.sleep(5)


def variant(label, checkout, port, *extra):
    # The desktop app's login secret, which the driver signs in with.
    run([str(py), "-m", "unsloth_cli", "studio", "provision-desktop-auth"], cwd = WORK)
    proc = boot(checkout, port)
    try:
        code = run(
            [str(pw_py), str(HERE / "npu_versions_ab.py"), f"http://127.0.0.1:{port}", str(HOME), label, str(OUT), *extra],
            check = False,
        )
    finally:
        stop(proc)
    return code


results = {}
results["A"] = variant("A", a, 18801)
results["B"] = variant("B", b, 18802)
# The weekly pin update, applied the way its PR would ship.
run([str(py), "scripts/update_lemonade_pins.py", "--write"], cwd = b, check = False)
run(["git", "-C", str(b), "diff", "--", "studio/lemonade_prebuilt_pins.json"])
pins = json.loads((b / "studio/lemonade_prebuilt_pins.json").read_text())
lemonade = pins["lemonade"]["version"]
results["B-bumped"] = variant("B-bumped", b, 18803, f"--expect=Lemonade v{lemonade}")

summary = ["## " + ("Windows" if WIN else "Linux"), "", f"exit codes: {results}", ""]
for label in ("A", "B", "B-bumped"):
    path = OUT / f"{label}.json"
    if path.exists():
        r = json.loads(path.read_text())
        st = r.get("status") or {}
        summary += [
            f"### {label}",
            f"- status.versions: `{st.get('versions', '(absent)')}`  state `{st.get('state')}` ready `{st.get('ready')}`",
            f"- Hub NPU credit on open: `{r.get('hub_credit_on_open', r.get('hub_credit'))}`",
            f"- Hub NPU credit: `{r.get('hub_credit')}`" + (f" (after {r['expect_seconds']} s, no reload)" if "expect_seconds" in r else ""),
            f"- Picker NPU credit: `{r.get('picker_credit')}`",
            f"- enable: `{json.dumps(r.get('enable', {}).get('http'))}` {str((r.get('status') or {}).get('error'))[:300]}",
            "",
        ]
text = "\n".join(summary)
print(text)
if os.environ.get("GITHUB_STEP_SUMMARY"):
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding = "utf-8") as fh:
        fh.write(text + "\n")
shutil.copy(HERE / "branch.patch", OUT / "branch.patch")
raise SystemExit(0 if all(v == 0 for v in results.values()) else 1)
