"""The installed Studio CLI: _launch itself, then `unsloth start codex` through every entry point."""
import os
import shutil
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import AB, T, newconsole_case, pty_case, run_show

home = os.environ["UNSLOTH_STUDIO_HOME"]
sp = os.path.join(home, "unsloth_studio", "Scripts", "python.exe")
bin_exe = os.path.join(home, "bin", "unsloth.exe")
bin_cmd = os.path.join(home, "bin", "unsloth.cmd")
venv_exe = os.path.join(home, "unsloth_studio", "Scripts", "unsloth.exe")
node = shutil.which("node")
pwsh = shutil.which("pwsh")
probe_py = os.path.join(AB, "probe.py")
probe_js = os.path.join(AB, "probe.js")
out = os.path.join(AB, "probe.json")
model = os.environ["AB_MODEL"]
for path in (sp, bin_exe, bin_cmd, venv_exe):
    print(path, "exists" if os.path.exists(path) else "MISSING")
run_show([bin_exe, "--version"])

resolve_py = os.path.join(AB, "resolve.py")
with open(resolve_py, "w") as f:
    f.write(
        "import os, shutil\n"
        "from unsloth_cli.commands import start as s\n"
        "print('managed node tools:', s._managed_node_tools())\n"
        "s._augment_path_with_install_dirs()\n"
        "w = shutil.which('codex')\n"
        "print('shutil.which codex:', w)\n"
        "env = dict(os.environ)\n"
        "print('resolved launch command:', s._resolved_launch_command(w, ['--oss'], env))\n"
        "print('PATH:', os.environ['PATH'])\n"
    )
run_show([sp, resolve_py])

codex_home = os.path.join(T, "codex-home-b")
os.makedirs(codex_home, exist_ok=True)
launch_codex = os.path.join(AB, "launch_codex.py")
with open(launch_codex, "w") as f:
    f.write(
        "import sys\n"
        "from unsloth_cli.commands.start import _launch\n"
        f"sys.exit(_launch(['codex'], {{'CODEX_HOME': {codex_home!r}}}, install_hint='npm install -g @openai/codex'))\n"
    )
launch_probe = os.path.join(AB, "launch_probe.py")
with open(launch_probe, "w") as f:
    f.write(
        "import sys\n"
        "from unsloth_cli.commands.start import _launch\n"
        f"sys.exit(_launch(['node', {probe_js!r}, {sp!r}, {probe_py!r}, {out!r}], {{}}, install_hint='x'))\n"
    )

for backend, tag in ((0, "conpty"), (1, "winpty")):
    pty_case(f"B-{tag}-launch-probe", [sp, launch_probe], probe_out=out, backend=backend)
    pty_case(f"B-{tag}-launch-codex", [sp, launch_codex], backend=backend)

# Server + model, outside the pty, so the agent cases below only attach.
log = open(os.path.join(AB, "studio.log"), "w")
server = subprocess.Popen([bin_exe, "studio", "-H", "127.0.0.1", "-p", "8888"], stdout=log,
                          stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                          creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
try:
    healthy = False
    for _ in range(600):
        try:
            with urllib.request.urlopen("http://127.0.0.1:8888/api/health", timeout=2):
                healthy = True
                break
        except Exception:
            time.sleep(1)
    print("studio healthy:", healthy, flush=True)
    if healthy:
        run_show([bin_exe, "start", "codex", "--model", model, "--no-launch"], timeout=1500)
        start = ["start", "codex", "--model", model]
        for backend, tag in ((0, "conpty"), (1, "winpty")):
            pty_case(f"B-{tag}-bin-exe-start-codex", [bin_exe, *start], ready_marker="Unsloth ready", alive_secs=20, cap=600, backend=backend)
            pty_case(f"B-{tag}-venv-exe-start-codex", [venv_exe, *start], ready_marker="Unsloth ready", alive_secs=20, cap=600, backend=backend)
            pty_case(f"B-{tag}-cmd-bin-cmd-start-codex", ["cmd.exe", "/d", "/c", bin_cmd, *start], ready_marker="Unsloth ready", alive_secs=20, cap=600, backend=backend)
            ps_cmd = f"& '{bin_exe}' start codex --model {model}"
            pty_case(f"B-{tag}-powershell-bin-exe-start-codex", ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_cmd], ready_marker="Unsloth ready", alive_secs=20, cap=600, backend=backend)
            if pwsh:
                pty_case(f"B-{tag}-pwsh-bin-exe-start-codex", [pwsh, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_cmd], ready_marker="Unsloth ready", alive_secs=20, cap=600, backend=backend)
        newconsole_case("B-newconsole-bin-exe-start-codex", [bin_exe, *start], wait=90)
finally:
    kill_tree_pid = server.pid
    subprocess.run(["taskkill", "/T", "/F", "/PID", str(kill_tree_pid)], capture_output=True)
    log.close()
    with open(os.path.join(AB, "studio.log"), encoding="utf-8", errors="replace") as f:
        print("---- studio.log tail ----")
        print(f.read()[-5000:])
