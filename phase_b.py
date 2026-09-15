"""The installed Studio CLI. Subcommands so a lost runner still shows which step it died in."""
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
pwsh = shutil.which("pwsh")
probe_py = os.path.join(AB, "probe.py")
probe_js = os.path.join(AB, "probe.js")
out = os.path.join(AB, "probe.json")
model = os.environ["AB_MODEL"]
base = os.environ["UNSLOTH_STUDIO_URL"]
port = base.rsplit(":", 1)[1]
start = ["start", "codex", "--model", model]
step = sys.argv[1]
os.makedirs(AB, exist_ok=True)


def backends(fn):
    for backend, tag in ((0, "conpty"), (1, "winpty")):
        fn(backend, tag)


if step == "launch":
    from lib import write_probes

    write_probes()
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
            "print('resolved launch command:', s._resolved_launch_command(w, ['--oss'], dict(os.environ)))\n"
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

    def launch_cases(backend, tag):
        pty_case(f"B-{tag}-launch-probe", [sp, launch_probe], probe_out=out, backend=backend)
        pty_case(f"B-{tag}-launch-codex", [sp, launch_codex], backend=backend)

    backends(launch_cases)

elif step == "server-start":
    log = open(os.path.join(AB, "studio.log"), "w")
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    server = subprocess.Popen([bin_exe, "studio", "-H", "127.0.0.1", "-p", port], stdout=log,
                              stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=flags)
    with open(os.path.join(AB, "server.pid"), "w") as f:
        f.write(str(server.pid))
    for i in range(600):
        try:
            with urllib.request.urlopen(f"{base}/api/health", timeout=2):
                print(f"studio healthy after {i}s", flush=True)
                break
        except Exception:
            time.sleep(1)
    else:
        print("studio never became healthy", flush=True)
    with open(os.path.join(AB, "studio.log"), encoding="utf-8", errors="replace") as f:
        print(f.read()[-4000:])

elif step == "preload":
    run_show([bin_exe, *start, "--no-launch"], timeout=1500)

elif step == "key":
    pty_case("C-conpty-bin-exe-start-codex", [bin_exe, *start], ready_marker="Unsloth ready", alive_secs=20, cap=600, backend=0)

elif step == "variants":
    ps_cmd = f"& '{bin_exe}' start codex --model {model}"

    def variant_cases(backend, tag):
        if tag == "winpty":
            pty_case(f"C-{tag}-bin-exe-start-codex", [bin_exe, *start], ready_marker="Unsloth ready", alive_secs=20, cap=600, backend=backend)
        pty_case(f"C-{tag}-venv-exe-start-codex", [venv_exe, *start], ready_marker="Unsloth ready", alive_secs=20, cap=600, backend=backend)
        pty_case(f"C-{tag}-cmd-bin-cmd-start-codex", ["cmd.exe", "/d", "/c", bin_cmd, *start], ready_marker="Unsloth ready", alive_secs=20, cap=600, backend=backend)
        pty_case(f"C-{tag}-powershell-start-codex", ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_cmd], ready_marker="Unsloth ready", alive_secs=20, cap=600, backend=backend)
        if pwsh:
            pty_case(f"C-{tag}-pwsh-start-codex", [pwsh, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_cmd], ready_marker="Unsloth ready", alive_secs=20, cap=600, backend=backend)

    backends(variant_cases)
    newconsole_case("C-newconsole-bin-exe-start-codex", [bin_exe, *start], wait=90)

elif step == "server-stop":
    pid_file = os.path.join(AB, "server.pid")
    if os.path.exists(pid_file):
        subprocess.run(["taskkill", "/T", "/F", "/PID", open(pid_file).read().strip()], capture_output=True)
    run_show([bin_exe, "studio", "stop"], timeout=60)
    log_path = os.path.join(AB, "studio.log")
    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8", errors="replace") as f:
            print(f.read()[-6000:])
