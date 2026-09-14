"""Codex alone: which hop of node / cmd shim / PowerShell shim / Python subprocess loses the console."""
import glob
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import AB, T, newconsole_case, pty_case, run_show, write_probes

write_probes()
prefix = os.path.join(T, "npm-global")
codex_cmd = os.path.join(prefix, "codex.cmd")
codex_js = os.path.join(prefix, "node_modules", "@openai", "codex", "bin", "codex.js")
codex_exe = (glob.glob(os.path.join(prefix, "node_modules", "**", "codex.exe"), recursive=True) or [None])[0]
node = shutil.which("node")
py = sys._base_executable
pwsh = shutil.which("pwsh")
probe_py = os.path.join(AB, "probe.py")
probe_js = os.path.join(AB, "probe.js")
out = os.path.join(AB, "probe.json")
codex_home = os.path.join(T, "codex-home-a")
os.makedirs(codex_home, exist_ok=True)
env = dict(os.environ, CODEX_HOME=codex_home)

print("codex.exe:", codex_exe)
print("node:", node, "python:", py, "pwsh:", pwsh)
with open(codex_cmd, encoding="utf-8") as f:
    print("---- codex.cmd ----\n" + f.read() + "\n-------------------")
run_show([node, "--version"])
run_show([codex_exe, "--version"], env=env)

sub = "import subprocess, sys; sys.exit(subprocess.run(sys.argv[1:]).returncode)"
spawn_py = os.path.join(AB, "spawn.py")
with open(spawn_py, "w") as f:
    f.write(sub + "\n")

for backend, tag in ((0, "conpty"), (1, "winpty")):
    pty_case(f"A-{tag}-probe-direct", [py, probe_py, out], probe_out=out, backend=backend)
    pty_case(f"A-{tag}-probe-node", [node, probe_js, py, probe_py, out], probe_out=out, backend=backend)
    pty_case(f"A-{tag}-probe-py-node", [py, spawn_py, node, probe_js, py, probe_py, out], probe_out=out, backend=backend)
    pty_case(f"A-{tag}-codex-exe", [codex_exe], env=env, backend=backend)
    pty_case(f"A-{tag}-node-codex-js", [node, codex_js], env=env, backend=backend)
    pty_case(f"A-{tag}-cmd-codex-cmd", ["cmd.exe", "/d", "/c", codex_cmd], env=env, backend=backend)
    pty_case(f"A-{tag}-powershell-codex", ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", "codex"], env=env, backend=backend)
    if pwsh:
        pty_case(f"A-{tag}-pwsh-codex", [pwsh, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", "codex"], env=env, backend=backend)
    pty_case(f"A-{tag}-py-codex-exe", [py, spawn_py, codex_exe], env=env, backend=backend)
    pty_case(f"A-{tag}-py-node-codex-js", [py, spawn_py, node, codex_js], env=env, backend=backend)
    pty_case(f"A-{tag}-py-codex-cmd", [py, spawn_py, codex_cmd], env=env, backend=backend)

newconsole_case("A-newconsole-probe-py-node", [py, spawn_py, node, probe_js, py, probe_py, out], probe_out=out, wait=30)
newconsole_case("A-newconsole-codex-exe", [codex_exe], env=env, wait=20)
newconsole_case("A-newconsole-py-node-codex-js", [py, spawn_py, node, codex_js], env=env, wait=20)
