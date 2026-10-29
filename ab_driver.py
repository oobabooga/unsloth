"""Run the real torchao step of install_python_stack against the live indexes.

usage: python ab_driver.py <studio_dir> <scenario> [torch_version]
  fresh   step 4 on a venv without torchao (first install)
  update  step 4 again with install evidence present (what every later update does)
  resync  the post-repair re-selection (_resync_torch_coupled_packages)
"""
import importlib.metadata as md
import sys
import time

studio, scenario = sys.argv[1], sys.argv[2]
sys.path.insert(0, studio)
import install_python_stack as m

m.USE_UV = m._bootstrap_uv()
commands = []
_try, _inst = m.pip_install_try, m.pip_install


def _log(kind, fn):
    def run(label, *args, **kwargs):
        t = time.monotonic()
        result = fn(label, *args, **kwargs)
        idx = args[args.index("--index-url") + 1] if "--index-url" in args else "default"
        ok = "ok" if result is None or result else "FAILED"
        commands.append(
            f"{kind:<16} index={idx:<40} {ok:<6} {time.monotonic() - t:6.1f}s  {' '.join(args)}"
        )
        return result

    return run


m.pip_install_try = _log("pip_install_try", _try)
m.pip_install = _log("pip_install", _inst)


def _installed():
    try:
        return md.version("torchao")
    except md.PackageNotFoundError:
        return None


torch_version = sys.argv[3] if len(sys.argv) > 3 else m._probe_installed_torch_version()
print(f"platform={sys.platform} uv={m.USE_UV} torch={torch_version} torchao_before={_installed()}")
if m.IS_WINDOWS:
    # step 4 only reaches the pinned branch when this is False
    print(f"step 4 takes the Windows ROCm branch: {m._installed_torch_is_windows_rocm()}")
t0 = time.monotonic()
if scenario == "update":
    m._PASS_EVIDENCE = {"pass_inputs": {}, "step_results": {}}
if scenario == "resync":
    m._resync_torch_coupled_packages("2.12.0+cu128")
else:
    m._install_torchao_for_torch(torch_version)
elapsed = time.monotonic() - t0
print(
    f"--- {scenario}: step={m._STEP_RESULTS.get('torchao', '-')} elapsed={elapsed:.1f}s "
    f"torchao_after={_installed()} installer_commands={len(commands)}"
)
for line in commands:
    print("   ", line)
