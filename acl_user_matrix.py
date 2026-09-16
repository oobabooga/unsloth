"""The #9928 scenario as a real unprivileged Windows user.

An elevated token bypasses deny ACEs (MoveFileEx opens directories with backup
semantics, and SeRestorePrivilege then ignores the DACL), so an admin-only probe
can never reproduce a permissions-blocked rename. #9928's reporter was not
elevated. This script is driven in four phases by the workflow, switching identity
between them:

  setup   (admin)     build a realistic install tree and deny the probe user
  probe   (user)      the rename fails; capture exactly what the installer prints
  repair  (admin)     run the commands the installer printed, verbatim
  verify  (user)      the rename now succeeds

Usage: python acl_user_matrix.py <phase> <repo> <workdir> [user]
"""

import errno
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

PHASE = sys.argv[1]
REPO = Path(sys.argv[2]).resolve()
WORK = Path(sys.argv[3]).resolve()
USER = sys.argv[4] if len(sys.argv) > 4 else ""

STATE = WORK / "_state.json"


def load_module():
    sys.path.insert(0, str(REPO / "studio"))
    sys.path.insert(0, str(REPO / "studio" / "backend"))
    spec = importlib.util.spec_from_file_location(
        "install_llama_prebuilt", REPO / "studio" / "install_llama_prebuilt.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["install_llama_prebuilt"] = m
    spec.loader.exec_module(m)
    return m


def sh(*args):
    return subprocess.run(args, capture_output=True, text=True)


def populate(root: Path, marker="OLD"):
    for sub in ("build/bin", "build/bin/Release", "gguf-py"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    for name in ("llama-server.exe", "llama-quantize.exe"):
        for d in (root, root / "build" / "bin", root / "build" / "bin" / "Release"):
            (d / name).write_text(marker)
    (root / "convert_hf_to_gguf.py").write_text(f"# {marker}")
    (root / "UNSLOTH_PREBUILT_INFO.json").write_text('{"marker": "%s"}' % marker)


def windows_host(M):
    import dataclasses
    fields = {f.name for f in dataclasses.fields(M.HostInfo)}
    vals = dict(system="Windows", machine="AMD64", is_windows=True, is_linux=False,
                is_macos=False, is_x86_64=True, is_arm64=False, nvidia_smi=None,
                driver_cuda_version=None, compute_caps=[], visible_cuda_devices=None,
                has_physical_nvidia=False, has_usable_nvidia=False)
    return M.HostInfo(**{k: v for k, v in vals.items() if k in fields})


# --------------------------------------------------------------- phases
def phase_setup():
    WORK.mkdir(parents=True, exist_ok=True)
    root = WORK / "home"
    install = root / "llama.cpp"
    if install.exists():
        sh("icacls", str(root), "/reset", "/T", "/C")
        sh("cmd", "/c", "rmdir", "/S", "/Q", str(root))
    install.mkdir(parents=True)
    populate(install, "OLD")

    # the probe user must be able to read and traverse everything...
    print(sh("icacls", str(WORK), "/grant", f"{USER}:(OI)(CI)(RX,W)").stdout.strip())
    # ...but must not be able to remove llama.cpp from its parent. Renaming a
    # directory needs DELETE on it OR FILE_DELETE_CHILD on the parent, so deny both.
    print(sh("icacls", str(install), "/deny", f"{USER}:(D)").stdout.strip())
    print(sh("icacls", str(root), "/deny", f"{USER}:(DC)").stdout.strip())
    print(sh("icacls", str(install)).stdout.strip())
    STATE.write_text(json.dumps({"install": str(install), "root": str(root)}))
    print("SETUP OK")


def phase_probe():
    M = load_module()
    st = json.loads(STATE.read_text())
    install, root = Path(st["install"]), Path(st["root"])

    print(f"probe running as: {sh('whoami').stdout.strip()}")
    import ctypes
    print(f"elevated: {bool(ctypes.windll.shell32.IsUserAnAdmin())}")

    # 1. the bare rename, so the WinError is recorded unambiguously
    try:
        os.replace(install, root / "llama.cpp.rollback-probe")
        bare = {"ok": True, "winerror": None}
    except OSError as exc:
        bare = {"ok": False, "winerror": getattr(exc, "winerror", None),
                "errno": exc.errno, "msg": str(exc)[:160]}
    print("BARE RENAME: " + json.dumps(bare))

    # 2. lstat, the call _confirmed_reparse_point depends on
    try:
        stt = install.lstat()
        lst = f"SUCCEEDED st_file_attributes={getattr(stt, 'st_file_attributes', None)}"
    except OSError as exc:
        lst = f"RAISED {type(exc).__name__} winerror={getattr(exc, 'winerror', None)}"
    print("LSTAT: " + lst)
    # main has no such helper; keep the control run alive so both sides print.
    probe_fn = getattr(M, "_confirmed_reparse_point", None)
    print("CONFIRMED_REPARSE_POINT: "
          + (str(probe_fn(install)) if probe_fn else "<absent on this revision>"))

    # 3. the whole installer path, capturing every line a user would see
    staging = M.create_install_staging_dir(install)
    populate(staging, "NEW")
    lines = []
    M.log = lines.append
    M.log_lines = lambda ls: [lines.append(x) for x in ls]
    try:
        M.activate_install_tree(staging, install, windows_host(M))
        summary = "<no exception>"
    except BaseException as exc:
        summary = f"{type(exc).__name__}: {exc}"

    print("---- INSTALLER OUTPUT ----")
    for line in lines:
        print("[llama-prebuilt] " + line)
    print("---- SUMMARY ----")
    print(summary)

    commands = [l.strip() for l in lines
                if l.strip().startswith(("takeown ", "icacls "))]
    st.update({"bare": bare, "lstat": lst, "summary": summary,
               "commands": commands, "lines": lines})
    STATE.write_text(json.dumps(st))
    print("---- COMMANDS THE INSTALLER PRINTED ----")
    for c in commands:
        print(c)
    print("PROBE OK")


def phase_repair():
    st = json.loads(STATE.read_text())
    commands = st.get("commands") or []
    if not commands:
        print("NO COMMANDS WERE PRINTED -- nothing to run")
        return 1
    for line in commands:
        r = subprocess.run(line, capture_output=True, text=True, shell=True)
        print(f"$ {line}\n  rc={r.returncode} {(r.stdout + r.stderr).strip()[:300]}")
    print("REPAIR OK")
    return 0


def phase_verify():
    st = json.loads(STATE.read_text())
    install, root = Path(st["install"]), Path(st["root"])
    print(f"verify running as: {sh('whoami').stdout.strip()}")
    if not install.exists():
        print("VERIFY: install path is gone; the probe must have moved it")
        return 1
    try:
        os.replace(install, root / "llama.cpp.rollback-verify")
        print("VERIFY: rename SUCCEEDED after the printed repair")
        return 0
    except OSError as exc:
        print(f"VERIFY: rename STILL BLOCKED winerror={getattr(exc, 'winerror', None)} :: {exc}")
        return 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rc = {"setup": phase_setup, "probe": phase_probe,
          "repair": phase_repair, "verify": phase_verify}[PHASE]()
    sys.exit(rc or 0)
