"""Real-Windows validation matrix for unslothai/unsloth#10182.

Runs on an actual Windows host against real NTFS ACLs, real junctions and real
open file handles -- the things a Linux simulation can only fake. Three questions:

  1. Do the WinError codes the PR branches on actually come back from Windows
     for the situations the PR attributes to them?
  2. Does the PR's code produce the right message for each real situation?
  3. Does the repair the PR prints actually unblock the rename? (#9928's claim)

Nothing here is mocked. Every failure mode is manufactured on disk.

With --control the same script runs against a checkout of main instead: it repeats
the pure-Windows probes (group A, which must agree) and captures what main prints
for the identical broken-ACL update, so the before/after comes from one machine.

Usage: python winacl_matrix.py <path-to-repo-checkout> [--control]
"""

import ctypes
import errno
import importlib.util
import io
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

CONTROL = "--control" in sys.argv[1:]
REPO = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(REPO / "studio"))
sys.path.insert(0, str(REPO / "studio" / "backend"))
_spec = importlib.util.spec_from_file_location(
    "install_llama_prebuilt", REPO / "studio" / "install_llama_prebuilt.py"
)
M = importlib.util.module_from_spec(_spec)
sys.modules["install_llama_prebuilt"] = M
_spec.loader.exec_module(M)

RESULTS = []
FACTS = []


def record(group, name, ok, detail=""):
    RESULTS.append((group, name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {group} {name}")
    if detail:
        for line in str(detail).splitlines():
            print("        " + line)
    sys.stdout.flush()


def fact(name, value):
    FACTS.append((name, value))
    print(f"[FACT] {name}: {value}")
    sys.stdout.flush()


def sh(*args, **kw):
    return subprocess.run(args, capture_output=True, text=True, shell=False, **kw)


WHOAMI = sh("whoami").stdout.strip()


# ---------------------------------------------------------------- helpers
def populate(root: Path, marker="OLD"):
    # Windows confirm_install_tree looks under build\bin\Release as well, so a tree
    # missing it fails validation before the aside-move is ever reached.
    for sub in ("build/bin", "build/bin/Release", "gguf-py"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    for name in ("llama-server.exe", "llama-quantize.exe"):
        for d in (root, root / "build" / "bin", root / "build" / "bin" / "Release"):
            (d / name).write_text(marker)
    (root / "convert_hf_to_gguf.py").write_text(f"# {marker}")
    (root / "UNSLOTH_PREBUILT_INFO.json").write_text('{"marker": "%s"}' % marker)


def break_acls(path: Path, strategy: str):
    """Manufacture a permission-denied directory. Returns (cmds, rc, output).

    Renaming a directory is authorized by DELETE on the directory OR
    FILE_DELETE_CHILD on its PARENT, so denying the child alone leaves an
    administrator able to rename it. The strategies that work deny the parent too.
    """
    parent = path.parent
    if strategy == "deny-child-and-parent-delete":
        cmds = [["icacls", str(path), "/deny", f"{WHOAMI}:(D)"],
                ["icacls", str(parent), "/deny", f"{WHOAMI}:(DC)"]]
    elif strategy == "deny-everyone-child-and-parent":
        cmds = [["icacls", str(path), "/deny", "Everyone:(D)"],
                ["icacls", str(parent), "/deny", "Everyone:(DC)"]]
    elif strategy == "deny-parent-full":
        cmds = [["icacls", str(parent), "/deny", f"{WHOAMI}:(OI)(CI)(F)"]]
    elif strategy == "deny-user-full":
        cmds = [["icacls", str(path), "/inheritance:r", "/deny", f"{WHOAMI}:(OI)(CI)(F)"]]
    else:
        raise ValueError(strategy)
    rcs, outs = [], []
    for cmd in cmds:
        r = sh(*cmd)
        rcs.append(r.returncode)
        outs.append((r.stdout + r.stderr).strip()[:120])
    return "; ".join(" ".join(c) for c in cmds), max(rcs), " | ".join(outs)


def restore_acls(path: Path):
    for target in (path.parent, path):
        sh("icacls", str(target), "/remove:d", WHOAMI, "Everyone")
        sh("takeown", "/F", str(target), "/R", "/D", "Y")
        sh("icacls", str(target), "/reset", "/T", "/C")
        sh("icacls", str(target), "/grant", f"{WHOAMI}:(OI)(CI)(F)")


def try_rename(src: Path, dst: Path):
    """Return (ok, winerror, errno, message)."""
    try:
        os.replace(src, dst)
        return True, None, None, ""
    except OSError as exc:
        return False, getattr(exc, "winerror", None), exc.errno, str(exc)[:160]


def make_junction(link: Path, target: Path):
    r = sh("cmd", "/c", "mklink", "/J", str(link), str(target))
    return r.returncode == 0, (r.stdout + r.stderr).strip()[:160]


class Capture:
    """Collect every [llama-prebuilt] line the installer emits."""

    def __enter__(self):
        self.lines = []
        self._log, self._log_lines = M.log, getattr(M, "log_lines", None)
        M.log = self.lines.append
        if self._log_lines is not None:
            M.log_lines = lambda ls: [self.lines.append(l) for l in ls]
        return self

    def __exit__(self, *a):
        M.log = self._log
        if self._log_lines is not None:
            M.log_lines = self._log_lines
        return False

    @property
    def text(self):
        return "\n".join(self.lines)


def windows_host():
    import dataclasses
    fields = {f.name for f in dataclasses.fields(M.HostInfo)}
    vals = dict(system="Windows", machine="AMD64", is_windows=True, is_linux=False,
                is_macos=False, is_x86_64=True, is_arm64=False, nvidia_smi=None,
                driver_cuda_version=None, compute_caps=[], visible_cuda_devices=None,
                has_physical_nvidia=False, has_usable_nvidia=False)
    return M.HostInfo(**{k: v for k, v in vals.items() if k in fields})


# ================================================================ GROUP A
# Does Windows actually return the codes the PR attributes to each situation?
# ========================================================================
def group_a(tmp: Path):
    # A1 -- an open handle inside the directory
    d = tmp / "a1_src"
    d.mkdir()
    populate(d)
    holder = open(d / "build" / "bin" / "llama-server.exe", "rb")
    try:
        ok, we, en, msg = try_rename(d, tmp / "a1_dst")
    finally:
        holder.close()
    fact("A1 open-handle rename", f"ok={ok} winerror={we} errno={en} :: {msg}")
    record("A", "an open handle blocks the rename with a code the installer retries",
           (not ok) and we in (5, 32, 145), f"winerror={we}")
    globals()["_HANDLE_CODE"] = we

    # A2 -- ACLs denied. Renaming a dir needs DELETE on it OR FILE_DELETE_CHILD on
    # the parent, so a child-only deny leaves an admin able to rename.
    denied = {}
    for strategy in ("deny-child-and-parent-delete", "deny-everyone-child-and-parent",
                     "deny-parent-full", "deny-user-full"):
        holder = tmp / f"a2_{strategy}"
        holder.mkdir()
        d = holder / "llama.cpp"
        d.mkdir()
        populate(d)
        cmd, rc, out = break_acls(d, strategy)
        ok, we, en, msg = try_rename(d, holder / "moved")
        denied[strategy] = we
        fact(f"A2 {strategy}", f"icacls_rc={rc} rename_ok={ok} winerror={we} :: {msg}")
        restore_acls(d)
    record("A", "at least one ACL break produces WinError 5 (the PR's premise)",
           5 in denied.values(), f"per strategy: {denied}")
    globals()["_ACL_STRATEGY"] = next((s for s, w in denied.items() if w == 5), None)

    # A3 -- non-empty destination. Characterized, not asserted: MOVEFILE_REPLACE_EXISTING
    # is not honoured for directories, so os.replace may answer 5 rather than 145.
    src, dst = tmp / "a3_src", tmp / "a3_dst"
    src.mkdir()
    populate(src)
    dst.mkdir()
    (dst / "occupied.txt").write_text("x")
    ok, we, en, msg = try_rename(src, dst)
    fact("A3 non-empty destination", f"ok={ok} winerror={we} errno={en} :: {msg}")
    record("A", "a non-empty destination is refused (code characterized, not asserted)",
           not ok, f"winerror={we} -- 145 is the documented code; this build answers {we}")

    # A5 -- is a genuine sharing violation (32) reachable at all? Open with no
    # sharing via CreateFileW, which is what a scanner effectively does.
    d = tmp / "a5_src"
    d.mkdir()
    populate(d)
    target = str(d / "build" / "bin" / "llama-server.exe")
    GENERIC_READ, OPEN_EXISTING = 0x80000000, 3
    h = ctypes.windll.kernel32.CreateFileW(
        ctypes.c_wchar_p(target), GENERIC_READ, 0, None, OPEN_EXISTING, 0, None)
    try:
        ok, we, en, msg = try_rename(d, tmp / "a5_dst")
    finally:
        if h != -1:
            ctypes.windll.kernel32.CloseHandle(h)
    fact("A5 exclusive (share-none) handle", f"handle_ok={h != -1} ok={ok} winerror={we} :: {msg}")
    record("A", "an exclusive handle also blocks with a retried code",
           (not ok) and we in (5, 32, 145), f"winerror={we}")

    # A4 -- the ordinary case still works
    src, dst = tmp / "a4_src", tmp / "a4_dst"
    src.mkdir()
    populate(src)
    ok, we, en, msg = try_rename(src, dst)
    record("A", "an unobstructed rename still succeeds", ok, msg)


# ================================================================ GROUP B
# Does the PR's message logic answer each REAL situation correctly?
# ========================================================================
def group_b(tmp: Path):
    plain = tmp / "b_plain"
    plain.mkdir()
    populate(plain)

    h5 = M.blocked_replace_hint(5, plain)
    record("B", "code 5 on a plain dir prints takeown + icacls",
           f'takeown /F "{plain}" /R /D Y' in h5 and f'icacls "{plain}" /reset /T /C' in h5,
           h5)

    h32 = M.blocked_replace_hint(32, plain)
    record("B", "code 32 keeps the scanner wording and prints no repair",
           "scanner" in h32 and "takeown" not in h32, h32)

    h145 = M.blocked_replace_hint(145, plain)
    record("B", "code 145 says the destination is not empty, no repair",
           "not empty" in h145 and "takeown" not in h145 and "scanner" not in h145, h145)

    # real junction
    target = tmp / "b_external"
    target.mkdir()
    populate(target)
    link = tmp / "b_junction"
    made, out = make_junction(link, target)
    fact("B junction created", f"{made} :: {out}")
    if made:
        is_rp = M._confirmed_reparse_point(link)
        hj = M.blocked_replace_hint(5, link)
        record("B", "a REAL mklink /J junction is seen as a reparse point", is_rp, f"{is_rp}")
        record("B", "a junction root suppresses the recursive repair",
               "takeown" not in hj and "icacls" not in hj, hj)
    else:
        record("B", "a REAL mklink /J junction is seen as a reparse point", False,
               "could not create a junction: " + out)

    record("B", "a plain directory is NOT seen as a reparse point",
           M._confirmed_reparse_point(plain) is False, "")

    # THE key assumption I could not test off-Windows: a broken-ACL directory must
    # NOT be mistaken for a link, or the repair is withheld from the one case it
    # exists for.
    strategy = globals().get("_ACL_STRATEGY") or "deny-user-full"
    acl = tmp / "b_denied"
    acl.mkdir()
    populate(acl)
    break_acls(acl, strategy)
    try:
        lstat_ok, lstat_detail = True, ""
        try:
            st = acl.lstat()
            lstat_detail = f"lstat SUCCEEDED, st_file_attributes={getattr(st, 'st_file_attributes', None)}"
        except OSError as exc:
            lstat_ok = False
            lstat_detail = f"lstat RAISED {type(exc).__name__} winerror={getattr(exc, 'winerror', None)}"
        fact("B broken-ACL lstat", lstat_detail)
        rp = M._confirmed_reparse_point(acl)
        hint = M.blocked_replace_hint(5, acl)
        record("B", "a broken-ACL dir is NOT read as a link, so it KEEPS the repair",
               rp is False and "takeown" in hint, f"reparse={rp}; {lstat_detail}")
    finally:
        restore_acls(acl)


# ================================================================ GROUP C
# End to end through the real installer entry point.
# ========================================================================
def group_c(tmp: Path):
    strategy = globals().get("_ACL_STRATEGY") or "deny-user-full"

    # C1 -- the #9928 case, for real
    root = tmp / "c1"
    root.mkdir()
    install = root / "llama.cpp"
    install.mkdir()
    populate(install, "OLD")
    staging = M.create_install_staging_dir(install)
    populate(staging, "NEW")
    break_acls(install, strategy)
    summary = ""
    try:
        with Capture() as cap:
            try:
                M.activate_install_tree(staging, install, windows_host())
                summary = "<no exception>"
            except M.BusyInstallConflict as exc:
                summary = f"BusyInstallConflict: {exc}"
            except BaseException as exc:
                summary = f"{type(exc).__name__}: {exc}"
        text = cap.text
    finally:
        restore_acls(install)

    print("        ---- captured installer output ----")
    for line in text.splitlines():
        print("        [llama-prebuilt] " + line)
    print("        ---- summary ----")
    print("        " + summary)

    takeowns = [l for l in cap.lines if "takeown" in l]
    record("C", "the real broken-ACL update prints the repair exactly once",
           len(takeowns) == 1, f"{len(takeowns)} takeown line(s)")
    record("C", "the repair names the install dir, not the rollback path",
           any(str(install) in l for l in takeowns) and
           not any("rollback" in l for l in takeowns),
           takeowns[0] if takeowns else "(none)")
    record("C", "the terminal summary widens to mention permissions",
           "or has broken permissions" in summary, summary[:200])
    record("C", "no log line carries an embedded newline (prefix preserved)",
           not any("\n" in l for l in cap.lines), "")
    record("C", "the previous install is left in place",
           install.exists() and (install / "UNSLOTH_PREBUILT_INFO.json").exists(), "")

    # C2 -- a real sharing violation must NOT get the ACL repair
    root = tmp / "c2"
    root.mkdir()
    install = root / "llama.cpp"
    install.mkdir()
    populate(install, "OLD")
    staging = M.create_install_staging_dir(install)
    populate(staging, "NEW")
    holder = open(install / "build" / "bin" / "llama-server.exe", "rb")
    try:
        with Capture() as cap:
            try:
                M.activate_install_tree(staging, install, windows_host())
                out = "<no exception>"
            except BaseException as exc:
                out = f"{type(exc).__name__}: {exc}"
    finally:
        holder.close()
    fact("C2 held-handle outcome", out[:160])
    # Windows answers a held handle with the SAME code as broken ACLs on this build,
    # so the installer cannot tell them apart. What it must do is lead with the
    # handle theory rather than assert permissions outright.
    cause_lines = [l for l in cap.lines if "blocked (" in l or "still blocked (" in l]
    leads_with_handle = all(
        "scanner, indexer or running process still holding a handle" in l or "scanner" in l
        for l in cause_lines) if cause_lines else False
    record("C", "a real held handle still leads with the held-handle cause",
           leads_with_handle, "\n".join(cause_lines[:2]))
    repair = [l for l in cap.lines if "takeown" in l]
    fact("C2 repair also offered for a held handle",
         f"{len(repair)} line(s) -- expected, since this Windows build returns "
         f"code {globals().get('_HANDLE_CODE')} for a held handle too")

    # C3 -- the ordinary update on real NTFS still works
    root = tmp / "c3"
    root.mkdir()
    install = root / "llama.cpp"
    install.mkdir()
    populate(install, "OLD")
    staging = M.create_install_staging_dir(install)
    populate(staging, "NEW")
    with Capture() as cap:
        try:
            M.activate_install_tree(staging, install, windows_host())
            ok, why = True, ""
        except BaseException as exc:
            ok, why = False, f"{type(exc).__name__}: {exc}"
    record("C", "an ordinary update still succeeds and prints no repair",
           ok and (install / "UNSLOTH_PREBUILT_INFO.json").read_text().find("NEW") >= 0
           and not [l for l in cap.lines if "takeown" in l], why or "")


# ================================================================ GROUP D
# Adversarial paths and shapes.
# ========================================================================
def group_d(tmp: Path):
    cases = {
        "spaces": "dir with spaces",
        "nonascii": "拒绝访问",          # the reporter's locale
        "accents": "café-über",
        "dollar-backtick": "we$ird`name",
        "bracket": "dir[1]",
        "ampersand": "a&b",
    }
    for label, name in cases.items():
        try:
            d = tmp / name
            d.mkdir()
            populate(d)
        except OSError as exc:
            record("D", f"path shape '{label}' usable", False, f"mkdir failed: {exc}")
            continue
        hint = M.blocked_replace_hint(5, d)
        quoted_ok = f'takeown /F "{d}" /R /D Y' in hint
        one_per_line = [l.strip() for l in hint.splitlines()]
        record("D", f"'{label}' renders one quoted command per line",
               quoted_ok and f'takeown /F "{d}" /R /D Y' in one_per_line,
               "\n".join(one_per_line))

    # D-long: a path beyond MAX_PATH
    deep = tmp
    try:
        for i in range(12):
            deep = deep / ("longsegment" * 2 + str(i))
        deep.mkdir(parents=True)
        hint = M.blocked_replace_hint(5, deep)
        record("D", "a >260 char path still produces a hint without raising",
               "takeown" in hint, f"len={len(str(deep))}")
    except OSError as exc:
        record("D", "a >260 char path still produces a hint without raising", False,
               f"could not create: {exc}")

    # D-nested: a junction INSIDE an otherwise ordinary install root (finding #2)
    root = tmp / "d_nested"
    root.mkdir()
    populate(root)
    external = tmp / "d_external"
    external.mkdir()
    (external / "precious.txt").write_text("not ours")
    made, out = make_junction(root / "nested_link", external)
    fact("D nested junction created", f"{made} :: {out}")
    if made:
        rp_root = M._confirmed_reparse_point(root)
        hint = M.blocked_replace_hint(5, root)
        record("D", "a nested junction does not make the ROOT a reparse point "
                    "(documents the known limit)",
               rp_root is False and "takeown" in hint,
               f"root_is_reparse={rp_root}; the printed takeown /R would walk "
               f"into {external}")

    # D-readonly: the read-only attribute rather than an ACL
    ro = tmp / "d_readonly"
    ro.mkdir()
    populate(ro)
    sh("cmd", "/c", "attrib", "+R", str(ro / "*"), "/S")
    ok, we, en, msg = try_rename(ro, tmp / "d_readonly_moved")
    fact("D read-only attribute rename", f"ok={ok} winerror={we} :: {msg}")
    record("D", "the read-only attribute is characterized (not a PR claim)", True,
           f"ok={ok} winerror={we}")

    # D-parent: the PARENT directory denies access
    parent = tmp / "d_parent"
    parent.mkdir()
    child = parent / "llama.cpp"
    child.mkdir()
    populate(child)
    break_acls(parent, globals().get("_ACL_STRATEGY") or "deny-user-full")
    try:
        rp = M._confirmed_reparse_point(child)
        hint = M.blocked_replace_hint(5, child)
        record("D", "a denied PARENT still keeps the repair (never read as a link)",
               rp is False and "takeown" in hint, f"reparse={rp}")
    finally:
        restore_acls(parent)


# ================================================================ GROUP E
# Does the printed repair actually unblock the rename? (#9928's whole claim)
# ========================================================================
def group_e(tmp: Path):
    strategy = globals().get("_ACL_STRATEGY")
    if strategy is None:
        record("E", "the printed repair actually unblocks the rename", False,
               "no ACL strategy produced WinError 5 on this host; cannot test")
        return

    install = tmp / "e_llama.cpp"
    install.mkdir()
    populate(install)
    break_acls(install, strategy)

    ok, we, en, msg = try_rename(install, tmp / "e_rollback")
    record("E", "precondition: the rename is blocked with WinError 5",
           (not ok) and we == 5, f"ok={ok} winerror={we} :: {msg}")
    if ok or we != 5:
        restore_acls(install)
        return

    # take the commands the installer would PRINT and run exactly those
    hint = M.blocked_replace_hint(5, install)
    commands = [l.strip() for l in hint.splitlines()
                if l.strip().startswith(("takeown", "icacls"))]
    fact("E commands taken from the hint", commands)

    outputs = []
    for line in commands:
        r = subprocess.run(line, capture_output=True, text=True, shell=True)
        outputs.append(f"{line} -> rc={r.returncode} {(r.stdout + r.stderr).strip()[:120]}")
    for o in outputs:
        print("        " + o)

    ok2, we2, en2, msg2 = try_rename(install, tmp / "e_rollback")
    record("E", "after running the PRINTED repair, the rename succeeds",
           ok2, f"ok={ok2} winerror={we2} :: {msg2}\n" + "\n".join(outputs))


def group_control(tmp: Path):
    """main's code, same machine, same manufactured failure: what does it say?"""
    strategy = globals().get("_ACL_STRATEGY")
    if strategy is None:
        record("CTRL", "main reproduces the misleading hint", False,
               "no ACL strategy produced WinError 5 on this host")
        return
    root = tmp / "ctrl"
    root.mkdir()
    install = root / "llama.cpp"
    install.mkdir()
    populate(install, "OLD")
    staging = M.create_install_staging_dir(install)
    populate(staging, "NEW")
    break_acls(install, strategy)
    summary = ""
    try:
        with Capture() as cap:
            try:
                M.activate_install_tree(staging, install, windows_host())
                summary = "<no exception>"
            except BaseException as exc:
                summary = f"{type(exc).__name__}: {exc}"
    finally:
        restore_acls(install)

    print("        ---- what MAIN prints for the identical failure ----")
    for line in cap.lines:
        print("        [llama-prebuilt] " + line)
    print("        ---- summary ----")
    print("        " + summary)

    scanner = [l for l in cap.lines if "scanner is likely still holding" in l]
    record("CTRL", "main blames a scanner for a permissions failure (the bug)",
           bool(scanner), scanner[0] if scanner else "(not found)")
    record("CTRL", "main never prints a permissions repair",
           not [l for l in cap.lines if "takeown" in l or "icacls" in l], "")
    record("CTRL", "main's summary does not mention permissions",
           "or has broken permissions" not in summary, summary[:200])


def main():
    print("=" * 78)
    print("PR #10182 -- real Windows validation matrix"
          + ("  [CONTROL: main]" if CONTROL else "  [PR HEAD]"))
    print(f"repo        : {REPO}")
    print(f"python      : {sys.version.split()[0]}  {sys.platform}")
    print(f"os.name     : {os.name}")
    print(f"whoami      : {WHOAMI}")
    print(f"admin       : {bool(ctypes.windll.shell32.IsUserAnAdmin())}")
    ver = sh("cmd", "/c", "ver").stdout.strip()
    print(f"windows     : {ver}")
    print(f"filesystem  : {sh('cmd', '/c', 'fsutil', 'fsinfo', 'volumeinfo', 'C:').stdout.strip()[:200]}")
    print("=" * 78)

    base = Path(os.environ.get("RUNNER_TEMP") or tempfile.gettempdir()) / (
        "pr10182_control" if CONTROL else "pr10182")
    base.mkdir(parents=True, exist_ok=True)
    groups = ([("A", group_a), ("CTRL", group_control)] if CONTROL else
              [("A", group_a), ("B", group_b), ("C", group_c),
               ("D", group_d), ("E", group_e)])
    for label, fn in groups:
        print("\n" + "=" * 78)
        print(f"GROUP {label}")
        print("=" * 78)
        work = Path(tempfile.mkdtemp(prefix=f"g{label}_", dir=str(base)))
        try:
            fn(work)
        except Exception:
            record(label, "<group crashed>", False, traceback.format_exc())
        finally:
            try:
                restore_acls(work)
            except Exception:
                pass

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    failed = [(g, n, d) for g, n, ok, d in RESULTS if not ok]
    for g, n, ok, d in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {g} {n}")
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if failed:
        print("\nFAILURES:")
        for g, n, d in failed:
            print(f"  {g} {n}\n    {str(d)[:500]}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
