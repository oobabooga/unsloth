"""CI driver: clone unslothai/unsloth at BASE, build main's and the patched dropdown-surround
modules, then run smoke_ab.py. Cross-platform (Linux, macOS, Windows).

usage: run_ci.py <engines> [sizes] [reps]
"""
import os, shutil, subprocess, sys
from pathlib import Path

BASE = "08f87d0a39c1115d25e4e8954b3a0eeec11fd737"
HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "dc024"
REPO = WORK / "unsloth"
ENGINES = sys.argv[1]
SIZES = sys.argv[2] if len(sys.argv) > 2 else "100000,300000,1000000"
REPS = sys.argv[3] if len(sys.argv) > 3 else "7"
NPM = shutil.which("npm") or "npm"
NPX = shutil.which("npx") or "npx"


def run(*cmd, cwd=None):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)


WORK.mkdir(parents=True, exist_ok=True)
if not REPO.exists():
    run("git", "init", "-q", REPO)
    run("git", "config", "core.autocrlf", "false", cwd=REPO)
    run("git", "remote", "add", "origin", "https://github.com/unslothai/unsloth.git", cwd=REPO)
    run("git", "fetch", "-q", "--depth", "1", "origin", BASE, cwd=REPO)
    run("git", "checkout", "-q", "FETCH_HEAD", cwd=REPO)
frontend = REPO / "studio" / "frontend"
lib = frontend / "src" / "lib" / "dropdown-surround.ts"
mods = WORK / "mods"
mods.mkdir(exist_ok=True)
shutil.copy(lib, mods / "old.ts")
# A Windows checkout of this branch may have turned the patch's LFs into CRLFs.
patch = WORK / "fix.patch"
patch.write_bytes((HERE / "fix.patch").read_bytes().replace(b"\r\n", b"\n"))
run("git", "apply", "--check", patch, cwd=REPO)
run("git", "apply", patch, cwd=REPO)
shutil.copy(lib, mods / "new.ts")
run("git", "checkout", "--", "studio/frontend/src/lib/dropdown-surround.ts", cwd=REPO)  # page code stays main's
run(NPM, "ci", "--no-audit", "--no-fund", cwd=frontend)
run(NPX, "tsc", mods / "old.ts", mods / "new.ts", "--target", "es2022", "--module", "esnext",
    "--lib", "es2022,dom,dom.iterable", "--skipLibCheck", "--outDir", mods / "out", cwd=frontend)
run(sys.executable, HERE / "smoke_ab.py", frontend, mods / "out" / "old.js", mods / "out" / "new.js",
    WORK / "result.json", ENGINES, SIZES, REPS)
print((WORK / "result.json").read_text())
