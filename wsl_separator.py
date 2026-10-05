"""Show what wsl.exe does with Studio's put() command line after `--` and after `--exec`, on a
throwaway distro: put() is `sh -c 'cat > "$0" && chmod 644 "$0"' PATH` with the text on stdin.

Usage: wsl_separator.py DISTRO
"""

import subprocess
import sys

name = sys.argv[1]


def wsl(*argv, data = None):
    result = subprocess.run(["wsl.exe", "-d", name, "-u", "root", *argv], input = data, capture_output = True)
    return result.returncode, (result.stdout + result.stderr).decode("utf-8", "replace").replace("\x00", "").strip()


for separator, target in (("--exec", "/tmp/exec.txt"), ("--", "/tmp/dashdash.txt")):
    code, output = wsl(
        "--cd", "/root", separator, "/usr/bin/env", "sh", "-c", 'cat > "$0" && chmod 644 "$0"', target,
        data = b"written by put\n",
    )
    print(f"{separator}: exit {code} {output[-300:]!r}")
    code, output = wsl("--exec", "sh", "-c", f"ls -l {target} 2>&1; cat {target} 2>&1; head -c 64 /bin/bash | od -c | head -n 2")
    print(f"  after {separator}: {output!r}")
