"""Which ways of handing uv a uv.toml does it honour on Windows? usage: probe_project.py PYTHON"""

import os
import shutil
import subprocess
import sys
import tempfile
import time

TARGET = sys.argv[1]
URL = (
    "https://github.com/unslothai/unsloth/releases/download/prebuilt-wheels-cu13/"
    "causal_conv1d-1.7.0+cu13torch2.13cxx11abiTRUE-cp313-cp313-win_amd64.whl"
)
TOML = (
    "[[dependency-metadata]]\n"
    'name = "causal-conv1d"\n'
    'version = "1.7.0+cu13torch2.13cxx11abitrue"\n'
    "requires-dist = []\n"
)
LOG = os.path.abspath("logs/proxy.log")


def asset_bytes_since(offset):
    with open(LOG, encoding = "utf-8") as fh:
        lines = fh.readlines()[offset:]
    return sum(
        int(line.split("down=")[1].split()[0]) for line in lines if "close release-assets" in line
    )


def line_count():
    with open(LOG, encoding = "utf-8") as fh:
        return len(fh.readlines())


def config_dir(where):
    if where == "temp":
        path = tempfile.mkdtemp(prefix = "unsloth-uv-")
    elif where == "temp-long":
        path = os.path.realpath(tempfile.mkdtemp(prefix = "unsloth-uv-"))
    else:
        path = os.path.abspath(f"cfg-{time.time_ns()}")
        os.makedirs(path)
    with open(os.path.join(path, "uv.toml"), "w", encoding = "utf-8") as fh:
        fh.write(TOML)
    return path


VARIANTS = [
    ("no config", None, None),
    ("cwd = temp dir", "temp", "cwd"),
    ("--project temp dir (as tempfile returns it)", "temp", "project"),
    ("--project temp dir (realpath)", "temp-long", "project"),
    ("--project workspace dir", "workspace", "project"),
]

for label, where, how in VARIANTS:
    cache = os.path.abspath(f"cache-{time.time_ns()}")
    cmd = ["uv", "-v"]
    cwd = None
    if where:
        path = config_dir(where)
        if how == "project":
            cmd += ["--project", path]
        else:
            cwd = path
    cmd += ["pip", "install", "--reinstall", "--python", TARGET, "--no-deps", URL]
    env = dict(os.environ, UV_CACHE_DIR = cache)
    before = line_count()
    result = subprocess.run(
        cmd, cwd = cwd, env = env, stdout = subprocess.PIPE, stderr = subprocess.STDOUT,
        text = True, encoding = "utf-8", errors = "replace",
    )
    time.sleep(2)
    markers = [
        line.strip()[:200]
        for line in result.stdout.splitlines()
        if "configuration" in line or "Range requests" in line
    ]
    print(f"=== {label}")
    if where:
        print(f"    config dir: {path}")
    print(f"    rc={result.returncode} release-asset bytes={asset_bytes_since(before):,}")
    for marker in markers:
        print(f"    {marker}")
    shutil.rmtree(cache, ignore_errors = True)
