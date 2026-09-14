"""Base vs head differential for unslothai/unsloth#10902 on the current OS."""
import http.server, functools, json, os, shutil, subprocess, sys, tempfile, threading
from pathlib import Path

BASE_SHA, HEAD_SHA = sys.argv[1], sys.argv[2]
PIP_VERSIONS = sys.argv[3].split(",")
HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("RUNNER_TEMP") or tempfile.gettempdir()) / "pr10902"
shutil.rmtree(WORK, ignore_errors=True)
WORK.mkdir(parents=True)
WIN = os.name == "nt"
UV = shutil.which("uv")
assert UV, "uv not on PATH"


def sh(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise SystemExit(f"failed: {cmd}")
    return r.stdout


def tree(sha, name):
    d = WORK / name
    d.mkdir()
    sh(["git", "init", "-q", str(d)])
    sh(["git", "-C", str(d), "fetch", "-q", "--depth", "1", "https://github.com/unslothai/unsloth", sha])
    sh(["git", "-C", str(d), "checkout", "-q", "FETCH_HEAD"])
    return d


trees = {"main": tree(BASE_SHA, "base"), "head": tree(HEAD_SHA, "head")}

# Local index: an sdist-only package, a wheel package, and the build backend.
index = WORK / "index" / "simple"
for name, kind, deps in (
    ("probe_sdist", "--sdist", ""),
    ("probe_wheel", "--wheel", ""),
    ("rocm", "--sdist", ""),  # AMD publishes rocm as an sdist alone
    ("probe_torch", "--wheel", 'dependencies = ["rocm==1.0"]\n'),
):
    src = WORK / "src" / name
    (src / name).mkdir(parents=True)
    (src / name / "__init__.py").write_text("")
    (src / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["setuptools>=61"]\nbuild-backend = "setuptools.build_meta"\n'
        f'[project]\nname = "{name.replace("_", "-")}"\nversion = "1.0"\n{deps}'
    )
    out = index / name.replace("_", "-")
    sh([UV, "build", "-q", kind, "-o", str(out)], cwd=src)
for name in ("setuptools", "wheel"):
    sh([sys.executable, "-m", "pip", "download", "-q", "--no-deps", "-d", str(index / name), name])
class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


handler = functools.partial(Quiet, directory=str(WORK / "index"))
server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
URL = f"http://127.0.0.1:{server.server_address[1]}/simple"

SCENARIOS = {
    "S0_clean": ("", {}),
    "S1_file_hardened": ("[global]\nonly-binary = :all:\nrequire-hashes = true\nno-index = true\n", {}),
    "S2_env_hardened": ("", {"PIP_ONLY_BINARY": ":all:", "PIP_REQUIRE_HASHES": "1"}),
    "S3_sections": ("[global]\nonly-binary = :all:\n[install]\nonly-binary = :none:\n    probe-wheel\n", {}),
    "S5_names_rocm": ("[global]\nonly-binary = :all:,rocm\n", {}),
    "S6_file_and_env": ("[global]\nonly-binary = probe-sdist\n", {"PIP_ONLY_BINARY": "probe-wheel"}),
    "S4_hashes_nonpinned": ("[global]\nrequire-hashes = true\n", {"PIP_INDEX_URL": URL, "UV_INDEX_URL": URL}),
}


def clean_env(home, extra):
    keep = ("PATH", "SYSTEMROOT", "SystemRoot", "COMSPEC", "PATHEXT", "WINDIR", "TEMP", "TMP", "LANG", "LC_ALL")
    env = {k: v for k, v in os.environ.items() if k in keep}
    for k in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "XDG_CONFIG_HOME"):
        env[k] = str(home)
    env["UV_CACHE_DIR"] = str(home / "uvc")
    env["PIP_NO_CACHE_DIR"] = "1"
    env["DRIVER_URL"] = URL
    env.update(extra)
    return env


def venv(pipv):
    d = Path(tempfile.mkdtemp(dir=WORK))
    sh([UV, "venv", "-q", "-p", f"{sys.version_info[0]}.{sys.version_info[1]}", str(d / "v")])
    py = d / "v" / ("Scripts/python.exe" if WIN else "bin/python")
    sh([UV, "pip", "install", "-q", "--python", str(py), f"pip=={pipv}"])
    return d, py


def run(script, scen, pipv, *args, keep=None):
    conf, extra = SCENARIOS[scen]
    d, py = keep if keep else venv(pipv)
    if conf:
        (d / "v" / ("pip.ini" if WIN else "pip.conf")).write_text(conf)
    home = d / "home"
    home.mkdir(exist_ok=True)
    r = subprocess.run([str(py), str(HERE / script), *args], capture_output=True, text=True,
                       env=clean_env(home, extra), timeout=900)
    line = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]
    if not keep:
        shutil.rmtree(d, ignore_errors=True)
    if not line:
        return {"error": (r.stdout + r.stderr)[-800:]}
    return json.loads(line[-1][7:])


# Expected (res, installed) per (scenario, fn, pinned, pkg) for main and head.
def expect(scen, fn, pinned, pkg):
    ok = (True, sorted([pkg, "rocm"]) if pkg == "probe-torch" else [pkg])
    refused = ("exit" if fn == "full" else False, [])
    hardened = scen in ("S1_file_hardened", "S2_env_hardened", "S5_names_rocm", "S6_file_and_env") and pinned == "1"
    if hardened and pkg == "probe-sdist":
        return ok, refused
    if pkg == "probe-torch" and scen == "S5_names_rocm":
        return ok, refused  # the operator named rocm, so no exemption
    return ok, ok


failures = []
rows = []
cases = []
for pipv in PIP_VERSIONS:
    for scen in ("S0_clean", "S1_file_hardened", "S2_env_hardened", "S3_sections"):
        for leg in ("uv", "pip"):
            for pkg in ("probe-sdist", "probe-wheel"):
                cases.append((pipv, scen, leg, "try", "1", pkg))
    for leg in ("uv", "pip"):
        cases.append((pipv, "S6_file_and_env", leg, "try", "1", "probe-sdist"))
    for scen in ("S0_clean", "S1_file_hardened", "S2_env_hardened", "S5_names_rocm"):
        for leg in ("uv", "pip"):
            cases.append((pipv, scen, leg, "try", "1", "probe-torch"))
        cases.append((pipv, scen, "uv", "full", "1", "probe-torch"))
    for leg in ("uv", "pip"):
        cases.append((pipv, "S4_hashes_nonpinned", leg, "try", "0", "probe-sdist"))
    cases.append((pipv, "S1_file_hardened", "uv", "full", "1", "probe-sdist"))
    cases.append((pipv, "S1_file_hardened", "uv", "full", "1", "probe-wheel"))
for pipv, scen, leg, fn, pinned, pkg in cases:
    exp = dict(zip(("main", "head"), expect(scen, fn, pinned, pkg)))
    for name, t in trees.items():
        got = run("driver.py", scen, pipv, str(t), leg, fn, pinned, pkg)
        good = "error" not in got and (got["res"], got["installed"]) == exp[name]
        rows.append(f"{'ok ' if good else 'BAD'} pip{pipv} {scen:20} {leg:3} {fn:4} pinned={pinned} {pkg:11} {name}: res={got.get('res')} inst={got.get('installed')}")
        if not good:
            failures.append((rows[-1], got))

# Clean-environment byte identity of argv and child env, base vs head.
for scen in ("S0_clean",):
    for pipv in PIP_VERSIONS[:1]:
        shared = venv(pipv)  # one interpreter, so sys.executable paths compare equal
        outs = {n: run("envdiff.py", scen, pipv, str(t), keep=shared) for n, t in trees.items()}
        for shape in outs["main"]:
            same = outs["main"][shape] == outs["head"][shape]
            allowed = shape == "pip_uninstall_wheel"
            good = same or allowed
            rows.append(f"{'ok ' if good else 'BAD'} envdiff {scen} {shape:22} identical={same}{' (documented, inert)' if allowed and not same else ''}")
            if not good:
                failures.append((rows[-1], {"main": outs["main"][shape], "head": outs["head"][shape]}))

print("\n".join(rows))
for row, got in failures:
    print("\nFAILURE:", row, "\n", json.dumps(got, indent=1)[:3000])
print(f"\n{len(rows) - len(failures)}/{len(rows)} checks as expected")
sys.exit(1 if failures else 0)
