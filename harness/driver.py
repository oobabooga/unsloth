import os, sys, json, io, contextlib, importlib, importlib.metadata as md
tree, leg, fn, pinned, pkg = sys.argv[1:6]
sys.path.insert(0, os.path.join(tree, "studio"))
import install_python_stack as ips
ips.USE_UV = leg == "uv"
ips.VERBOSE = True
URL = os.environ["DRIVER_AMD_URL"] if pkg == "probe-torch" else os.environ["DRIVER_URL"]
pkg = pkg.split("@")[0]
args = [pkg, "--no-cache-dir"] + ([] if pkg == "probe-torch" else ["--no-deps"])
if pinned == "1":
    args += ["--index-url", URL]
buf = io.StringIO()
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
    try:
        if fn == "try":
            res = ips.pip_install_try("probe", *args, constrain=False, force_pip=(leg == "pip"))
        else:
            ips.pip_install("probe", *args, constrain=False); res = True
    except SystemExit as e:
        res = "exit"
importlib.invalidate_caches()
inst = sorted(d.metadata["Name"] for d in md.distributions() if (d.metadata["Name"] or "").startswith(("probe", "rocm")))
print("RESULT " + json.dumps({"res": res, "installed": inst, "tail": buf.getvalue().strip().splitlines()[-2:]}))
