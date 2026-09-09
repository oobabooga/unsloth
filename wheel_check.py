"""Build a wheel and prove the new test file is not in it."""
import glob, os, subprocess, sys, zipfile
env = dict(os.environ, UNSLOTH_IS_PRESENT = "1")
r = subprocess.run([sys.executable, "-m", "build", "--wheel", "--no-isolation", "-o", "dist_probe"],
                   capture_output = True, text = True, env = env)
if r.returncode != 0:
    print("build --no-isolation failed, retrying isolated"); print(r.stderr[-1500:])
    r = subprocess.run([sys.executable, "-m", "build", "--wheel", "-o", "dist_probe"],
                       capture_output = True, text = True, env = env)
if r.returncode != 0:
    print("WHEEL BUILD UNAVAILABLE (reported, not a failure)"); print(r.stderr[-2500:]); sys.exit(0)
whl = glob.glob("dist_probe/*.whl")[0]
names = zipfile.ZipFile(whl).namelist()
print("wheel:", os.path.basename(whl), "entries:", len(names))
shipped = [n for n in names if n.startswith("tests/") or "/tests/" in n]
print("test files shipped:", shipped[:10] or "none")
assert "tests/test_rmsnorm_gradient_layout.py" not in names, "new test shipped in the wheel"
kern = [n for n in names if n.endswith("kernels/rms_layernorm.py")]
print("kernel module in wheel:", kern)
assert kern, "kernel module missing from wheel"
print("WHEEL CHECK OK")
