"""Exercise PR 17463's wheel step on a Windows runner laid out like a chat-only Studio install.

  wheels_test.py <studio_dir> <install_dir> <driver_dir>

<install_dir>/build/bin/Release holds the published CUDA bundle and <install_dir> a prebuilt marker.
The Python running this script is the "Studio env". The runner has no NVIDIA driver, so <driver_dir>
may hold a stand-in nvcuda.dll, the same for both states.
"""
import ast, importlib.util, os, subprocess, sys
from pathlib import Path

sys.dont_write_bytecode = True
studio, install_dir, driver = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve()
bundle = install_dir / "build" / "bin" / "Release"
sys.path.insert(0, str(studio))
spec = importlib.util.spec_from_file_location("ilp", studio / "install_llama_prebuilt.py")
m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m)

# Studio's own search-dir builder, lifted from llama_cpp.py without importing the backend package.
src = (studio / "backend" / "core" / "inference" / "llama_cpp.py").read_text(encoding="utf-8")
fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "_windows_pip_nvidia_dll_dirs")
fn.decorator_list = []
ns = {"Path": Path, "os": os}
exec(compile(ast.Module(body=[fn], type_ignores=[]), "llama_cpp.py", "exec"), ns)
pip_dirs = ns["_windows_pip_nvidia_dll_dirs"]

SYSTEM32 = Path(os.environ["SystemRoot"]) / "System32"
LOAD = (
    "import ctypes, sys\n"
    "k = ctypes.WinDLL('kernel32', use_last_error=True)\n"
    "k.LoadLibraryExW.restype = ctypes.c_void_p\n"
    "k.LoadLibraryExW.argtypes = [ctypes.c_wchar_p, ctypes.c_void_p, ctypes.c_uint32]\n"
    "k.SetErrorMode(0x8003)\n"
    "h = k.LoadLibraryExW(sys.argv[1], None, 0x8)\n"
    "print('loaded' if h else f'FAILED WinError {ctypes.get_last_error()}')\n"
)

def state(label):
    dirs = pip_dirs(sys.prefix)
    print(f"RESULT {label} wheels the PR considers missing: {m._missing_cuda_runtime_wheels(install_dir) or 'none'}")
    print(f"RESULT {label} Studio pip nvidia search dirs: {[str(Path(d).relative_to(sys.prefix)) for d in dirs] or 'none'}")
    found = sorted(p.name for d in dirs for p in Path(d).glob("*.dll") if p.name.lower().startswith(("cudart", "cublas")))
    print(f"RESULT {label} runtime DLLs in those dirs: {found or 'none'}")
    env = {k: v for k, v in os.environ.items() if k.upper() not in ("PATH", "CUDA_PATH", "CUDA_HOME")}
    env["PATH"] = os.pathsep.join([str(bundle), *dirs, str(SYSTEM32), str(SYSTEM32.parent), str(driver)])
    r = subprocess.run([sys.executable, "-c", LOAD, str(bundle / "ggml-cuda.dll")], env=env, capture_output=True, text=True)
    print(f"RESULT {label} LoadLibrary(ggml-cuda.dll): {(r.stdout + r.stderr).strip()}")
    r = subprocess.run([str(bundle / "llama-server.exe"), "--list-devices"], env=env, cwd=bundle, capture_output=True, text=True, timeout=300)
    lines = (r.stdout + r.stderr).strip().splitlines()
    print(f"RESULT {label} llama-server --list-devices rc={r.returncode}:")
    for line in lines[-6:]:
        print(f"    | {line[:200]}")

print(f"RESULT marker: {m.load_prebuilt_metadata(install_dir)}")
state("BEFORE")
m.ensure_cuda_runtime_wheels(install_dir)
state("AFTER")
subprocess.run([sys.executable, "-m", "pip", "list", "--disable-pip-version-check"], check=False)
