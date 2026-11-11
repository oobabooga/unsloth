"""Run Studio's real llama.cpp installer on a Windows runner as a chat-only NVIDIA host would.

  win_e2e.py <studio_dir> <install_dir> <driver_major.minor> <compute_cap> <driver_dir> <label> <mode>

Only host detection is replaced (the runner has no NVIDIA GPU): everything else, selection, download,
validation, the marker and the wheel step, is the installer's own code. The Python running this is
the "Studio env". mode: install | check | probe
"""
import ast, dataclasses, importlib.util, os, subprocess, sys
from pathlib import Path

sys.dont_write_bytecode = True
studio, install_dir = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
driver_ver = tuple(int(x) for x in sys.argv[3].split("."))
cap, driver, label, mode = sys.argv[4], Path(sys.argv[5]).resolve(), sys.argv[6], sys.argv[7]
sys.path.insert(0, str(studio))
spec = importlib.util.spec_from_file_location("ilp", studio / "install_llama_prebuilt.py")
m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m)

_real_detect = m.detect_host
def fake_detect(*a, **k):
    host = _real_detect(*a, **k)
    fields = {f.name for f in dataclasses.fields(host)}
    want = dict(nvidia_smi="nvidia-smi", driver_cuda_version=driver_ver, compute_caps=[cap],
                physical_compute_caps=[cap], visible_cuda_devices=None,
                has_physical_nvidia=True, has_usable_nvidia=True)
    return dataclasses.replace(host, **{k: v for k, v in want.items() if k in fields})
m.detect_host = fake_detect

def pip_dirs():
    src = (studio / "backend" / "core" / "inference" / "llama_cpp.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "_windows_pip_nvidia_dll_dirs")
    fn.decorator_list = []
    ns = {"Path": Path, "os": os}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "llama_cpp.py", "exec"), ns)
    return ns["_windows_pip_nvidia_dll_dirs"](sys.prefix)

if mode == "install":
    sys.argv = ["install_llama_prebuilt.py", "--install-dir", str(install_dir), "--llama-tag", "latest", "--published-repo", "unslothai/llama.cpp"]
    try:
        rc = m.main()
    except SystemExit as exc:
        rc = exc.code
    except m.PrebuiltFallback as exc:
        print(f"PrebuiltFallback: {exc}"); rc = "fallback"
    print(f"RESULT {label} installer exit code: {rc}")
elif mode == "check":
    sys.argv = ["install_llama_prebuilt.py", "--check-installed", str(install_dir)]
    try:
        rc = m.main()
    except SystemExit as exc:
        rc = exc.code
    print(f"RESULT {label} --check-installed exit code: {rc}")
else:
    marker = m.load_prebuilt_metadata(install_dir) or {}
    print(f"RESULT {label} marker: asset={marker.get('asset')} backend={marker.get('backend')} install_kind={marker.get('install_kind')} keys={len(marker)}")
    dirs = pip_dirs()
    wheels = subprocess.run([sys.executable, "-m", "pip", "list", "--disable-pip-version-check", "--format=freeze"], capture_output=True, text=True).stdout
    print(f"RESULT {label} nvidia wheels in the env: {[w for w in wheels.split() if w.lower().startswith('nvidia')] or 'none'}")
    found = sorted(p.name for d in dirs for p in Path(d).glob("*.dll") if p.name.lower().startswith(("cudart", "cublas")))
    print(f"RESULT {label} runtime DLLs in Studio's search dirs: {found or 'none'}")
    hits = list(install_dir.rglob("ggml-cuda.dll"))
    if not hits:
        print(f"RESULT {label} LoadLibrary(ggml-cuda.dll): no ggml-cuda.dll installed"); sys.exit(0)
    bundle = hits[0].parent
    system32 = Path(os.environ["SystemRoot"]) / "System32"
    env = {k: v for k, v in os.environ.items() if k.upper() not in ("PATH", "CUDA_PATH", "CUDA_HOME")}
    env["PATH"] = os.pathsep.join([str(bundle), *dirs, str(system32), str(system32.parent), str(driver)])
    load = ("import ctypes, sys\nk = ctypes.WinDLL('kernel32', use_last_error=True)\nk.LoadLibraryExW.restype = ctypes.c_void_p\n"
            "k.LoadLibraryExW.argtypes = [ctypes.c_wchar_p, ctypes.c_void_p, ctypes.c_uint32]\nk.SetErrorMode(0x8003)\n"
            "h = k.LoadLibraryExW(sys.argv[1], None, 0x8)\nprint('loaded' if h else f'FAILED WinError {ctypes.get_last_error()}')\n")
    r = subprocess.run([sys.executable, "-c", load, str(hits[0])], env=env, capture_output=True, text=True)
    print(f"RESULT {label} LoadLibrary(ggml-cuda.dll): {(r.stdout + r.stderr).strip()}")
    r = subprocess.run([str(bundle / "llama-server.exe"), "--list-devices"], env=env, cwd=bundle, capture_output=True, text=True, timeout=300)
    print(f"RESULT {label} llama-server --list-devices rc={r.returncode}:")
    for line in (r.stdout + r.stderr).strip().splitlines()[-5:]:
        print(f"    | {line[:200]}")
