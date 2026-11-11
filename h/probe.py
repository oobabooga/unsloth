"""Loader probe for a Windows CUDA llama.cpp bundle.

  probe.py gen-stub <ggml-cuda.dll> <out.c>   write a C stub exporting what the DLL imports from nvcuda.dll
  probe.py probe <bundle_dir> <driver_dir> <label>

The runner has no NVIDIA driver, so nvcuda.dll is supplied by a stub in <driver_dir>, the same one
for every bundle probed. Everything else resolves from the bundle directory and System32 only.
"""
from __future__ import annotations

import os
import struct
import subprocess
import sys
from pathlib import Path

SYSTEM32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"


def pe_imports(path: Path) -> dict[str, list[str]]:
    """{imported dll name: [symbol names]} from the import and delay-import tables of a PE32+ file."""
    data = path.read_bytes()
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    assert data[pe:pe + 4] == b"PE\0\0", path
    nsec = struct.unpack_from("<H", data, pe + 6)[0]
    optsize = struct.unpack_from("<H", data, pe + 20)[0]
    opt = pe + 24
    assert struct.unpack_from("<H", data, opt)[0] == 0x20B, "PE32+ only"
    dirs = opt + 112
    sections = []
    for i in range(nsec):
        s = opt + optsize + 40 * i
        vsize, va, rsize, raw = struct.unpack_from("<IIII", data, s + 8)
        sections.append((va, max(vsize, rsize), raw))

    def off(rva: int) -> int:
        for va, size, raw in sections:
            if va <= rva < va + size:
                return raw + rva - va
        raise ValueError(f"rva {rva:#x} outside sections")

    def cstr(rva: int) -> str:
        o = off(rva)
        return data[o:data.index(b"\0", o)].decode("ascii", "replace")

    def thunks(rva: int) -> list[str]:
        names, o = [], off(rva)
        while True:
            v = struct.unpack_from("<Q", data, o)[0]
            if v == 0:
                return names
            names.append(f"#{v & 0xFFFF}" if v >> 63 else cstr((v & 0x7FFFFFFF) + 2))
            o += 8

    out: dict[str, list[str]] = {}
    imp_rva = struct.unpack_from("<I", data, dirs + 8 * 1)[0]
    if imp_rva:
        o = off(imp_rva)
        while True:
            oft, _, _, name, ft = struct.unpack_from("<IIIII", data, o)
            if not name:
                break
            out.setdefault(cstr(name), []).extend(thunks(oft or ft))
            o += 20
    delay_rva = struct.unpack_from("<I", data, dirs + 8 * 13)[0]
    if delay_rva:
        o = off(delay_rva)
        while True:
            _attr, name, _hmod, _iat, int_rva = struct.unpack_from("<IIIII", data, o)
            if not name:
                break
            out.setdefault(cstr(name), []).extend(thunks(int_rva) if int_rva else [])
            o += 32
    return out


def unresolved(root: Path, dirs: list[Path]) -> list[str]:
    """DLL names in the import closure of `root` that are in none of `dirs` (API sets count as present)."""
    index: dict[str, Path] = {}
    for d in dirs:
        for p in d.glob("*.dll"):
            index.setdefault(p.name.lower(), p)
    missing, seen, todo = set(), set(), [root]
    while todo:
        cur = todo.pop()
        for dll in pe_imports(cur):
            low = dll.lower()
            if low in seen or low.startswith(("api-ms-win-", "ext-ms-")):
                continue
            seen.add(low)
            hit = index.get(low)
            if hit is None:
                missing.add(dll)
            elif hit.parent != SYSTEM32:
                todo.append(hit)
    return sorted(missing, key=str.lower)


def main() -> int:
    if sys.argv[1] == "gen-stub":
        syms = pe_imports(Path(sys.argv[2])).get("nvcuda.dll", [])
        if not syms:
            print("ggml-cuda.dll imports nothing from nvcuda.dll; no stand-in needed")
            return 0
        assert not any(s.startswith("#") for s in syms), syms
        Path(sys.argv[3]).write_text("".join(f"__declspec(dllexport) int {s}(void) {{ return 999; }}\n" for s in syms))
        print("nvcuda.dll symbols imported by ggml-cuda.dll:", ", ".join(syms))
        return 0

    bundle, driver, label = Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve(), sys.argv[4]
    cuda = bundle / "ggml-cuda.dll"
    print(f"===== {label}: {bundle}")
    runtime = sorted(p.name for p in bundle.glob("*.dll") if p.name.lower().startswith(("cudart", "cublas")))
    print(f"RESULT {label} runtime DLLs in bundle: {runtime or 'none'}")
    for name in runtime:
        print(f"         {name}  {(bundle / name).stat().st_size / 1048576:.1f} MiB")
    print(f"RESULT {label} ggml-cuda.dll imports: {sorted(d for d in pe_imports(cuda) if not d.lower().startswith('api-ms'))}")
    no_driver = unresolved(cuda, [bundle, SYSTEM32])
    with_driver = unresolved(cuda, [bundle, driver, SYSTEM32])
    print(f"RESULT {label} unresolved without an NVIDIA driver: {no_driver or 'none'}")
    print(f"RESULT {label} unresolved with nvcuda.dll present:  {with_driver or 'none'}")

    env = {k: v for k, v in os.environ.items() if k.upper() not in ("PATH", "CUDA_PATH", "CUDA_HOME")}
    env["PATH"] = os.pathsep.join([str(SYSTEM32), str(SYSTEM32.parent), str(driver)])
    load = (
        "import ctypes, sys\n"
        "k = ctypes.WinDLL('kernel32', use_last_error=True)\n"
        "k.LoadLibraryExW.restype = ctypes.c_void_p\n"
        "k.LoadLibraryExW.argtypes = [ctypes.c_wchar_p, ctypes.c_void_p, ctypes.c_uint32]\n"
        "k.SetErrorMode(0x8003)\n"
        "h = k.LoadLibraryExW(sys.argv[1], None, 0x8)\n"
        "print('loaded' if h else f'FAILED WinError {ctypes.get_last_error()}: {ctypes.FormatError(ctypes.get_last_error())}')\n"
    )
    r = subprocess.run([sys.executable, "-c", load, str(cuda)], env=env, capture_output=True, text=True)
    print(f"RESULT {label} LoadLibrary(ggml-cuda.dll): {(r.stdout + r.stderr).strip()}")
    r = subprocess.run([str(bundle / "llama-server.exe"), "--list-devices"], env=env, cwd=bundle,
                       capture_output=True, text=True, timeout=300)
    text = (r.stdout + r.stderr).strip().splitlines()
    print(f"RESULT {label} llama-server --list-devices rc={r.returncode}, {len(text)} lines:")
    for line in text[-25:]:
        print(f"    | {line[:220]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
