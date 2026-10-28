"""Lay out AMD HIP SDK 7.2 files exactly where its MSIs put them, without running AMD's installer
(which also installs a display driver). The MSIs set only HIP_PATH / HIP_PATH_72 and never touch PATH.

usage: python prep_sdk.py <installer.exe> <7zr.exe> <work dir> <install root to create>
"""

import shutil
import subprocess
import sys
from pathlib import Path

exe, sevenzr, work, root = (Path(a) for a in sys.argv[1:5])
work.mkdir(parents = True, exist_ok = True)

# The installer is a 7z SFX: copy the archive that follows the PE stub.
sig = b"7z\xbc\xaf\x27\x1c"
with open(exe, "rb") as f:
    head = f.read(8 << 20)
    off = head.find(sig)
    assert off > 0, "no 7z payload"
    f.seek(off)
    archive = work / "payload.7z"
    with open(archive, "wb") as out:
        shutil.copyfileobj(f, out, 16 << 20)
print("7z payload at offset", off)

pkgs = "Packages/Apps/ROCmSDKPackages"
subprocess.run(
    [str(sevenzr), "x", "-y", f"-o{work / 'x'}", str(archive),
     f"{pkgs}/SDKCore/ROCm_SDK_Core.msi", f"{pkgs}/LibrariesRuntime/ROCm_Libs_RT.msi"],
    check = True, stdout = subprocess.DEVNULL,
)
archive.unlink()
for msi in ("SDKCore/ROCm_SDK_Core.msi", "LibrariesRuntime/ROCm_Libs_RT.msi"):
    dest = work / Path(msi).stem
    subprocess.run(
        [sys.executable, "-m", "pymsi", "extract", "-o", str(dest), str(work / "x" / pkgs / msi)], check = True
    )
    tree = next(dest.rglob("ROCm")) / "7.2"
    shutil.copytree(tree, root, dirs_exist_ok = True)
    shutil.rmtree(dest)
shutil.rmtree(work / "x")
bindir = root / "bin"
for dll in ("hipblas.dll", "rocblas.dll", "hipblaslt.dll", "amdhip64_7.dll"):
    print(dll, (bindir / dll).is_file())
print("rocblas gfx1151 kernels:", len(list((bindir / "rocblas" / "library").glob("*gfx1151*"))))
