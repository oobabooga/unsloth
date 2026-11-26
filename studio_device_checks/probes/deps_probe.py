"""Does the installed venv carry what this backend tree needs? Compares the tree's own requirement files with the
installed distributions and imports the modules the checks use.
usage: deps_probe.py --out FILE --modules a,b,c"""

from __future__ import annotations

import importlib
import importlib.metadata as md
import os
import sys

import _dc

REQ_FILES = ("studio.txt", "no-torch-runtime.txt", "diffusers-pin.txt")


def _requirements(path: str):
    try:
        from packaging.requirements import Requirement
    except Exception:  # noqa: BLE001
        return None, "packaging is not importable"
    reqs = []
    with open(path, encoding = "utf-8") as fh:
        for raw in fh:
            line = raw.split(" #", 1)[0].strip()
            if not line or line.startswith(("#", "-", "--")):
                continue
            try:
                reqs.append(Requirement(line))
            except Exception:  # noqa: BLE001 - not a plain requirement line
                continue
    return reqs, None


def main() -> None:
    out = _dc.out_path()
    args = sys.argv[3:]
    modules = []
    if "--modules" in args:
        modules = [m for m in args[args.index("--modules") + 1].split(",") if m]
    rec = {"tree": _dc.enter_tree(), "missing": [], "mismatch": [], "import_errors": {}, "req_errors": []}
    backend = _dc.tree_backend()
    for name in REQ_FILES:
        path = os.path.join(backend, "requirements", name)
        if not os.path.isfile(path):
            continue
        reqs, problem = _requirements(path)
        if problem:
            rec["req_errors"].append(problem)
            break
        for req in reqs:
            try:
                if req.marker is not None and not req.marker.evaluate():
                    continue
            except Exception:  # noqa: BLE001
                continue
            try:
                have = md.version(req.name)
            except md.PackageNotFoundError:
                rec["missing"].append({"name": req.name, "wants": str(req), "file": name})
                continue
            if req.specifier and not req.specifier.contains(have, prereleases = True):
                rec["mismatch"].append({"name": req.name, "wants": str(req.specifier), "have": have, "file": name})
    for mod in modules:
        try:
            importlib.import_module(mod)
        except ModuleNotFoundError as exc:
            rec["import_errors"][mod] = {"error": _dc.err(exc), "missing_module": exc.name}
        except Exception as exc:  # noqa: BLE001
            rec["import_errors"][mod] = {"error": _dc.err(exc), "missing_module": None, "tb": _dc.tb(1500)}
    _dc.write(out, rec)


if __name__ == "__main__":
    main()
