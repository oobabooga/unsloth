"""`tests` must not be a declared package, so a new test file cannot reach an install."""
import sys, pathlib
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

cfg = tomllib.loads(pathlib.Path("pyproject.toml").read_text(encoding = "utf-8"))
st = cfg.get("tool", {}).get("setuptools", {})
packages = st.get("packages", {})
print("tool.setuptools.packages:", packages)
if isinstance(packages, dict):
    find = packages.get("find", {})
    include, exclude = find.get("include", []), find.get("exclude", [])
    print("  include:", include)
    print("  exclude:", exclude)
    assert not any(p.split(".")[0] == "tests" or p in ("*", "") for p in include), include
elif isinstance(packages, list):
    assert not any(p.split(".")[0] == "tests" for p in packages), packages
print("package-data:", st.get("package-data", {}))
assert any(p == "tests*" or p == "tests" for p in exclude), exclude

man = pathlib.Path("MANIFEST.in")
assert man.exists(), "MANIFEST.in missing"
lines = [ln.strip() for ln in man.read_text(encoding = "utf-8").splitlines()]
print("MANIFEST prune lines:", [ln for ln in lines if ln.startswith("prune")])
assert "prune tests" in lines, "MANIFEST.in does not prune the root tests/ tree"
print("PACKAGING CHECK OK: tests/ excluded by packages.find AND pruned from the sdist")
