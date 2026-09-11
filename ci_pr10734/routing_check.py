"""Deterministic routing: the only non-test callers of the changed loader are unsloth/__init__.py,
unsloth/_gpu_init.py (re-exports) and unsloth-cli.py. Nothing under studio/, installers, or Desktop."""
import os, re, sys

PAT = re.compile(r"RawTextDataLoader|smart_chunk_text|dataprep\.raw_text|dataprep import raw_text")
ALLOWED = {
    "unsloth/__init__.py", "unsloth/_gpu_init.py", "unsloth-cli.py", "unsloth/dataprep/raw_text.py",
}
hits = {}
for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in {".git", "node_modules", "ci_pr10734", "__pycache__"}]
    for f in files:
        p = os.path.relpath(os.path.join(root, f)).replace(os.sep, "/")
        try:
            text = open(p, encoding = "utf-8").read()
        except (UnicodeDecodeError, OSError):
            continue
        if PAT.search(text):
            hits[p] = len(PAT.findall(text))
non_test = sorted(p for p in hits if not p.startswith("tests/"))
print("all references:", hits)
print("non-test references:", non_test)
assert set(non_test) <= ALLOWED, set(non_test) - ALLOWED
studio = [p for p in hits if p.startswith(("studio/", "install", "unsloth_cli/"))]
assert not studio, studio
print("ROUTING CHECK PASS")
