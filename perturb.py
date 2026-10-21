"""Deterministically un-format a sample of eligible files (spaces around '=' removed)."""
import random, re, subprocess, sys
from pathlib import Path

root, exclude, n = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
files = subprocess.run(["git", "-C", str(root), "ls-files", "*.py"], capture_output=True, text=True, check=True).stdout.split()
files = sorted(f for f in files if not re.search(exclude, f) and f != "scripts/enforce_kwargs_spacing.py")
pat = re.compile(r"(?<=[\w\)\]])[ ]=[ ](?=[^=\s])")
for f in random.Random(14174).sample(files, min(n, len(files))):
    p = root / f
    t = p.read_text(encoding="utf-8")
    p.write_text(pat.sub("=", t), encoding="utf-8", newline="")
print(f"perturbed {min(n, len(files))} of {len(files)} files")
