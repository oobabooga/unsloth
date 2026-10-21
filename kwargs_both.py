"""asottile's 'python-based formatting' as one hook: the pre pass, then the post pass, per file."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import enforce_kwargs_spacing as k

self_path = Path(k.__file__).resolve()
for name in sys.argv[1:]:
    p = Path(name)
    if p.resolve() == self_path or not p.is_file():
        continue
    k.process_file(p, pre=True)
    k.process_file(p, pre=False)
