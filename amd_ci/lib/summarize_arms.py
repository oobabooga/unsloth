#!/usr/bin/env python3
"""One table over every obs_*.json under out/D1 and out/controls, via the criteria's table()."""
import glob, importlib.util, json, sys
from pathlib import Path
root = Path(sys.argv[1]); crit_path = sys.argv[2]
spec = importlib.util.spec_from_file_location("crit", crit_path); crit = importlib.util.module_from_spec(spec); spec.loader.exec_module(crit)
obs = {}
for f in sorted(glob.glob(str(root / "D1" / "obs_*.json"))) + sorted(glob.glob(str(root / "controls" / "obs_*.json"))):
    try:
        d = json.loads(Path(f).read_text()); obs[d.get("state") or Path(f).stem] = d
    except Exception as e:  # noqa: BLE001
        obs[Path(f).stem] = {"setup_error": f"unreadable: {e}"}
print("## Every arm, one table\n"); print(crit.table(obs) if obs else "(no observations)"); print("\nfingerprints:\n")
for n, d in obs.items():
    fp = d.get("fingerprint") or {}
    print(f"- {n}: sha256 {str(fp.get('llama-server_sha256', '?'))[:16]}; libs {fp.get('backend_libs')}")
