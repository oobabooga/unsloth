"""Print every step and note from a downloaded evidence directory."""
import json, pathlib, sys

for d in sys.argv[1:]:
    d = pathlib.Path(d)
    print("=" * 30, d)
    for f in sorted(d.rglob("results*.json")):
        print("--", f.parent)
        for r in json.loads(f.read_text()):
            v = r.get("value") if r["ok"] else r.get("error")
            print(f"{'OK  ' if r['ok'] else 'FAIL'} {r['test']:<42} {r['seconds']:>7}s  {json.dumps(v, default=str)[:400]}")
    for f in sorted(d.rglob("notes*.json")):
        print("-- notes", f.parent)
        for n in json.loads(f.read_text()):
            print("  *", n["note"][:500])
