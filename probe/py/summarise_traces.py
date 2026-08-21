"""Group @@@MEM_GET_INFO@@@ stacks under the @@@REQUEST ...@@@ marker they follow."""
import sys

req = "(startup / background warm)"
kind = "?"
stacks = []
cur = None
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    if line.startswith("@@@REQUEST "):
        req = line.strip()[len("@@@REQUEST "):-3]
        continue
    if line.startswith("@@@BASELINE_AFTER_WARM@@@"):
        req = "(idle, after warm)"
        continue
    if line.startswith("@@@MEM_GET_INFO@@@") or line.startswith("@@@AMD_SMI@@@"):
        cur = []
        kind = line.strip().strip("@")
        continue
    if line.startswith("@@@END@@@"):
        if cur is not None:
            stacks.append((req + "  [" + kind + "]", cur))
        cur = None
        continue
    if cur is not None:
        cur.append(line.rstrip("\n"))

print("total traced native calls:", len(stacks))
seen = {}
for req, frames in stacks:
    interesting = [f for f in frames if "site-packages/torch" not in f and 'File "<' not in f]
    key = (req, "\n".join(interesting[-14:]))
    seen[key] = seen.get(key, 0) + 1
for (req, frames), n in sorted(seen.items(), key=lambda kv: -kv[1]):
    print("\n" + "=" * 70)
    print("REQUEST:", req, " calls:", n)
    print(frames)
