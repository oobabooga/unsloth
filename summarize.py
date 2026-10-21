import csv, statistics, sys
from collections import defaultdict
rows = []
for f in sys.argv[1:]:
    rows += list(csv.DictReader(open(f), delimiter="\t"))
g = defaultdict(list)
for r in rows:
    g[(r["cpus"], r["scenario"], r["variant"])].append(r)
print("| cpus | tree | variant | n | wall s (median) | CPU s (median) | CPU s (min..max) | peak ruff procs | peak ruff threads | peak kwarg pythons | rc | output sha |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for k in sorted(g):
    rs = g[k]
    cpu = [float(r["user"]) + float(r["sys"]) for r in rs]
    wall = [float(r["wall"]) for r in rs]
    print(f"| {k[0]} | {k[1]} | {k[2]} | {len(rs)} | {statistics.median(wall):.1f} | {statistics.median(cpu):.1f} | {min(cpu):.1f}..{max(cpu):.1f} | {max(int(r['peak_ruff']) for r in rs)} | {max(int(r['peak_ruff_threads']) for r in rs)} | {max(int(r['peak_kwargs_py']) for r in rs)} | {','.join(sorted(set(r['rc'] for r in rs)))} | {','.join(sorted(set(r['diff_sha'] for r in rs)))} |")
