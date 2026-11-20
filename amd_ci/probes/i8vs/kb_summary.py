#!/usr/bin/env python3
"""Compact per card x shape verdict grids from kb.json: python kb_summary.py CARD=kb.json [...] > summary.md
Speed cell: t_sage / t_int8 and F (INT8 faster) / S (INT8 slower) / T (tie: inside max(A/A of both arms, 1%)).
Accuracy cell: relL1_sage / relL1_int8 on probe | sink inputs (> 1 = INT8 closer to fp32)."""
from __future__ import annotations

import json
import math
import sys

from kb_table import av, sv

SAGES = ("sage_hub", "sage2_pip", "sage3", "sage3_hnd")


def main():
    for arg in sys.argv[1:]:
        card, path = arg.split("=", 1)
        kb = json.load(open(path))
        rows = [r for r in kb["rows"] if r["ms"].get("int8")]
        toks = sorted({r["S"] for r in rows})
        present = [s for s in SAGES if any(s in r["ms"] for r in rows)]
        for s in present:
            print(f"#### {card}: INT8 vs `{s}` (speed: t_{s} / t_int8, verdict; accuracy: rel-L1 ratio probe | sink)\n")
            print("| dtype | D | H | " + " | ".join(f"{t // 1024}k speed" for t in toks) + " | "
                  + " | ".join(f"{t // 1024}k acc" for t in toks) + " |")
            print("|" + "---|" * (3 + 2 * len(toks)))
            keys = sorted({(r["dtype"], r["D"], r["H"]) for r in rows}, key = lambda x: (x[0] != "bfloat16", x[1], x[2]))
            for key in keys:
                cells, accs = [], []
                for t in toks:
                    r = next((x for x in rows if (x["dtype"], x["D"], x["H"], x["S"]) == (*key, t)), None)
                    v = sv(r["ms"], s) if r else None
                    cells.append("-" if v is None else f"{v[0]:.2f} {v[1][0].upper()}")
                    ap = av(r["acc"].get("probe", {}), s) if r else None
                    ak = av(r["acc"].get("sink", {}), s) if r else None
                    accs.append("-" if ap is None else f"{ap[0]:.2f} / {ak[0]:.2f}" if ak else f"{ap[0]:.2f}")
                print(f"| {key[0]} | {key[1]} | {key[2]} | " + " | ".join(cells) + " | " + " | ".join(accs) + " |")
            # aggregates
            rs = [sv(r["ms"], s) for r in rows if s in r["ms"]]
            rs4 = [sv(r["ms"], s) for r in rows if s in r["ms"] and r["S"] >= 4096]
            gm = lambda xs: math.exp(sum(math.log(x[0]) for x in xs) / len(xs)) if xs else None  # noqa: E731
            cnt = {k: sum(1 for x in rs if x[1] == k) for k in ("faster", "tie", "slower")}
            cnt4 = {k: sum(1 for x in rs4 if x[1] == k) for k in ("faster", "tie", "slower")}
            ap = [av(r["acc"]["probe"], s)[0] for r in rows if s in r["acc"].get("probe", {}) and "int8" in r["acc"]["probe"]]
            ak = [av(r["acc"]["sink"], s)[0] for r in rows if s in r["acc"].get("sink", {}) and "int8" in r["acc"]["sink"]]
            print(f"\nAll cells: {cnt}, geomean {gm(rs):.3f}x; from 4k tokens: {cnt4}, geomean {gm(rs4):.3f}x, "
                  f"range {min(x[0] for x in rs4):.2f}-{max(x[0] for x in rs4):.2f}x. Accuracy rel-L1 ratio (sage / int8): "
                  f"probe {min(ap):.2f}-{max(ap):.2f}, sink {min(ak):.2f}-{max(ak):.2f}.\n")
        # INT8 vs the best torch SDPA reference
        print(f"#### {card}: INT8 vs best torch SDPA kernel (flash / cuDNN / efficient / unpinned), t_ref / t_int8\n")
        print("| dtype | D | H | " + " | ".join(f"{t // 1024}k" for t in toks) + " |")
        print("|" + "---|" * (3 + len(toks)))
        keys = sorted({(r["dtype"], r["D"], r["H"]) for r in rows}, key = lambda x: (x[0] != "bfloat16", x[1], x[2]))
        for key in keys:
            cells = []
            for t in toks:
                r = next((x for x in rows if (x["dtype"], x["D"], x["H"], x["S"]) == (*key, t)), None)
                refs = {n: r["ms"][n]["med"] for n in ("flash", "cudnn", "efficient", "sdpa") if r and n in r["ms"]}
                if not refs:
                    cells.append("-")
                    continue
                n = min(refs, key = refs.get)
                cells.append(f"{refs[n] / r['ms']['int8']['med']:.2f} ({n})")
            print(f"| {key[0]} | {key[1]} | {key[2]} | " + " | ".join(cells) + " |")
        print()


if __name__ == "__main__":
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    main()
