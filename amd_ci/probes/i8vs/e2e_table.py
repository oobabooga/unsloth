#!/usr/bin/env python3
"""Part 2 tables: python e2e_table.py CARD=RUN/art/out [...] > e2e.md

Per family and arm letter: processes, s/step (denoise seconds / steps, median of the process medians, min-max over
processes), vs D, A/A (D processes' spread), peak VRAM (nvidia-smi per process, torch allocated), engagement
(status backend, INT8 calls per render, attention kernels of the profiled 2-step render), LPIPS vs D1 and vs N."""
from __future__ import annotations

import glob
import json
import os
import statistics as S
import sys
from collections import defaultdict

ORDER = ["D", "S", "SP", "U", "G", "GP", "N", "K", "K2"]


def letter(tag):
    if tag in ("N", "K", "K2"):
        return tag
    t = tag.rstrip("0123456789")
    return t if t else tag


def main():
    for arg in sys.argv[1:]:
        card, out = arg.split("=", 1)
        recs = {}
        for p in glob.glob(os.path.join(out, "*/record.json")):
            r = json.load(open(p))
            recs[r["tag"]] = r
        fams = sorted({t.split("_", 1)[0] for t in recs})
        for fam in fams:
            fr = {t.split("_", 1)[1]: r for t, r in recs.items() if t.startswith(fam + "_")}
            sc = {}
            sp = os.path.join(out, f"scores_{fam}.json")
            if os.path.exists(sp):
                s = json.load(open(sp))
                for x in s.get("vs_ref", []) + s.get("pairs", []):
                    sc[(x["a"].split("_", 1)[1], x["b"].split("_", 1)[1])] = x
            any_r = next(iter(fr.values()))
            print(f"### {card} / {fam} ({any_r.get('gpu')}, params {any_r.get('params')})\n")
            by = defaultdict(list)
            for t, r in fr.items():
                by[letter(t)].append((t, r))
            dmed = None
            rows = []
            for L in ORDER + sorted(set(by) - set(ORDER)):
                if L not in by:
                    continue
                procs = sorted(by[L])
                per = []
                vram, alloc, i8, backs, kern, errs = [], [], [], set(), [], []
                bydev = {}
                for t, r in procs:
                    if r.get("verdict") != "ok":
                        errs.append(f"{t}: {str(r.get('error'))[:200]}")
                        continue
                    steps = (r.get("params") or {}).get("steps") or 1
                    xs = [x["denoise_s"] / steps for x in r.get("renders", []) if x.get("denoise_s")]
                    if xs:
                        per.append(S.median(xs))
                    vram += [x.get("smi_gib") for x in r.get("renders", []) if x.get("smi_gib")]
                    alloc += [x.get("alloc_gib") for x in r.get("renders", []) if x.get("alloc_gib")]
                    i8 += [x.get("int8_calls") for x in r.get("renders", [])]
                    for dv, n in (r.get("int8_by_device") or {}).items():
                        bydev[dv] = bydev.get(dv, 0) + n
                    st = r.get("status") or {}
                    backs.add(str(st.get("attention_backend")))
                    pr = r.get("profile") or {}
                    if pr.get("attn_kernels"):
                        kern = [k[0][:60] for k in pr["attn_kernels"][:3]]
                med = S.median(per) if per else None
                if L == "D":
                    dmed = med
                aa = (max(per) - min(per)) / med if per and len(per) > 1 and med else None
                vs = f"{(med / dmed - 1) * 100:+.1f}%" if med and dmed else "-"
                d1 = sc.get((f"{procs[0][0]}", "D1")) if procs[0][0] != "D1" else sc.get(("D2", "D1"))
                n1 = sc.get((procs[0][0], "N")) if procs[0][0] != "N" else None
                rows.append([L, len(per), f"{med:.4f}" if med else "-",
                             f"{min(per):.4f}-{max(per):.4f}" if per else "-", vs,
                             f"{aa * 100:.2f}%" if aa is not None else "-",
                             f"{max(vram):.2f} / {max(alloc):.2f}" if vram and alloc else "-",
                             ",".join(sorted(backs)), str(sorted(set(i8))[:3]), "; ".join(kern) or "-",
                             f"{d1['lpips_mean']:.4f} ({d1['lpips_max']:.4f})" if d1 and d1.get("lpips_mean") is not None else "-",
                             f"{n1['lpips_mean']:.4f} ({n1['lpips_max']:.4f})" if n1 and n1.get("lpips_mean") is not None else "-",
                             json.dumps(bydev) if bydev else "-",
                             "; ".join(errs)[:300] or ""])
            print("| arm | n ok | s/step median | per-process range | vs D | spread | peak VRAM GiB smi / alloc | "
                  "status backend | INT8 calls/render | attention kernels (profiled) | LPIPS vs D1 mean (max) | "
                  "LPIPS vs N mean (max) | INT8 calls by device (all processes) | errors |")
            print("|" + "---|" * 14)
            for r in rows:
                print("| " + " | ".join(map(str, r)) + " |")
            aa_pairs = [(k, v) for k, v in sc.items() if k in (("D2", "D1"), ("D3", "D1"), ("U2", "D1"))]
            if aa_pairs:
                print("\nA/A and repeat LPIPS: " + "; ".join(f"{a}:{b} {v['lpips_mean']}" for (a, b), v in aa_pairs))
            print()


if __name__ == "__main__":
    main()
