#!/usr/bin/env python3
"""Part 1 tables + verdicts from kb.json files: python kb_table.py CARD=PATH/kb.json [...] > tables.md

Speed verdict per cell and Sage build: r = t_sage / t_int8 (> 1 = INT8 faster). Tie when |r - 1| <= band, band =
max(A/A spread of the two arms, 1%), A/A spread = (max - min) / median of an arm's three per-pass medians.
Accuracy verdict (probe and sink inputs separately): a = relL1_sage / relL1_int8; INT8 more accurate when a > 1.1,
Sage when a < 1/1.1, else similar.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

SAGES = ("sage_hub", "sage2_pip", "sage1_pip", "sage3", "sage3_hnd")
REF = ("flash", "cudnn", "efficient", "sdpa")


def band(ms, a, b):
    return max(ms[a].get("aa") or 0, ms[b].get("aa") or 0, 0.01)


def sv(ms, s):
    """speed verdict of INT8 vs arm s: (ratio, 'faster'|'slower'|'tie', band)"""
    if "int8" not in ms or s not in ms:
        return None
    r = ms[s]["med"] / ms["int8"]["med"]
    bd = band(ms, "int8", s)
    return r, ("tie" if abs(r - 1) <= bd else ("faster" if r > 1 else "slower")), bd


def av(acc, s):
    if "int8" not in acc or s not in acc:
        return None
    a = acc[s]["rel_l1"] / max(acc["int8"]["rel_l1"], 1e-30)
    return a, ("int8" if a > 1.1 else ("sage" if a < 1 / 1.1 else "similar"))


def fmt(x, n = 3):
    return "-" if x is None else f"{x:.{n}f}"


def main():
    cards = []
    for arg in sys.argv[1:]:
        name, path = arg.split("=", 1)
        cards.append((name, json.load(open(path))))
    summary = defaultdict(lambda: defaultdict(int))
    for name, kb in cards:
        info = kb["info"]
        print(f"## {name}: {info['gpu']} (sm{''.join(map(str, info['cap']))}), torch {info['torch']}\n")
        print("Builds:")
        for b, x in info["builds"].items():
            if x.get("ok"):
                desc = x.get("metadata") or {k: x.get(k) for k in ("version", "commit", "nvcc") if x.get(k)}
                print(f"- `{b}`: loaded. {json.dumps(desc, default = str)[:400]}")
            else:
                print(f"- `{b}`: cannot load: {x.get('why', '')[:400]}")
        print()
        print("INT8 verdicts (verdict(), per dtype / heads / head_dim): " + "; ".join(
            f"{k} ok={v.get('ok')} impl={v.get('impl')} min_tokens={v.get('min_tokens')}"
            for k, v in kb["verdicts"].items()) + "\n")
        mx = info.get("max_clocks") or {}
        sm = [r["clk0"].get("sm_mhz") for r in kb["rows"] if r.get("clk0")] + \
             [r["clk1"].get("sm_mhz") for r in kb["rows"] if r.get("clk1")]
        sm = [x for x in sm if x]
        procs = {r["clk0"].get("procs") for r in kb["rows"] if r.get("clk0")}
        print(f"Clocks: SM {min(sm) if sm else '-'}-{max(sm) if sm else '-'} MHz at cell edges (max {mx.get('sm')}), "
              f"compute processes seen on the card: {sorted(x for x in procs if x is not None)}\n")
        present = [s for s in SAGES if any(s in r["ms"] for r in kb["rows"])]
        refs = [s for s in REF if any(s in r["ms"] for r in kb["rows"])]
        hdr = ["dtype", "D", "H", "S"] + ["int8 ms"] + [f"{s} ms" for s in present] + [f"{s} ms" for s in refs] + \
              [f"{s}/int8" for s in present] + ["best ref/int8", "A/A int8"]
        print("### Speed (ms per call, median; ratio > 1 = INT8 faster)\n")
        print("| " + " | ".join(hdr) + " |")
        print("|" + "---|" * len(hdr))
        for r in kb["rows"]:
            ms = r["ms"]
            if not ms:
                continue
            cells = [r["dtype"], r["D"], r["H"], r["S"], fmt(ms.get("int8", {}).get("med"))]
            cells += [fmt(ms.get(s, {}).get("med")) for s in present]
            cells += [fmt(ms.get(s, {}).get("med")) for s in refs]
            for s in present:
                v = sv(ms, s)
                cells.append("-" if v is None else f"{v[0]:.2f}x {v[1]}")
                if v:
                    summary[(name, s)][v[1]] += 1
            rb = [ms[x]["med"] for x in refs if x in ms]
            cells.append(f"{min(rb) / ms['int8']['med']:.2f}x" if rb and "int8" in ms else "-")
            cells.append(fmt(ms.get("int8", {}).get("aa"), 4))
            print("| " + " | ".join(map(str, cells)) + " |")
        print()
        print("### Accuracy vs fp32 (cosine / rel-L1 / max-abs), probe and sink inputs\n")
        arms = ["int8"] + present + ["flash", "efficient"]
        hdr = ["dtype", "D", "H", "S", "inputs"] + arms + [f"{s} rel-L1 / int8" for s in present]
        print("| " + " | ".join(hdr) + " |")
        print("|" + "---|" * len(hdr))
        for r in kb["rows"]:
            for inp in ("probe", "sink"):
                acc = r["acc"].get(inp) or {}
                if not acc:
                    continue
                cells = [r["dtype"], r["D"], r["H"], r["S"], inp]
                for s in arms:
                    x = acc.get(s)
                    cells.append("-" if x is None else f"{x['cos']:.6f} / {x['rel_l1']:.4f} / {x['max_abs']:.3g}")
                for s in present:
                    v = av(acc, s)
                    cells.append("-" if v is None else f"{v[0]:.2f}x ({v[1]})")
                    if v:
                        summary[(name, s, inp)][v[1]] += 1
                print("| " + " | ".join(map(str, cells)) + " |")
        print()
        errs = {k: v for r in kb["rows"] for k, v in r["err"].items()}
        fat = [(r["dtype"], r["D"], r["H"], r["S"], r["fatal"]) for r in kb["rows"] if r.get("fatal")]
        skips = sorted({f"{k}: {v}" for r in kb["rows"] for k, v in r["skip"].items()})
        print("Arm errors (first per arm/input): " + "; ".join(f"`{k}` {v[:160]}" for k, v in sorted(errs.items())[:20]))
        print("\nSkipped: " + "; ".join(skips[:12]))
        if fat:
            print("\nFatal cells: " + "; ".join(map(str, fat[:10])))
        print("\nKernels profiled (dtype_h_d_s/arm: top CUDA kernels by time):")
        for k, v in kb.get("kernels", {}).items():
            if isinstance(v, list):
                print(f"- `{k}`: " + ", ".join(f"{n[:70]} ({t} ms x{c})" for n, t, c in v[:3]))
            else:
                print(f"- `{k}`: {v}")
        print(f"\nSTATS: {kb.get('int8_stats')}; Sage 2 dispatch counts: {kb.get('sage_dispatch_counts')}\n")
    print("## Verdict counts\n")
    for k, v in summary.items():
        print(f"- {k}: {dict(v)}")


if __name__ == "__main__":
    main()
