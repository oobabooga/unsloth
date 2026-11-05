#!/usr/bin/env python3
"""k19_tab.py IN_DIR OUT_DIR: tables from the k19 AMD artifacts (rows.jsonl of each GPU job + the hosted SRC_CHECK / build logs).
md5 groups per cell, speed medians per arm (A/A spread beside), engagement counts, test-backend-ops failing IDs head vs base, archs,
exe tests, PolyStrata. Exit 1 only when no GPU rows came back (a non-result); findings never fail it."""
import collections, glob, json, os, statistics, sys

IN, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
L = []
P = lambda *x: L.append(" ".join(str(y) for y in x))


def rows(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return [json.loads(l) for l in fh if l.strip()]


R = {}
for f in sorted(glob.glob(os.path.join(IN, "**", "rows.jsonl"), recursive=True)):
    osn = "windows" if "windows" in f else "linux"
    R.setdefault(osn, []).extend(rows(f))
P("# k19-int3 on gfx1151\n")
for f in sorted(glob.glob(os.path.join(IN, "**", "SRC_CHECK_*"), recursive=True)):
    with open(f, encoding="utf-8", errors="replace") as fh:
        P("- `%s`: %s" % (os.path.basename(f), " | ".join(l.strip() for l in fh if l.strip())[:600]))
med = lambda v: round(statistics.median(v), 2) if v else None
for osn, rs in sorted(R.items()):
    P("\n## %s (%d rows)\n" % (osn, len(rs)))
    P("### server cells: md5 (dec / p2k / p8k text), tok/s medians\n")
    P("| cell | arm | n | ok | dec tg | p2k pp | p2k tg | p8k tg | dec md5 | p2k md5 | p8k md5 | tiers / engagement | err |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    cells = collections.OrderedDict()
    for r in rs:
        if r.get("kind") == "server":
            cells.setdefault((r["model"], r["mode"], r.get("arm")), []).append(r)
    for (m, mode, arm), v in cells.items():
        g = lambda c, k: [x[c][k] for x in v if x.get(c) and x[c].get(k) is not None]
        md = lambda c: ",".join(sorted({(x.get(c) or {}).get("md5", "-")[:8] for x in v}))
        st = v[0].get("stats", {})
        eng = "sd%d p8%d pm%d hb%d dcpu%d cm%d/%d uma%d" % (st.get("small_dev_n", 0), st.get("place8_n", 0), st.get("placemid_n", 0),
                                                         st.get("hostbound_n", 0), st.get("draft_cpu_n", 0), st.get("cpu_miss_on_n", 0),
                                                         st.get("cpu_miss_off_n", 0), len(st.get("uma", [])))
        err = ";".join(sorted({(x.get("err") or "")[:60] for x in v if x.get("err")} | {"abort" for x in v if x.get("stats", {}).get("abort")}))
        P("| %s:%s | %s | %d | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (m, mode, arm, len(v), sum(1 for x in v if x.get("ok")),
          med(g("dec", "tg_tps")), med(g("p2k", "pp_tps")), med(g("p2k", "tg_tps")), med(g("p8k", "tg_tps")), md("dec"), md("p2k"), md("p8k"), eng, err))
    P("\n### tier lines (first launch of each arm)\n")
    seen = set()
    for (m, mode, arm), v in cells.items():
        tl = v[0].get("stats", {}).get("tier_lines") or []
        if tl and (m, mode, arm) not in seen:
            seen.add((m, mode, arm))
            P("- %s:%s %s: %s" % (m, mode, arm, " || ".join(tl[:5])[:900]))
    P("\n### test-backend-ops (failing IDs)\n")
    for r in rs:
        if r.get("kind") == "tbo":
            P("- %s %s rc=%s passed=%s n_fail=%d aborted=%s ids=%s" % (r.get("bin"), r.get("op")[:60], r.get("rc"), r.get("passed"), r.get("n_fail", 0),
                                                                    r.get("aborted"), (r.get("fail_ids") or [])[:12]))
    tb = {r["bin"]: set(r.get("fail_ids") or []) for r in rs if r.get("kind") == "tbo" and r.get("op") == "ALL"}
    if "t_i3" in tb and "t_up" in tb:
        P("- full suite: head-only failures %s; base-only %s" % (sorted(tb["t_i3"] - tb["t_up"])[:40], sorted(tb["t_up"] - tb["t_i3"])[:40]))
    P("\n### archs / exe\n")
    for r in rs:
        if r.get("kind") == "archs":
            for e in ("test-llama-archs", "test-llama-context"):
                d = r.get(e) or {}
                P("- archs %s %s rc=%s models=%s fail_n=%s rocm=%s %s" % (r.get("bin"), e, d.get("rc"), d.get("n_models"), d.get("fail_n"), d.get("rocm"),
                                                                     (d.get("fail_lines") or [])[:4]))
        if r.get("kind") == "exe":
            P("- exe %s %s rc=%s %s" % (r.get("bin"), r.get("exe"), r.get("rc"), (r.get("lines") or [])[-3:]))
        if r.get("kind") == "poly":
            P("- poly %s r%s ok=%s load=%s dec tg=%s p2k pp=%s tg=%s err=%s lines=%s" % (r.get("model"), r.get("rep"), r.get("ok"), r.get("load_s"),
              (r.get("dec") or {}).get("tg_tps"), (r.get("p2k") or {}).get("pp_tps"), (r.get("p2k") or {}).get("tg_tps"), r.get("err"),
              (r.get("poly_lines") or [])[:6]))
        if r.get("kind") == "devinfo":
            P("- devinfo %s %s %s" % (r.get("arm"), r.get("devs"), (r.get("version") or "")[-120:]))
with open(os.path.join(OUT, "TABLES.md"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(L) + "\n")
print("\n".join(L))
sys.exit(0 if R else 1)
