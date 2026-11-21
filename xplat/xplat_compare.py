#!/usr/bin/env python3
"""Judge base vs head results from xplat_tests.py for one platform leg. Stdlib only.

  python xplat_compare.py --out OUT --leg NAME [--build-rc base=0,head=0] [--expect-banded yes|no|any]

Verdict rules (head vs base, same job, same host):
  - both builds must succeed (a base build failure = VOID: the harness, not the change)
  - every step's set of failing test IDs on head must be a subset of base's (no new FAIL)
  - head must not lose passing tests on an op (n_ok head >= n_ok base, except where tests were added)
  - tiny-model perplexity: every head run exits 0 with a finite PPL; quantized KV PPL within 5% of the
    same state's f16 PPL; the generated models must be byte-identical across states (same seed)
  - --expect-banded no: FLASH_ATTN_EXT_BANDED must have zero OK tests on head (op unsupported, dense path)
Writes OUT/VERDICT_<leg>.md and OUT/verdict_<leg>.json; exit 0 = NO_REGRESSION, 1 = REGRESSION, 2 = VOID.
"""
import argparse
import json
import math
from pathlib import Path


def load(out, s):
    p = Path(out) / s / "results.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--leg", required=True)
    ap.add_argument("--build-rc", default="")
    ap.add_argument("--expect-banded", default="any", choices=["yes", "no", "any"])
    ap.add_argument("--ppl-tol", type=float, default=0.05)
    ap.add_argument("--expect-quant-banded", default="any", choices=["yes", "no", "any"],
                    help="yes: head quantized-KV compute buffer == head f16 (banded engaged) and base's is larger "
                         "(dense); no: head quantized-KV compute buffer == base's (path unchanged)")
    a = ap.parse_args()
    brc = dict(kv.split("=") for kv in a.build_rc.split(",") if kv)
    base, head = load(a.out, "base"), load(a.out, "head")
    lines, problems, void = [f"# {a.leg}: base vs head", ""], [], []
    lines.append(f"build rc: base={brc.get('base', '?')} head={brc.get('head', '?')}")
    if brc.get("base", "0") != "0" or base is None:
        void.append("base did not build or produced no results")
    if brc.get("head", "0") != "0" or head is None:
        problems.append("head did not build or produced no results")
    if base and head:
        lines += ["", "| step | base ok/fail/unsup | head ok/fail/unsup | new FAIL on head |", "|---|---|---|---|"]
        for k in sorted(set(base["steps"]) | set(head["steps"])):
            if not k.startswith("tbo_"):
                continue
            bs, hs = base["steps"].get(k), head["steps"].get(k)
            if not hs:
                problems.append(f"{k}: missing on head")
                continue
            new = sorted(set(hs["fail"]) - set(bs["fail"] if bs else []))
            fmt = lambda s: f"{s['n_ok']}/{s['n_fail']}/{s['unsupported']} rc={s['rc']}" if s else "-"
            lines.append(f"| {k} | {fmt(bs)} | {fmt(hs)} | {len(new)} |")
            if new:
                problems.append(f"{k}: {len(new)} new FAIL on head: " + "; ".join(new[:20]))
            if hs["rc"] not in (0,) and not (bs and bs["rc"] == hs["rc"]):
                problems.append(f"{k}: head rc={hs['rc']} (base rc={bs['rc'] if bs else '-'})")
            if bs and hs["n_ok"] < bs["n_ok"]:
                problems.append(f"{k}: head passes fewer tests ({hs['n_ok']} < {bs['n_ok']})")
            if k == "tbo_FLASH_ATTN_EXT_BANDED" and a.expect_banded == "no" and hs["n_ok"] > 0:
                problems.append(f"{k}: banded op unexpectedly supported on this backend ({hs['n_ok']} OK)")
            if k == "tbo_FLASH_ATTN_EXT_BANDED" and a.expect_banded == "yes" and hs["n_ok"] == 0:
                problems.append(f"{k}: banded op expected to run on this backend but 0 OK")
            if hs.get("n_ok", 0) == 0 and hs.get("n_fail", 0) == 0 and a.expect_banded != "no":
                lines.append(f"  note: {k} ran no case on any non-CPU backend (CPU-only build or unsupported)")
        ba, ha = base["steps"].get("test-llama-archs"), head["steps"].get("test-llama-archs")
        if ha:
            newf = sorted(set(ha["fail_rows"]) - set(ba["fail_rows"] if ba else []))
            lines += ["", f"test-llama-archs: base rc={ba['rc'] if ba else '-'} rows={ba['rows'] if ba else '-'} "
                      f"fails={len(ba['fail_rows']) if ba else '-'} {ba['summary'] if ba else ''}; head rc={ha['rc']} "
                      f"rows={ha['rows']} fails={len(ha['fail_rows'])} {ha['summary']}"]
            lines += [f"  head {r}" for r in ha["inkling_rows"]]
            if newf:
                problems.append("test-llama-archs new FAIL rows: " + "; ".join(newf))
            if ha["rc"] != 0 and not (ba and ba["rc"] == ha["rc"]):
                problems.append(f"test-llama-archs head rc={ha['rc']}")
            for r in (ba or {}).get("fail_rows", []):
                lines.append(f"  pre-existing (base) FAIL row: {r}")
        bt, ht = base["steps"].get("tiny", {}), head["steps"].get("tiny", {})
        if ht:
            lines += ["", "| model | kv | base PPL | head PPL | base compute MiB | head compute MiB | base/head graph nodes |", "|---|---|---|---|---|---|---|"]
        for arch, he in ht.items():
            be = bt.get(arch, {})
            if not he.get("files"):
                problems.append(f"tiny {arch}: no model generated on head (rc={he.get('gen_rc')})")
            for fn, hf in he.get("files", {}).items():
                bf = be.get("files", {}).get(fn)
                if bf and bf["sha256"] != hf["sha256"]:
                    void.append(f"tiny {fn}: generated model differs base vs head ({bf['sha256']} vs {hf['sha256']})")
                ref = {}
                for st, f in (("base", bf), ("head", hf)):
                    if f and "f16_faon" in f["kv"] and f["kv"]["f16_faon"].get("finite"):
                        ref[st] = float(f["kv"]["f16_faon"]["ppl"])
                for kv, hp in hf["kv"].items():
                    bp = (bf or {}).get("kv", {}).get(kv, {})
                    cb = lambda p: ", ".join(f"{d}={m}" for d, m in p.get("compute_buffers", [])) or "-"
                    lines.append(f"| {fn} | {kv} | {bp.get('ppl', '-')} (rc={bp.get('rc', '-')}) | "
                                 f"{hp.get('ppl', '-')} (rc={hp.get('rc')}) | {cb(bp)} | {cb(hp)} | "
                                 f"{bp.get('graph_nodes', '-')} / {hp.get('graph_nodes', '-')} |")
                    if hp.get("rc") != 0 or not hp.get("finite"):
                        problems.append(f"tiny {fn} {kv}: head rc={hp.get('rc')} ppl={hp.get('ppl')} "
                                        f"{hp.get('suspicious_lines', [])[:3]}")
                    elif "head" in ref and abs(math.log(float(hp["ppl"]) / ref["head"])) > a.ppl_tol:
                        problems.append(f"tiny {fn} {kv}: head PPL {hp['ppl']} vs head f16 {ref['head']} "
                                        f"outside {a.ppl_tol:.0%}")
                    if kv != "f16_faon" and bp and hp.get("rc") == 0:
                        hf16 = hf["kv"].get("f16_faon", {}).get("compute_buffers")
                        bf16 = (bf or {}).get("kv", {}).get("f16_faon", {}).get("compute_buffers")
                        hc, bc = hp.get("compute_buffers"), bp.get("compute_buffers")
                        eng = "banded (== f16)" if hc == hf16 else "dense (!= f16)"
                        lines.append(f"  {fn} {kv}: head path {eng}; base path "
                                     f"{'banded (== f16)' if bc == bf16 else 'dense (!= f16)'}; "
                                     f"head PPL == base PPL: {hp.get('ppl') == bp.get('ppl')}")
                        if a.expect_quant_banded == "yes" and hc != hf16:
                            problems.append(f"tiny {fn} {kv}: head did not take the banded path ({hc} vs f16 {hf16})")
                        if a.expect_quant_banded == "yes" and bc == bf16:
                            void.append(f"tiny {fn} {kv}: base already matches f16 ({bc}): no defect shown on base")
                        if a.expect_quant_banded == "no" and hc != bc:
                            problems.append(f"tiny {fn} {kv}: path changed on a backend without the banded op "
                                            f"({bc} -> {hc})")
                    if kv == "f16_faon" and bp:
                        lines.append(f"  {fn} f16: head PPL == base PPL: {hp.get('ppl') == bp.get('ppl')} "
                                     f"({bp.get('ppl')} vs {hp.get('ppl')})")
                    if hp.get("suspicious_lines"):
                        lines.append(f"  head {fn} {kv} suspicious: {hp['suspicious_lines'][:3]}")
    verdict = "VOID" if void else ("REGRESSION" if problems else "NO_REGRESSION")
    lines += ["", f"VERDICT: {verdict}"] + [f"- VOID: {v}" for v in void] + [f"- {p}" for p in problems]
    md = "\n".join(lines) + "\n"
    Path(a.out, f"VERDICT_{a.leg}.md").write_text(md, encoding="utf-8")
    Path(a.out, f"verdict_{a.leg}.json").write_text(json.dumps(
        {"leg": a.leg, "verdict": verdict, "void": void, "problems": problems}, indent=1), encoding="utf-8")
    print(md)
    return {"NO_REGRESSION": 0, "REGRESSION": 1, "VOID": 2}[verdict]


if __name__ == "__main__":
    raise SystemExit(main())
