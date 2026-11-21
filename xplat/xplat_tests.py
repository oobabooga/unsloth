#!/usr/bin/env python3
"""Run the llama.cpp cross-platform checks for one built state (base or head) and write JSON.

Same script on Linux, macOS and Windows (stdlib only). Observes, never judges: xplat_compare.py judges.

  python xplat_tests.py --bin BUILD/bin --state head --out OUT [--ops FLASH_ATTN_EXT_BANDED,FLASH_ATTN_EXT]
                        [--kv f16,q8_0,q4_0] [--archs-filter REGEX] [--tiny-archs inkling2,inkling2_loop]
                        [--ngl 99] [--timeout 2400]

Writes OUT/<state>/results.json plus every raw log next to it.
"""
import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")
EXE = ".exe" if os.name == "nt" else ""


def run(cmd, log, timeout, env=None):
    t0 = time.time()
    with open(log, "w", encoding="utf-8", errors="replace") as f:
        f.write("$ " + " ".join(map(str, cmd)) + "\n")
        f.flush()
        try:
            p = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, timeout=timeout,
                               env=env)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = "TIMEOUT"
        except FileNotFoundError as e:
            rc = f"MISSING: {e}"
    return rc, round(time.time() - t0, 1)


def read(log):
    return ANSI.sub("", Path(log).read_text(encoding="utf-8", errors="replace"))


def parse_tbo(text):
    res = {"ok": [], "fail": [], "unsupported": 0, "skipped_large": 0, "backends": [], "summaries": []}
    cur = None
    for line in text.splitlines():
        m = re.match(r"^Backend \d+/\d+: (\S+)", line)
        if m:
            cur = m.group(1)
            res["backends"].append(cur)
            continue
        m = re.match(r"^  (\w+\(.*\)): (.*)$", line)
        if m:
            name, st = m.group(1), m.group(2).strip()
            key = f"{cur}:{name}"
            if st.endswith("OK"):
                res["ok"].append(key)
            elif st.endswith("FAIL"):
                res["fail"].append(key)
            elif st.startswith("not supported"):
                res["unsupported"] += 1
            elif st.startswith("skipping large"):
                res["skipped_large"] += 1
            continue
        m = re.match(r"^\s*(\d+)/(\d+) (tests|backends) passed", line)
        if m:
            res["summaries"].append(line.strip())
    res["n_ok"], res["n_fail"] = len(res["ok"]), len(res["fail"])
    res["ok_sha"] = hashlib.sha256("\n".join(sorted(res["ok"])).encode()).hexdigest()[:16]
    del res["ok"]  # ids of passing tests are only kept as a hash; failures are listed
    return res


def parse_archs(text):
    rows, fails = [], []
    for line in text.splitlines():
        if line.startswith("|") and line.count("|") >= 6:
            cells = [c.strip() for c in line.strip("|").split("|")]
            if cells[0] in ("arch", "") or set(cells[0]) <= set("-:"):
                continue
            rows.append(cells)
            if any(c.startswith("FAIL") for c in cells[3:]):
                fails.append(" | ".join(cells))
    summ = [l.strip() for l in text.splitlines() if l.startswith("Summary")]
    ink = [" | ".join(c) for c in rows if c[0].startswith("inkling")]
    return {"rows": len(rows), "fail_rows": fails, "summary": summ, "inkling_rows": ink}


def text_corpus(path):
    # deterministic plain text, enough tokens for a few 256-token chunks with any tiny vocab
    words = ("the quick brown fox jumps over a lazy dog while seven bright stars rise above the quiet "
             "harbor and a small boat drifts toward the old stone lighthouse near the river").split()
    out, x = [], 12345
    for i in range(60000):
        x = (1103515245 * x + 12345) % (1 << 31)
        out.append(words[x % len(words)])
        if i % 17 == 16:
            out.append(".\n")
    Path(path).write_text(" ".join(out), encoding="utf-8")


def parse_ppl(text):
    r = {}
    m = re.search(r"Final estimate: PPL = ([0-9.eE+\-naninf]+) \+/- ([0-9.eE+\-naninf]+)", text)
    if m:
        r["ppl"], r["ppl_err"] = m.group(1), m.group(2)
    r["chunks"] = re.findall(r"\[(\d+)\]([0-9.eE+\-naninf]+)", text)
    r["compute_buffers"] = sorted(set(re.findall(r"(\S+) compute buffer size = +([0-9.]+) MiB", text)))
    r["kv_buffers"] = sorted(set(re.findall(r"(\S+) KV buffer size = +([0-9.]+) MiB", text)))
    r["offload"] = re.findall(r"offloaded (\d+/\d+) layers to GPU", text)
    r["model_buffers"] = sorted(set(re.findall(r"(\S+) model buffer size = +([0-9.]+) MiB", text)))
    r["graph_nodes"] = sorted(set(re.findall(r"graph: nodes = (\d+), splits = (\d+)", text)))
    r["fa"] = re.findall(r"flash_attn\s*=\s*(\S+)", text)[:1]
    r["suspicious_lines"] = [l for l in text.splitlines()
                             if re.search(r"GGML_ASSERT|Segmentation fault|\berror:|\bnan\b|\binf\b", l, re.I)
                             and not l.startswith("$ ") and "llama_model_loader" not in l and "llama_perf" not in l][:10]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ops", default="FLASH_ATTN_EXT_BANDED,FLASH_ATTN_EXT")
    ap.add_argument("--tbo-backend", default="", help="test-backend-ops -b NAME (default: every non-CPU)")
    ap.add_argument("--kv", default="f16,q8_0,q4_0")
    ap.add_argument("--tiny-archs", default="", help="also generate stock tiny models with test-llama-archs -o")
    ap.add_argument("--models", default="", help="comma list of tiny GGUF files to run (shipped with the kit)")
    ap.add_argument("--archs-filter", default="", help="test-llama-archs -a REGEX (default: all archs)")
    ap.add_argument("--ngl", default="99")
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--skip", default="", help="comma list of steps to skip: tbo,archs,tiny")
    a = ap.parse_args()

    bindir = Path(a.bin).resolve()
    out = Path(a.out).resolve() / a.state
    out.mkdir(parents=True, exist_ok=True)
    skip = set(filter(None, a.skip.split(",")))
    res = {"state": a.state, "bin": str(bindir), "platform": sys.platform, "steps": {}}

    def b(name):
        return bindir / f"{name}{EXE}"

    if "tbo" not in skip:
        for op in filter(None, a.ops.split(",")):
            log = out / f"tbo_{op}.log"
            cmd = [b("test-backend-ops"), "-o", op] + (["-b", a.tbo_backend] if a.tbo_backend else [])
            rc, dt = run(cmd, log, a.timeout)
            r = parse_tbo(read(log))
            r.update(rc=rc, seconds=dt)
            res["steps"][f"tbo_{op}"] = r
            print(f"[{a.state}] test-backend-ops -o {op}: rc={rc} ok={r['n_ok']} fail={r['n_fail']} "
                  f"unsupported={r['unsupported']} backends={r['backends']} {dt}s", flush=True)

    if "archs" not in skip:
        log = out / "test-llama-archs.log"
        cmd = [b("test-llama-archs")] + (["-a", a.archs_filter] if a.archs_filter else [])
        rc, dt = run(cmd, log, a.timeout)
        r = parse_archs(read(log))
        r.update(rc=rc, seconds=dt)
        res["steps"]["test-llama-archs"] = r
        print(f"[{a.state}] test-llama-archs: rc={rc} rows={r['rows']} fail_rows={len(r['fail_rows'])} "
              f"{r['summary']} {dt}s", flush=True)

    if "tiny" not in skip:
        tm = out / "tiny"
        tm.mkdir(exist_ok=True)
        corpus = out / "corpus.txt"
        text_corpus(corpus)
        tiny = {}
        groups = [(f"file:{Path(m).name}", None, [Path(m).resolve()]) for m in filter(None, a.models.split(","))]
        groups += [(arch, arch, None) for arch in filter(None, a.tiny_archs.split(","))]
        for key, arch, files in groups:
            rc = "shipped"
            if arch:
                d = tm / arch
                d.mkdir(exist_ok=True)
                rc, dt = run([b("test-llama-archs"), "-a", f"^{arch}$", "-s", "42", "-o", d],
                             out / f"gen_{arch}.log", 600)
                files = sorted(d.glob("*.gguf"))
            ggufs = files
            ent = {"gen_rc": rc, "files": {}}
            for g in ggufs:
                fe = {"sha256": hashlib.sha256(g.read_bytes()).hexdigest()[:16], "bytes": g.stat().st_size,
                      "kv": {}}
                for kv in filter(None, a.kv.split(",")):
                    for fa in ("on",):
                        log = out / f"ppl_{g.stem}_{kv}_fa{fa}.log"
                        cmd = [b("llama-perplexity"), "-m", g, "-f", corpus, "-c", "256", "-b", "256",
                               "--chunks", "4", "-fa", fa, "-ctk", kv, "-ctv", kv, "-ngl", a.ngl, "-t", "4",
                               "--seed", "42", "-lv", "4"]
                        rc2, dt2 = run(cmd, log, 900)
                        p = parse_ppl(read(log))
                        p.update(rc=rc2, seconds=dt2)
                        try:
                            p["finite"] = math.isfinite(float(p.get("ppl", "nan")))
                        except ValueError:
                            p["finite"] = False
                        fe["kv"][f"{kv}_fa{fa}"] = p
                        print(f"[{a.state}] {g.name} kv={kv} fa={fa}: rc={rc2} PPL={p.get('ppl')} "
                              f"compute={p['compute_buffers']} {dt2}s", flush=True)
                ent["files"][g.name] = fe
            tiny[key] = ent
        res["steps"]["tiny"] = tiny

    (out / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"wrote {out / 'results.json'}")


if __name__ == "__main__":
    main()
