#!/usr/bin/env python3
"""Portability harness: is a kernel / quantized path correct, engaged and faster on THIS device?

A case is a named Python callable ``module:function`` taking a ``portlib.Ctx`` and returning (or appending to
``ctx.rows``) rows made with ``ctx.row(...)``. Each row says: runs / fallback / fails / refuses / skipped, whether the
fast path actually engaged, its error against a float32/float64 eager reference, and its speed against the device's
base dtype (bf16, fp16 on pre-Ampere NVIDIA) eager and compiled.

    python harness.py --list
    python harness.py --cases all --quick                      # -> outputs/portability/<device>/
    python harness.py --cases int8_gemm,conv_lowering --out DIR
    python harness.py --case-def mine=my_pkg.my_mod:run        # add a case without editing this file

Every case runs in its own subprocess (a Triton fault or illegal address in one cannot take down the others) and
writes ``<out>/parts/<case>.json``; the parent merges them into ``<out>/results.json`` and ``<out>/report.md``.
``--bundle`` points at a snapshot built by ``snapshot.py`` (PR kernel sources + verifier); default: the newest local one.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# name -> "module:callable". Order is run order.
CASES = {
    "int8_gemm": "cases.int8_gemm:run",
    "h3vae_triton": "cases.h3vae_triton:run",
    "nvfp4_gate": "cases.nvfp4_gate:run",
    "conv_lowering": "cases.conv_lowering:run",
    "studio_compile": "cases.studio_compile:run",
}

WS = os.environ.get("WORKSPACE") or os.path.abspath(os.path.join(HERE, "..", ".."))


def resolve(spec: str):
    import importlib

    mod, _, fn = spec.partition(":")
    return getattr(importlib.import_module(mod), fn or "run")


def run_child(args) -> int:
    import portlib

    info = portlib.device_facts()
    ctx = portlib.Ctx(args.child, args.out, args.quick, args.iters, args.bundle, info)
    t0 = time.time()
    status = "ok"
    try:
        fn = resolve(args.spec)
        extra = fn(ctx)
        if isinstance(extra, list):
            for r in extra:
                if r not in ctx.rows:
                    ctx.rows.append(r)
    except Exception as exc:  # noqa: BLE001
        status = "error"
        ctx.fail("<case>", exc)
    portlib.dump(os.path.join(args.out, "parts", f"{args.child}.json"),
                 {"case": args.child, "spec": args.spec, "status": status, "seconds": round(time.time() - t0, 1),
                  "device": info, "rows": ctx.rows})
    return 0


def _manifest(bundle):
    try:
        with open(os.path.join(bundle, "MANIFEST.json"), encoding = "utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def merge(out: str, device: dict, case_meta: dict, bundle: str | None = None) -> dict:
    rows = []
    for name in case_meta:
        p = os.path.join(out, "parts", f"{name}.json")
        if os.path.exists(p):
            with open(p, encoding = "utf-8") as f:
                part = json.load(f)
            rows += part["rows"]
            case_meta[name].update({"status": part.get("status"), "seconds": part.get("seconds")})
        else:
            rows.append({"case": name, "variant": "<case>", "status": "fails",
                         "note": f"child died: {case_meta[name].get('exit')}; {case_meta[name].get('stderr_tail', '')[-300:]}"})
    res = {"device": device, "cases": case_meta, "rows": rows, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "bundle": _manifest(bundle)}
    with open(os.path.join(out, "results.json"), "w", encoding = "utf-8") as f:
        json.dump(res, f, indent = 1, default = str)
    write_report(out, res)
    return res


def write_report(out: str, res: dict) -> None:
    from portlib import _f

    d = res["device"]
    L = [f"# Portability: {d.get('name')} ({d.get('arch')}), {res['time']}", "",
         f"torch {d.get('torch')}, triton {d.get('triton')}, torchao {d.get('torchao')}, cuda {d.get('cuda_runtime')}, "
         f"hip {d.get('hip')}, {d.get('platform')}", "",
         "| case | variant | shape | dtype | status | engaged | max abs | rel rms | ms | x eager | x compiled | note |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        e = r.get("err") or {}
        note = str(r.get("note", "")).replace("|", "/").replace("\n", " ")[:160]
        L.append(f"| {r.get('case')} | {r.get('variant')} | {r.get('shape', '')} | {r.get('dtype', '')} | "
                 f"{r.get('status')} | {_f(r.get('engaged'))} | {_f(e.get('max_abs'))} | {_f(e.get('rel_rms'))} | "
                 f"{_f(r.get('ms'))} | {_f(r.get('speedup_vs_eager'))} | {_f(r.get('speedup_vs_compiled'))} | {note} |")
    with open(os.path.join(out, "report.md"), "w", encoding = "utf-8") as f:
        f.write("\n".join(L) + "\n")


def default_bundle() -> str | None:
    p = os.path.join(WS, "temp", "portability_bundle")
    return p if os.path.isdir(p) else None


def main() -> int:
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default = "all", help = "comma list of case names, or all")
    ap.add_argument("--case-def", action = "append", default = [], help = "extra case NAME=module:callable")
    ap.add_argument("--out", default = None, help = "output dir (default outputs/portability/<device>)")
    ap.add_argument("--quick", action = "store_true", help = "fewer / smaller shapes (Colab, CI)")
    ap.add_argument("--iters", type = int, default = 20)
    ap.add_argument("--bundle", default = None, help = "snapshot dir from snapshot.py")
    ap.add_argument("--timeout", type = int, default = 3600, help = "seconds per case")
    ap.add_argument("--no-isolate", action = "store_true", help = "run cases in-process (debugging)")
    ap.add_argument("--list", action = "store_true")
    ap.add_argument("--child", default = None, help = argparse.SUPPRESS)
    ap.add_argument("--spec", default = None, help = argparse.SUPPRESS)
    args = ap.parse_args()

    cases = dict(CASES)
    for d in args.case_def:
        k, _, v = d.partition("=")
        cases[k.strip()] = v.strip()
    if args.list:
        for k, v in cases.items():
            print(f"{k:<16} {v}")
        return 0
    args.bundle = args.bundle or default_bundle()
    if args.child:
        return run_child(args)

    import portlib

    info = portlib.device_facts()
    tag = portlib.device_tag(info)
    args.out = args.out or os.path.join(WS, "outputs", "portability", tag)
    os.makedirs(os.path.join(args.out, "parts"), exist_ok = True)
    print(f"device {tag}: {json.dumps(info)}", flush = True)
    print(f"bundle {args.bundle}  out {args.out}", flush = True)
    names = list(cases) if args.cases == "all" else [c.strip() for c in args.cases.split(",") if c.strip()]
    meta = {}
    for name in names:
        spec = cases[name]
        meta[name] = {"spec": spec}
        stale = os.path.join(args.out, "parts", f"{name}.json")
        if os.path.exists(stale):
            os.remove(stale)
        t0 = time.time()
        print(f"\n===== case {name} ({spec}) =====", flush = True)
        if args.no_isolate:
            ns = argparse.Namespace(**vars(args))
            ns.child, ns.spec = name, spec
            run_child(ns)
            meta[name]["exit"] = 0
            continue
        cmd = [sys.executable, "-u", os.path.abspath(__file__), "--child", name, "--spec", spec, "--out", args.out,
               "--iters", str(args.iters)] + (["--quick"] if args.quick else []) + (
            ["--bundle", args.bundle] if args.bundle else [])
        tail = []
        try:
            p = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, text = True,
                                 encoding = "utf-8", errors = "replace", cwd = args.out)
            import threading

            timer = threading.Timer(args.timeout, p.kill)
            timer.start()
            for line in p.stdout:
                sys.stdout.write(line)
                tail.append(line)
                tail = tail[-40:]
            rc = p.wait()
            if not timer.is_alive():
                tail.append(f"TIMEOUT after {args.timeout}s (killed)\n")
            timer.cancel()
        except Exception as exc:  # noqa: BLE001
            rc = f"spawn failed: {exc}"
        meta[name].update({"exit": rc, "wall_s": round(time.time() - t0, 1), "stderr_tail": "".join(tail[-12:])})
        sys.stdout.flush()
    res = merge(args.out, info, meta, args.bundle)
    bad = [r for r in res["rows"] if r.get("status") == "fails"]
    print(f"\nwrote {args.out}/results.json and report.md: {len(res['rows'])} rows, {len(bad)} fails", flush = True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
