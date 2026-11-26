#!/usr/bin/env python3
"""Criteria for probes/a7_probe.py: round-7 A7 gfx1151 measurements. Arms differ by config / env / tree inside the
head step, so nothing is judged between states: the gates prove each requested item ran on gfx1151 with enough
arms (and that the GEMM checker can flag a corrupted row); KEEP / KILL is read from the observations
(outputs/a7/REPORT.md)."""

from __future__ import annotations

import statistics as st

TITLE = "gfx1151: round-7 A7 (GEMM corruption probe, #368 V^T layout, native mmap access, HSA env, batched Conv2d, FlyDSL)"
MODE = "regression"
NEEDS = ["gpu", "rocm", "exclusive_gpu", "nvidia", "windows", "multi_gpu"]


def _ok(rs):
    return [r for r in rs or [] if not r.get("error") and r.get("rc") == 0 and not r.get("missing")]


def gates(obs: dict) -> list:
    h = obs.get("head") or {}
    arch = (h.get("env") or {}).get("arch") or ""
    items = ((h.get("args") or {}).get("items") or "").split(",")
    out = [("gfx1151", arch.startswith("gfx1151"), arch)]
    for item in items:
        if item == "gemm":
            g = _ok(h.get("gemm"))
            sens = all(((r.get("selftest_mm") or {}).get("bad_rows") or 0) >= 1 for r in g)
            out.append(("gemm arms ok + checker flags a zeroed row", len(g) >= 2 and sens and
                        all(r.get("n_cases_ok", 0) >= 7 for r in g), f"{len(g)} ok, sens {sens}"))
        elif item == "flydsl":
            f = h.get("flydsl") or []
            ans = [r for r in f if r.get("imported") or r.get("import_error")]
            out.append(("flydsl answered (imported or import error recorded)", bool(ans), f"{len(ans)} of {len(f)}"))
        elif item == "vtb":
            n = len(_ok(h.get("vtb")))
            out.append(("vtb arms ok", n >= 25, f"{n} ok of {len(h.get('vtb') or [])}"))
        elif item == "conv":
            c = [r for r in _ok(h.get("conv")) if (r.get("e2e") or {}).get("speedup")]
            out.append(("conv e2e ran", len(c) >= 1, f"{len(c)}"))
        elif item == "mmap":
            n = len(_ok(h.get("mmap")))
            out.append(("mmap arms ok", n >= 6, f"{n} ok of {len(h.get('mmap') or [])}"))
        elif item == "vaee2e":
            rs = (h.get("vaee2e") or {}).values()
            n = sum(1 for r in rs if r.get("verdict") == "ok")
            out.append(("vaee2e arms ok", n >= 4, f"{n} ok"))
        elif item == "hsa":
            rs = (h.get("hsa") or {}).values()
            n = sum(1 for r in rs if r.get("verdict") == "ok")
            out.append(("hsa arms ok", n >= 5, f"{n} ok"))
    return out


def head_is_worse(base: dict, head: dict) -> tuple:
    return False, "measurement only (arms differ by config / env / tree inside the head step)"


def table(obs: dict) -> str:
    h = obs.get("head") or {}
    L = [f"env {h.get('env')} platform {h.get('platform')}", f"item seconds {h.get('item_s')}", ""]
    for r in h.get("gemm") or []:
        bad = [(c.get("lib"), c.get("op"), c.get("dtype"), c.get("bad_elems"), c.get("bad_rows_past_split"))
               for c in r.get("cases") or [] if c.get("bad_elems") or c.get("error")]
        L.append(f"gemm {r.get('arm')}: torch {r.get('torch')} hip {r.get('hip')} any_bad {r.get('any_bad')} "
                 f"cases {r.get('n_cases_ok')} blas {r.get('blas_lib')} bad/err {bad} err {str(r.get('error'))[-200:]}")
    for r in h.get("flydsl") or []:
        L.append(f"flydsl: imported {r.get('imported')} geomean {r.get('geomean_speedup_fly')} "
                 f"import_error {str(r.get('import_error'))[-300:]}")
    by: dict = {}
    for r in _ok(h.get("vtb")):
        by.setdefault((r.get("shape"), r.get("label")), []).append(r)
    for (shape, label), rs in sorted(by.items()):
        L.append(f"vtb {shape} {label}: med ms {[round(x['med_ms'], 2) for x in rs]} -> "
                 f"{round(st.median([x['med_ms'] for x in rs]), 2)} sha {sorted({x.get('out_sha') for x in rs})} "
                 f"plan {rs[0].get('plan_meta')}")
    for r in h.get("conv") or []:
        L.append(f"conv {r.get('model')}: e2e {r.get('e2e')} err {str(r.get('error'))[-200:]}")
        for m in r.get("micro") or []:
            L.append(f"  micro x{m.get('x')} w{m.get('w')} n{m.get('count')}: {m.get('stock_ms')} -> "
                     f"{m.get('lowered_ms')} ms ({m.get('speedup')}) rel {m.get('max_rel')}")
    for r in h.get("mmap") or []:
        L.append(f"mmap {r.get('model')} {r.get('mode')} {r.get('arm')}: cold {r.get('cold')} warm {r.get('warm')} "
                 f"build_s {r.get('build_s')} after_build {r.get('after_build')} to_cuda {r.get('to_cuda_gibps')} "
                 f"chk {r.get('checksum')} err {str(r.get('error'))[-200:]}")
    for tag, r in (h.get("hsa") or {}).items():
        rd = (r.get("renders") or [{}])[0]
        L.append(f"hsa {tag} {r.get('env')}: load_s {r.get('load_s')} render {rd.get('wall_s')} md5 {r.get('frames_md5')}"
                 f" kmsg {r.get('kmsg_before')} -> {r.get('kmsg_after')} verdict {r.get('verdict')}")
    for tag, r in (h.get("vaee2e") or {}).items():
        L.append(f"vaee2e {tag}: load_s {r.get('load_s')} renders "
                 f"{[(x.get('wall_s'), x.get('vae_plus_post_s'), x.get('denoise_s')) for x in r.get('renders') or []]} "
                 f"unfold lines {[x for x in r.get('engage_lines') or [] if 'unfold' in x][:2]} verdict {r.get('verdict')}")
    L.append(f"vaee2e scores {str(h.get('vaee2e_scores'))[:1500]}")
    return "\n".join(L)
