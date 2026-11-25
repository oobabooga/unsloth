#!/usr/bin/env python3
"""Criteria: same-host XDNA2 NPU (Ryzen AI 1.9 + sd-sandbox) vs gfx1151 iGPU (diffusers ROCm),
SD-Turbo / SDXL-Turbo 512px 1 step, plus the Ryzen AI Linux tgz GenAI-SD search.

Exploration, not a PR differential: the heavy part ignores the checkout (NO_REGRESSION by
construction). Gates require both arms to have produced timings for SD-Turbo, so a broken
arm reads INCONCLUSIVE instead of a one-sided number. Pairs with probes/npu_vs_igpu_probe.py.
"""

from __future__ import annotations

import re
import statistics

TITLE = "XDNA2 NPU vs gfx1151 iGPU, SD-Turbo / SDXL-Turbo 512px 1 step (same host)"
MODE = "regression"
NEEDS: list[str] = ["gpu"]


def _heavy(obs):
    h = (obs.get("head") or {}).get("heavy")
    return h if isinstance(h, dict) else {}


def _npu(run: dict) -> dict:
    t = ((run or {}).get("profile") or {}).get("tail", "")
    o = {}
    m = re.search(r"average execution time \(excl\. 1st Gen\)\s*:\s*([\d.]+)s", t)
    if m:
        o["avg_s"] = float(m[1])
    m = re.search(r"warm-up execution time \(1st Gen\)\s*:\s*([\d.]+)s", t)
    if m:
        o["first_s"] = float(m[1])
    perf = t.rsplit("Pipeline Performance Summary", 1)[-1] if "Pipeline Performance Summary" in t else ""
    for comp in ("text_encoder", "unet", "transformer", "vae_decoder"):
        m = re.search(rf"\] {comp}\s+(\w+)\s+[\d.]+\s+([\d.]+)", perf)
        if m:
            o[comp] = f"{float(m[2]):.3f} ({m[1]})"
    return o


def _igpu(run: dict) -> dict:
    rs = ((run or {}).get("result") or {}).get("runs") or []
    w = rs[1:]
    if not w:
        return {"n": 0}
    med = lambda k: statistics.median(r[k] for r in w)  # noqa: E731
    return {"n": len(w), "first_s": rs[0]["total_s"], "total_s": med("total_s"), "te_s": med("te_s"),
            "denoise_s": med("denoise_s"), "vae_s": med("vae_s"),
            "spread_s": max(r["total_s"] for r in w) - min(r["total_s"] for r in w),
            "vae_dtype": w[0].get("vae_dtype"), "nan": any(r.get("nan") for r in rs)}


def gates(obs):
    h = _heavy(obs)
    n = _npu(((h.get("sandbox") or {}).get("runs") or {}).get("sd_turbo_512_1"))
    g = _igpu(((h.get("igpu") or {}).get("runs") or {}).get("sd_turbo_512_1"))
    return [("NPU SD-Turbo profiled", "avg_s" in n, str(n)),
            ("iGPU SD-Turbo timed (>= 4 warm runs)", g.get("n", 0) >= 4, str(g))]


def head_is_worse(base, head):
    return False, "exploration probe: measurement only"


def table(obs):
    h = _heavy(obs)
    sv = (obs.get("head") or {}).get("survey") or {}
    rai = h.get("rai") or {}
    rows = ["| item | value |", "|---|---|"]
    for k in ("os_release", "uname", "lspci_npu", "amdxdna_version", "flm_path", "lemonade", "xdna_procs"):
        rows.append(f"| {k} | {str(sv.get(k, '-')).replace(chr(10), ' / ').replace('|', '/')[:220]} |")
    for k in ("tgz_genai_dirs", "tgz_inner_hits", "install_script_genai", "genai_lib_copy", "venv_genai"):
        rows.append(f"| {k} | {str(rai.get(k, '-')).replace(chr(10), ' / ').replace('|', '/')[:400] or '(empty)'} |")
    sb = h.get("sandbox") or {}
    rows.append(f"| genai_lib_used | {sb.get('genai_lib_used')} {sb.get('genai_lib_from_venv', '')} |")
    rows += ["", "| model | NPU steady s/img | NPU 1st | NPU parts (avg s) | iGPU warm s/img (median) | iGPU 1st | iGPU te / denoise / vae | iGPU spread | NPU power | iGPU power |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    nr = sb.get("runs") or {}
    gr = (h.get("igpu") or {}).get("runs") or {}
    for tag in sorted(set(nr) | set(gr)):
        a, b = nr.get(tag) or {}, gr.get(tag) or {}
        n, g = _npu(a), _igpu(b)
        nstat = a.get("skipped") or ("cold rc=%s" % (a.get("cold") or {}).get("rc") if not n.get("avg_s") else "")
        parts = ", ".join(f"{k} {v}" for k, v in n.items() if k not in ("avg_s", "first_s"))
        gp = (b.get("power") or {})
        npw = ((a.get("profile") or {}).get("power") or {})
        rows.append(f"| {tag} | {n.get('avg_s', nstat or '-')} | {n.get('first_s', '-')} | {parts or '-'} | "
                    + (f"{g['total_s']:.3f} | {g['first_s']:.2f} | {g['te_s']:.3f} / {g['denoise_s']:.3f} / {g['vae_s']:.3f} ({g['vae_dtype']}, nan={g['nan']}) | {g['spread_s']:.3f} | "
                       if g.get("n") else f"- ({b.get('result_error') or (b.get('run') or {}).get('rc', 'not run')}) | - | - | - | ")
                    + f"{npw} | {gp} |")
    for tag, a in nr.items():
        c = a.get("cold") or {}
        if c.get("rc") not in (0, None):
            rows.append(f"\n{tag} NPU cold tail:\n```\n{str(c.get('tail', ''))[-1500:]}\n```")
    for tag, b in gr.items():
        if not _igpu(b).get("n"):
            rows.append(f"\n{tag} iGPU tail:\n```\n{str((b.get('run') or {}).get('out', ''))[-1500:]}\n```")
    ig = h.get("igpu") or {}
    rows.append(f"\niGPU torch: {(ig.get('torch') or {}).get('out', '').strip()[:200]}; deps: {(ig.get('deps') or {}).get('out', '').strip()[-200:]}")
    return "\n".join(rows)
