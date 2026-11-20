#!/usr/bin/env python3
"""Criteria for probes/i8auto_probe.py: should Studio's INT8 attention (Triton kernel) engage automatically on gfx1151?

The rule is NVIDIA's: a family engages with the switch unset where the kernel is accurate there and more than 2% faster
end to end. Accurate = the module's own accuracy check passes, and the teacher-forced per-step error of the DiT output
(rel L2 vs native bf16 attention, median and max over prompts x 3 steps) is within what the family's shipped INT8
weights cost (the bar the NVIDIA AUTO_DITS were held to, measured on NVIDIA: BAR below). Faster = forced INT8's median
s/image over the timed arms below RACE_MARGIN x native's, by more than native's own A/A spread.

Defect (base, read from the head observation, which runs the base tree's arms): some family is accurate and faster
with INT8 forced (F) while the base with the switch unset (N) runs it native (0 INT8 calls).
Fixed (head): every such family engages with the switch unset at the head (A: INT8 calls > 0) and is faster than N by
the same rule; no family where forced INT8 is inaccurate engages at the head; no family is slower at the head than N
beyond noise; no INT8 attention test that passes at the base fails at the head.

``gates`` sees both states; ``base_shows_defect`` gets the base observation only, so the head's (where the arms run)
is kept from ``gates`` for it.
"""

from __future__ import annotations

import statistics as st

TITLE = "gfx1151: Studio INT8 attention with the switch unset (auto) vs native vs forced, end to end (exclusive GPU)"
MODE = "differential"
NEEDS = ["gpu", "rocm", "exclusive_gpu", "nvidia", "discrete_gpu", "windows", "windows_rocm_wddm", "multi_gpu"]
RACE_MARGIN = 0.98
MIN_NOISE = 0.01
# per-step rel L2 vs bf16 attention of the shipped INT8 weights (median, max), NVIDIA teacher-forced runs:
# outputs/i8def/val2_g4_tables.md (FLUX.3 Klein, Z-Image-Turbo), outputs/int8_attn_families/REPORT.md (Qwen-Image-2.1-
# Turbo, Wan 2.2 TI2V-5B at 1280x704x121)
BAR = {"klein": (0.0118, 0.0155), "zimage": (0.0240, 0.0734), "q21t": (0.0084, 0.27), "wan": (0.0135, 0.031)}
NAMES = {"klein": "FLUX.3 Klein", "zimage": "Z-Image-Turbo", "q21t": "Qwen-Image-2.1-Turbo", "wan": "Wan2.2-TI2V-5B"}
_KEEP: dict = {}


def _arms(fobs: dict, side: str) -> list:
    return [r for t, r in sorted((fobs.get("arms") or {}).items()) if r.get("side") == side]


def _timed(rec: dict) -> list:
    return [x for x in rec.get("renders") or [] if x.get("kind") == "timed"]


def _arm_mean(rec: dict):
    w = [x["wall_s"] for x in _timed(rec) if x.get("wall_s")]
    return st.mean(w) if rec.get("verdict") == "ok" and w else None


def _calls(rec: dict, key: str = "int8_calls") -> list:
    return [x.get(key, 0) for x in _timed(rec)]


def _tf(rec: dict):
    rel = [lf["rel_l2"] for row in rec.get("tf_rows") or [] for lf in row.get("leaves") or [] if "rel_l2" in lf
           and lf.get("leaf", 0) == 0]
    return (st.median(rel), max(rel), len(rel)) if rel else None


def _lpips(fobs: dict, tag: str):
    for r in (fobs.get("scores") or {}).get("vs_ref") or []:
        if r.get("a", "").endswith("_" + tag):
            return r.get("lpips_mean"), r.get("lpips_max")
    return None, None


def _check(rec: dict) -> dict:
    """The module's own verdict lines from an arm's log tap (accuracy check + race)."""
    lines = [x for x in (rec.get("i8_log") or []) + (rec.get("i8_lines") or []) if "heads=" in x and "ok=" in x]
    return {"lines": lines[:6], "ok": bool(lines) and all("ok=True" in x for x in lines)} if lines else {}


def _probe(rec: dict) -> dict:
    rows = [r for r in rec.get("probe_rows") or [] if "rel_kernel" in r and r["k"][1] >= 1024]
    if not rows:
        return {}
    ker = [r["rel_kernel"] for r in rows]
    out = {"calls": len(rows), "kernel_med": st.median(ker), "kernel_max": max(ker),
           "sdpa_med": st.median([r["rel_sdpa"] for r in rows]),
           "sink_share": st.mean([r.get("sink_heads", 0) for r in rows])}
    fl = [r["rel_flash"] for r in rows if "rel_flash" in r]
    if fl:
        out["flash_med"] = st.median(fl)
    races = {}
    for r in rec.get("probe_rows") or []:
        if r.get("race_ms"):
            races[f"{r['q'][1]}x{r['k'][1]}x{r['q'][2]}"] = r["race_ms"]
    out["races"] = races
    return out


def family(fobs: dict, fam: str) -> dict:
    n = [m for m in (_arm_mean(r) for r in _arms(fobs, "N")) if m]
    f = [m for m in (_arm_mean(r) for r in _arms(fobs, "F")) if m]
    a = [m for m in (_arm_mean(r) for r in _arms(fobs, "A")) if m]
    res: dict = {"n": n, "f": f, "a": a}
    if not n:
        return res
    nat = st.median(n)
    within = [(max(w) - min(w)) / st.median(w) for w in ([x["wall_s"] for x in _timed(r)] for r in _arms(fobs, "N"))
              if len(w) > 1]
    spread = (max(n) - min(n)) / nat if len(n) > 1 else 0.0
    res.update(native = nat, spread = spread, within = max(within) if within else None,
               noise = max(spread, MIN_NOISE))
    for side, vals in (("f", f), ("a", a)):
        if vals:
            res[side + "_s"] = st.median(vals)
            res[side + "_ratio"] = st.median(vals) / nat
    res["n_int8"] = sum(sum(_calls(r)) for r in _arms(fobs, "N"))
    res["f_int8"] = [c for r in _arms(fobs, "F") for c in _calls(r)]
    res["a_int8"] = [c for r in _arms(fobs, "A") for c in _calls(r)]
    res["a_native"] = [c for r in _arms(fobs, "A") for c in _calls(r, "native_calls")]
    f1 = next((r for r in _arms(fobs, "F") if r.get("round") == 1), {})
    a1 = next((r for r in _arms(fobs, "A") if r.get("round") == 1), {})
    res["tf_f"], res["tf_a"] = _tf(f1), _tf(a1)
    res["lpips_f"] = [_lpips(fobs, f"F{i}")[0] for i in (1, 2, 3)]
    res["lpips_a"] = [_lpips(fobs, f"A{i}")[0] for i in (1, 2, 3)]
    res["lpips_aa"] = [_lpips(fobs, f"N{i}")[0] for i in (2, 3)]
    res["check_f"] = _check(f1)
    res["check_a"] = _check(a1)
    res["probe"] = _probe((fobs.get("arms") or {}).get("P") or {})
    bar = BAR.get(fam)
    tf = res["tf_f"]
    res["accurate"] = bool(tf and bar and tf[0] <= bar[0] and tf[1] <= bar[1]
                           and res["check_f"].get("ok", True))
    res["faster"] = bool(f and res["f_ratio"] < RACE_MARGIN and 1 - res["f_ratio"] > res["noise"])
    res["a_engaged"] = bool(res["a_int8"]) and all(c > 0 for c in res["a_int8"])
    res["a_faster"] = bool(a and res["a_ratio"] < RACE_MARGIN and 1 - res["a_ratio"] > res["noise"])
    res["a_slower"] = bool(a and res["a_ratio"] > 1 + res["noise"])
    res["defect"] = res["accurate"] and res["faster"] and res["n_int8"] == 0
    return res


def _fams(head: dict) -> dict:
    return {fam: family(fobs, fam) for fam, fobs in (head.get("families") or {}).items()}


def _pytest_new_failures(base: dict, head: dict) -> list:
    b = (base.get("pytest") or {}).get("cases") or {}
    h = (head.get("pytest") or {}).get("cases") or {}
    return sorted(k for k, v in h.items() if v == "failed" and b.get(k) == "passed")


def gates(obs: dict) -> list:
    head, base = obs.get("head") or {}, obs.get("base") or {}
    _KEEP["head"], _KEEP["base"] = head, base
    fams = _fams(head)
    out = [("head pytest ran", bool((head.get("pytest") or {}).get("cases")),
            f"{(head.get('pytest') or {}).get('passed')} passed / {(head.get('pytest') or {}).get('failed')} failed"),
           ("base pytest ran", bool((base.get("pytest") or {}).get("cases")),
            f"{(base.get('pytest') or {}).get('passed')} passed / {(base.get('pytest') or {}).get('failed')} failed"),
           ("gfx1151", "gfx1151" in str(head.get("arch")), str(head.get("arch")))]
    out.append(("at least one family measured", any(len(r["n"]) >= 2 and len(r["f"]) >= 2 for r in fams.values()),
                ", ".join(f"{k}: N {len(r['n'])} F {len(r['f'])} A {len(r['a'])}" for k, r in fams.items())))
    for fam, r in fams.items():
        if len(r["n"]) < 2 or len(r["f"]) < 2:
            continue
        out.append((f"{fam}: forced arms ran the INT8 kernel in every render", bool(r["f_int8"]) and all(
            c > 0 for c in r["f_int8"]), f"INT8 calls per render {r['f_int8']}"))
        out.append((f"{fam}: base unset arms ran native", r["n_int8"] == 0, f"INT8 calls {r['n_int8']}"))
        out.append((f"{fam}: teacher-forced rows", r["tf_f"] is not None, str(r["tf_f"])))
    return out


def base_shows_defect(base: dict) -> bool:
    return any(r["defect"] for r in _fams(_KEEP.get("head") or {}).values())


def head_is_fixed(head: dict) -> bool:
    if _pytest_new_failures(_KEEP.get("base") or {}, head):
        return False
    fams = _fams(head)
    for r in fams.values():
        if not r.get("n"):
            continue
        if r["defect"] and not (r["a_engaged"] and r["a_faster"]):
            return False
        if not r["accurate"] and any(c > 0 for c in r["a_int8"]):
            return False
        if r["a_slower"]:
            return False
    return True


def _f(x, nd = 3):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def table(obs: dict) -> str:
    head, base = obs.get("head") or {}, obs.get("base") or {}
    fams = _fams(head)
    L = ["| family | native s/img (A/A) | forced INT8 s/img | forced/native | auto s/img | auto/native | INT8 calls "
         "per render forced / auto | LPIPS forced / auto vs native (A/A) | per-step rel L2 forced med / max (bar) | "
         "accurate | faster | defect | auto engaged |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for fam, r in fams.items():
        if not r.get("n"):
            L.append(f"| {fam} | no native arm | | | | | | | | | | | |")
            continue
        bar = BAR.get(fam, (None, None))
        tf = r.get("tf_f") or (None, None, 0)

        def lp(xs):
            xs = [x for x in xs if x is not None]
            return _f(st.mean(xs), 4) if xs else "-"
        L.append(f"| {NAMES.get(fam, fam)} | {_f(r['native'])} ({_f(100 * r['spread'], 1)}%) | {_f(r.get('f_s'))} | "
                 f"{_f(r.get('f_ratio'))} | {_f(r.get('a_s'))} | {_f(r.get('a_ratio'))} | "
                 f"{sorted(set(r['f_int8']))} / {sorted(set(r['a_int8']))} | {lp(r['lpips_f'])} / {lp(r['lpips_a'])} "
                 f"({lp(r['lpips_aa'])}) | {_f(tf[0], 4)} / {_f(tf[1], 4)} ({bar[0]} / {bar[1]}) | "
                 f"{'yes' if r['accurate'] else 'NO'} | {'yes' if r['faster'] else 'no'} | "
                 f"{'yes' if r['defect'] else 'no'} | {'yes' if r['a_engaged'] else 'no'} |")
    L += ["", "Per call on the real activations (arm P: forced, compile off; calls with >= 1024 keys): rel L2 vs fp32 "
          "on 256 query rows, and each distinct shape raced on its own inputs (ms, median):", "",
          "| family | calls | kernel med / max | SDPA med | flash med | heads with a V sink | races (shape: int8 / sdpa / "
          "flash ms) |", "|---|---|---|---|---|---|---|"]
    for fam, r in fams.items():
        p = r.get("probe") or {}
        if not p:
            L.append(f"| {fam} | - | | | | | |")
            continue
        races = "; ".join(f"{k}: {v.get('int8')} / {v.get('sdpa')} / {v.get('flash', '-')}"
                          for k, v in list(p.get("races", {}).items())[:6])
        L.append(f"| {NAMES.get(fam, fam)} | {p['calls']} | {_f(p['kernel_med'], 4)} / {_f(p['kernel_max'], 4)} | "
                 f"{_f(p['sdpa_med'], 4)} | {_f(p.get('flash_med'), 4)} | {_f(p['sink_share'], 3)} | {races} |")
    sh = head.get("shapes") or {}
    if sh.get("rows"):
        L += ["", f"Per call, synthetic probe inputs at real shapes (head tree, forced; impl {sh.get('impl')}):", "",
              "| shape | int8 ms | sdpa ms | flash ms | int8 / best native | int8 rel L2 | sdpa rel L2 | first call s |",
              "|---|---|---|---|---|---|---|---|"]
        for row in sh["rows"]:
            ms = row.get("ms") or {}
            i8 = (ms.get("int8") or {}).get("median")
            nats = [(ms.get(k) or {}).get("median") for k in ("sdpa", "flash") if ms.get(k)]
            best = min(nats) if nats else None
            L.append(f"| {row['shape']} | {_f(i8)} | {_f((ms.get('sdpa') or {}).get('median'))} | "
                     f"{_f((ms.get('flash') or {}).get('median'))} | {_f(i8 / best if i8 and best else None)} | "
                     f"{_f((row.get('int8') or {}).get('rel_l2'), 4)} | {_f((row.get('sdpa') or {}).get('rel_l2'), 4)} | "
                     f"{_f(row.get('first_call_s'))} |" + (f" {row['error'][:120]}" if row.get("error") else ""))
        for d, c in (sh.get("checks") or {}).items():
            L.append(f"- module accuracy check head_dim {d}: ok {c.get('ok')} cos {_f(c.get('cos'), 5)} rel_l1 "
                     f"{_f(c.get('rel_l1'), 4)} sharp_cos {_f(c.get('sharp_cos'), 4)} race ms {c.get('ms')} min_tokens "
                     f"{c.get('min_tokens')} long {(c.get('long') or {}).get('ok')} ({c.get('s')} s incl. compile)")
    for fam, r in fams.items():
        for side in ("f", "a"):
            c = r.get("check_" + side) or {}
            for line in c.get("lines", [])[:2]:
                L.append(f"- {fam} {side.upper()}1 log: {line[:300]}")
        fobs = (head.get("families") or {}).get(fam) or {}
        a1 = next((x for x in _arms(fobs, "F") if x.get("round") == 1), {})
        n1 = next((x for x in _arms(fobs, "N") if x.get("round") == 1), {})
        st_ = (n1.get("status") or {})
        L.append(f"- {fam}: speed_optims {st_.get('speed_optims')} attention {st_.get('attention_backend')} quant "
                 f"{st_.get('transformer_quant')} cuda_graphs {st_.get('cuda_graphs')}; triton {a1.get('triton')} "
                 f"available {a1.get('triton_available')}; load s N1 {n1.get('load_s')} F1 {a1.get('load_s')}; "
                 f"warm s N1 {[w.get('wall_s') for w in n1.get('warm') or []]} F1 "
                 f"{[w.get('wall_s') for w in a1.get('warm') or []]}; params {n1.get('params')}")
        for t, x in sorted((fobs.get("arms") or {}).items()):
            if x.get("verdict") != "ok":
                L.append(f"- {fam} arm {t}: {x.get('verdict')} rc {x.get('rc')} {str(x.get('error'))[:400]}")
    nf = _pytest_new_failures(base, head)
    L.append(f"- pytest base {(base.get('pytest') or {}).get('passed')} passed / "
             f"{(base.get('pytest') or {}).get('failed')} failed; head {(head.get('pytest') or {}).get('passed')} "
             f"passed / {(head.get('pytest') or {}).get('failed')} failed; new at head: {nf[:10]}")
    return "\n".join(L)
