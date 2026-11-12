#!/usr/bin/env python3
"""Diagnostic (regression mode, never judges the PR): check C of rand4x_proof.py across torch versions on
gfx1151. Pairs with probes/zoo2241c_probe.py (head state only; base / merge record `skipped`).

Per lane (t212 2.12.1, t214 2.14.1, t215 2.15.0, each a fresh venv) and arm (stock, patched, rand4x_on,
fallback_random; on t214 also the warm pairs stock_fill_a -> patched_warm (PR head patch) and stock_fill_b ->
patched_fixed_warm (cache key tag in every thread)) it reports A / B / C / D as rand4x_proof.py computes them, C cold vs warm, C with the no-grad
forward routed through the training graph (C_fixgrad), inference-graph vs training-graph mask agreement, each
mask against eager tl.rand / rand4x references from the expected seed, and seed sensitivity.
Gates: each lane's venv is torch X.Y on HIP gfx1151 and every arm produced the C numbers.
"""

from __future__ import annotations

TITLE = "Diagnostic: rand4x check C (inference vs training graph dropout masks) on gfx1151, torch 2.12 / 2.14 / 2.15"
MODE = "regression"
NEEDS = ["rocm", "gpu", "nvidia", "exclusive_gpu"]
EXPECT = {"t212": (2, 12), "t214": (2, 14), "t215": (2, 15)}


def _vt(s):
    try:
        return tuple(int(x) for x in str(s).split("+")[0].split(".")[:2])
    except ValueError:
        return (0, 0)


def _bad(r):
    try:
        return {
            "A": r["A_dup_pct"] > 60 or abs(r["A_sigma"]) > 5,
            "B": r["B_agree_pct"] < 99.99,
            "C": r["C_fwd_rel_err"] > 1e-3 or r["C_gradA_rel_err"] > 1e-2,
            "D": abs(r["D_sigma"]) > 5 or abs(r["D_lag1_corr"]) > 0.01,
        }
    except (KeyError, TypeError):
        return None


def gates(obs):
    out = []
    h = obs.get("head") or {}
    out.append(("head has the patch module", h.get("has_patch_module") is True, str(h.get("patch_file"))))
    for lane, want in EXPECT.items():
        rec = (h.get("lanes") or {}).get(lane) or {}
        v = rec.get("venv") or {}
        env = v.get("env") or {}
        out.append((f"{lane}: fresh venv on HIP gfx1151, torch {want[0]}.{want[1]}",
                    not v.get("error") and bool(env.get("hip")) and str(env.get("arch") or "").startswith("gfx1151")
                    and env.get("cuda_available") is True and _vt(env.get("torch")) == want,
                    f"{v.get('error') or ''} {env} {v.get('pip_list')} {str(v.get('log_tail') or '')[-400:]}"))
        arms = rec.get("arms") or {}
        missing = [a for a, r in arms.items() if not r.get("skipped") and _bad(r) is None]
        out.append((f"{lane}: every arm produced A-D", bool(arms) and not missing,
                    f"rc {rec.get('rc')} arms {sorted(arms)} missing {missing} "
                    + " ".join(str((arms.get(a) or {}).get('error') or '') + str((arms.get(a) or {}).get('C_error') or '')[-300:]
                               for a in missing) + str(rec.get('arms_error') or '') + str(rec.get('log_tail') or '')[-300:]))
    return out


def _f(x, fmt):
    try:
        return format(x, fmt)
    except (TypeError, ValueError):
        return str(x)


def table(obs):
    h = obs.get("head") or {}
    rows = ["| lane | torch | arm | 4x path / cfg | rand4x in code | A dup % | B agree % | C fwd / grad (cold) | C fwd / grad (warm) "
            "| C_fixgrad fwd / grad | mask infer=train % (cold / warm) | infer = ref tl / 4x % | train = ref tl / 4x % "
            "| seed sens % | D sigma / lag1 | rng changed cold / warm | bad |",
            "|" + "---|" * 17]
    for lane in EXPECT:
        rec = (h.get("lanes") or {}).get(lane) or {}
        for arm, r in (rec.get("arms") or {}).items():
            if r.get("skipped") or r.get("error"):
                rows.append(f"| {lane} | | {arm} | {r.get('skipped') or r.get('error')} |" + " |" * 13)
                continue
            rows.append(
                f"| {lane} | {r.get('torch')} | {arm}{' (patched ' + str(r.get('patched')) + ')' if arm == 'patched' else ''} "
                f"| {r.get('has_4x_path')} / {r.get('use_rand4x_config')} | {r.get('code_rand4x')} "
                f"| {_f(r.get('A_dup_pct'), '.2f')} | {_f(r.get('B_agree_pct'), '.3f')} "
                f"| {_f(r.get('C_cold_fwd_rel_err'), '.2e')} / {_f(r.get('C_cold_gradA_rel_err'), '.2e')} "
                f"| {_f(r.get('C_warm_fwd_rel_err'), '.2e')} / {_f(r.get('C_warm_gradA_rel_err'), '.2e')} "
                f"| {_f(r.get('C_fixgrad_fwd_rel_err'), '.2e')} / {_f(r.get('C_fixgrad_gradA_rel_err'), '.2e')} "
                f"| {_f(r.get('mask_cold_infer_vs_train_agree_pct'), '.3f')} / {_f(r.get('mask_warm_infer_vs_train_agree_pct'), '.3f')} "
                f"| {_f(r.get('mask_cold_infer_eq_ref_tl_pct'), '.2f')} / {_f(r.get('mask_cold_infer_eq_ref_4x_pct'), '.2f')} "
                f"| {_f(r.get('mask_cold_train_eq_ref_tl_pct'), '.2f')} / {_f(r.get('mask_cold_train_eq_ref_4x_pct'), '.2f')} "
                f"| {_f(r.get('seed_sens_agree_pct'), '.2f')} | {_f(r.get('D_sigma'), '.2f')} / {_f(r.get('D_lag1_corr'), '.4f')} "
                f"| {r.get('rng_cold_changed_bytes')} / {r.get('rng_warm_changed_bytes')} | {_bad(r)} |")
    rows.append("")
    for lane in EXPECT:
        rec = (h.get("lanes") or {}).get(lane) or {}
        for arm, r in (rec.get("arms") or {}).items():
            errs = {k: str(v)[-400:] for k, v in r.items() if k.endswith("_error") and v}
            if errs:
                rows.append(f"{lane}/{arm} errors: {errs}")
            if r.get("code_rand_lines"):
                rows.append(f"{lane}/{arm} rand lines: {r.get('code_rand_lines')[:12]}")
    return "\n".join(rows)


def head_is_worse(base, head):
    lanes = (head or {}).get("lanes") or {}
    summ = []
    for lane, rec in lanes.items():
        for arm, r in (rec.get("arms") or {}).items():
            b = _bad(r)
            if b is not None:
                summ.append(f"{lane}/{arm} C={'bad' if b['C'] else 'ok'}")
    return False, "diagnostic only: " + ", ".join(summ)
