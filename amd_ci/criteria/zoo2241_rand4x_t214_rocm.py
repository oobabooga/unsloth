#!/usr/bin/env python3
"""Criteria (unsloth-zoo PR 2241, Inductor rand4x off), lane `t214`: torch 2.14.1 ROCm on gfx1151, base (zoo main)
vs head. Pairs with probes/zoo2241_probe.py --lane t214.

Differential mode. Thresholds are rand4x_proof.py's `bad()`:
  A  dropout after transpose().contiguous(): channel-copy rate > 60 % (independent = 50 %) or |keep sigma| > 5
  B  checkpoint(use_reentrant=False) forward vs backward mask agreement < 99.99 %
  C  Unsloth-style reentrant checkpoint of a LoRA dropout: fwd rel err > 1e-3 or grad rel err > 1e-2
  D  plain contiguous dropout control: |keep sigma| > 5 or |lag-1 corr| > 0.01
Base shows the defect: after importing the base's unsloth_zoo, A, B or C is bad (a statistical defect, not just
`rand4x(` in the generated code). If not, the defect does not reproduce on gfx1151: VOID.
Head is fixed: importing the head's unsloth_zoo replaced the 4x path and tagged the cache key; A-D all clean and no
`rand4x(` in the generated code; and rand4x_proof.py against the head's patch module exits 0 (base arm bad,
patched cold and warm-cache arms clean).
Gates: venv on HIP gfx1151 with torch 2.14; the 4x path exists before the zoo import in both states; each state
imported its own unsloth_zoo; the dropout cell finished in both states on the GPU; head has the patch module and
base does not; the head proof ran (exit 0, 1 or 3).
"""

from __future__ import annotations

TITLE = "unsloth-zoo PR 2241 (rand4x off) on gfx1151 torch 2.14.1: dropout defect at base, fixed at head"
MODE = "differential"
NEEDS = ["rocm", "gpu", "nvidia", "exclusive_gpu"]


def _vt(s):
    try:
        return tuple(int(x) for x in str(s).split("+")[0].split(".")[:2])
    except ValueError:
        return (0, 0)


def _bad(r):
    r = r or {}
    try:
        return {
            "A": r["A_dup_pct"] > 60 or abs(r["A_sigma"]) > 5,
            "B": r["B_agree_pct"] < 99.99,
            "C": r["C_fwd_rel_err"] > 1e-3 or r["C_gradA_rel_err"] > 1e-2,
            "D": abs(r["D_sigma"]) > 5 or abs(r["D_lag1_corr"]) > 0.01,
            "code": bool(r["code_rand4x"]),
        }
    except (KeyError, TypeError):
        return None


def _drop(o):
    return ((o or {}).get("dropout") or {}).get("dropout")


def _state(o):
    return ((o or {}).get("dropout") or {}).get("state") or ((o or {}).get("import_state") or {}).get("state") or {}


def gates(obs):
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        v = o.get("venv") or {}
        env = v.get("env") or {}
        out.append((f"{name} venv on HIP gfx1151, torch 2.14",
                    not v.get("error") and bool(env.get("hip")) and str(env.get("arch") or "").startswith("gfx1151")
                    and env.get("cuda_available") is True and _vt(env.get("torch")) == (2, 14) and env.get("matmul_smoke") is not None,
                    f"{v.get('error') or ''} {env} steps {v.get('steps')} triton {v.get('triton_fixup')} {str(v.get('log_tail') or '')[-400:]}"))
        ck = str(o.get("checkout") or "<none>")
        st = _state(o)
        f, f2 = str(o.get("unsloth_zoo_file") or ""), str(st.get("unsloth_zoo_file") or "")
        out.append((f"{name} imported its own unsloth_zoo (probe and cell)", bool(f) and f.startswith(ck) and f2.startswith(ck),
                    f"{f or o.get('unsloth_zoo_import_error')} | cell {f2}"))
        out.append((f"{name}: torch 2.14 has the 4x path", st.get("has_4x_path_before_import") is True,
                    f"has_4x_path {st.get('has_4x_path_before_import')} use_rand4x config {st.get('upstream_use_rand4x_config')}"))
        d = o.get("dropout") or {}
        out.append((f"{name} dropout cell finished on the gfx1151 GPU",
                    d.get("stage") == "done" and _bad(_drop(o)) is not None
                    and str(((st.get("device") or {}).get("arch")) or "").startswith("gfx1151"),
                    f"stage {d.get('stage')} rc {d.get('rc')} fatal {d.get('fatal')} device {st.get('device')} "
                    f"{str(d.get('traceback') or d.get('log_tail') or '')[-500:] if d.get('stage') != 'done' else ''}"))
    h, b = obs.get("head") or {}, obs.get("base") or {}
    out.append(("head has the patch module, base does not",
                h.get("has_patch_module") is True and b.get("has_patch_module") is False,
                f"head {h.get('has_patch_module')} base {b.get('has_patch_module')}"))
    pr = h.get("proof") or {}
    out.append(("head rand4x_proof.py ran (exit 0 / 1 / 3)", pr.get("rc") in (0, 1, 3),
                f"rc {pr.get('rc')} {str(pr.get('tail') or '')[-300:]}"))
    return out


def base_shows_defect(base):
    v = _bad(_drop(base))
    return bool(v) and (v["A"] or v["B"] or v["C"])


def head_is_fixed(head):
    st, v = _state(head), _bad(_drop(head))
    return (bool(v) and st.get("patched") is True and st.get("tag_present") is True
            and not any(v.values()) and (head.get("proof") or {}).get("rc") == 0)


def table(obs):
    rows = ["| state | patched | tag | A dup % / sigma | B agree % | C fwd / grad rel | D sigma / lag1 | rand4x in code | bad |",
            "|" + "---|" * 9]
    for name in ("base", "head"):
        o = obs.get(name) or {}
        st, r = _state(o), _drop(o) or {}
        if not r:
            rows.append(f"| {name} | {st.get('patched')} | {st.get('tag_present')} | no data | | | | | |")
            continue
        rows.append(f"| {name} | {st.get('patched')} | {st.get('tag_present')} | {r.get('A_dup_pct'):.2f} / {r.get('A_sigma'):.2f} | "
                    f"{r.get('B_agree_pct'):.4f} | {r.get('C_fwd_rel_err'):.2e} / {r.get('C_gradA_rel_err'):.2e} | "
                    f"{r.get('D_sigma'):.2f} / {r.get('D_lag1_corr'):.4f} | {r.get('code_rand4x')} | {_bad(r)} |")
    pr = (obs.get("head") or {}).get("proof") or {}
    rows += ["", f"head rand4x_proof.py exit {pr.get('rc')} (0 pass / 1 patched arm still bad / 3 void: base arm clean)",
             "```", str(pr.get("tail") or "")[-3500:], "```"]
    for name in ("base", "head"):
        p = (obs.get(name) or {}).get("pytest") or {}
        bad = sorted(set(p.get("failed") or []) | set(p.get("errors") or []))
        rows.append(f"{name} pytest {p.get('n_passed')}/{p.get('n_failed')}/{p.get('n_skipped')} (passed/failed/skipped), failing: {bad or 'none'}")
    v = (obs.get("head") or {}).get("venv") or {}
    rows += ["", f"env: {v.get('env')}; {v.get('pip_list')}; triton fixup {v.get('triton_fixup')}"]
    return "\n".join(rows)
