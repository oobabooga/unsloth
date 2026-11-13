#!/usr/bin/env python3
"""Criteria for probes/sdpa_bwd_layout_probe.py (unsloth#17548). Regression mode: head must not fail a
compiled attention case or tiny Qwen3.5 step that base passed, must not turn off an SDPA backend whose
raw backward strides match its fake kernel, and must not add more than 3 s to `import unsloth`. The
table also reports whether the defect reproduces at all on this host (raw stride mismatches, compiled
stride asserts without unsloth)."""

from __future__ import annotations

TITLE = "ROCm SDPA backward strides vs fake kernels, and compiled attention, base versus head"
MODE = "regression"
NEEDS = ["rocm", "gpu", "discrete_gpu", "multi_gpu"]
IMPORT_SLACK_S = 3.0


def _states(obs: dict) -> dict:
    return {k: v for k, v in obs.items() if not k.startswith("_")}


def _envs(state: dict) -> dict:
    return {k: v for k, v in (state.get("envs") or {}).items() if "unavailable" not in v}


def _raw_mismatch(env: dict) -> list[str]:
    rows = (env.get("raw") or {}).get("raw_ops") or []
    return sorted({r["backend"] for r in rows if r.get("mismatch")})


def _ok_cases(u: dict) -> dict:
    out = {f"compiled {c['shape']}": bool(c.get("ok")) for c in u.get("compiled_default") or []}
    for i, s in enumerate((u.get("tiny_qwen35") or {}).get("steps") or []):
        out[f"qwen3.5 step {i} bs={s['bs']}"] = "loss" in s
    return out


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name, st in _states(obs).items():
        envs = _envs(st)
        out.append((f"{name}: probe ran", st.get("_probe_rc") == 0 and bool(envs), f"rc={st.get('_probe_rc')} envs={list(envs)}"))
        for tag, e in envs.items():
            r, u = e.get("raw") or {}, e.get("unsloth") or {}
            out.append((f"{name}/{tag}: ROCm torch on a GPU", bool(r.get("hip")) and bool(r.get("available")),
                        f"{r.get('torch')} hip={r.get('hip')} {r.get('device')}"))
            out.append((f"{name}/{tag}: unsloth imported from this state's checkout", bool(u.get("unsloth_from_checkout")),
                        f"{u.get('unsloth_file')} fatal={u.get('fatal', '')[:200]}"))
            ran = [x for x in (r.get("raw_ops") or []) if "real" in x]
            out.append((f"{name}/{tag}: at least one fused backward ran for real", bool(ran),
                        f"{len(ran)}/{len(r.get('raw_ops') or [])} raw cases ran"))
    return out


def table(obs: dict) -> str:
    rows = ["| state | env | torch / arch / FA lib | raw stride mismatch | compiled, no unsloth (forced+default) | flags after import unsloth | compiled after import | tiny Qwen3.5 | import s |",
            "|---|---|---|---|---|---|---|---|---|"]
    for name, st in _states(obs).items():
        for tag, e in (st.get("envs") or {}).items():
            if "unavailable" in e:
                rows.append(f"| {name} | {tag} | {e['unavailable']} | | | | | | |")
                continue
            r, u = e.get("raw") or {}, e.get("unsloth") or {}
            nou = [f"{c['backend'][:5]}{c['shape'][2:4]}:{'ok' if c.get('ok') else ('STRIDE' if c.get('stride_assert') else 'err')}"
                   for c in (r.get("compiled_forced") or []) + (r.get("compiled_default_no_unsloth") or [])]
            comp = [f"{c['shape'][2:4]}:{'ok' if c.get('ok') else ('STRIDE' if c.get('stride_assert') else 'err')}"
                    for c in u.get("compiled_default") or []]
            tq = u.get("tiny_qwen35") or {}
            tqs = tq.get("setup_error", "")[:80] or " ".join(
                ("ok" if "loss" in s else ("STRIDE" if s.get("stride_assert") else "err")) for s in tq.get("steps") or [])
            rows.append(f"| {name} | {tag} | {r.get('torch')} {(r.get('device') or [None, None])[1]} {r.get('fa_library')} | "
                        f"{', '.join(_raw_mismatch(e)) or 'none'} | {' '.join(nou)} | {u.get('flags')} | {' '.join(comp)} | {tqs} | "
                        f"{u.get('import_unsloth_s')} |")
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    bad = []
    benvs = _envs(base)
    for tag, he in _envs(head).items():
        be = benvs.get(tag)
        if not be:
            continue
        bu, hu = be.get("unsloth") or {}, he.get("unsloth") or {}
        b_ok, h_ok = _ok_cases(bu), _ok_cases(hu)
        for case, ok in b_ok.items():
            if ok and not h_ok.get(case, False):
                bad.append(f"{tag}: {case} passes at base, fails at head")
        needed = set(_raw_mismatch(he))
        flags = hu.get("flags") or {}
        bflags = bu.get("flags") or {}
        for be_name in ("flash", "mem_efficient"):
            if bflags.get(be_name) and not flags.get(be_name) and be_name not in needed:
                bad.append(f"{tag}: head turned {be_name} SDPA off though its raw strides match the fake kernel")
        bi, hi = bu.get("import_unsloth_s"), hu.get("import_unsloth_s")
        if bi is not None and hi is not None and hi > bi + IMPORT_SLACK_S:
            bad.append(f"{tag}: import unsloth {hi}s vs base {bi}s")
    if bad:
        return True, "; ".join(bad)
    return False, "no compiled case or tiny Qwen3.5 step regressed; no backend turned off without a measured mismatch"
