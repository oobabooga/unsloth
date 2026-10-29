"""Criteria: unsloth-zoo compiled-module key (follow-up to #2227) on gfx1151, torch.compile on.

base (unsloth main + zoo main) must show the defect, for every model: in cell b (fp16 load then bf16 load in one
process) the bf16 model runs the forced float32 compiled Gemma 3 module, and its losses differ from the fresh bf16
process (d16) by more than the base A/A spread.
head (same unsloth + zoo fix) is fixed when, for every model:
  - b's losses equal d16 bitwise, and b's compiled module is the bf16 one (not forced, same normalised hash as d16);
  - r (bf16 then fp16 in one process) equals c16 bitwise with a forced module;
  - xb (bf16 process on a cache an fp16 process wrote) equals d16, xr (the reverse) equals c16;
  - c16 / c4 / d16 / d4 (one dtype per process) losses and tokens are bit-identical to base, and the dynamo counters
    (calls_captured, unique_graphs) are equal to base;
  - the fix's CPU test passes.
"""

from __future__ import annotations

import math

TITLE = "unsloth-zoo compiled Gemma 3 module keyed on UNSLOTH_FORCE_FLOAT32 (gfx1151 ROCm, torch.compile on)"
MODE = "differential"
NEEDS = ["rocm", "gpu", "nvidia", "exclusive_gpu"]
MODELS = ("tiny", "270m")
ALONE = ("c16", "c4", "d16", "d4")
_CTX: dict = {}


def _arm(o):
    arms = o.get("arms") or {}
    return next(iter(arms.values()), {}) if arms else {}


def _c(o, model, cell):
    return (_arm(o).get("cells") or {}).get(f"{model}/{cell}") or {}


def _done(c):
    return c.get("stage") == "done" and c.get("rc") == 0


def _finite(c):
    ls = c.get("losses") or []
    return bool(ls) and all(isinstance(x, float) and math.isfinite(x) for x in ls)


def _maxdiff(x, y):
    if not x or not y or len(x) != len(y):
        return None
    return max(abs(a - b) for a, b in zip(x, y))


def _states(obs):
    return [k for k in obs if not k.startswith("_")]


def gates(obs):
    out = []
    for st in _states(obs):
        o = obs[st]
        if st == "merge":
            continue
        cells = list((_arm(o).get("cells") or {}).values())
        ck = o.get("checkout") or ""
        uf = {c.get("unsloth_file") for c in cells if c.get("unsloth_file")}
        out.append((f"{st} cells imported the state's unsloth", bool(uf) and all(f.startswith(ck) for f in uf),
                    ", ".join(sorted(uf))[:200]))
        zp = ((o.get("zoo") or {}).get(_arm(o).get("zoo")) or {})
        zf = {c.get("zoo_file") for c in cells if c.get("zoo_file")}
        out.append((f"{st} cells imported the arm's unsloth_zoo", bool(zf) and all(f.startswith(zp.get("path") or "?") for f in zf),
                    f"{_arm(o).get('zoo')} {str(zp.get('sha'))[:9]} {zp.get('diffstat', '')[-120:]}"))
        if st == "head":
            out.append(("head arm carries the zoo fix (patch applied, compiler.py changed)",
                        bool(o.get("patch_present")) and "compiler.py" in str(zp.get("diffstat")), str(zp.get("diffstat"))[:200]))
        vs = [c.get("versions") or {} for c in cells if c.get("versions")]
        out.append((f"{st} ran on HIP gfx1151", bool(vs) and all(v.get("hip") and "gfx1151" in str(v.get("arch")) for v in vs),
                    f"arch {sorted({str(v.get('arch')) for v in vs})}"))
        comp = [c for c in cells if c.get("env_at_import")]
        out.append((f"{st} torch.compile on (no disable env, dynamo on, compiled cache written)",
                    bool(comp) and all(not c["env_at_import"].get("TORCHDYNAMO_DISABLE") and not c["env_at_import"].get("UNSLOTH_COMPILE_DISABLE")
                                       and not (c.get("versions") or {}).get("dynamo_disable") for c in comp)
                    and all((c.get("compiled_cache_files") or 0) > 0 for c in cells if _done(c)),
                    f"cells with compiled cache: {sum(1 for c in cells if (c.get('compiled_cache_files') or 0) > 0)}/{len(cells)}"))
    base = obs.get("base") or {}
    ctl = [f"{m}/{n}" for m in MODELS for n in ALONE + ("bb",) if _done(_c(base, m, n)) and _finite(_c(base, m, n))]
    out.append(("base controls trained", len(ctl) == len(MODELS) * (len(ALONE) + 1), str(ctl)))
    aa = {}
    for m in MODELS:
        for n in ("c16", "d16"):
            aa[f"{m}/{n}"] = _maxdiff(_c(base, m, n).get("losses"), _c(base, m, n + "_r2").get("losses"))
    _CTX["aa"], _CTX["base"] = aa, base
    return out


def base_shows_defect(base):
    ok = True
    for m in MODELS:
        b, d16 = _c(base, m, "b"), _c(base, m, "d16")
        d = _maxdiff(b.get("losses"), d16.get("losses"))
        aa = (_CTX.get("aa") or {}).get(f"{m}/d16")
        _CTX[f"base {m} b-vs-d16"] = d
        ok = ok and _done(b) and (b.get("module") or {}).get("forced") is True and d is not None and aa is not None and d > aa
    return ok


def head_is_fixed(head):
    if head.get("state") == "merge":
        return True
    base = _CTX.get("base") or {}
    parts = {}
    for m in MODELS:
        c = {n: _c(head, m, n) for n in ("b", "bb", "r", "c16", "c4", "d16", "d4", "xb", "xr")}
        for n in ("b", "r", "xb", "xr"):
            ref = c["d16" if n in ("b", "xb") else "c16"]
            d = _maxdiff(c[n].get("losses"), ref.get("losses"))
            _CTX[f"head {m} {n}-vs-{'d16' if n in ('b', 'xb') else 'c16'}"] = d
            mod, rmod = c[n].get("module") or {}, ref.get("module") or {}
            parts[f"{m} {n} losses == fresh"] = _done(c[n]) and _finite(c[n]) and d == 0.0
            parts[f"{m} {n} module == fresh (normalised)"] = bool(mod.get("present")) and mod.get("norm_sha") == rmod.get("norm_sha") \
                and mod.get("forced") == (n in ("r", "xr"))
            gen = (c[n].get("loads") or [{}])[-1].get("gen_ids")
            parts[f"{m} {n} tokens == fresh"] = bool(gen) and gen == (ref.get("loads") or [{}])[0].get("gen_ids")
        for n in ALONE:
            bc, hc = _c(base, m, n), c[n]
            tol = (_CTX.get("aa") or {}).get(f"{m}/{n}") or 0.0
            d = _maxdiff(bc.get("losses"), hc.get("losses"))
            _CTX[f"{m} {n} base-vs-head"] = d
            parts[f"{m} {n} losses base==head"] = _done(hc) and d is not None and d <= tol
            parts[f"{m} {n} tokens base==head"] = [ld.get("gen_ids") for ld in bc.get("loads") or []] == \
                [ld.get("gen_ids") for ld in hc.get("loads") or []]
            parts[f"{m} {n} module sha base==head"] = (bc.get("module") or {}).get("sha") == (hc.get("module") or {}).get("sha")
            parts[f"{m} {n} dynamo counters base==head"] = bool(hc.get("dynamo_frames")) and \
                bc.get("dynamo_frames") == hc.get("dynamo_frames")
    r = (head.get("pytest") or {}).get("zoo") or {}
    parts["pytest fix test"] = r.get("rc") == 0 and len(r.get("passed") or []) == 2 and not r.get("failed")
    _CTX["head_parts"] = parts
    return all(parts.values())


def table(obs):
    if "head" in obs:
        head_is_fixed(obs["head"])
    rows = ["| state | model/cell | stage | s | dtypes | embed | losses | gen (last load) | module forced / norm sha | dynamo | pre | error |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for st in _states(obs):
        for key, c in (_arm(obs[st]).get("cells") or {}).items():
            loads = c.get("loads") or []
            mod = c.get("module") or {}
            pre = f"{c.get('pre_stage')} {(c.get('pre_module') or {}).get('forced')}" if "pre_rc" in c else ""
            rows.append(f"| {st} | {key} | {c.get('stage')} | {c.get('seconds')} | {[ld.get('dtype') for ld in loads]} | "
                        f"{[ld.get('embed_out_dtype') for ld in loads]} | {[round(x, 6) for x in c.get('losses') or []]} | "
                        f"{str((loads or [{}])[-1].get('gen_ids'))[:40]} | {mod.get('forced')} {mod.get('norm_sha')} | "
                        f"{c.get('dynamo_frames')} | {pre} | {str(c.get('error') or '')[:120]} |")
    rows += ["", "| state | test | rc | passed | failed |", "|---|---|---|---|---|"]
    for st in _states(obs):
        for k, r in (obs[st].get("pytest") or {}).items():
            rows.append(f"| {st} | {k} | {r.get('rc')} | {len(r.get('passed') or [])} | {r.get('failed')} |")
    rows += ["", f"- A/A (base vs base repeat) max loss diff: {_CTX.get('aa')}"]
    rows.append("- diffs: " + str({k: v for k, v in _CTX.items() if "-vs-" in k}))
    if "head_parts" in _CTX:
        bad = [k for k, v in _CTX["head_parts"].items() if not v]
        rows.append(f"- head parts failing: {bad}")
        rows.append(f"- head parts: {_CTX['head_parts']}")
    return "\n".join(rows)
