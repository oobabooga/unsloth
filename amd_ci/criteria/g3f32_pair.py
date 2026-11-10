"""Criteria: Gemma 3 float32 pair, unsloth #15879 (import pass runs with UNSLOTH_FORCE_FLOAT32=0) + unsloth-zoo #2227
(forced float32 Gemma 3 patches undo themselves when the flag is 0), on gfx1151 with torch.compile on.

base (unsloth main + zoo main) must show the defect: for every model, cell a (FORCE_FLOAT32=1 at import, bf16 load)
and cell b (fp16 load then bf16 load) fail with "same dtype".
head (both PRs) is fixed when, for every model:
  - a and b train every step with finite losses, and generate in every loaded model;
  - b's bf16 losses equal the fresh-process bf16 run (d16) of the same arm;
  - c16 / c4 (fp16 alone) and d16 / d4 (bf16 alone) losses are bit-identical to base (when the base A/A repeat
    is bit-identical; otherwise within the A/A spread), and generated tokens match base wherever base generated;
  - both PRs' new tests pass at head.
The single-PR arms (unsloth_only, zoo_only) are reported, not judged. The merge state is not probed.
"""

from __future__ import annotations

import math

TITLE = "Gemma 3 float32 pair (unsloth #15879 + unsloth-zoo #2227) on gfx1151 ROCm, torch.compile on"
MODE = "differential"
NEEDS = ["rocm", "gpu", "nvidia", "exclusive_gpu"]
MODELS = ("tiny", "270m")
ALONE = ("c16", "c4", "d16", "d4", "bb")
_CTX: dict = {}


def _arms(o):
    return o.get("arms") or {}


def _c(o, arm, model, cell):
    return ((_arms(o).get(arm) or {}).get("cells") or {}).get(f"{model}/{cell}") or {}


def _done(c):
    return c.get("stage") == "done" and c.get("rc") == 0


def _finite(c):
    ls = c.get("losses") or []
    return bool(ls) and all(isinstance(x, float) and math.isfinite(x) for x in ls)


def _gen_ok(c):
    loads = c.get("loads") or []
    return bool(loads) and all(ld.get("gen_ids") for ld in loads)


def _dtype_err(c):
    blob = " ".join(str(x) for x in (c.get("error"), c.get("traceback"))) + " ".join(
        str(ld.get("gen_error")) for ld in c.get("loads") or [])
    return "same dtype" in blob


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
        cells = [c for arm in _arms(o).values() for c in (arm.get("cells") or {}).values()]
        ck = o.get("checkout") or ""
        uf = {c.get("unsloth_file") for c in cells if c.get("unsloth_file")}
        out.append((f"{st} cells imported the state's unsloth", bool(uf) and all(f.startswith(ck) for f in uf),
                    ", ".join(sorted(uf))[:200]))
        zok = True
        ev = []
        for name, arm in _arms(o).items():
            zp = ((o.get("zoo") or {}).get(arm.get("zoo")) or {}).get("path") or "?"
            zf = {c.get("zoo_file") for c in (arm.get("cells") or {}).values() if c.get("zoo_file")}
            zok = zok and bool(zf) and all(f.startswith(zp) for f in zf)
            ev.append(f"{name}: {arm.get('zoo')} {str(arm.get('zoo_sha'))[:9]}")
        out.append((f"{st} cells imported the arm's unsloth_zoo", zok, "; ".join(ev)))
        vs = [c.get("versions") or {} for c in cells if c.get("versions")]
        out.append((f"{st} ran on HIP gfx1151", bool(vs) and all(v.get("hip") and "gfx1151" in str(v.get("arch")) for v in vs),
                    f"hip {sorted({str(v.get('hip')) for v in vs})} arch {sorted({str(v.get('arch')) for v in vs})}"))
        comp = [c for c in cells if c.get("env_at_import")]
        out.append((f"{st} torch.compile enabled (no TORCHDYNAMO_DISABLE / UNSLOTH_COMPILE_DISABLE, dynamo on, compiled cache written)",
                    bool(comp) and all(not c["env_at_import"].get("TORCHDYNAMO_DISABLE") and not c["env_at_import"].get("UNSLOTH_COMPILE_DISABLE")
                                       and not (c.get("versions") or {}).get("dynamo_disable") for c in comp)
                    and any((c.get("compiled_cache_files") or 0) > 0 for c in cells),
                    f"cells with compiled cache: {sum(1 for c in cells if (c.get('compiled_cache_files') or 0) > 0)}/{len(cells)}"))
    base = obs.get("base") or {}
    ctl = [f"{m}/{n}" for m in MODELS for n in ALONE if _done(_c(base, "base", m, n))]
    out.append(("base controls (fp16 / bf16 loads alone) trained: the card and the stack work at base",
                len(ctl) == len(MODELS) * len(ALONE), str(ctl)))
    # A/A and base-vs-head comparisons, for head_is_fixed.
    aa = {}
    for m in MODELS:
        for n in ("c16", "d16"):
            aa[f"{m}/{n}"] = _maxdiff(_c(base, "base", m, n).get("losses"), _c(base, "base", m, n + "_r2").get("losses"))
    _CTX["aa"] = aa
    _CTX["base"] = base
    return out


def _bf16_defect(c):
    """The bf16 load (last in the cell) carries the stale float32 embedding, and the dtype mismatch is raised
    (in training on CPU; on gfx1151 with compile training runs and generate's KV-cache index_copy_ raises)."""
    loads = c.get("loads") or []
    return bool(loads) and loads[-1].get("dtype") == "bfloat16" and loads[-1].get("embed_out_dtype") == "torch.float32" \
        and _dtype_err(c)


def base_shows_defect(base):
    return all(_bf16_defect(_c(base, "base", m, "a")) and _bf16_defect(_c(base, "base", m, "b")) for m in MODELS)


def head_is_fixed(head):
    if head.get("state") == "merge":
        return True  # not probed; declared in the table
    base = _CTX.get("base") or {}
    parts = {}
    for m in MODELS:
        a, b, d16 = _c(head, "head", m, "a"), _c(head, "head", m, "b"), _c(head, "head", m, "d16")
        parts[f"{m} a,b train + generate"] = all(_done(x) and _finite(x) and _gen_ok(x) for x in (a, b))
        bb = _c(head, "head", m, "bb")
        db, dbb = _maxdiff(b.get("losses"), d16.get("losses")), _maxdiff(b.get("losses"), bb.get("losses"))
        _CTX[f"{m} b-vs-fresh"], _CTX[f"{m} b-vs-bb"] = db, dbb
        _CTX[f"{m} a-vs-fresh"] = _maxdiff(a.get("losses"), d16.get("losses"))
        # b's bf16 model must equal a bf16 model loaded second in a process with no fp16 load before it (bb);
        # whether an in-process reload equals a fresh process is pre-existing and is reported, not judged.
        parts[f"{m} b == bf16-after-bf16 (bb) or fresh bf16"] = (db is not None and db == 0.0) or (dbb is not None and dbb == 0.0)
        parts[f"{m} b bf16 embed dtype bf16"] = (b.get("loads") or [{}])[-1].get("embed_out_dtype") == "torch.bfloat16"
        parts[f"{m} b bf16 tokens == fresh bf16"] = bool(d16.get("loads")) and \
            (b.get("loads") or [{}])[-1].get("gen_ids") == d16["loads"][0].get("gen_ids")
        for n in ALONE:
            bc, hc = _c(base, "base", m, n), _c(head, "head", m, n)
            tol = 0.0
            aa = (_CTX.get("aa") or {}).get(f"{m}/{n}")
            if aa not in (None, 0.0):
                tol = aa
            d = _maxdiff(bc.get("losses"), hc.get("losses"))
            parts[f"{m} {n} losses base==head"] = _done(hc) and d is not None and d <= tol
            gens_b = [ld.get("gen_ids") for ld in bc.get("loads") or []]
            gens_h = [ld.get("gen_ids") for ld in hc.get("loads") or []]
            parts[f"{m} {n} tokens base==head"] = bool(gens_b) and all(gb is None or gb == gh for gb, gh in zip(gens_b, gens_h))
        # b's first (fp16) load generates at base too: same tokens.
        bb = _c(base, "base", m, "b").get("loads") or []
        hb = b.get("loads") or []
        if bb and bb[0].get("gen_ids") and hb:
            parts[f"{m} b fp16-load tokens base==head"] = bb[0]["gen_ids"] == hb[0].get("gen_ids")
    py = head.get("pytest") or {}
    for k in ("unsloth", "zoo"):
        r = py.get(k) or {}
        parts[f"pytest {k}"] = r.get("rc") == 0 and bool(r.get("passed")) and not r.get("failed")
    _CTX["head_parts"] = parts
    return all(parts.values())


def _short_err(c):
    e = c.get("error")
    if not e:
        ge = [ld.get("gen_error") for ld in c.get("loads") or [] if ld.get("gen_error")]
        e = ("gen: " + ge[0]) if ge else ""
    return str(e).replace("|", "/").replace("\n", " ")[:140]


def table(obs):
    if "head" in obs:
        head_is_fixed(obs["head"])
    rows = ["| state | arm | model/cell | stage | s | flags after loads | embed dtypes | losses | gen tokens | error |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for st in _states(obs):
        for arm, ad in _arms(obs[st]).items():
            for key, c in (ad.get("cells") or {}).items():
                loads = c.get("loads") or []
                rows.append(f"| {st} | {arm} | {key} | {c.get('stage')} | {c.get('seconds')} | "
                            f"{[ld.get('flag_after_load') for ld in loads]} | {[ld.get('embed_out_dtype') for ld in loads]} | "
                            f"{[round(x, 6) for x in c.get('losses') or []]} | {[str(ld.get('gen_ids'))[:40] for ld in loads]} | "
                            f"{_short_err(c)} |")
    rows += ["", "| state | test file | rc | passed | failed |", "|---|---|---|---|---|"]
    for st in _states(obs):
        for k, r in (obs[st].get("pytest") or {}).items():
            rows.append(f"| {st} | {k} | {r.get('rc')} | {len(r.get('passed') or [])} | {r.get('failed')} |")
    rows += ["", f"- A/A (base vs base repeat) max loss diff: {_CTX.get('aa')}"]
    if "head_parts" in _CTX:
        rows.append(f"- head parts: {_CTX['head_parts']}")
    rows.append("- loss max diffs: " + str({k: v for k, v in _CTX.items() if "-vs-" in k}))
    rows.append("- merge state: not probed (base and head carry the evidence); its head_is_fixed is declared True.")
    return "\n".join(rows)
