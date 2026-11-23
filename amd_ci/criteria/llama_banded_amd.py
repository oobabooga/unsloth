#!/usr/bin/env python3
"""Criteria: Inkling 2 banded attention on AMD WMMA (unslothai/llama.cpp#406). Judges only.

Defect (base = #386 head 6348de261): on AMD the banded relative-bias attention runs the
scalar banded kernels, so prompt processing slows down sharply with prompt length.
Observable as base pp2048 < BASE_LEN_RATIO * base pp512 (llama-bench, same run).

Fixed (head = base + #406): every test-backend-ops eval case of the banded op and of the
fused ops passes against CPU, head output stays close to base (KLD vs the base logits,
short and wrapped-SWA long context), pp2048 is at least MIN_PP_SPEEDUP x base and tg128
does not regress beyond MAX_TG_LOSS.

Thresholds are fixed here, before any run, and are not tuned after seeing results.
"""

from __future__ import annotations

TITLE = "Inkling 2 banded attention on gfx1151 (ROCm): #386 vs #406"
MODE = "differential"
NEEDS = ["rocm", "gpu", "discrete_gpu", "windows", "nvidia", "multi_gpu"]

BASE_LEN_RATIO = 0.85   # base: pp2048 / pp512 below this = the length slowdown of the scalar path
MIN_PP_SPEEDUP = 1.30   # head pp2048 / base pp2048
MAX_TG_LOSS    = 0.03   # head tg128 >= (1 - this) * base tg128
MAX_KLD_MEAN   = 0.01   # head vs base logits, per context
MIN_TOP1_PCT   = 97.0
EVAL_OPS_HEAD  = ["FLASH_ATTN_EXT_BANDED", "SSM_CONV", "SSM_CONV_RES_SCALE", "TANH_GATE_MIX", "DEPTH_MIX",
                  "MUL_MAT_SHARED_SRC1", "CONCAT"]


def _states(obs: dict) -> dict:
    return {n: v for n, v in obs.items() if not n.startswith("_")}


def _bench(state: dict, key: str) -> list[float]:
    return [b[key] for b in (state.get("bench") or []) if isinstance(b.get(key), (int, float))]


def _med(xs: list[float]) -> float | None:
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else None


def _eval_ok(state: dict, op: str) -> bool:
    e = (state.get("eval") or {}).get(op) or {}
    return e.get("total", 0) > 0 and e.get("passed") == e.get("total") and e.get("n_fail_lines", 1) == 0


_CTX: dict = {"base": {}}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    st = _states(obs)
    out = [("both states probed", {"base", "head"} <= set(st), ", ".join(st))]
    out.append(("ROCm device is gfx1151 in every state", all(v.get("gfx1151") for v in st.values()),
                ", ".join(f"{k}={v.get('gfx1151')}" for k, v in st.items())))
    libs = {k: v.get("hip_lib") or "" for k, v in st.items()}
    out.append(("each state ran its OWN libggml-hip",
                all(v and v.startswith(st[k].get("libdir", "?")) for k, v in libs.items()) and len(set(libs.values())) == len(libs),
                ", ".join(f"{k}={v}" for k, v in libs.items())))
    n = {k: ((v.get("eval") or {}).get("FLASH_ATTN_EXT_BANDED") or {}).get("total", 0) for k, v in st.items()}
    out.append(("FLASH_ATTN_EXT_BANDED eval cases ran (> 0) in every state", all(c > 0 for c in n.values()),
                ", ".join(f"{k}={c}" for k, c in n.items())))
    ctrl = {k: _eval_ok(v, "FLASH_ATTN_EXT_BANDED") for k, v in st.items()}
    out.append(("control: base passes its own FLASH_ATTN_EXT_BANDED cases", ctrl.get("base", False),
                ", ".join(f"{k}={c}" for k, c in ctrl.items())))
    ppl = {f"{k}:{c}": isinstance(((v.get("ppl") or {}).get(f"{c}:score") or {}).get("kld_mean"), float)
           for k, v in st.items() for c in ("short", "long")}
    out.append(("KLD scored vs the base logits for every state and context", all(ppl.values()),
                ", ".join(f"{k}={c}" for k, c in ppl.items())))
    aa = [((st.get("base", {}).get("ppl") or {}).get(f"{c}:score") or {}).get("kld_mean") for c in ("short", "long")]
    out.append(("A/A: base vs its own reference has KLD <= 1e-4 (deterministic floor)",
                all(isinstance(x, float) and x <= 1e-4 for x in aa), f"base A/A KLD short/long = {aa}"))
    bench = {k: len(_bench(v, "pp2048")) == 2 and len(_bench(v, "tg128")) == 2 and len(_bench(v, "pp512")) == 2
             for k, v in st.items()}
    out.append(("llama-bench pp512/pp2048/tg128 x2 in every state", all(bench.values()),
                ", ".join(f"{k}={c}" for k, c in bench.items())))
    _CTX["base"] = st.get("base") or {}
    return out


def table(obs: dict) -> str:
    rows = ["| state | banded eval | fused evals | PPL short (KLD, top1) | PPL long (KLD, top1) | pp512 | pp2048 | tg128 |",
            "|---|---|---|---|---|---|---|---|"]
    for name, v in _states(obs).items():
        e = v.get("eval") or {}
        f = lambda op: f"{(e.get(op) or {}).get('passed', 0)}/{(e.get(op) or {}).get('total', 0)}"
        p = v.get("ppl") or {}
        g = lambda c: (f"{(p.get(c + ':score') or {}).get('ppl', float('nan')):.4f} "
                       f"({(p.get(c + ':score') or {}).get('kld_mean', float('nan')):.6f}, "
                       f"{(p.get(c + ':score') or {}).get('top1_pct', float('nan')):.2f}%)")
        b = lambda k: " / ".join(f"{x:.1f}" for x in _bench(v, k))
        rows.append(f"| {name} | {f('FLASH_ATTN_EXT_BANDED')} | "
                    + ", ".join(f"{op} {f(op)}" for op in EVAL_OPS_HEAD[1:]) + f" | {g('short')} | {g('long')} | "
                    f"{b('pp512')} | {b('pp2048')} | {b('tg128')} |")
    rows.append("")
    rows.append(f"Defect: base pp2048/pp512 < {BASE_LEN_RATIO}. Fixed: all head evals pass, KLD mean <= {MAX_KLD_MEAN} "
                f"and top-1 >= {MIN_TOP1_PCT}% (short -c 512 and wrapped-SWA long -c 4096 -ub 512), "
                f"head pp2048 >= {MIN_PP_SPEEDUP}x base, head tg128 >= {1 - MAX_TG_LOSS:.2f}x base (medians of 2 runs x 3 reps).")
    return "\n".join(rows)


def base_shows_defect(base: dict):
    a, b = _med(_bench(base, "pp512")), _med(_bench(base, "pp2048"))
    r = (b / a) if a and b else None
    return (r is not None and r < BASE_LEN_RATIO), f"base pp2048/pp512 = {r}"


def head_is_fixed(head: dict):
    evals = {op: _eval_ok(head, op) for op in EVAL_OPS_HEAD}
    p = head.get("ppl") or {}
    kl = {c: (p.get(f"{c}:score") or {}) for c in ("short", "long")}
    kld_ok = all(isinstance(k.get("kld_mean"), float) and k["kld_mean"] <= MAX_KLD_MEAN
                 and (k.get("top1_pct") or 0) >= MIN_TOP1_PCT for k in kl.values())
    base = _CTX["base"]
    bp, hp = _med(_bench(base, "pp2048")), _med(_bench(head, "pp2048"))
    bt, ht = _med(_bench(base, "tg128")), _med(_bench(head, "tg128"))
    sp = hp / bp if bp and hp else None
    tg = ht / bt if bt and ht else None
    ok = all(evals.values()) and kld_ok and sp is not None and sp >= MIN_PP_SPEEDUP and tg is not None and tg >= 1 - MAX_TG_LOSS
    return ok, f"evals={evals} kld={ {c: (k.get('kld_mean'), k.get('top1_pct')) for c, k in kl.items()} } pp2048x={sp} tg128x={tg}"
