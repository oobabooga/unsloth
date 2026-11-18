#!/usr/bin/env python3
"""Criteria (unsloth #18313): ROCm packed rows on torch's flash varlen, gfx1151, base (merge base) vs head.

Differential mode.
  base_shows_defect: torch's varlen_attn is loaded (scale / window_size) yet the base refuses the route on ROCm
    (_torch_varlen(cuda:0) is None), and the base's packed rows go to dense-mask SDPA: the real wrapper's
    fast counter stays 0 with fallback > 0 (and, where training ran, every packed arm had 0 fast / compiled calls).
  head_is_fixed:
    * route on; windowed route on (window 512 / 1024)
    * self-check fails closed: an injected raising varlen, a document-leaking varlen and a raising aten entry
      each keep the route off; the real one restores it; 4 racing threads agree and do not raise
    * kernels: flash varlen (eager varlen_attn and the aten op) out / dq / dk / dv max relative error vs an fp32
      per-document reference within max(2 x dense-mask SDPA's, 1e-2), all finite
    * the real wrapper: fast >= 1 and fallback 0 for windowless and window-512 packed rows, its error within
      max(2 x the wrapped sdpa's, 1e-2)
    * training (where both states ran it): head packed arms engaged (fast + compiled > 0, fallback 0); per-step
      loss vs base packed within max(3 x base A/A spread, 1e-3 x |loss|); grad norm within max(3 x A/A, 2%);
      head packed useful tok/s >= 1.15 x head padded and > base packed by more than max(5%, 3 x base A/A spread)
  Notes (reported, not judged): self-check verdict under autocast / no_grad / inference_mode, check cost, cached
  per-call cost, kernel timings, efficient varlen stays off (head dim 512), padded tok/s base vs head.
Gates: base is merge base f5df9fa088, head tree is PR head 97c124bc2e's; gfx1151 HIP torch with varlen_attn
(scale / window_size); head has the self-check, base does not; each state imported its own checkout.
"""
from __future__ import annotations

import statistics

TITLE = "unsloth #18313: ROCm packed rows on flash varlen (gfx1151), base vs head"
MODE = "differential"
NEEDS = ["rocm", "gpu", "exclusive_gpu", "nvidia", "multi_gpu", "windows"]
EXPECT_BASE = "f5df9fa088a43055e2ba590b3d5a2dc41209ce2a"
EXPECT_HEAD_TREE = None  # filled from the observations when unknown; set below
_CTX: dict = {}
TOL_KERNEL = 1e-2


def _k(o):
    k = (o or {}).get("kernel") or {}
    if k.get("stage") != "done" and (o or {}).get("kernel_stub"):
        return (o or {}).get("kernel_stub") or {}
    return k


def _tr(o, name):
    return (((o or {}).get("train") or {}).get(name)) or {}


def _done(c):
    return c.get("stage") == "done" and bool(c.get("losses"))


def _tps(c, first = 3):
    toks, ts = c.get("tokens") or [], c.get("step_s") or []
    vals = [toks[i][0] / ts[i] for i in range(min(len(toks), len(ts))) if i + 1 >= first and ts[i] > 0]
    return statistics.median(vals) if vals else None


def _ls(c, key = "losses"):
    return {int(k): float(v) for k, v in (c.get(key) or {}).items()}


def _spread(a, b, key = "losses", relative = False):
    x, y = _ls(a, key), _ls(b, key)
    common = set(x) & set(y)
    if not common:
        return None
    return max(abs(x[s] - y[s]) / (abs(x[s]) if relative and x[s] else 1.0) for s in common)


def _own(o, f):
    f = str(f or "").replace("\\", "/")
    ck = str((o or {}).get("checkout") or "").replace("\\", "/")
    return bool(f) and bool(ck) and f.startswith(ck)


def gates(obs):
    b, h = obs.get("base") or {}, obs.get("head") or {}
    _CTX["base"] = b
    out = [("base is the merge base f5df9fa088", str(b.get("commit")) == EXPECT_BASE, f"base={b.get('commit')}"),
           ("head is PR head 97c124bc2e (or its tree)", str(h.get("commit", "")).startswith("97c124bc2e")
            or bool(h.get("tree")), f"head commit={h.get('commit')} tree={h.get('tree')}")]
    for name, o in (("base", b), ("head", h)):
        k = _k(o)
        v = k.get("versions") or {}
        out.append((f"{name}: kernel cell ran on a gfx1151 HIP device", bool(v.get("hip")) and "gfx1151" in str(v.get("arch")),
                    f"stage={k.get('stage')} hip={v.get('hip')} arch={v.get('arch')} torch={v.get('torch')} err={str(k.get('error') or '')[:300]}"))
        out.append((f"{name}: torch varlen_attn loaded with scale / window_size",
                    isinstance(k.get("varlen_params"), list) and {"scale", "window_size"} <= set(k["varlen_params"])
                    and (k.get("route") or {}).get("fn_loaded") is True,
                    f"params={k.get('varlen_params')} fn_loaded={(k.get('route') or {}).get('fn_loaded')}"))
        out.append((f"{name}: hf_packed_attention loaded from its own checkout", _own(o, k.get("hpa_file")),
                    f"{k.get('hpa_file')}"))
    out.append(("head has the self-check, base does not", _k(h).get("has_selfcheck") is True and _k(b).get("has_selfcheck") is False,
                f"head={_k(h).get('has_selfcheck')} base={_k(b).get('has_selfcheck')}"))
    if b.get("train") is not None:
        cs = {n: c for n, c in (b.get("train") or {}).items() if n.endswith("_packed")}
        ok = any(_done(c) and _own(b, c.get("unsloth_file")) for c in cs.values())
        out.append(("base: at least one packed training arm finished on its own checkout", ok,
                    "; ".join(f"{n}: stage={c.get('stage')} file={c.get('unsloth_file')} err={str(c.get('error') or '')[:200]}" for n, c in cs.items())))
    return out


def base_shows_defect(base):
    k = _k(base)
    if (k.get("route") or {}).get("on") is not False:
        return False
    wr = k.get("wrapper") or {}
    rows = [wr.get(n) or {} for n in ("hd256_nowindow_2048", "hd128_window512_4096")]
    if wr:
        for r in rows:
            d = r.get("delta") or {}
            if d.get("fast", 0) != 0 or d.get("fallback", 0) < 1:
                return False
    for n, c in (base.get("train") or {}).items():
        if "packed" in n and _done(c):
            d = c.get("attn_stats_delta") or {}
            if (d.get("fast", 0) or 0) + (d.get("compiled", 0) or 0) != 0:
                return False
    return True


def _problems(head):
    base = _CTX.get("base") or {}
    bad = []
    k = _k(head)
    r = k.get("route") or {}
    if r.get("on") is not True:
        bad.append(f"route off: {r}")
    if r.get("window_512") is not True or r.get("window_1024") is not True:
        bad.append(f"window route off: 512={r.get('window_512')} 1024={r.get('window_1024')}")
    sc = k.get("selfcheck") or {}
    for key in ("injected_raise_route_on", "injected_leak_route_on", "injected_aten_raise_route_on"):
        if sc.get(key) is not False:
            bad.append(f"self-check did not fail closed: {key}={sc.get(key)}")
    if sc.get("restored_route_on") is not True:
        bad.append(f"route not restored after injection: {sc.get('restored_route_on')}")
    th = sc.get("threads") or {}
    if th.get("errors") or set(th.get("results") or []) != {True}:
        bad.append(f"threads: {th}")
    for name, row in (k.get("kernels") or {}).items():
        if row.get("error"):
            bad.append(f"kernel {name}: {row['error'][:200]}")
            continue
        dense = row.get("dense_sdpa_rel_err") or [0, 0, 0, 0]
        for arm in ("varlen_eager", "aten_flash"):
            errs = row.get(arm + "_rel_err")
            if errs is None or row.get(arm + "_finite") is not True:
                bad.append(f"kernel {name} {arm}: errs={errs} finite={row.get(arm + '_finite')} {row.get(arm + '_error', '')[:200]}")
                continue
            for i, (e, d) in enumerate(zip(errs, dense)):
                if e > max(2 * d, TOL_KERNEL):
                    bad.append(f"kernel {name} {arm} {'out dq dk dv'.split()[i]}: rel err {e:.3g} > max(2 x dense {d:.3g}, {TOL_KERNEL})")
    if not k.get("kernels"):
        bad.append("no kernel measurements")
    wr = k.get("wrapper")
    if wr is not None:
        for n in ("hd256_nowindow_2048", "hd128_window512_4096"):
            row = wr.get(n) or {}
            d = row.get("delta") or {}
            if row.get("error") or d.get("fast", 0) < 1 or d.get("fallback", 0) != 0:
                bad.append(f"wrapper {n}: delta={d} err={str(row.get('error') or '')[:200]}")
                continue
            for i, (e, o) in enumerate(zip(row.get("wrapper_rel_err") or [], row.get("orig_sdpa_rel_err") or [])):
                if e > max(2 * o, TOL_KERNEL):
                    bad.append(f"wrapper {n} {'out dq dk dv'.split()[i]}: {e:.3g} > max(2 x {o:.3g}, {TOL_KERNEL})")
    bt, ht = (base.get("pytest") or {}).get("cases"), (head.get("pytest_fixed") or head.get("pytest") or {}).get("cases")
    if bt is not None and ht is not None:
        bfail = {t for t, c in bt.items() if c["outcome"] in ("failed", "error")}
        new = [t.split("::")[-1] for t, c in ht.items() if c["outcome"] in ("failed", "error") and t not in bfail]
        if new:
            bad.append(f"pytest (fixed test file) fails at head only: {new}")
    for arch in ("qwen35", "gemma3"):
        bp, bp2, bpad = _tr(base, f"{arch}_packed"), _tr(base, f"{arch}_packed2"), _tr(base, f"{arch}_padded")
        hp, hp2, hpad = _tr(head, f"{arch}_packed"), _tr(head, f"{arch}_packed2"), _tr(head, f"{arch}_padded")
        if not (_done(bp) and _done(bpad)):
            continue  # base could not train this arch here: reported, not judged
        for nm, c in (("packed", hp), ("packed2", hp2), ("padded", hpad)):
            if not _done(c):
                bad.append(f"{arch} head {nm}: base finished, head {c.get('stage')} {str(c.get('error') or '')[:300]}")
        if not (_done(hp) and _done(hpad)):
            continue
        d = hp.get("attn_stats_delta") or {}
        if (d.get("fast", 0) or 0) + (d.get("compiled", 0) or 0) < 1 or (d.get("fallback", 0) or 0) != 0:
            bad.append(f"{arch} head packed not fully on varlen: {d}")
        aa = _spread(bp, bp2) if _done(bp2) else None
        aag = _spread(bp, bp2, "grad_norms", relative = True) if _done(bp2) else None
        bl, hl = _ls(bp), _ls(hp)
        for s in sorted(set(bl) & set(hl)):
            tol = max(3 * (aa or 0.0), 1e-3 * abs(bl[s]))
            if abs(hl[s] - bl[s]) > tol:
                bad.append(f"{arch} step {s}: head packed loss {hl[s]:.5f} vs base packed {bl[s]:.5f} (tol {tol:.3g})")
        bg, hg = _ls(bp, "grad_norms"), _ls(hp, "grad_norms")
        for s in sorted(set(bg) & set(hg)):
            tol = max(3 * (aag or 0.0), 0.02)
            if bg[s] and abs(hg[s] - bg[s]) / abs(bg[s]) > tol:
                bad.append(f"{arch} step {s}: grad norm {hg[s]:.4f} vs base {bg[s]:.4f} (rel tol {tol:.3g})")
        t_hp, t_hpad, t_bp = _tps(hp), _tps(hpad), _tps(bp)
        t_bp2 = _tps(bp2) if _done(bp2) else None
        noise = abs(t_bp - t_bp2) / t_bp if (t_bp and t_bp2) else 0.0
        if not (t_hp and t_hpad and t_bp):
            bad.append(f"{arch}: missing tok/s ({t_hp}, {t_hpad}, {t_bp})")
            continue
        if t_hp < 1.15 * t_hpad:
            bad.append(f"{arch}: head packed {t_hp:.0f} tok/s < 1.15 x head padded {t_hpad:.0f}")
        if t_hp <= t_bp * (1 + max(0.05, 3 * noise)):
            bad.append(f"{arch}: head packed {t_hp:.0f} tok/s not above base packed {t_bp:.0f} beyond noise {max(0.05, 3 * noise):.3f}")
    return bad


def head_is_fixed(head):
    return not _problems(head)


def _fmt(x, f = "{:.0f}"):
    return "-" if x is None else f.format(x)


def table(obs):
    b, h = obs.get("base") or {}, obs.get("head") or {}
    L = []
    for name, o in (("base", b), ("head", h)):
        k = _k(o)
        r = k.get("route") or {}
        L.append(f"- {name}: mode={k.get('mode')} torch={(k.get('versions') or {}).get('torch')} route_on={r.get('on')} "
                 f"first={_fmt(r.get('first_s'), '{:.3f}')}s second={_fmt(r.get('second_s'), '{:.6f}')}s "
                 f"window512={r.get('window_512')} ({_fmt(r.get('window_first_s'), '{:.3f}')}s) window1024={r.get('window_1024')} "
                 f"efficient_hd512={k.get('efficient_takes_hd512')} pip={o.get('pip')}")
        if k.get("selfcheck"):
            L.append(f"  - self-check: {k['selfcheck']}")
        for n, row in (k.get("wrapper") or {}).items():
            if isinstance(row, dict):
                L.append(f"  - wrapper {n}: delta={row.get('delta')} wrapper_err={row.get('wrapper_rel_err')} "
                         f"orig_err={row.get('orig_sdpa_rel_err')} {str(row.get('error') or '')[:200]}")
        for key in ("pytest", "pytest_fixed"):
          t = o.get(key) or {}
          if t:
            oc = {}
            for c in (t.get("cases") or {}).values():
                oc[c["outcome"]] = oc.get(c["outcome"], 0) + 1
            fails = [tid.split("::")[-1] for tid, c in (t.get("cases") or {}).items() if c["outcome"] in ("failed", "error")]
            L.append(f"  - {key} {TEST_NAME}: rc={t.get('rc')} {oc} failing={fails}")
    L += ["", "| kernel (head) | dense SDPA ms | varlen eager ms | aten flash ms | dense err | varlen err | aten err |",
          "|---|---|---|---|---|---|---|"]
    for n, row in (_k(h).get("kernels") or {}).items():
        def ms(arm):
            v = row.get(arm + "_ms")
            return ", ".join(f"{x[0]:.1f}" for x in v) if v else str(row.get(arm + "_error", "-"))[:80]

        def er(arm):
            v = row.get(arm + "_rel_err")
            return "-" if not v else f"{max(v):.2e}"
        L.append(f"| {n} | {ms('dense_sdpa')} | {ms('varlen_eager')} | {ms('aten_flash')} | {er('dense_sdpa')} | {er('varlen_eager')} | {er('aten_flash')} |")
    L += ["", "| arm | state | stage | median s/step | tokens/step | useful tok/s (median steps 3+) | first losses | fast / compiled / fallback | peak GB |",
          "|---|---|---|---|---|---|---|---|---|"]
    for arch in ("qwen35", "gemma3"):
        for nm in ("padded", "packed", "packed2"):
            for st, o in (("base", b), ("head", h)):
                c = _tr(o, f"{arch}_{nm}")
                if not c:
                    continue
                ts = c.get("step_s") or []
                toks = c.get("tokens") or []
                d = c.get("attn_stats_delta") or {}
                ls = _ls(c)
                L.append(f"| {arch} {nm} | {st} | {c.get('stage')} {str(c.get('error') or '')[:120]} | "
                         f"{_fmt(statistics.median(ts[2:]) if len(ts) > 2 else None, '{:.2f}')} | "
                         f"{_fmt(statistics.median([x[0] for x in toks]) if toks else None)} | {_fmt(_tps(c))} | "
                         f"{', '.join(f'{ls[s]:.4f}' for s in sorted(ls)[:3])} | {d.get('fast')} / {d.get('compiled')} / {d.get('fallback')} | "
                         f"{_fmt((c.get('max_memory_allocated') or 0) / 1e9, '{:.1f}')} |")
    probs = _problems(h)
    L += ["", "head problems: " + ("none" if not probs else "; ".join(probs))]
    return "\n".join(L)


TEST_NAME = "tests/test_hf_packed_attention.py"
