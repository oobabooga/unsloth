#!/usr/bin/env python3
"""Criteria (unsloth-zoo PR 1580): routed NF4 / BF16 MoE decode kernels, on gfx1151 ROCm.

Regression mode. The PR adds Triton routed-expert decode kernels taken for decode-sized
no-grad experts calls (UNSLOTH_MOE_ROUTED_KERNEL, default "1", "0" = off). There is no HIP
gate in the code, so on ROCm it either engages (Triton on HIP) or declines. Head is WORSE
than base when any of:
  * a head decode cell crashes or produces non-finite logits
  * the head's kill-switch arm ("0") is not bit-identical to the base (same cell): "off" must
    be exactly the old path
  * a head default arm that did NOT engage (0 routed returns) is not bit-identical to the base
  * a prefill-sized call engaged the routed path (it must keep the current path)
  * a head default arm that ENGAGED:
      - |default - 0| exceeds the PR's own model-level tripwire (max <= 3x, mean <= 2x the
        "0" arm's error vs the fp32 twin), or
      - is less accurate than "0" vs the fp32 twin (max > 1.5x, or mean > 1.25x), or
      - flips a teacher-forced greedy argmax where the "0" arm's top-2 margin is wider than
        2x that row's own |default - 0| (a flip inside it is a near tie, reported only)
  * a test that ran at the base fails at the head, or a test the PR adds fails at the head

Gates (non-vacuity): each state imported unsloth_zoo from its own checkout; every base cell
finished with finite logits; base has no moe_routed module and its default arm equals its "0"
arm (one path); the head has moe_routed; pytest ran (exit 0 / 1) and collected at both states.
"""

from __future__ import annotations

TITLE = "unsloth-zoo PR 1580: routed NF4 / BF16 MoE decode on gfx1151 ROCm, base vs head"
MODE = "regression"
# nvidia: the PR's kernels were tuned / validated on CUDA (B200); NVIDIA behaviour unreachable here.
NEEDS = ["rocm", "gpu", "nvidia"]

_RAN = (0, 1)


def _cells(o: dict) -> dict:
    return (o or {}).get("decode") or {}


def _done(c: dict) -> bool:
    return c.get("stage") == "done" and c.get("ok") is True and not c.get("error") and \
        (c.get("arm0") or {}).get("finite") is True and (c.get("default_forced") or {}).get("finite") is True


def _dec(c: dict, arm: str) -> dict:
    return ((c.get(arm) or {}).get("counts") or {}).get("decode") or {}


def _pre(c: dict, arm: str) -> dict:
    return ((c.get(arm) or {}).get("counts") or {}).get("prefill") or {}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        ck = str(o.get("checkout") or "<none>")
        cells = _cells(o)
        files = {str(c.get("unsloth_zoo_file") or "") for c in cells.values()}
        out.append((f"{name} cells imported the state's unsloth_zoo", bool(cells) and all(f.startswith(ck) and f for f in files),
                    ", ".join(sorted(files)) or "no cells"))
        p = o.get("pytest") or {}
        out.append((f"{name} pytest ran", p.get("rc") in _RAN,
                    f"rc={p.get('rc')} " + (str(p.get("stderr_tail") or "")[-200:] if p.get("rc") not in _RAN else "")))
        out.append((f"{name} pytest collected", (p.get("n_passed", 0) + p.get("n_failed", 0) + p.get("n_skipped", 0)) > 0,
                    f"passed {p.get('n_passed', 0)}, failed {p.get('n_failed', 0)}, skipped {p.get('n_skipped', 0)}"))
    b, h = _cells(obs.get("base")), _cells(obs.get("head"))
    bad = [k for k, c in b.items() if not _done(c)]
    out.append(("every base decode cell finished, finite", bool(b) and not bad,
                "; ".join(f"{k}: {b[k].get('error') or b[k].get('stage')} rc={b[k].get('rc')}" for k in bad) or f"{len(b)} cells"))
    out.append(("base has no moe_routed; head has it",
                bool(b) and bool(h) and not any(c.get("has_moe_routed") for c in b.values())
                and all(c.get("has_moe_routed") for c in h.values() if c.get("stage") != "import"),
                f"base {[c.get('has_moe_routed') for c in b.values()]}, head {[c.get('has_moe_routed') for c in h.values()]}"))
    same = [k for k, c in b.items() if _done(c) and (c.get("arm0") or {}).get("logits_sha") == (c.get("default_forced") or {}).get("logits_sha")]
    out.append(("base default arm == base '0' arm (one path at the base)", len(same) == len(b) and bool(b),
                f"{len(same)}/{len(b)} cells identical"))
    return out


def _f(v, p = 4):
    return f"{v:.{p}g}" if isinstance(v, float) else str(v)


def table(obs: dict) -> str:
    rows = ["| state | cell | routed calls / returned (decode, default arm) | nf4_routed | bf16_routed | prefill returned "
            "| '0' arm returned | max abs default-vs-0 | mean | '0' vs fp32 max / mean | default vs fp32 max / mean "
            "| greedy tokens equal (free run) | forced flips (unexplained) | '0' sha | default sha |",
            "|" + "---|" * 15]
    vers = None
    for name in ("base", "head", "merge"):
        for key, c in _cells(obs.get(name)).items():
            vers = vers or c.get("versions")
            if not _done(c):
                rows.append(f"| {name} | {key} | CRASH: {c.get('error') or c.get('stage')} rc={c.get('rc')} |" + " |" * 13)
                continue
            d, p0, cmp_ = _dec(c, "default_forced"), _pre(c, "default_forced"), c.get("compare") or {}
            rows.append(
                f"| {name} | {key} | {d.get('routed_moe_forward_calls')} / {d.get('routed_moe_forward_returned')} "
                f"| {d.get('nf4_routed')} | {d.get('bf16_routed')} | {p0.get('routed_moe_forward_returned')} "
                f"| {_dec(c, 'arm0').get('routed_moe_forward_returned')} "
                f"| {_f(cmp_.get('max_abs_default_vs_0'))} | {_f(cmp_.get('mean_abs_default_vs_0'))} "
                f"| {_f(cmp_.get('max_abs_0_vs_fp32'))} / {_f(cmp_.get('mean_abs_0_vs_fp32'))} "
                f"| {_f(cmp_.get('max_abs_default_vs_fp32'))} / {_f(cmp_.get('mean_abs_default_vs_fp32'))} "
                f"| {cmp_.get('tokens_equal_free_run')} | {cmp_.get('n_forced_flips')} ({cmp_.get('n_unexplained_flips')}) "
                f"| {(c.get('arm0') or {}).get('logits_sha')} | {(c.get('default_forced') or {}).get('logits_sha')} |")
    rows += ["", "| state | pytest passed | failed | skipped | failing tests |", "|---|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        p = (obs.get(name) or {}).get("pytest")
        if not p:
            continue
        bad = ", ".join(f"`{t.split('::')[-1]}`" for t in p.get("failed", []) + p.get("errors", [])) or "none"
        rows.append(f"| {name} | {p.get('n_passed')} | {p.get('n_failed')} | {p.get('n_skipped')} | {bad} |")
    hp = (obs.get("head") or {}).get("pytest") or {}
    reasons: dict = {}
    for tid, msg in (hp.get("skipped") or {}).items():
        reasons.setdefault(msg, []).append(tid.split("::")[-1])
    if reasons:
        rows += ["", "Head skip reasons:"]
        for msg, tids in sorted(reasons.items(), key = lambda kv: -len(kv[1])):
            rows.append(f"- {len(tids)}x `{msg}` (e.g. `{tids[0]}`)")
    if vers:
        rows += ["", "Versions: " + ", ".join(f"{k}={v}" for k, v in vers.items())]
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    problems, notes = [], []
    bc, hc = _cells(base), _cells(head)
    engaged, declined = [], []
    for key, c in hc.items():
        if not _done(c):
            problems.append(f"head {key} did not finish: {c.get('error') or ('rc=' + str(c.get('rc')) + ' stage=' + str(c.get('stage')))}")
            continue
        b = bc.get(key) or {}
        h0 = (c.get("arm0") or {}).get("logits_sha")
        hd = (c.get("default_forced") or {}).get("logits_sha")
        b0 = (b.get("arm0") or {}).get("logits_sha")
        if h0 != b0:
            problems.append(f"{key}: head UNSLOTH_MOE_ROUTED_KERNEL=0 logits ({h0}) not bit-identical to base ({b0})")
        if _dec(c, "arm0").get("routed_moe_forward_returned"):
            problems.append(f"{key}: the '0' arm still took the routed path ({_dec(c, 'arm0')})")
        if _pre(c, "default_forced").get("routed_moe_forward_returned"):
            problems.append(f"{key}: a prefill-sized call took the routed path ({_pre(c, 'default_forced')})")
        n = _dec(c, "default_forced").get("routed_moe_forward_returned") or 0
        cmp_ = c.get("compare") or {}
        if n == 0:
            declined.append(key)
            if hd != b0:
                problems.append(f"{key}: routed path declined but default logits ({hd}) differ from base ({b0})")
            continue
        engaged.append(f"{key} ({n}/{_dec(c, 'default_forced').get('routed_moe_forward_calls')})")
        mx, mn = cmp_.get("max_abs_default_vs_0"), cmp_.get("mean_abs_default_vs_0")
        e0x, e0m = cmp_.get("max_abs_0_vs_fp32"), cmp_.get("mean_abs_0_vs_fp32")
        e1x, e1m = cmp_.get("max_abs_default_vs_fp32"), cmp_.get("mean_abs_default_vs_fp32")
        if None in (mx, mn, e0x, e0m, e1x, e1m):
            problems.append(f"{key}: comparison missing {cmp_}")
            continue
        if mx > 3 * e0x or mn > 2 * e0m:
            problems.append(f"{key}: |routed - off| max {mx:.4g} / mean {mn:.4g} beyond the PR's tripwire "
                            f"(3x / 2x the off arm's fp32 error {e0x:.4g} / {e0m:.4g})")
        if e1x > 1.5 * e0x or e1m > 1.25 * e0m:
            problems.append(f"{key}: routed less accurate vs fp32 than off (max {e1x:.4g} vs {e0x:.4g}, mean {e1m:.4g} vs {e0m:.4g})")
        if cmp_.get("n_unexplained_flips"):
            problems.append(f"{key}: greedy argmax flipped at a wide margin: {c.get('forced_flips')}")
        elif cmp_.get("n_forced_flips"):
            notes.append(f"{key}: {cmp_['n_forced_flips']} forced argmax flip(s), all near ties {c.get('forced_flips')}")
        if not cmp_.get("tokens_equal_free_run"):
            notes.append(f"{key}: free-run greedy tokens diverged (after a near-tie flip)")
    notes.insert(0, "routed path ENGAGED on this host in " + (", ".join(engaged) or "no cell")
                 + "; declined (bit-identical to base required) in " + (", ".join(declined) or "no cell"))
    bp, hp = base.get("pytest") or {}, head.get("pytest") or {}
    b = set(bp.get("failed", [])) | set(bp.get("errors", []))
    h = set(hp.get("failed", [])) | set(hp.get("errors", []))
    new = sorted(h - b)
    if new:
        problems.append("failing at the head (not failing at the base): " + ", ".join(f"`{n.split('::')[-1]}`" for n in new))
    if b & h:
        notes.append("failing at BOTH states (pre-existing): " + ", ".join(f"`{n.split('::')[-1]}`" for n in sorted(b & h)))
    if problems:
        return True, "; ".join(problems) + ". Also: " + "; ".join(notes)
    return False, "; ".join(notes)
