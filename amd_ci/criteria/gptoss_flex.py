#!/usr/bin/env python3
"""Criteria: zoo #2153 GPT-OSS flex inference on gfx1151, zoo base vs head. Pairs with probes/gptoss_flex_probe.py.

Regression by test ID over tests/test_gpt_oss_flex_inference.py (the base runs the head's copy of the file against
its own zoo). Non-vacuity gates, all must hold or the verdict is INCONCLUSIVE:
  - pytest exited 0/1 and collected at base and head; each state imported unsloth_zoo from its own checkout
  - head: torch.cuda.is_available(), torch.version.hip set and arch gfx1151 inside the pytest process
  - head: test_generate_engages_flex_and_kill_switch_restores_eager PASSED (flex inference called during generate,
    UNSLOTH_GPT_OSS_EAGER_INFERENCE=1 gives 0 flex calls)
  - base: at least one GPU flex test fails (the tests discriminate: the base has no flex inference path)
  - head: the eager-vs-flex sanity ran to completion with eager and flex rows
Timing / memory numbers from the sanity are reported, not gated (a shared APU; noise is shown per round).
"""

from __future__ import annotations

import os
import statistics

TITLE = "unsloth-zoo #2153 GPT-OSS flex inference on gfx1151, zoo base vs head"
MODE = "regression"
NEEDS = ["rocm", "gpu", "discrete_gpu", "windows", "nvidia", "multi_gpu", "mig", "xpu", "mlx"]
_RAN = (0, 1)
_ENGAGE = "test_generate_engages_flex_and_kill_switch_restores_eager"
_GPU_TESTS = ("test_inference_matches_fp32_sink_oracle", "test_greedy_tokens_match_eager_and_stock_hf",
              "test_prefill_and_chunked_prefill_match", _ENGAGE, "test_decode_does_not_recompile_after_warmup",
              "test_train_generate_train")


def _states(obs):
    return {n: v for n, v in obs.items() if not n.startswith("_") and not (v or {}).get("skipped")}


def _norm(p: str) -> str:
    return os.path.normcase(os.path.abspath(p)) if p else ""


def _bad(o):
    return set(o.get("failed") or []) | set(o.get("errors") or [])


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name, o in _states(obs).items():
        rc = o.get("rc")
        n = sum((o.get(k) or 0) for k in ("n_passed", "n_failed", "n_skipped", "n_errors"))
        out.append((f"{name}: pytest ran and collected", rc in _RAN and n > 0,
                    f"rc={rc}, {n} outcomes; {o.get('test_missing') or o.get('note') or ''} "
                    f"{str(o.get('stderr_tail'))[-200:]}"))
        f, co = o.get("unsloth_zoo_file") or "", o.get("checkout") or ""
        out.append((f"{name}: unsloth_zoo imported from its own checkout", bool(f) and _norm(f).startswith(_norm(co)),
                    f"{f} (checkout {co}, zoo {str(o.get('zoo_sha'))[:9]})"))
    head, base = obs.get("head") or {}, obs.get("base") or {}
    t = head.get("torch") or {}
    out.append(("head: GPU visible in the pytest process (HIP, gfx1151)",
                bool(t.get("is_available")) and bool(t.get("hip")) and "gfx1151" in str(t.get("arch")),
                f"is_available={t.get('is_available')} hip={t.get('hip')} arch={t.get('arch')} "
                f"name={t.get('device_name')} torch={t.get('version')} triton={t.get('triton')} {t.get('error') or ''}"))
    hb = _bad(head)
    skipped = [k for k in (head.get("skip_reasons") or {}) if _ENGAGE in k]
    engaged = (head.get("n_passed") or 0) > 0 and not any(_ENGAGE in k for k in hb) and not skipped
    out.append((f"head: {_ENGAGE} passed", engaged,
                f"failing at head: {sorted(k.split('::')[-1] for k in hb if _ENGAGE in k) or 'none'}; "
                f"skipped: {skipped or 'no'}"))
    bb = sorted(k.split("::")[-1] for k in _bad(base) if any(g in k for g in _GPU_TESTS))
    out.append(("base: the GPU flex tests fail at the base (they discriminate)", len(bb) > 0,
                f"{len(bb)} failing at the base: {', '.join(bb[:8])}{' ...' if len(bb) > 8 else ''}; "
                f"test copied from head: {bool(base.get('test_copied_from_head'))}"))
    s = head.get("sanity") or {}
    modes = {r.get("mode") for r in s.get("rows") or []}
    out.append(("head: eager-vs-flex sanity ran (eager and flex rows)", s.get("rc") == 0 and {"eager", "flex"} <= modes,
                f"rc={s.get('rc')} modes={sorted(m for m in modes if m)}; {str(s.get('stderr_tail'))[-300:]}"))
    return out


def _sanity_table(s: dict) -> str:
    rows = [r for r in s.get("rows") or [] if r.get("mode") in ("eager", "flex")]
    if not rows:
        return "\n\nSanity: no rows. stderr tail:\n```\n" + str(s.get("stderr_tail"))[-1500:] + "\n```"
    keys = [k for k in rows[0] if k != "mode"]
    lines = ["", "", "Head sanity, tiny GPT-OSS (64 q heads, 8 kv heads, head_dim 64, batch 4 left-padded), "
             "alternating eager / flex (round 1 includes compile):", "",
             "| round | mode | " + " | ".join(keys) + " |", "|---|---|" + "---|" * len(keys)]
    count = {}
    for r in rows:
        count[r["mode"]] = count.get(r["mode"], 0) + 1
        lines.append(f"| {count[r['mode']]} | {r['mode']} | " + " | ".join(str(r.get(k)) for k in keys) + " |")
    med = []
    for k in keys:
        e = [r[k] for r in rows if r["mode"] == "eager" and isinstance(r.get(k), (int, float))][1:]
        f = [r[k] for r in rows if r["mode"] == "flex" and isinstance(r.get(k), (int, float))][1:]
        if e and f:
            me, mf = statistics.median(e), statistics.median(f)
            spread = (max(e) - min(e)) / me * 100 if me else 0.0
            med.append(f"| {k} | {me} | {mf} | {mf / me:.3f} | {spread:.1f}% |" if me else f"| {k} | {me} | {mf} | - | - |")
    if med:
        lines += ["", "Rounds 2-3 (warm) medians:", "", "| metric | eager | flex | flex/eager | eager spread |",
                  "|---|---|---|---|---|"] + med
    tr = [r for r in s.get("rows") or [] if r.get("mode") == "train"]
    if tr:
        lines += ["", f"Training path peak (flex_attention_with_sink): {dict((k, v) for k, v in tr[0].items() if k != 'mode')}"]
    k = s.get("kernels") or {}
    dec = k.get("decisions") or []
    summary = {}
    for d in dec:
        key = (d.get("q_len"), "flex_decoding" if d.get("flex_decoding") else "main flex kernel")
        summary[key] = summary.get(key, 0) + 1
    lines += ["", f"Inductor flex lowering decisions (q_len, kernel): {dict(sorted(summary.items(), key = str))}; "
              f"options seen: {sorted({str(d.get('kernel_options')) for d in dec})[:4]}; "
              f"zoo kernel_options={k.get('kernel_options')}; wrap errors={k.get('errors')}"]
    return "\n".join(lines)


def table(obs: dict) -> str:
    rows = ["| state | zoo | passed | failed | errors | skipped | failing |", "|---|---|---|---|---|---|---|"]
    for name, o in _states(obs).items():
        bad = ", ".join(f"`{x.split('::')[-1]}`" for x in sorted(_bad(o))) or "none"
        rows.append(f"| {name} | {str(o.get('zoo_sha'))[:9]} | {o.get('n_passed', 0)} | {o.get('n_failed', 0)} | "
                    f"{o.get('n_errors', 0)} | {o.get('n_skipped', 0)} | {bad} |")
    head = obs.get("head") or {}
    sk = head.get("skip_reasons") or {}
    if sk:
        rows.append("\nHead skips: " + "; ".join(f"`{k.split('::')[-1]}`: {v}" for k, v in list(sk.items())[:12]))
    return "\n".join(rows) + _sanity_table(head.get("sanity") or {})


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    new = sorted(_bad(head) - _bad(base))
    if new:
        return True, "failing at the head and not at the base: " + ", ".join(f"`{n}`" for n in new)
    detail = f"head {head.get('n_passed')} passed, 0 new failures"
    both = _bad(head) & _bad(base)
    if both:
        detail += "; failing at BOTH states (pre-existing): " + ", ".join(f"`{n}`" for n in sorted(both))
    return False, detail
