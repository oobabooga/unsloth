#!/usr/bin/env python3
"""Criteria (unsloth #17780): native FSDP2 resume of expert LoRA that Zoo saved in vanilla PEFT's layout.

The defect: Zoo's wrapped get_peft_model_state_dict saves expert LoRA in vanilla PEFT's layout (swapped factors,
legacy [out, E, R] lora_B); native_load_model_checkpoint copied the file straight into the FSDP2-sharded params,
so the swap / legacy cases fail with a size mismatch (or restore wrong values).

Differential mode.
  base_shows_defect: the PR's test (same bytes in both states) ran 3 cases at the base; [swap] AND [legacy] failed,
    at least one naming a size / shape mismatch; [noswap] passed (the harness itself works at the base).
  head_is_fixed: all 3 cases passed at the head, AND the non-FSDP2 smoke (single-GPU tiny Qwen3-MoE LoRA SFT, save
    at step 3, resume) finished both phases with every LoRA tensor restored bit-exact and resumed losses at steps
    4-6 within 2e-2 relative of the uninterrupted run, i.e. the PR leaves the ordinary PEFT resume path alone.
Gates: each state imported its own fsdp2_native; head has _peft_trained_layout, base does not; the installed Zoo has
_from_vanilla (else the test skips: vacuous); the test bytes match the head's; base smoke finished (so a head
smoke failure is attributable to the change, not the environment).
"""
from __future__ import annotations

TITLE = "unsloth #17780: native FSDP2 resume of vanilla-layout expert LoRA, gfx1151, base vs head"
MODE = "differential"
NEEDS = ["rocm", "gpu", "multi_gpu", "nvidia", "windows"]

MODES = ("noswap", "swap", "legacy")


def _cases(o):
    return ((o or {}).get("test") or {}).get("cases") or {}


def _case(o, mode):
    for tid, c in _cases(o).items():
        if tid.endswith(f"[{mode}]"):
            return c
    return {}


def _sft(o):
    return (o or {}).get("sft") or {}


def _smoke_ok(o):
    s = _sft(o)
    t, r = s.get("train") or {}, s.get("resume") or {}
    if t.get("stage") != "done" or r.get("stage") != "done":
        return False, f"train={t.get('stage')} {t.get('error') or ''} resume={r.get('stage')} {r.get('error') or ''}"[:400]
    if r.get("resume_max_abs_diff") != 0 or not r.get("resume_n_compared"):
        return False, f"restore max abs diff {r.get('resume_max_abs_diff')} over {r.get('resume_n_compared')} tensors"
    lt = {int(k): v for k, v in (t.get("losses") or {}).items()}
    lr = {int(k): v for k, v in (r.get("losses") or {}).items()}
    steps = [s for s in (4, 5, 6) if s in lt and s in lr]
    if len(steps) != 3:
        return False, f"resumed steps {sorted(lr)} vs uninterrupted {sorted(lt)}"
    rel = max(abs(lt[s] - lr[s]) / max(abs(lt[s]), 1e-6) for s in steps)
    return rel <= 2e-2, f"exact restore of {r['resume_n_compared']} LoRA tensors; loss rel diff steps 4-6 {rel:.2e}"


def gates(obs):
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        imp = o.get("imports") or {}
        f = str(imp.get("fsdp2_native_file") or "")
        ck = str(o.get("checkout") or "")
        out.append((f"{name} imported its own fsdp2_native", bool(f) and bool(ck) and
                    f.replace("\\", "/").startswith(ck.replace("\\", "/")), f or str(imp)[:300]))
        out.append((f"{name} Zoo has _from_vanilla", imp.get("zoo_has_from_vanilla") is True,
                    f"unsloth_zoo {imp.get('zoo_version')} {imp.get('zoo_error') or ''}"))
        t = o.get("test") or {}
        out.append((f"{name} test ran all 3 cases", t.get("rc") in (0, 1) and len(_cases(o)) == 3,
                    f"rc={t.get('rc')} cases={len(_cases(o))} {t.get('stderr_tail', '')[-200:] if t.get('rc') not in (0, 1) else ''}"))
    h = obs.get("head") or {}
    ht = h.get("test") or {}
    out.append(("test bytes = head's test", ht.get("state_had_test") is True and
                ht.get("state_test_sha256") == ht.get("shipped_sha256"), str(ht.get("state_test_sha256"))[:16]))
    out.append(("head has _peft_trained_layout", (h.get("imports") or {}).get("has_peft_trained_layout") is True, ""))
    out.append(("base lacks _peft_trained_layout",
                ((obs.get("base") or {}).get("imports") or {}).get("has_peft_trained_layout") is False, ""))
    for name in ("base", "head"):
        t = (_sft(obs.get(name)).get("train") or {})
        v = t.get("versions") or {}
        out.append((f"{name} SFT smoke ran on the gfx1151 GPU", str(t.get("model_device", "")).startswith("cuda")
                    and "gfx1151" in str(v.get("arch")), f"{t.get('model_device')} {v.get('device')} {v.get('arch')}"))
    ok, ev = _smoke_ok(obs.get("base"))
    out.append(("base SFT save+resume smoke finished cleanly", ok, ev))
    return out


def base_shows_defect(base):
    sw, lg, ns = _case(base, "swap"), _case(base, "legacy"), _case(base, "noswap")
    msgs = (sw.get("message", "") + lg.get("message", "")).lower()
    return (sw.get("outcome") == "failed" and lg.get("outcome") == "failed" and ns.get("outcome") == "passed"
            and ("size" in msgs or "shape" in msgs))


def head_is_fixed(head):
    return all(_case(head, m).get("outcome") == "passed" for m in MODES) and _smoke_ok(head)[0]


def table(obs):
    rows = ["| state | commit | zoo | " + " | ".join(MODES) + " | SFT smoke (non-FSDP2) | device |",
            "|---|---|---|" + "---|" * len(MODES) + "---|---|"]
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if not o:
            continue
        cells = []
        for m in MODES:
            c = _case(o, m)
            msg = " ".join(c.get("message", "").split())
            i = msg.lower().find("size")
            cells.append(f"{c.get('outcome')}" + (f": {msg[max(0, i - 60):i + 120]}" if c.get("outcome") == "failed" else ""))
        ok, ev = _smoke_ok(o)
        v = ((_sft(o).get("train") or {}).get("versions") or {})
        rows.append(f"| {name} | {str(o.get('commit'))[:10]} | {(o.get('imports') or {}).get('zoo_version')} | "
                    + " | ".join(c.replace("|", "/") for c in cells)
                    + f" | {'ok' if ok else 'FAIL'}: {ev} | {v.get('device')} {v.get('arch')} torch {v.get('torch')} |")
    lines = ["\n".join(rows), "", "Per-test ids:"]
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if o:
            lines += [f"- {name}: " + ", ".join(f"`{t.split('::')[-1]}` {c.get('outcome')}" for t, c in sorted(_cases(o).items()))]
    return "\n".join(lines)
