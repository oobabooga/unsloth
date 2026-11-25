#!/usr/bin/env python3
"""Criteria for probes/rocm_secondary_sdpa_probe.py (PR 14655).

Defect at the base, either of:
  1. checking a math-only candidate secondary changes the process-wide flash / mem_efficient SDPA switches the
     primary runs under (or the second identical check keeps the card the first rejected);
  2. the tiny MiniMax-H3 denoiser (fp32 stem, bf16 attention) is probed at a dtype other than its attention's.
Fixed at the head: neither, and the homogeneous all-real pair is unchanged.

Gates: real ROCm torch on a gfx target, not the CPU smoke mode, every scenario ran, each state imported its own
tree, the real cuda:0 probe answered at bf16 with a fused kernel (else nothing can be shown).
"""

from __future__ import annotations

TITLE = "ROCm layer split: secondary SDPA probe vs process-wide switches, and probe dtype"
MODE = "differential"
NEEDS = ["gpu", "rocm", "multi_gpu", "discrete_gpu"]

_SC = ("kernels", "mixed_switches", "real_spoof_switches", "mixed_h3_dtype")
_FUSED = ("flash", "mem_efficient", "cudnn")


def _sc(o: dict, name: str) -> dict:
    return ((o or {}).get("scenarios") or {}).get(name) or {}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        for sc in _SC:
            r = _sc(o, sc)
            out.append((f"{name}: {sc} ran", bool(r.get("ok")),
                        (r.get("error") or r.get("stderr_tail") or "ok")[-300:].replace("\n", " ").replace("|", "/")))
        k = _sc(o, "kernels")
        out.append((f"{name}: real ROCm torch, not smoke", bool(k.get("hip")) and not k.get("smoke"),
                    f"torch {k.get('torch')} hip {k.get('hip')} arch {k.get('arch')}"))
        imp = (k.get("imported_from") or {}).values()
        root = str(o.get("checkout") or "")
        out.append((f"{name}: imported its own tree", bool(root) and all(str(p).startswith(root) for p in imp)
                    and bool(imp), ", ".join(imp)))
        bf16 = (k.get("real_kernels") or {}).get("torch.bfloat16") or []
        out.append((f"{name}: real cuda:0 bf16 probe has a fused kernel", any(x in bf16 for x in _FUSED),
                    str(bf16)))
        m = _sc(o, "mixed_switches")
        real0 = [c for c in m.get("child_calls", []) if c.get("device") == "cuda:0"
                 and c.get("source") == "real child probe"]
        out.append((f"{name}: mixed scenario used the real cuda:0 probe", bool(real0), f"{len(real0)} real calls"))
    return out


def _flags(r: dict, key: str) -> str:
    v = r.get(key)
    return "n/a" if v is None else f"flash={v[0]}, mem_eff={v[1]}"


def table(obs: dict) -> str:
    rows = ["| observation | base | head |", "|---|---|---|"]

    def row(label, f):
        cells = []
        for n in ("base", "head"):
            try:
                cells.append(str(f(obs.get(n) or {})))
            except Exception as e:  # noqa: BLE001
                cells.append(f"error {e!r}")
        rows.append(f"| {label} | " + " | ".join(cells) + " |")

    for dt in ("torch.float16", "torch.bfloat16", "torch.float32"):
        row(f"real gfx1151 SDPA kernels at {dt}", lambda o, dt=dt: _sc(o, "kernels")["real_kernels"].get(dt))
    row("flash_attn package runs at bf16 / fp32", lambda o: _sc(o, "kernels").get("flash_attn_runs"))
    m = "mixed_switches"
    row("mixed: switches at start", lambda o: _flags(_sc(o, m), "flags_start"))
    row("mixed: after the primary's own probe", lambda o: _flags(_sc(o, m), "flags_after_primary_probe"))
    row("mixed: after 1st secondary check", lambda o: _flags(_sc(o, m), "flags_after_first_secondary_check"))
    row("mixed: kept cards, 1st / 2nd identical check",
        lambda o: f"{_sc(o, m).get('kept_first')} / {_sc(o, m).get('kept_second')}")
    row("mixed: after 2nd secondary check", lambda o: _flags(_sc(o, m), "flags_after_second_secondary_check"))
    row("mixed: primary reads math-only afterwards", lambda o: _sc(o, m).get("primary_math_only_after"))
    row("mixed: primary's real bf16 SDPA op afterwards",
        lambda o: ", ".join(x for x in _sc(o, m).get("primary_sdpa_ops_after", []) if "scaled_dot" in x))
    s = "real_spoof_switches"
    row("all-real (cuda:1 = spoofed same card): switches after / kept",
        lambda o: f"{_flags(_sc(o, s), 'flags_after_second_secondary_check')}; "
                  f"{_sc(o, s).get('kept_first')} / {_sc(o, s).get('kept_second')}")
    h = "mixed_h3_dtype"
    row("H3 model", lambda o: _sc(o, h).get("model"))
    row("H3 first param / attention projection dtype",
        lambda o: f"{_sc(o, h).get('first_param')} / {_sc(o, h).get('attention_projection_dtype')}")
    row("H3 SDPA probe dtype", lambda o: _sc(o, h).get("probe_dtype"))
    row("H3 primary math-only at that dtype (real)", lambda o: _sc(o, h).get("primary_math_only_at_probe_dtype"))
    row("H3 kept secondaries (cuda:1 math-only)", lambda o: _sc(o, h).get("kept"))
    row("H3 flash-branch dtype / flash_attn runs there",
        lambda o: f"{_sc(o, h).get('flash_branch_dtype')} / {_sc(o, h).get('flash_runs_at_flash_branch_dtype')}")
    return "\n".join(rows)


def _switch_defect(o: dict) -> list[str]:
    m = _sc(o, "mixed_switches")
    bad = []
    if m.get("flags_after_first_secondary_check") != m.get("flags_after_primary_probe"):
        bad.append("secondary check changed the process-wide switches")
    if m.get("flags_after_second_secondary_check") != m.get("flags_after_primary_probe"):
        bad.append("switches changed after the second check")
    if m.get("kept_first") != m.get("kept_second"):
        bad.append("identical checks disagree")
    if m.get("kept_first") != []:
        bad.append("math-only secondary kept")
    return bad


def _dtype_defect(o: dict) -> list[str]:
    h = _sc(o, "mixed_h3_dtype")
    bad = []
    want = h.get("attention_projection_dtype")
    if h.get("probe_dtype") != want:
        bad.append(f"probe at {h.get('probe_dtype')} not {want}")
    if h.get("flash_branch_dtype") != want:
        bad.append(f"flash branch at {h.get('flash_branch_dtype')} not {want}")
    return bad


def base_shows_defect(base: dict):
    bad = _switch_defect(base) + _dtype_defect(base)
    return (bool(bad), "; ".join(bad))


def head_is_fixed(head: dict):
    bad = _switch_defect(head) + _dtype_defect(head)
    s = _sc(head, "real_spoof_switches")
    if s.get("flags_after_second_secondary_check") != s.get("flags_start"):
        bad.append("all-real pair changed the switches")
    return (not bad, "; ".join(bad) or "fixed")
