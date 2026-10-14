#!/usr/bin/env python3
"""Criteria: does every visibility-mask cell pick the same AMD target at base and head?

For PR 11965 (setup.sh and install_llama_prebuilt.py stop reapplying
ROCR_VISIBLE_DEVICES over rocminfo output, which ROCr already filtered). The PR's
intended behaviour changes need at least two AMD devices (ROCR=1,0 over a filtered
rocminfo list, amd-smi survivor prefixes over several cards, UUID over unlike
adapters). On this ONE-GPU host no cell can tell the old and new rules apart, so any
difference is a regression finding, not an intended change.

Regression mode. Compared per (arm, cell): setup.sh _setup_gfx, _setup_mkt,
_setup_amd_detected and any error; detect_host() has_rocm, rocm_gfx_target,
rocm_gfx_targets and any error. _setup_amd_probe is shown but NOT compared: it is a
variable the PR introduces, so it is necessarily empty at the base.

Gates, so a match cannot be vacuous:
  * both probes wrote observations and the setup.sh block lifted and parsed;
  * the host's real rocminfo lists gfx1151, unmasked;
  * the unmasked rocminfo cell resolves gfx1151 in BOTH consumers at BOTH states
    (proves the real tools were read, not a CPU / NVIDIA / empty fallback);
  * head's setup.sh records _setup_amd_probe=rocminfo on that cell (the new
    bookkeeping is live, so the amd-smi-only filter is correctly bypassed);
  * amdsmi arm: when the host HAS amd-smi, the unmasked cell detected an AMD GPU in
    both consumers at both states (proves the fall-through reached the real amd-smi).
    When amd-smi is absent the arm is reported as unmeasured in the table and the
    verdict text, not passed off as covered.

Pairs with probes/amd_mask_selection_probe.py.
"""

from __future__ import annotations

TITLE = "AMD gfx selection under visibility masks on real gfx1151 tools, base versus head"
MODE = "regression"
# What the change touches: Linux ROCm selection across several AMD devices, amd-smi
# ordinals, iGPU + dGPU repicks. Only linux, rocm, gpu and amd_smi exist here.
NEEDS = ["linux", "rocm", "gpu", "amd_smi", "multi_gpu_amd", "discrete_gpu"]

EXPECT_GFX = "gfx1151"
UNMASKED = "none"
SETUP_KEYS = ("gfx", "mkt", "amd_detected", "error")
DETECT_KEYS = ("has_rocm", "rocm_gfx_target", "rocm_gfx_targets", "error")


def _cells(o: dict) -> dict:
    return (o or {}).get("cells") or {}


def _cell(o: dict, key: str) -> dict:
    return _cells(o).get(key) or {}


def resolved(o: dict) -> dict:
    out = {}
    for key, c in _cells(o).items():
        s, d = c.get("setup_sh") or {}, c.get("detect") or {}
        out[key] = {
            "setup": tuple(s.get(k) for k in SETUP_KEYS),
            "detect": tuple(str(d.get(k)) if k == "rocm_gfx_targets" else d.get(k)
                            for k in DETECT_KEYS),
        }
    return out


def _amd_smi(o: dict) -> bool:
    return bool(((o or {}).get("host") or {}).get("amd_smi_present"))


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        probe_ok = (o.get("_probe_rc") == 0 and not o.get("_missing_output")
                    and not o.get("_parse_error") and bool(_cells(o)))
        out.append((f"{name} probe wrote observations", probe_ok,
                    f"rc={o.get('_probe_rc')}, {len(_cells(o))} cells"
                    + ("; missing output" if o.get("_missing_output") else "")
                    + (f"; {o['_parse_error']}" if o.get("_parse_error") else "")))
        lift = o.get("lift") or {}
        out.append((f"{name} setup.sh block lifted and parsed",
                    not lift.get("error") and lift.get("bash_n_rc") == 0
                    and bool(lift.get("nvidia_pinned_false")),
                    lift.get("error") or f"{lift.get('block_lines')} lines, NVIDIA pinned "
                    f"false={lift.get('nvidia_pinned_false')}"))
        u = _cell(o, f"rocminfo|{UNMASKED}")
        view = (u.get("rocminfo_view") or {}).get("gpu_names")
        out.append((f"{name} host rocminfo lists {EXPECT_GFX} unmasked",
                    EXPECT_GFX in (view or []), f"rocminfo GPU agents {view}"))
        s, d = u.get("setup_sh") or {}, u.get("detect") or {}
        out.append((f"{name} unmasked setup.sh resolves {EXPECT_GFX}",
                    s.get("gfx") == EXPECT_GFX,
                    f"gfx={s.get('gfx')!r} mkt={s.get('mkt')!r} err={s.get('error')}"))
        out.append((f"{name} unmasked detect_host resolves {EXPECT_GFX}",
                    d.get("rocm_gfx_target") == EXPECT_GFX,
                    f"target={d.get('rocm_gfx_target')!r} targets={d.get('rocm_gfx_targets')} "
                    f"err={d.get('error')}"))
        if _amd_smi(o):
            a = _cell(o, f"amdsmi|{UNMASKED}")
            sa, da = a.get("setup_sh") or {}, a.get("detect") or {}
            out.append((f"{name} amd-smi arm reached the real amd-smi (rocminfo hidden)",
                        sa.get("amd_detected") == "true" and da.get("has_rocm") is True,
                        f"setup detected={sa.get('amd_detected')} gfx={sa.get('gfx')!r}; "
                        f"detect has_rocm={da.get('has_rocm')}"))
    h = _cell(obs.get("head") or {}, f"rocminfo|{UNMASKED}").get("setup_sh") or {}
    out.append(("head setup.sh records _setup_amd_probe=rocminfo on the rocminfo arm",
                h.get("amd_probe") == "rocminfo", f"{h.get('amd_probe')!r}"))
    return out


def _fmt(v) -> str:
    return str(v).replace("|", "\\|")[:120]


def table(obs: dict) -> str:
    base, head = obs.get("base") or {}, obs.get("head") or {}
    b, h = resolved(base), resolved(head)
    rows = ["| arm | cell | rocminfo agents | setup.sh base (gfx, mkt, detected, err) | "
            "setup.sh head | head probe | detect_host base (has_rocm, target, targets, err) | "
            "detect_host head | same |",
            "|---|---|---|---|---|---|---|---|---|"]
    for key in b:
        c = _cell(head, key) or _cell(base, key)
        arm, cell = key.split("|", 1)
        view = (c.get("rocminfo_view") or {}).get("gpu_names", "-") if arm == "rocminfo" else "hidden"
        hv = h.get(key) or {}
        same = b[key] == hv
        rows.append(f"| {arm} | `{cell}` | {_fmt(view)} | `{_fmt(b[key]['setup'])}` | "
                    f"`{_fmt(hv.get('setup'))}` | "
                    f"`{_fmt((c.get('setup_sh') or {}).get('amd_probe'))}` | "
                    f"`{_fmt(b[key]['detect'])}` | `{_fmt(hv.get('detect'))}` | "
                    f"{'yes' if same else 'NO'} |")
    if not _amd_smi(head):
        rows += ["", "amd-smi is NOT on PATH on this runner: the amdsmi arm resolved "
                 "nothing in either state and measured nothing about amd-smi survivors."]
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    b, h = resolved(base), resolved(head)
    missing = sorted(set(b) ^ set(h))
    moved = [k for k in b if k in h and b[k] != h[k]]
    if missing or moved:
        return True, ("selection moved at the head on this 1-GPU host (no PR-intended "
                      "difference is reachable with one device): "
                      + ", ".join(f"`{k}`" for k in moved + missing))
    note = "" if _amd_smi(head) else " (amdsmi arm unmeasured: amd-smi absent)"
    return False, (f"all {len(b)} (arm, mask) cells resolve identically in setup.sh and "
                   f"detect_host; unmasked both pick `{EXPECT_GFX}`{note}")
