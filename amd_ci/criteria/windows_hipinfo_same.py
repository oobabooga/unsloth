#!/usr/bin/env python3
"""Criteria: does install_llama_prebuilt.detect_host() pick the same gfx target per mask cell at
base (round 7 reverted) and head on a real one-GPU Windows gfx1151 host?

Round 7 changed Windows hipinfo mask handling (#13925: hipinfo output is already HIP-filtered;
#14019: an all-hiding ROCR mask means no ROCm target). With one GPU, the only intended
difference is that ROCR "" / "-1" with HIP unset must resolve no ROCm target at head. Every
other cell moving is a regression finding.

Regression mode. Pairs with probes/windows_hipinfo_probe.py.
"""

from __future__ import annotations

TITLE = "Windows hipinfo gfx selection per mask on real gfx1151, round 7 reverted versus head"
MODE = "regression"
NEEDS = ["windows", "rocm", "gpu", "multi_gpu_amd"]
EXPECT_GFX = "gfx1151"
KEYS = ("has_rocm", "rocm_gfx_target", "rocm_gfx_targets", "error")
INTENDED = {"rocr_empty", "rocr_minus1"}


def _cells(o):
    return (o or {}).get("cells") or {}


def _r(c):
    return tuple(str(c.get(k)) if k == "rocm_gfx_targets" else c.get(k) for k in KEYS)


def gates(obs):
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        cells = _cells(o)
        out.append((f"{name} probe wrote observations", o.get("_probe_rc") == 0 and bool(cells),
                    f"rc={o.get('_probe_rc')}, {len(cells)} cells"))
        u = cells.get("none") or {}
        out.append((f"{name} unmasked detect_host resolves {EXPECT_GFX}",
                    u.get("rocm_gfx_target") == EXPECT_GFX,
                    f"target={u.get('rocm_gfx_target')!r} targets={u.get('rocm_gfx_targets')} err={u.get('error')}"))
    return out


def table(obs):
    b, h = _cells(obs.get("base")), _cells(obs.get("head"))
    rows = ["| cell | env | base (has_rocm, target, targets, err) | head | same |", "|---|---|---|---|---|"]
    for k in b:
        same = _r(b[k]) == _r(h.get(k) or {})
        rows.append(f"| `{k}` | `{b[k].get('env')}` | `{_r(b[k])}` | `{_r(h.get(k) or {})}` | "
                    f"{'yes' if same else ('intended' if k in INTENDED else 'NO')} |")
    host = (obs.get("head") or {}).get("host") or {}
    rows += ["", f"hipinfo: `{host.get('hipinfo_on_path')}` HIP_PATH=`{host.get('hip_path')}` "
             f"lines: `{host.get('hipinfo_gcn_lines')}`"]
    return "\n".join(rows)


def head_is_worse(base, head):
    b, h = _cells(base), _cells(head)
    moved = [k for k in b if k not in INTENDED and _r(b[k]) != _r(h.get(k) or {})]
    bad_intended = [k for k in INTENDED if k in h and h[k].get("rocm_gfx_target")]
    errors = [k for k in h if h[k].get("error") and not (b.get(k) or {}).get("error")]
    if moved or bad_intended or errors:
        return True, ("moved: " + ", ".join(moved) + "; all-hiding ROCR still picked a target: "
                      + ", ".join(bad_intended) + "; new errors: " + ", ".join(errors))
    return False, f"{len(b)} cells: every non-ROCR-hiding cell identical; unmasked picks {EXPECT_GFX}"
