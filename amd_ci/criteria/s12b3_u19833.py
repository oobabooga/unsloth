#!/usr/bin/env python3
"""Criteria: unsloth PR 19833 on gfx1151. The ROCm Triton probe kernel moves to module level because Triton < 3.4
resolves a kernel's names through fn.__globals__ only: base's function-local `tl` is a NameError there, so
triton_runs_on_device answers False and FlexAttention is turned off.

Defect (base, old-torch arm, Triton < 3.4): `_triton_probe(0)` raises NameError naming `tl`, and
triton_runs_on_device(0) is False. Fixed (head / merge): old arm `_triton_probe(0)` returns True and
triton_runs_on_device(0) is True; new arm (current ROCm torch, Triton >= 3.4) also True.
Gates: every arm a HIP torch on gfx1151, old arm Triton < 3.4, new arm Triton >= 3.4, functions read from the
state's own tree, head's kernel module loaded from its own tree, base new arm True (no-regression baseline)."""

from __future__ import annotations

TITLE = "unsloth PR 19833: module-level ROCm Triton probe kernel (Triton < 3.4 NameError), gfx1151"
MODE = "differential"
# Real `import unsloth` path not exercised (functions exec'd from source); Windows triton-windows not reached.
NEEDS = ["rocm", "gpu", "windows"]


def _ver(s):
    try:
        return tuple(int(x) for x in str(s).split("+")[0].split(".")[:2])
    except Exception:
        return None


def _arm(o, k):
    return ((o or {}).get("arms") or {}).get(k) or {}


def _true(r, key):
    x = r.get(key) or {}
    return x.get("ok") is True and x.get("value") is True


def gates(obs):
    out = []
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if o is None:
            continue
        for k in ("old", "new"):
            r = _arm(o, k)
            env = r.get("env") or {}
            out.append((f"{name}/{k}: cell ran", r.get("done") is True,
                        f"rc {r.get('rc')} venv {(r.get('venv') or {}).get('error')} {str(r.get('log_tail'))[-300:] if not r.get('done') else ''}"))
            out.append((f"{name}/{k}: HIP torch on gfx1151", bool(env.get("hip")) and "gfx1151" in str(env.get("arch")),
                        f"torch {env.get('torch')} hip {env.get('hip')} {env.get('device')} {env.get('arch')}"))
            tv = _ver(env.get("triton"))
            ok = tv is not None and ((tv < (3, 4)) if k == "old" else (tv >= (3, 4)))
            out.append((f"{name}/{k}: triton {'< 3.4' if k == 'old' else '>= 3.4'}", ok, str(env.get("triton"))))
            f = str(r.get("utils_file") or "")
            out.append((f"{name}/{k}: _utils.py from its own tree", f.startswith(str(o.get("checkout")) + "/")
                        and set(r.get("functions_found") or []) == {"_is_hip_build", "_triton_probe", "triton_runs_on_device"}, f))
            if name != "base":
                kf = str(r.get("kernel_module_file") or "")
                out.append((f"{name}/{k}: kernel module from its own tree", kf.startswith(str(o.get("checkout")) + "/"), kf))
    b = _arm(obs.get("base"), "new")
    out.append(("base/new: probe works on current Triton (baseline)", _true(b, "triton_probe") and _true(b, "triton_runs_on_device"),
                str(b.get("triton_probe"))[:300]))
    return out


def base_shows_defect(base):
    r = _arm(base, "old")
    tp = r.get("triton_probe") or {}
    named = tp.get("ok") is False and "NameError" in (str(tp.get("type")) + str(tp.get("msg")) + str(tp.get("tb"))) \
        and "tl" in str(tp.get("msg")) + str(tp.get("tb"))
    gate_off = (r.get("triton_runs_on_device") or {}).get("value") is False
    return (named and gate_off, f"probe {tp.get('type')}: {str(tp.get('msg'))[:200]}; runs_on_device {(r.get('triton_runs_on_device') or {}).get('value')}")


def head_is_fixed(head):
    o, n = _arm(head, "old"), _arm(head, "new")
    ok = all(_true(r, k) for r in (o, n) for k in ("triton_probe", "triton_runs_on_device"))
    return (ok, f"old {o.get('triton_probe')} / new {n.get('triton_probe')}"[:500])


def state_is_fixed(name, obs):
    return head_is_fixed(obs)


def table(obs):
    rows = ["| state | arm | torch | triton | arch | _triton_probe(0) | triton_runs_on_device(0) |", "|---|---|---|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if not o:
            continue
        for k in ("old", "new"):
            r = _arm(o, k)
            env = r.get("env") or {}

            def show(x):
                x = x or {}
                return repr(x.get("value")) if x.get("ok") else f"{x.get('type')}: {str(x.get('msg')).strip().splitlines()[-1][:200] if x.get('msg') else ''}"
            rows.append(f"| {name} {str(o.get('commit'))[:10]} | {k} | {env.get('torch')} (hip {env.get('hip')}) | {env.get('triton')} | "
                        f"{env.get('arch')} | {show(r.get('triton_probe'))} | {show(r.get('triton_runs_on_device'))} |")
    return "\n".join(rows)
