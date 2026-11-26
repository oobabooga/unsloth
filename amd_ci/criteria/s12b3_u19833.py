#!/usr/bin/env python3
"""Criteria: unsloth PR 19833 on gfx1151. The ROCm Triton probe kernel moves to module level because Triton < 3.4
resolves a kernel's names through fn.__globals__ only: base's function-local `tl` is a NameError there, so
triton_runs_on_device answers False and FlexAttention is turned off.

Arms: `old` / `old711` = torch 2.7.1 ROCm (Triton 3.3.1), `new` = current ROCm torch (Triton >= 3.4). Each arm runs
the stages torch_launch, triton_driver, compile_only (triton.compile for the card's target, no launch), probe
(`_triton_probe(0)`), runs_on_device (`triton_runs_on_device(0)`), each in its own process.

Defect (base): on at least one old arm, compile_only raises NameError naming `tl`.
Fixed (head / merge): on every old arm where base showed it, compile_only compiles; and where the arm's
torch_launch (a plain torch kernel, no Triton) and triton_driver stages work in that state, probe and
runs_on_device return True there (an arm whose torch cannot launch any kernel is reported, not judged at launch); new arm probe + runs_on_device True.
Gates: HIP torch on gfx1151, Triton < 3.4 on old arms and >= 3.4 on new, functions from the state's own tree,
head kernel module from its own tree, base new arm probe True (current-Triton baseline)."""

from __future__ import annotations

TITLE = "unsloth PR 19833: module-level ROCm Triton probe kernel (Triton < 3.4 NameError), gfx1151"
MODE = "differential"
# The real `import unsloth` path is not exercised (functions exec'd from source); Windows triton-windows not reached.
NEEDS = ["rocm", "gpu", "windows"]
OLD = ("old", "old711")
STAGES = ("torch_launch", "triton_driver", "compile_only", "probe", "runs_on_device")


def _ver(s):
    try:
        return tuple(int(x) for x in str(s).split("+")[0].split(".")[:2])
    except Exception:
        return None


def _st(o, arm, stage):
    return ((((o or {}).get("arms") or {}).get(arm) or {}).get("stages") or {}).get(stage) or {}


def _res(o, arm, stage):
    return _st(o, arm, stage).get("result") or {}


def _true(o, arm, stage):
    r = _res(o, arm, stage)
    return r.get("ok") is True and r.get("value") is True


def _nameerror_tl(o, arm, stage):
    r = _res(o, arm, stage)
    blob = f"{r.get('type')} {r.get('msg')} {r.get('tb')}"
    return r.get("ok") is False and "NameError" in blob and ("'tl'" in blob or "tl is not defined" in blob)


def _compiled(o, arm):
    r = _res(o, arm, "compile_only")
    return r.get("ok") is True and (r.get("value") or {}).get("compiled") is True


def _launch_ok(o, arm):
    """A plain torch kernel launch works in this venv (no Triton): only then can the probe's launch be judged."""
    return _true(o, arm, "torch_launch") and _res(o, arm, "triton_driver").get("ok") is True


def gates(obs):
    out = []
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if o is None:
            continue
        for arm in OLD + ("new",):
            a = ((o.get("arms") or {}).get(arm) or {})
            out.append((f"{name}/{arm}: venv built", bool(a) and not (a.get("venv") or {}).get("error"),
                        str((a.get("venv") or {}).get("error")) + " " + str((a.get("venv") or {}).get("log_tail") or "")[-300:]))
            st = _st(o, arm, "compile_only")
            env = st.get("env") or {}
            out.append((f"{name}/{arm}: HIP torch on gfx1151", bool(env.get("hip")) and "gfx1151" in str(env.get("arch")),
                        f"torch {env.get('torch')} hip {env.get('hip')} {env.get('device')} {env.get('arch')}"))
            tv = _ver(env.get("triton"))
            ok = tv is not None and ((tv < (3, 4)) if arm in OLD else (tv >= (3, 4)))
            out.append((f"{name}/{arm}: triton {'< 3.4' if arm in OLD else '>= 3.4'}", ok, str(env.get("triton"))))
            out.append((f"{name}/{arm}: compile_only stage finished", st.get("done") is True,
                        f"rc {st.get('rc')} reached {st.get('reached')} {str(st.get('log_tail'))[-300:] if not st.get('done') else ''}"))
            f = str(st.get("utils_file") or "")
            out.append((f"{name}/{arm}: _utils.py from its own tree", f.startswith(str(o.get("checkout")) + "/")
                        and set(st.get("functions_found") or []) == {"_is_hip_build", "_triton_probe", "triton_runs_on_device"}
                        and st.get("module_level_tl") is False, f))
            if name != "base":
                kf = str(st.get("kernel_module_file") or "")
                out.append((f"{name}/{arm}: kernel module from its own tree", kf.startswith(str(o.get("checkout")) + "/"), kf))
    b = obs.get("base")
    out.append(("base/new: probe + runs_on_device True on current Triton (baseline)",
                _true(b, "new", "probe") and _true(b, "new", "runs_on_device"), str(_res(b, "new", "probe"))[:300]))
    return out


def base_shows_defect(base):
    hit = [arm for arm in OLD if _nameerror_tl(base, arm, "compile_only")]
    return (bool(hit), f"compile_only NameError(tl) on {hit}" if hit else "no old arm raised NameError naming tl")


def head_is_fixed(head, base = None):
    why = []
    for arm in OLD:
        if not _compiled(head, arm):
            why.append(f"{arm}: compile_only {str(_res(head, arm, 'compile_only'))[:200]}")
        if _launch_ok(head, arm):
            for stage in ("probe", "runs_on_device"):
                if not _true(head, arm, stage):
                    why.append(f"{arm}: {stage} {str(_res(head, arm, stage))[:200]} rc {_st(head, arm, stage).get('rc')}")
    for stage in ("probe", "runs_on_device"):
        if not _true(head, "new", stage):
            why.append(f"new: {stage} {str(_res(head, 'new', stage))[:200]}")
    return (not why, "; ".join(why) or "compiles on Triton 3.3.1, probe True where the driver loads, True on current Triton")


def state_is_fixed(name, obs):
    return head_is_fixed(obs)


def _show(st):
    r = st.get("result") or {}
    if not st:
        return "-"
    if not st.get("done"):
        return f"CRASH rc {st.get('rc')} (reached {(st.get('reached') or ['-'])[-1]})"
    if r.get("ok"):
        v = r.get("value")
        return "compiled" if isinstance(v, dict) and v.get("compiled") else repr(v)[:80]
    m = str(r.get("msg") or "").strip().splitlines()
    return f"{r.get('type')}: {(m[-1] if m else '')[:160]}"


def table(obs):
    rows = ["| state | arm | torch / triton / arch | " + " | ".join(STAGES) + " |", "|---|---|---|" + "---|" * len(STAGES)]
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if not o:
            continue
        for arm in OLD + ("new",):
            env = _st(o, arm, "compile_only").get("env") or {}
            rows.append(f"| {name} {str(o.get('commit'))[:10]} | {arm} | {env.get('torch')} / {env.get('triton')} / {env.get('arch')} | "
                        + " | ".join(_show(_st(o, arm, s)) for s in STAGES) + " |")
    crash = []
    for name in ("base", "head"):
        for arm in OLD:
            for s in ("torch_launch", "triton_driver", "probe"):
                st = _st(obs.get(name), arm, s)
                if st and not st.get("done"):
                    crash.append(f"{name}/{arm}/{s}: rc {st.get('rc')} log tail: {str(st.get('log_tail'))[-700:]}")
    return "\n".join(rows) + ("\n\nCrashes:\n" + "\n".join("- " + c.replace("\n", " / ") for c in crash) if crash else "")
