#!/usr/bin/env python3
"""Criteria: does the Python tool still launch, and still see the GPU, after the change?

PR 10285 routes every default Python/Terminal launch through an OS sandbox
(Bubblewrap / LPAC / Seatbelt) and fails closed when none qualifies. Two
regressions are possible on a real GPU box and both matter:

  1. the tool no longer runs at all (Required mode refused: no bwrap, AppArmor
     userns restriction, LPAC probe failure), and
  2. the tool runs but the GPU is gone (the Bubblewrap profile mounts a fresh
     /dev with no /dev/kfd or /dev/dri; the LPAC token has no device access).

Whether (2) is a bug or an intended policy is the reviewer's call; the criteria
only says what changed. A box whose host torch cannot see a GPU cannot answer
(2), so that reading is reported as VOID for the GPU question, but (1) is still
judged. Pairs with probes/sandbox_tool_gpu_probe.py.
"""

from __future__ import annotations

TITLE = "Python tool launch and GPU visibility, base versus head"
MODE = "regression"
# Everything the change touches, not what this host has (capability.py renders the gaps).
NEEDS = ["gpu", "rocm", "nvidia", "windows", "windows_rocm_wddm", "docker", "multi_gpu", "mlx", "xpu"]


def _tool(state: dict, label: str) -> dict:
    return (state or {}).get(label) or {}


def _gpu(state: dict, label: str):
    parsed = _tool(state, label).get("parsed") or {}
    return parsed.get("cuda_available")


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        out.append((f"{name} probe imported the backend", "import_error" not in o and "error" not in o,
                    (o.get("import_error") or o.get("error") or "ok")[:200]))
    base = obs.get("base") or {}
    ran = _tool(base, "tool_default").get("ran", False)
    out.append(("base tool executed the payload", bool(ran),
                (_tool(base, "tool_default").get("raw") or "no output")[:200]))
    # Host torch is reported, not gated: a box without torch still answers the
    # launch question, and the GPU question is then VOID rather than INCONCLUSIVE.
    return out


def table(obs: dict) -> str:
    rows = ["| state | capability | default-mode tool | GPU in default tool | GPU in Full tool | host GPU | seconds |",
            "|---|---|---|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if not o:
            continue
        cap = o.get("capability") or {}
        cap_s = (f"{cap.get('backend')} / {cap.get('protection_state')} / env={cap.get('environment')}"
                 if cap else "n/a (no os_sandbox module)")
        d = _tool(o, "tool_default")
        if d.get("ran"):
            d_s = "ran"
        elif d.get("fail_closed"):
            d_s = "FAIL-CLOSED"
        else:
            d_s = "did not run"
        rows.append(f"| {name} | {cap_s} | {d_s} | {_gpu(o, 'tool_default')} | {_gpu(o, 'tool_full')} "
                    f"| {(o.get('host_torch') or {}).get('cuda_available')} | {d.get('seconds')} |")
    head = obs.get("head") or {}
    notes = []
    cap = head.get("capability") or {}
    if cap:
        notes.append(f"head capability reason: `{str(cap.get('reason'))[:300]}`")
    h = (head.get("host") or {})
    notes.append(f"host: bwrap={h.get('bwrap')} apparmor_restrict_unprivileged_userns="
                 f"{h.get('apparmor_restrict_unprivileged_userns')} dockerenv={h.get('dockerenv')} "
                 f"uid={h.get('uid')} /dev/kfd={h.get('dev_kfd')} /dev/dri={h.get('dev_dri')}")
    d = _tool(head, "tool_default")
    if d.get("raw"):
        notes.append("head default-mode tool output tail: `" + d["raw"][-300:].replace("\n", " ") + "`")
    return "\n".join(rows) + "\n\n" + "\n".join(notes)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    bd, hd = _tool(base, "tool_default"), _tool(head, "tool_default")
    if bd.get("ran") and not hd.get("ran"):
        why = "fail-closed (Required mode refused to launch)" if hd.get("fail_closed") else "did not run"
        reason = ((head.get("capability") or {}).get("reason") or hd.get("exception") or hd.get("raw") or "")
        return True, (f"the default Python tool ran at the base but {why} at the head: "
                      f"{str(reason)[:300]}")
    bg, hg = _gpu(base, "tool_default"), _gpu(head, "tool_default")
    host_gpu = (base.get("host_torch") or {}).get("cuda_available")
    if bg is True and hg is False:
        full = _gpu(head, "tool_full")
        return True, (f"the default Python tool saw the GPU at the base and not at the head "
                      f"(Full-access control at the head: cuda_available={full}); the sandbox hides "
                      f"the accelerator device nodes")
    if not host_gpu:
        return False, ("no regression in tool launch; the GPU question is VOID here because host "
                       f"torch reports cuda_available={host_gpu}")
    return False, (f"default tool ran at both states; GPU visible at base={bg} head={hg}; "
                   f"head capability={((head.get('capability') or {}).get('protection_state'))}")
