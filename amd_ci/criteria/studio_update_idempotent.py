#!/usr/bin/env python3
"""Criteria: does a no-op Studio update stop redoing work?

The defect (issue #10579 and the update bisection): an update that changes no
package still rebuilds sidecars, re-fetches payloads, re-validates binaries and
fails outright when the network is gone. On an AMD host there is a second edge:
setup.sh's AMD escape may throw a ROCm torch away and install it again.

Pairs with probes/studio_update_probe.py. Differential: the base must SHOW the
defect on this host, or the run is VOID.

    base shows the defect   the second no-op update (role `noop`) did visible work
                            or the offline no-op failed
    head is fixed           the second no-op is download-free and idempotent, the
                            offline one succeeds with zero connections, torch is
                            never reinstalled, and the AMD escape never fires

Judges only; every number it reads was written by the probe.
"""

from __future__ import annotations

TITLE = "No-op Studio update: work done, bytes fetched, offline behaviour (base vs head)"
MODE = "differential"
# Everything the change touches, not only what this host has: the report bounds
# itself by NEEDS minus the host, so an under-declared list bounds nothing.
NEEDS = ["rocm", "gpu", "windows", "windows_rocm_wddm", "nvidia", "mlx", "xpu"]

# Bytes below this from payload hosts are index chatter, not a fetch.
PAYLOAD_TOLERANCE = 64 * 1024
# Steps whose presence in the update log means the update redid work.
WORK_STEPS = ("sidecar_rebuilt", "llama_installed", "whisper_installed")


def _s(o: dict, key: str) -> dict:
    return (o or {}).get(key) or {}


def _torch_is_rocm(t: dict) -> bool:
    return bool(t.get("present")) and (bool(t.get("hip")) or "rocm" in str(t.get("version", "")))


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        inst = _s(o, "install")
        ok = inst.get("present") and inst.get("exit_code") == 0
        detail = o.get("error") or f"exit={inst.get('exit_code')} seconds={inst.get('seconds')}"
        if not inst.get("present"):
            detail += "; " + (o.get("log_tail") or "")[-300:]
        out.append((f"{name}: fresh install from this state's wheel succeeded", bool(ok), detail))

        t = o.get("torch_after_install") or {}
        detail = f"version={t.get('version')} hip={t.get('hip')} gpu_available={t.get('gpu_available')}"
        if not t.get("present"):
            detail = f"torch missing: {t.get('error')}"
        out.append((f"{name}: installed torch is a ROCm build (the AMD question was asked)",
                    _torch_is_rocm(t), detail))

        noop = _s(o, "noop")
        out.append((f"{name}: second no-op update ran under the proxy", bool(noop.get("present")),
                    f"exit={noop.get('exit_code')} connections={noop.get('connections_attempted')}"))
        off = _s(o, "offline")
        refused = off.get("connections_refused")
        answered = off.get("pypi_probe_answered")
        out.append((f"{name}: offline no-op was really offline (PyPI probe unanswered)",
                    bool(off.get("present")) and not answered,
                    f"refused={refused} pypi_probe_answered={answered}"))
    b, h = obs.get("base") or {}, obs.get("head") or {}
    bt, ht = (b.get("torch_after_install") or {}).get("version"), (h.get("torch_after_install") or {}).get("version")
    out.append(("both states installed the same torch build (comparable no-op cost)", bt == ht,
                f"base={bt} head={ht}"))
    return out


def _work_evidence(o: dict) -> list[str]:
    """Why a no-op update counts as having done work. Empty means it did none."""
    why: list[str] = []
    noop = _s(o, "noop")
    off = _s(o, "offline")
    if noop.get("exit_code") not in (0,):
        why.append(f"second no-op exited {noop.get('exit_code')}")
    if noop.get("payload_bytes_down", 0) > PAYLOAD_TOLERANCE:
        why.append(f"second no-op fetched {noop.get('payload_bytes_down', 0) / 1e6:.1f} MB of payloads")
    ran = set(noop.get("update_steps_ran") or [])
    hit = sorted(ran & set(WORK_STEPS))
    if hit:
        why.append("second no-op redid: " + ", ".join(hit))
    if noop.get("sidecars_changed"):
        why.append("sidecar venvs were rewritten by the second no-op")
    if noop.get("idempotent_relaxed") is False:
        why.append("second no-op was not idempotent: " + "; ".join(map(str, noop.get("relaxed_reasons") or []))[:300])
    if off.get("exit_code") != 0:
        why.append(f"offline no-op failed (exit {off.get('exit_code')})")
    if off.get("total_bytes_down", 0) > 0:
        why.append(f"offline no-op still moved {off.get('total_bytes_down')} bytes")
    if noop.get("amd_escape_fired") or off.get("amd_escape_fired"):
        why.append("the AMD escape forced a dependency pass on a no-op")
    m0, m1, m2 = (o.get("torch_record_mtime_after_settle"), o.get("torch_record_mtime_after_noop"),
                  o.get("torch_record_mtime_after_offline"))
    if m0 and (m1 != m0 or m2 != m0):
        why.append("torch was reinstalled by a no-op update")
    if noop.get("torch_install_lines"):
        why.append("update log shows torch being installed: " + noop["torch_install_lines"][0][:120])
    for label in ("noop", "offline"):
        t = o.get(f"torch_after_{label}") or {}
        if t.get("present") and not _torch_is_rocm(t):
            why.append(f"after the {label} update torch is no longer a ROCm build ({t.get('version')})")
    return why


def base_shows_defect(base: dict) -> tuple[bool, str]:
    why = _work_evidence(base)
    if why:
        return True, "; ".join(why)
    return False, "the base's no-op updates were already free and idempotent on this host"


def head_is_fixed(head: dict) -> tuple[bool, str]:
    why = _work_evidence(head)
    off = _s(head, "offline")
    if off.get("idempotent") is False:
        why.append("offline no-op changed the install (strict): " + "; ".join(map(str, off.get("idempotency_reasons") or []))[:300])
    if why:
        return False, "; ".join(why)
    return True, "no-op updates fetch nothing, redo nothing, succeed offline and leave torch alone"


def _fmt_mb(n) -> str:
    try:
        return f"{float(n) / 1e6:.1f}"
    except (TypeError, ValueError):
        return "-"


def table(obs: dict) -> str:
    rows = ["| state | install s | settle s | noop s | noop payload MB | noop steps | noop idempotent (relaxed) "
            "| offline exit | offline bytes | torch after | AMD escape |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if not o:
            continue
        inst, settle, noop, off = _s(o, "install"), _s(o, "settle"), _s(o, "noop"), _s(o, "offline")
        steps = ", ".join(noop.get("update_steps_ran") or []) or "none"
        t = o.get("torch_after_offline") or o.get("torch_after_noop") or o.get("torch_after_install") or {}
        esc = o.get("amd_escape_after_noop") or {}
        esc_txt = "n/a" if not esc.get("applicable") else (
            "would force pass" if esc.get("would_force_dependency_pass") else f"keeps fast path (rc {esc.get('rc')})")
        rows.append(
            f"| {name} | {inst.get('seconds', '-')} | {settle.get('seconds', '-')} | {noop.get('seconds', '-')} "
            f"| {_fmt_mb(noop.get('payload_bytes_down'))} | {steps} | {noop.get('idempotent_relaxed')} "
            f"| {off.get('exit_code', '-')} | {off.get('total_bytes_down', '-')} | {t.get('version', '-')} | {esc_txt} |")
    notes = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        hosts = _s(o, "noop").get("by_host") or {}
        if hosts:
            top = sorted(hosts.items(), key = lambda kv: -kv[1]["bytes_down"])[:6]
            notes.append(f"{name} noop bytes by host: " + ", ".join(
                f"{h}={_fmt_mb(v['bytes_down'])} MB/{v['connections']} conn" for h, v in top))
        w = o.get("wheel") or {}
        if w:
            notes.append(f"{name} wheel: unsloth=={w.get('version')} (zoo {w.get('zoo')}) on {o.get('machine')}")
    return "\n".join(rows) + ("\n\n" + "\n".join(notes) if notes else "")
