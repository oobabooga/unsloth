"""Criteria: on real Windows AMD hardware the installers' detection spawns amd-smi on base (HIP SDK
present or opted in) and never on head, while what they detect with the real hipinfo is unchanged."""

from __future__ import annotations

TITLE = "Windows installer detection: amd-smi spawns and detected arch, base vs head"
MODE = "differential"
NEEDS = ["gpu", "windows"]
_SAME = ("stack.has_rocm_gpu", "stack.windows_gfx_arch", "stack.amd_gfx_codes", "prebuilt.detect_host")
_CTX: dict = {}


def _calls(state: dict) -> int:
    return sum(len(s.get("amd_smi_calls") or []) for s in (state.get("scenarios") or {}).values())


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    states = {n: v for n, v in obs.items() if not n.startswith("_")}
    out = [("ran on Windows", all(v.get("os") == "win32" for v in states.values()),
            ", ".join(f"{n}={v.get('os')}" for n, v in states.items())),
           ("amd-smi shim resolves", all(v.get("shim_resolves") for v in states.values()), "")]
    loaded = all(not any(str(r).startswith("ERR") for k, r in (s.get("results") or {}).items() if k.endswith("_load"))
                 and s.get("results") for v in states.values() for s in (v.get("scenarios") or {}).values())
    out.append(("both installers imported in every scenario", loaded, ""))
    base, head = states.get("base", {}), states.get("head", {})
    b = ((base.get("scenarios") or {}).get("default") or {}).get("results") or {}
    h = ((head.get("scenarios") or {}).get("default") or {}).get("results") or {}
    diffs = [k for k in _SAME if b.get(k) != h.get(k)]
    _CTX["diffs"] = diffs
    out.append(("real-hipinfo detection unchanged base vs head", not diffs, ", ".join(diffs) or "identical"))
    out.append(("an AMD GPU was actually detected", bool(h.get("stack.windows_gfx_arch")), str(h.get("stack.windows_gfx_arch"))))
    return out


def table(obs: dict) -> str:
    rows = ["| state | scenario | amd-smi spawns | arch | has_rocm | rocm version | detect_host |", "|---|---|---|---|---|---|---|"]
    for n, v in obs.items():
        if n.startswith("_"):
            continue
        for sn, s in (v.get("scenarios") or {}).items():
            r = s.get("results") or {}
            rows.append(f"| {n} | {sn} | {len(s.get('amd_smi_calls') or [])} {s.get('amd_smi_calls')} | {r.get('stack.windows_gfx_arch')} | "
                        f"{r.get('stack.has_rocm_gpu')} | {r.get('stack.rocm_version')} | {r.get('prebuilt.detect_host')} |")
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    return _calls(base) > 0


def head_is_fixed(head: dict) -> bool:
    return _calls(head) == 0 and not _CTX.get("diffs")
