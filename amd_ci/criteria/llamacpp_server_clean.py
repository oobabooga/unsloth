#!/usr/bin/env python3
"""Criteria: does the head build clear a corruption the base build shows?

Pairs with probes/llamacpp_server_probe.py. A defect is a corruption signature
in a non-control cell, a failed request, a near-empty slot, a fault in the
server log, or two slots sharing output. The `single` cell is the control and
must be clean in every state. Every criterion returns `(bool, reason)`.
"""

from __future__ import annotations

TITLE = "llama-server output on gfx1151: base vs head prebuilt"
MODE = "differential"
# What such reports touch; the verdict lists those this host could not answer.
NEEDS = ["gpu", "rocm", "vulkan", "integrated_gpu", "discrete_gpu", "windows",
         "windows_rocm_wddm", "nvidia", "multi_gpu", "multi_gpu_amd"]

CONTROL_CELL = "single"
BAD_LOG = ("inconsistent sequence positions", "GGML_ASSERT", "failed to decode", "nan", "NaN",
           "Memory access fault", "HSA_STATUS_ERROR")
MIN_TOKENS = 8


def _states(obs: dict) -> dict:
    return {k: v for k, v in obs.items() if not k.startswith("_")}


def _cell(st: dict, name: str) -> dict:
    return ((st.get("cells") or {}).get(name)) or {}


def _defects(st: dict) -> list[str]:
    out: list[str] = []
    if st.get("setup_error"):
        return [f"setup: {st['setup_error']}"]
    if st.get("error"):
        out.append(f"probe error: {st['error']}")
    for name, c in (st.get("cells") or {}).items():
        if name == CONTROL_CELL:
            continue
        for i, p in enumerate(c.get("prompts") or []):
            if p.get("signatures"):
                out.append(f"{name}[{i}] {','.join(p['signatures'])}")
            if not p.get("ok"):
                out.append(f"{name}[{i}] request failed: {p.get('error')}")
            elif (p.get("n_tokens") or 0) < MIN_TOKENS:
                out.append(f"{name}[{i}] produced {p.get('n_tokens')} tokens")
        bad = {k: v for k, v in (c.get("log_sig_counts") or {}).items() if k in BAD_LOG}
        if bad:
            out.append(f"{name} server log: {bad}")
        if c.get("cross_slot_shared_60char"):
            out.append(f"{name} cross-slot shared text between slots "
                       + ", ".join(f"{i}/{j}" for i, j, _ in c["cross_slot_shared_60char"]))
    return out


def _control_clean(st: dict) -> tuple[bool, str]:
    ps = _cell(st, CONTROL_CELL).get("prompts") or []
    if not ps:
        return False, f"{CONTROL_CELL} never ran"
    p = ps[0]
    ok = bool(p.get("ok")) and (p.get("n_tokens") or 0) >= MIN_TOKENS and not p.get("signatures")
    return ok, f"n_tokens={p.get('n_tokens')} sigs={p.get('signatures')} err={p.get('error')}"


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    states = _states(obs)
    out = []
    launched = {n: bool(v.get("version")) and not v.get("setup_error") for n, v in states.items()}
    out.append(("every state ran its binary", all(launched.values()),
                "; ".join(f"{n}: {v.get('setup_error') or v.get('error') or (v.get('version') or ['?'])[0]}"
                          for n, v in states.items())))
    builds = {n: (v.get("version") or ["?"])[0] for n, v in states.items()}
    out.append(("base and head are different builds", len(set(builds.values())) == len(builds),
                "; ".join(f"{n}={b}" for n, b in builds.items())))
    ctl = {n: _control_clean(v) for n, v in states.items()}
    out.append((f"control cell ({CONTROL_CELL}) clean in every state", all(ok for ok, _ in ctl.values()),
                "; ".join(f"{n}: {d}" for n, (_, d) in ctl.items())))
    sent = {n: (v.get("sentinel") or {}) for n, v in states.items()}
    pre_ok = all((s.get("pre") or {}).get("clean", True) for s in sent.values())
    out.append(("known-good sentinel clean before each state (GPU not poisoned)", pre_ok,
                "; ".join(f"{n}: pre={(s.get('pre') or {}).get('clean', 'n/a')} "
                          f"post={(s.get('post') or {}).get('clean', 'n/a')}" for n, s in sent.items())))
    return out


def _env_summary(v: dict) -> str:
    env = v.get("env") or []
    unset = v.get("unset_env") or []
    parts = list(env) + [f"unset {k}" for k in unset]
    return ", ".join(parts) if parts else "inherited"


def table(obs: dict) -> str:
    states = _states(obs)
    cells: list[str] = []
    for v in states.values():
        for c in (v.get("cells") or {}):
            if c not in cells:
                cells.append(c)
    rows = ["| state | build | env | " + " | ".join(cells) + " | defects | sentinel pre/post |",
            "|---|---|---|" + "---|" * len(cells) + "---|---|"]
    for n, v in states.items():
        def tokens(name):
            ps = _cell(v, name).get("prompts") or []
            return "/".join(str(p.get("n_tokens")) for p in ps) if ps else "-"
        s = v.get("sentinel") or {}
        rows.append(f"| {n} | {(v.get('version') or ['?'])[0][:40]} | {_env_summary(v)} | "
                    + " | ".join(tokens(c) for c in cells)
                    + f" | {'; '.join(_defects(v)) or 'none'} | "
                    f"{(s.get('pre') or {}).get('clean', '-')}/{(s.get('post') or {}).get('clean', '-')} |")
    return "\n".join(rows)


def base_shows_defect(base: dict) -> tuple[bool, str]:
    d = _defects(base)
    return bool(d), "; ".join(d) or "no corruption signature, no decode error, every slot produced output"


def head_is_fixed(head: dict) -> tuple[bool, str]:
    d = _defects(head)
    return not d, "; ".join(d) or "clean on every cell"
