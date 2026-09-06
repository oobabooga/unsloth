#!/usr/bin/env python3
"""Criteria: can this HIP runtime load a model whose device buffer is larger than
the allocation cap?

Pairs with probes/llamacpp_server_probe.py run with `--cells single` on a model
chosen so its single largest device buffer straddles the cap the allocation
probe measured. The states are two HIP runtimes, not two llama.cpp builds.

A model load is the end-to-end form of the same question the hipMalloc bisection
asks directly, and it is the form users report: llama.cpp asks for one buffer per
device, so a runtime that refuses an allocation above a constant refuses the
model outright rather than placing fewer layers. `--fit off` is what keeps that
true; with fitting on, a refusal turns into a quieter, smaller load and the
measurement disappears.
"""

from __future__ import annotations

TITLE = "loading a model whose device buffer exceeds the allocation cap"
MODE = "differential"
NEEDS = ["gpu", "rocm", "integrated_gpu", "windows", "windows_rocm_wddm",
         "discrete_gpu", "nvidia", "multi_gpu", "multi_gpu_amd"]

MIN_TOKENS = 8
ALLOC_WORDS = ("out of memory", "hipmalloc", "failed to allocate", "unable to allocate",
               "cudamalloc", "buffer")


def _single(state: dict) -> dict:
    return ((state.get("cells") or {}).get("single") or {})


def _prompt(state: dict) -> dict:
    ps = _single(state).get("prompts") or []
    return ps[0] if ps else {}


def _loaded(state: dict) -> bool:
    return bool(_prompt(state))


def _answered(state: dict) -> tuple[bool, str]:
    p = _prompt(state)
    if not p:
        return False, "the server never answered"
    if not p.get("ok"):
        return False, f"the request failed: {p.get('error')}"
    if (p.get("n_tokens") or 0) < MIN_TOKENS:
        return False, f"only {p.get('n_tokens')} tokens came back"
    if p.get("signatures"):
        return False, f"the answer carries {','.join(p['signatures'])}"
    return True, f"{p.get('n_tokens')} clean tokens"


def _why_failed(state: dict) -> str:
    return str(state.get("error") or state.get("setup_error") or "")


def gates(obs: dict) -> list:
    out = []
    for name in ("base", "head"):
        st = obs.get(name) or {}
        out.append((f"{name}: the binaries ran at all", bool(st.get("version")),
                    _why_failed(st) or "no --version line"))
        # A poisoned GPU makes every later load fail for reasons that have
        # nothing to do with the cap, so the sentinel decides whether this
        # state's failure is even interpretable.
        pre = ((st.get("sentinel") or {}).get("pre") or {})
        out.append((f"{name}: the sentinel was clean before the cell", bool(pre.get("clean")),
                    pre.get("error") or pre.get("text", "")[:80]))
    return out


def base_shows_defect(base: dict) -> tuple[bool, str]:
    ok, why = _answered(base)
    if ok:
        return False, (f"the base loaded the model and answered ({why}), so nothing is being "
                       f"refused and this run has no defect to fix")
    err = _why_failed(base)
    if not err and not _loaded(base):
        return True, "the base did not load the model, and left no error to attribute it to"
    hint = "an allocation failure" if any(w in err.lower() for w in ALLOC_WORDS) else \
        "a failure that does not name an allocation; the server log is the evidence"
    return True, f"the base did not serve the model: {hint} ({err[:200]})"


def head_is_fixed(head: dict) -> tuple[bool, str]:
    ok, why = _answered(head)
    if ok:
        return True, f"loaded the model and answered: {why}"
    return False, f"did not serve the model: {why}; {_why_failed(head)[:200]}"


def table(obs: dict) -> str:
    rows = ["| state | build | loaded | tokens | error |", "|---|---|---|---|---|"]
    for name, st in obs.items():
        if name.startswith("_") or not isinstance(st, dict):
            continue
        p = _prompt(st)
        rows.append(f"| {name} | {' '.join(st.get('version') or [])[:60]} | "
                    f"{'yes' if _loaded(st) else 'NO'} | {p.get('n_tokens')} | "
                    f"{_why_failed(st)[:120] or '-'} |")
    return "\n".join(rows)
