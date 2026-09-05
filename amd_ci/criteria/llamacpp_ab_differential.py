#!/usr/bin/env python3
"""Criteria: does the newer llama.cpp build clear output corruption the older one shows?

Judges only; never observes. Pairs with probes/llamacpp_ab_probe.py, where the
BINARY is the variable and everything else is held: same model file shard for
shard, same flags, same environment, greedy, fixed seed, no prompt cache.

MODE is "differential" on purpose, and the harness rule applies in full: if the
base build does not exhibit the defect, the verdict is VOID and the job fails.
A clean head next to a clean base says only that this host, this model and these
requests never provoked the race, and that is not evidence for the fix.

"Defect" is defined by arithmetic, not by reading prose:

  * a planted code that does not come back (the prompt was corrupted on the way
    in, which is the mechanism ggml-org#25863 describes), or
  * a degenerate tail: a run of one character or one token past a threshold, or
    a quarter of the output being `/`, the field signature from lemonade#3160,
    ggml-org#26209 and ggml-org#27797.

Both counters are recorded by the probe and thresholded here, so the line can be
argued about in one place.

The gates invert two of the uma module's: the builds must DIFFER (that is the
variable), and the environments must be the SAME with the unified-memory name
absent from both (so the other known corruption cannot leak in).
"""

from __future__ import annotations

import json
import os

TITLE = "llama.cpp build A/B on gfx1151 (pass --title per cell)"
MODE = "differential"

NEEDS = [
    "gpu", "rocm", "vulkan", "integrated_gpu",
    "discrete_gpu", "multi_gpu", "multi_gpu_amd",
    "windows", "windows_rocm_wddm", "nvidia", "amdvlk",
]

EXPECT_ARCH = "gfx1151"
EXPECT_DEVICE = os.environ.get("AMD_CI_EXPECT_DEVICE", "ROCm")
# A thinking-mode cell may spend its whole budget thinking and never answer, so
# the missing code is not evidence there. The workflow says which cells demand it.
NONCE_REQUIRED = os.environ.get("AMD_CI_AB_NONCE_REQUIRED", "1") == "1"

MAX_CHAR_RUN = 16
MAX_TOKEN_RUN = 16
SLASH_FRAC = 0.25

UMA_VAR = "GGML_CUDA_ENABLE_UNIFIED_MEMORY"


def _states(obs: dict) -> dict:
    return {k: v for k, v in obs.items() if not k.startswith("_")}


def _ok(state: dict) -> list[dict]:
    return [c for c in (state.get("completions") or [])
            if c.get("status") == 200 and c.get("tokens_generated")]


def _degenerate(c: dict) -> str | None:
    if NONCE_REQUIRED and "nonce_ok" in c and not c.get("nonce_ok"):
        return f"code {c.get('code')} not echoed"
    if (c.get("max_char_run") or 0) >= MAX_CHAR_RUN:
        return f"run of {c.get('max_char_run')} identical characters"
    if (c.get("max_token_run") or 0) >= MAX_TOKEN_RUN:
        return f"run of {c.get('max_token_run')} identical tokens"
    if (c.get("slash_frac") or 0.0) >= SLASH_FRAC:
        return f"{round(100 * c['slash_frac'])}% of the output is `/`"
    return None


def _degenerates(state: dict) -> list[tuple[dict, str]]:
    out = []
    for c in _ok(state):
        why = _degenerate(c)
        if why:
            out.append((c, why))
    return out


def _env_signature(state: dict) -> str:
    return json.dumps({"set": dict(sorted((state.get("env_overrides") or {}).items())),
                       "unset": sorted(state.get("env_unset") or [])}, sort_keys = True)


def _placement(state: dict) -> str:
    return json.dumps(state.get("layers_by_device") or {}, sort_keys = True)


def _on_expected_device(state: dict) -> bool:
    return any(EXPECT_DEVICE.lower() in d.lower() for d in (state.get("layers_by_device") or {}))


def _gpu_majority(state: dict) -> bool:
    by_dev = state.get("layers_by_device") or {}
    if not by_dev:
        return False
    acc = sum(n for d, n in by_dev.items() if not d.upper().startswith("CPU"))
    return acc > sum(by_dev.values()) / 2


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    states = _states(obs)
    out: list[tuple[str, bool, str]] = []

    missing = [n for n, v in states.items() if v.get("setup_error") or not v.get("cmd")]
    out.append(("every state launched a binary", not missing,
                "; ".join(f"{n}: {states[n].get('setup_error', 'never ran')}" for n in missing)
                or ", ".join(sorted(states))))

    builds = {n: v.get("build") for n, v in states.items()}
    out.append(("the states ran DIFFERENT llama.cpp builds (that is the variable)",
                all(builds.values()) and len(set(builds.values())) == len(builds),
                ", ".join(f"{n}=b{b} ({(states[n].get('bin_dir') or '?').rsplit('/', 1)[-1]})"
                          for n, b in builds.items())))

    names = {n: v.get("gguf_name") for n, v in states.items()}
    shards = {n: json.dumps(v.get("model_files") or {}, sort_keys = True) for n, v in states.items()}
    out.append(("the states used the same model file, shard for shard",
                len(set(names.values())) == 1 and len(set(shards.values())) == 1
                and all(v.get("model_files") for v in states.values()),
                "; ".join(f"{n}={names[n]}, {sum((states[n].get('model_files') or {}).values())} bytes"
                          for n in states)))

    dirs = {n: v.get("gguf") for n, v in states.items()}
    out.append(("the states are separate directories", len(set(dirs.values())) == len(dirs),
                "; ".join(f"{n}={d}" for n, d in dirs.items())))

    sigs = {n: _env_signature(v) for n, v in states.items()}
    uma_absent = {n: not v.get("uma_env_present") for n, v in states.items()}
    out.append((f"the environments are identical and {UMA_VAR} is absent from both",
                len(set(sigs.values())) == 1 and all(uma_absent.values()),
                "; ".join(f"{n}: UMA={'ABSENT' if uma_absent[n] else states[n].get('uma_env_value')}"
                          for n in states)))

    dev_ok = {n: _on_expected_device(v) for n, v in states.items()}
    out.append((f"every state put layers on a {EXPECT_DEVICE} device", all(dev_ok.values()),
                "; ".join(f"{n}={states[n].get('layers_by_device')}" for n in states)))

    maj = {n: _gpu_majority(v) for n, v in states.items()}
    out.append(("every state put a clear majority of layers on the GPU", all(maj.values()),
                "; ".join(f"{n}={m}" for n, m in maj.items())))

    places = {n: _placement(v) for n, v in states.items()}
    out.append(("the states placed layers identically", len(set(places.values())) == 1,
                "; ".join(f"{n}={p}" for n, p in places.items())))

    archs = {n: (v.get("gfx") or "?") for n, v in states.items()}
    out.append((f"host is {EXPECT_ARCH}", all(EXPECT_ARCH in a for a in archs.values()),
                ", ".join(f"{n}={a}" for n, a in archs.items())))

    oom = {n: "out_of_memory" in (v.get("markers") or []) for n, v in states.items()}
    out.append(("no state ran out of memory", not any(oom.values()),
                ", ".join(f"{n}={o}" for n, o in oom.items())))

    specs = {n: json.dumps(v.get("request_spec") or {}, sort_keys = True) for n, v in states.items()}
    out.append(("every state was asked the same requests", len(set(specs.values())) == 1,
                "; ".join(f"{n}: mode={(states[n].get('request_spec') or {}).get('mode')}, "
                          f"concurrent={(states[n].get('request_spec') or {}).get('concurrent')}, "
                          f"repeats={(states[n].get('request_spec') or {}).get('repeats')}"
                          for n in states)))

    # A build that never served, or dropped requests, is a different finding from
    # corrupted output and must not be read as either side of this comparison.
    served = {n: (v.get("server_ready") is True
                  and v.get("completions_total")
                  and v.get("completions_ok") == v.get("completions_total"))
              for n, v in states.items()}
    out.append(("every state served every request with tokens", all(served.values()),
                "; ".join(f"{n}: ready={states[n].get('server_ready')}, "
                          f"{states[n].get('completions_ok')}/{states[n].get('completions_total')}"
                          for n in states)))
    return out


def _worst(state: dict) -> str:
    degs = _degenerates(state)
    if degs:
        c, why = degs[0]
        text = ((c.get("content") or "") + (c.get("reasoning") or "")).replace("\n", " ")[:60]
        return f"{why}: `{text}`"
    ok = _ok(state)
    if not ok:
        return "no completion"
    text = ((ok[0].get("content") or "") + (ok[0].get("reasoning") or "")).replace("\n", " ")[:60]
    return f"`{text}`"


def table(obs: dict) -> str:
    rows = ["| state | build | served | degenerate | code echoed | max `/` frac | "
            "longest run | sequential identical | sample |",
            "|---|---|---|---|---|---|---|---|---|"]
    for name, v in _states(obs).items():
        rows.append(
            f"| {name} | b{v.get('build')} | {v.get('completions_ok')}/{v.get('completions_total')} | "
            f"{len(_degenerates(v))} | {v.get('nonce_ok_count')}/{v.get('nonce_total')} | "
            f"{v.get('max_slash_frac')} | {v.get('max_char_run')} chars / "
            f"{v.get('max_token_run')} tokens | {v.get('sequential_identical')} | {_worst(v)} |")
    rows.append("")
    rows.append(f"Degenerate means: a planted code not echoed"
                f"{' (required in this cell)' if NONCE_REQUIRED else ' (NOT required in this cell)'}, "
                f"a run of {MAX_CHAR_RUN}+ identical characters or {MAX_TOKEN_RUN}+ identical "
                f"tokens, or {round(100 * SLASH_FRAC)}%+ of the output being `/`. Thresholds live "
                f"in criteria/llamacpp_ab_differential.py.")
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    return bool(_degenerates(base))


def head_is_fixed(head: dict) -> bool:
    if _degenerates(head):
        return False
    ok = _ok(head)
    if not ok:
        return False
    if NONCE_REQUIRED and head.get("nonce_total") and \
            head.get("nonce_ok_count") != head.get("nonce_total"):
        return False
    seq = [c for c in ok if c.get("phase") == "sequential"]
    if len(seq) > 1 and not head.get("sequential_identical"):
        return False
    raw = head.get("raw_repeats_identical") or {}
    if raw and not all(raw.values()):
        return False
    return True
