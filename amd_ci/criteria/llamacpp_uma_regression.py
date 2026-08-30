#!/usr/bin/env python3
"""Criteria: does setting GGML_CUDA_ENABLE_UNIFIED_MEMORY change what the model emits?

Judges only; it never observes. Pairs with probes/llamacpp_uma_probe.py.

The reported defect (HF unsloth/Qwen3.8-Flash-Next-GGUF discussion #30, and
independently ggml-org/llama.cpp#26148 on the same gfx1151 with a different
model) is silent output CORRUPTION, not a crash. So "worse" here cannot mean
"produced no tokens". It means the two states disagreed on the token IDs while
every input that could legitimately change them was held identical: same model
file, same binary, same flags, greedy decoding, fixed seed, no prompt cache.

MODE is "regression" for the same reason the mtp module gives: differential mode
VOIDs on "the base state did not exhibit the defect", and base here is the
CONTROL, which is supposed not to. Regression mode asks the question posed.

The gates are where this design earns its result. An env-var comparison has more
ways of looking decisive while measuring nothing than a model-file comparison
does, and two of them are inversions of the mtp module's gates:

  * The mtp module gates on "the states used DIFFERENT model files", because
    there the model was the variable. Here the model must be the SAME file, and
    two different models would be the spoof. Inherited unchanged, that gate
    would have waved through exactly the mistake it was written to catch.
  * A Vulkan cell would ignore the variable entirely - ggml only reads it in the
    CUDA/HIP allocator - and would return a confident, meaningless verdict. So
    the expected backend is asserted, and is a parameter of the cell rather than
    an assumption.

And the one that matters most: the control has to agree with ITSELF across
repeats before any claim that the treatment differs from it means anything.
Without that, ordinary run-to-run nondeterminism reads as the defect.
"""

from __future__ import annotations

import json
import os

TITLE = "GGML_CUDA_ENABLE_UNIFIED_MEMORY on gfx1151 (pass --title per cell)"
MODE = "regression"

# What the QUESTION touches, not what the host has.
#
# discrete_gpu is the load-bearing entry. This variable exists to spill an
# allocation from a separate VRAM pool into system RAM; gfx1151 has no separate
# pool, so the hardware where the setting is meant to help is precisely the
# hardware this run cannot speak for.
NEEDS = [
    "gpu", "rocm", "vulkan", "integrated_gpu",
    "discrete_gpu", "multi_gpu", "multi_gpu_amd",
    "windows", "windows_rocm_wddm", "nvidia", "amdvlk",
]

EXPECT_ARCH = "gfx1151"

# Which backend the cell requires the weights to land on. A parameter, because
# the Vulkan negative-control cell needs the opposite answer from the ROCm
# cells, and a criteria module that silently assumed ROCm would pass that cell
# for the wrong reason. Rendered into the table so the report states it.
EXPECT_DEVICE = os.environ.get("AMD_CI_EXPECT_DEVICE", "ROCm")

# Short label for the variable in tables and gate evidence.
UMA_SHORT = "UMA"

_CTX: dict = {}


def _states(obs: dict) -> dict:
    return {k: v for k, v in obs.items() if not k.startswith("_")}


def _vectors(state: dict) -> list:
    return [v for v in (state.get("token_vectors") or []) if v is not None]


def _env_signature(state: dict) -> str:
    """Everything about the child environment this cell varies, as one string."""
    sets = state.get("env_overrides") or {}
    unsets = state.get("env_unset") or []
    return json.dumps({"set": dict(sorted(sets.items())), "unset": sorted(unsets)},
                      sort_keys = True)


def _placement(state: dict) -> str:
    return json.dumps(state.get("layers_by_device") or {}, sort_keys = True)


def _on_expected_device(state: dict) -> bool:
    by_dev = state.get("layers_by_device") or {}
    return any(EXPECT_DEVICE.lower() in d.lower() for d in by_dev)


def _gpu_majority(state: dict) -> bool:
    by_dev = state.get("layers_by_device") or {}
    if not by_dev:
        return False
    acc = sum(n for d, n in by_dev.items() if not d.upper().startswith("CPU"))
    return acc > sum(by_dev.values()) / 2


def _first_divergence(a: list, b: list) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    states = _states(obs)
    out: list[tuple[str, bool, str]] = []

    missing = [n for n, v in states.items() if v.get("setup_error") or not v.get("cmd")]
    out.append(("every state launched the binary", not missing,
                "; ".join(f"{n}: {states[n].get('setup_error', 'never ran')}" for n in missing)
                or ", ".join(sorted(states))))

    builds = {n: v.get("build") for n, v in states.items()}
    out.append(("all states used the same llama.cpp build",
                len(set(builds.values())) == 1 and all(builds.values()),
                ", ".join(f"{n}={b}" for n, b in builds.items())))

    # The inversion of the mtp module's gate. Here the model is the CONSTANT, and
    # every shard is compared: this model loads from a ~10 MB shard-1 metadata
    # file, so checking only the named file would let two states differ by the
    # entire 87 GiB of weights and still look identical.
    names = {n: v.get("gguf_name") for n, v in states.items()}
    shards = {n: json.dumps(v.get("model_files") or {}, sort_keys = True)
              for n, v in states.items()}
    same_model = (len(set(names.values())) == 1 and len(set(shards.values())) == 1
                  and all(v.get("model_files") for v in states.values()))
    out.append(("the states used the same model file, shard for shard", same_model,
                "; ".join(f"{n}={names[n]} over {len(states[n].get('model_files') or {})} "
                          f"shard(s), {sum((states[n].get('model_files') or {}).values())} bytes"
                          for n in states)))

    # ...and the state directories must still be distinct, or lib/write_states.py
    # would have refused them and nothing here would be comparing two launches.
    dirs = {n: v.get("gguf") for n, v in states.items()}
    out.append(("the states are separate directories", len(set(dirs.values())) == len(dirs),
                "; ".join(f"{n}={d}" for n, d in dirs.items())))

    # Non-vacuity: the environments really differed, read back from the child
    # environment the probe actually built rather than from what was requested.
    sigs = {n: _env_signature(v) for n, v in states.items()}
    uma = {n: (v.get("uma_env_value") if v.get("uma_env_present") else "ABSENT")
           for n, v in states.items()}
    out.append(("the states really differ in the environment under test",
                len(set(sigs.values())) == len(sigs),
                "; ".join(f"{n}: {UMA_SHORT}={uma[n]}"
                          f"{', HIP_LAUNCH_BLOCKING=' + str(v.get('launch_blocking')) if v.get('launch_blocking') else ''}"
                          for n, v in states.items())))

    dev_ok = {n: _on_expected_device(v) for n, v in states.items()}
    out.append((f"every state put layers on a {EXPECT_DEVICE} device", all(dev_ok.values()),
                "; ".join(f"{n}={states[n].get('layers_by_device')}" for n in states)))

    maj = {n: _gpu_majority(v) for n, v in states.items()}
    out.append(("every state put a clear majority of layers on the GPU", all(maj.values()),
                "; ".join(f"{n}={m}" for n, m in maj.items())))

    # A placement difference would confound the comparison: different layers on
    # different devices can legitimately change arithmetic, and that is not the
    # variable under test.
    places = {n: _placement(v) for n, v in states.items()}
    out.append(("the states placed layers identically", len(set(places.values())) == 1,
                "; ".join(f"{n}={p}" for n, p in places.items())))

    archs = {n: (v.get("gfx") or "?") for n, v in states.items()}
    out.append((f"host is {EXPECT_ARCH}", all(EXPECT_ARCH in a for a in archs.values()),
                ", ".join(f"{n}={a}" for n, a in archs.items())))

    oom = {n: "out_of_memory" in (v.get("markers") or []) for n, v in states.items()}
    out.append(("no state ran out of memory", not any(oom.values()),
                ", ".join(f"{n}={o}" for n, o in oom.items())))

    prompts = {n: (v.get("prompt_text"), v.get("n_predict")) for n, v in states.items()}
    out.append(("every state asked the same question", len(set(prompts.values())) == 1,
                "; ".join(f"{n}={p!r}" for n, p in prompts.items())))

    # The gate this whole design rests on.
    base = states.get("base") or {}
    b_vecs = _vectors(base)
    out.append(("the control agreed with itself across repeats",
                bool(b_vecs) and base.get("repeats_identical") is True,
                f"base returned {len(b_vecs)} completions, "
                f"{base.get('distinct_vectors')} distinct token vector(s)"))

    return out


def table(obs: dict) -> str:
    rows = [f"| state | {UMA_SHORT} | launch_blocking | ok | identical repeats | tok/s | "
            f"first 60 chars |",
            "|---|---|---|---|---|---|---|"]
    for name, v in _states(obs).items():
        uma = v.get("uma_env_value") if v.get("uma_env_present") else "unset"
        comps = v.get("completions") or []
        rate = next((c.get("predicted_per_second") for c in comps
                     if c.get("predicted_per_second")), None)
        head = (v.get("contents") or [""])
        text = (head[0] or "").replace("\n", " ")[:60] if head else ""
        rows.append(
            f"| {name} | {uma} | {v.get('launch_blocking') or '-'} | "
            f"{v.get('completions_ok')}/{v.get('repeats_requested')} | "
            f"{v.get('repeats_identical')} | "
            f"{round(rate, 1) if isinstance(rate, (int, float)) else '-'} | `{text}` |")
    rows.append("")
    rows.append(f"Weights were required on a **{EXPECT_DEVICE}** device: ggml reads "
                f"`GGML_CUDA_ENABLE_UNIFIED_MEMORY` only in the CUDA/HIP allocator "
                f"(`ggml/src/ggml-cuda/ggml-cuda.cu:139-166`), so a Vulkan run would ignore "
                f"it and any verdict from one would be meaningless.")
    rows.append("")
    rows.append("The comparison is over token IDs, not prose. `UMA=unset` means the name was "
                "absent from the child environment, which is the only true control: ggml tests "
                "`getenv(...) != nullptr`, so `=0` enables it too.")
    return "\n".join(rows)


def _describe(state: dict) -> str:
    markers = state.get("markers") or []
    if markers:
        return "markers " + ", ".join(markers)
    sig = state.get("signal")
    if sig:
        return f"killed by signal {sig}" + (" (SIGABRT)" if sig == 6 else "")
    if not state.get("server_ready"):
        return "server never became ready"
    if not state.get("completions_ok"):
        return "server ready but no completion returned tokens"
    return f"exited rc={state.get('rc')} with no diagnostic in the log"


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    b_vecs, h_vecs = _vectors(base), _vectors(head)

    if not b_vecs:
        return False, (f"the control produced no usable completion - {_describe(base)}. "
                       f"Nothing the head did can be called worse than that")
    if not h_vecs:
        return True, (f"the control produced {len(b_vecs)} completions and the head produced "
                      f"none - {_describe(head)}. That is a harder failure than the reported "
                      f"corruption, and is still a difference the variable caused")

    b0, h0 = b_vecs[0], h_vecs[0]
    idx = _first_divergence(b0, h0)
    if idx is None:
        return False, (f"both states emitted the identical {len(b0)}-token sequence under "
                       f"greedy decoding; setting the variable changed nothing on this host "
                       f"in this configuration")

    b_txt = (base.get("contents") or [""])[0] or ""
    h_txt = (head.get("contents") or [""])[0] or ""
    stable = "" if head.get("repeats_identical") else \
        " The head was NOT self-consistent across its repeats either, so the corruption is " \
        "not even deterministic here."
    return True, (f"the states diverged at token index {idx} of {min(len(b0), len(h0))} "
                  f"compared, with the model file, binary, flags, seed and prompt held "
                  f"identical and prompt caching off. Control: {b_txt[:120]!r}. "
                  f"Treatment: {h_txt[:120]!r}.{stable} "
                  f"Read the verdict word as 'the states disagreed': regression mode has no "
                  f"oracle for which output is correct, and in a cell where the head adds a "
                  f"suspected WORKAROUND rather than the suspected cause, a divergence means "
                  f"the workaround changed something. The two texts above say which way round "
                  f"it went")
