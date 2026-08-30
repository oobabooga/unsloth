#!/usr/bin/env python3
"""Criteria: is the head GGUF numerically worse than the base one on this backend?

Judges only; it never observes. Pairs with probes/llamacpp_ppl_probe.py.

MODE is "regression" for the reason the other llama.cpp modules give: differential
mode VOIDs on "the base state did not exhibit the defect", and base here is the
revision reported to WORK, which is supposed not to.

Two quantisations of the same model at nearly the same file size are directly
comparable on perplexity over a fixed corpus, which is what makes this cell
possible at all. They are not identical files, so a small difference is expected
and is not the finding: the threshold is set where a recipe change stops being a
recipe change and becomes a broken kernel. ggml-org/llama.cpp#27506 measured
ratios of 100x to 900x on this hardware class when a backend was actually wrong,
and the honest quantisation-to-quantisation spread between neighbouring recipes
is a few percent. A 25% threshold sits far from both.

The gate that matters most is that the BASE produced a finite, plausible
perplexity. A head compared against a base that returned nan, or no number at
all, is not a comparison.
"""

from __future__ import annotations

import json
import math
import os

TITLE = "GGUF revision perplexity on gfx1151 (pass --title per cell)"
MODE = "regression"

# What the QUESTION touches, not what the host has. The reports this cell exists
# to explain are Windows, discrete RDNA3/RDNA4 and in several cases two cards;
# none of that is reachable here, and the report must say so.
NEEDS = [
    "gpu", "rocm", "vulkan", "integrated_gpu",
    "discrete_gpu", "multi_gpu", "multi_gpu_amd",
    "windows", "windows_rocm_wddm", "amdvlk", "nvidia",
]

EXPECT_ARCH = "gfx1151"
EXPECT_DEVICE = os.environ.get("AMD_CI_EXPECT_DEVICE", "ROCm")

# Ratio above which the head is called worse. See the module docstring.
PPL_RATIO_LIMIT = float(os.environ.get("AMD_CI_PPL_RATIO_LIMIT", "1.25"))
# A base perplexity outside this range means the cell measured something other
# than the model: a wrong corpus, a truncated run, or a backend already broken.
PPL_SANE = (1.0, 100.0)


def _states(obs: dict) -> dict:
    return {k: v for k, v in obs.items() if not k.startswith("_")}


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def _on_expected_device(state: dict) -> bool:
    return any(EXPECT_DEVICE.lower() in d.lower()
               for d in (state.get("layers_by_device") or {}))


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
    out.append(("every state ran llama-perplexity", not missing,
                "; ".join(f"{n}: {states[n].get('setup_error', 'never ran')}" for n in missing)
                or ", ".join(sorted(states))))

    builds = {n: v.get("build") for n, v in states.items()}
    out.append(("all states used the same llama.cpp build",
                len(set(builds.values())) == 1 and all(builds.values()),
                ", ".join(f"{n}={b}" for n, b in builds.items())))

    # The model IS the variable here, so unlike the unified-memory module the
    # gate runs the other way: two states holding the same file would be
    # comparing a revision with itself.
    shards = {n: json.dumps(v.get("model_files") or {}, sort_keys = True)
              for n, v in states.items()}
    out.append(("the states really used different model files",
                len(set(shards.values())) == len(shards)
                and all(v.get("model_files") for v in states.values()),
                "; ".join(f"{n}={sum((states[n].get('model_files') or {}).values())} bytes"
                          for n in states)))

    corpora = {n: (v.get("corpus_bytes"), v.get("ctx_size"), v.get("chunks_requested"))
               for n, v in states.items()}
    out.append(("every state scored the same corpus at the same settings",
                len(set(corpora.values())) == 1,
                "; ".join(f"{n}={c}" for n, c in corpora.items())))

    done = {n: v.get("chunks_done") for n, v in states.items()}
    out.append(("every state completed the requested chunks",
                all(d == states[n].get("chunks_requested") for n, d in done.items()),
                "; ".join(f"{n}={d}/{states[n].get('chunks_requested')}"
                          for n, d in done.items())))

    dev_ok = {n: _on_expected_device(v) for n, v in states.items()}
    out.append((f"every state put layers on a {EXPECT_DEVICE} device", all(dev_ok.values()),
                "; ".join(f"{n}={states[n].get('layers_by_device')}" for n in states)))

    maj = {n: _gpu_majority(v) for n, v in states.items()}
    out.append(("every state put a clear majority of layers on the GPU", all(maj.values()),
                "; ".join(f"{n}={m}" for n, m in maj.items())))

    archs = {n: (v.get("gfx") or "?") for n, v in states.items()}
    out.append((f"host is {EXPECT_ARCH}", all(EXPECT_ARCH in a for a in archs.values()),
                ", ".join(f"{n}={a}" for n, a in archs.items())))

    oom = {n: "out_of_memory" in (v.get("markers") or []) for n, v in states.items()}
    out.append(("no state ran out of memory", not any(oom.values()),
                ", ".join(f"{n}={o}" for n, o in oom.items())))

    timeouts = {n: bool(v.get("timed_out")) for n, v in states.items()}
    out.append(("no state timed out", not any(timeouts.values()),
                ", ".join(f"{n}={t}" for n, t in timeouts.items())))

    # The gate this design rests on.
    base = states.get("base") or {}
    b = base.get("ppl")
    out.append(("the control produced a finite, plausible perplexity",
                _finite(b) and PPL_SANE[0] <= b <= PPL_SANE[1],
                f"base PPL = {b}, sane range {PPL_SANE}"))

    return out


def table(obs: dict) -> str:
    rows = ["| state | model bytes | PPL | +/- | chunks | tok dev | seconds |",
            "|---|---:|---:|---:|---|---|---:|"]
    for name, v in _states(obs).items():
        rows.append(
            f"| {name} | {sum((v.get('model_files') or {}).values())} | "
            f"{v.get('ppl')} | {v.get('ppl_stderr')} | "
            f"{v.get('chunks_done')}/{v.get('chunks_requested')} | "
            f"{v.get('layers_by_device')} | {v.get('seconds')} |")
    rows.append("")
    rows.append(f"Weights were required on a **{EXPECT_DEVICE}** device, asserted from "
                f"`load_tensors:` lines rather than from `system_info`, which names ROCm "
                f"and never Vulkan.")
    rows.append("")
    rows.append(f"`head_is_worse` fires above a ratio of **{PPL_RATIO_LIMIT}x**, or on a "
                f"non-finite head. Two different quantisations of one model differ by a few "
                f"percent honestly; a broken backend on this hardware class has measured "
                f"100x to 900x (ggml-org/llama.cpp#27506).")
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    b, h = base.get("ppl"), head.get("ppl")

    if not _finite(b):
        return False, (f"the control produced no usable perplexity (PPL = {b}, "
                       f"{head.get('chunks_done')} chunks, rc = {base.get('rc')}). Nothing "
                       f"the head did can be called worse than that")
    if not _finite(h):
        return True, (f"the control scored PPL = {b:.4f} and the head produced no finite "
                      f"number (PPL = {h}, rc = {head.get('rc')}, markers "
                      f"{head.get('markers')}). That is a harder failure than a quality "
                      f"regression and is still a difference the head caused")

    ratio = h / b
    series_b = [v for _, v in (base.get("chunk_series") or [])][:3]
    series_h = [v for _, v in (head.get("chunk_series") or [])][:3]
    where = (f" First chunks: control {series_b}, head {series_h} - a head that is already "
             f"wrong at chunk 1 is a broken kernel rather than a degraded recipe."
             if series_b and series_h else "")

    if ratio > PPL_RATIO_LIMIT:
        return True, (f"the head scored PPL = {h:.4f} against the control's {b:.4f}, a ratio "
                      f"of {ratio:.2f}x, over the same corpus, chunk count, context and "
                      f"binary on a {EXPECT_DEVICE} device.{where}")
    return False, (f"the head scored PPL = {h:.4f} against the control's {b:.4f}, a ratio of "
                   f"{ratio:.3f}x, inside the {PPL_RATIO_LIMIT}x band. On this backend, at "
                   f"this corpus and chunk count, the head revision is not numerically "
                   f"broken. That is a statement about arithmetic, not about the crash the "
                   f"field reports: a load-time or first-token crash on another OS and "
                   f"another GPU is not something perplexity here can see")
