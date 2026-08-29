#!/usr/bin/env python3
"""Criteria: with two torch devices on this host, does PR 8791 drop the uncovered one?

Issue 8792 is an iGPU the torch wheel has no kernels for sitting beside a covered
card: `auto_select_gpu_ids` ranks on free VRAM, keeps the iGPU, `device_map="balanced"`
shards onto it, and the load dies with `hipErrorInvalidKernelFile`. Every earlier leg
on this runner could only show non-regression, because one GPU cannot express the bug.

  base_shows_defect -- device 1 presents gfx1036 and the base still selects it.
  head_is_fixed     -- the head excludes it and keeps device 0.

HOW MUCH OF THIS IS REAL is the whole question, and the gate below decides it rather
than the prose. `GGML_CUDA_DEVICES` gives llama.cpp genuine virtual devices, but it is
a ggml-backend feature and creates no torch devices, so it cannot reach this code path.
The probe therefore tries several mechanisms and reports which, if any, gave
`torch.cuda.device_count() >= 2`.

  - If one did, the devices, their HIP contexts, the enumeration and the whole
    selection path are real, and only the ARCH STRING on device 1 is faked. That is a
    materially stronger claim than the PR's own tests, which fake torch entirely.
  - If none did, the probe falls back to mocking the count, and `table()` says so in
    the verdict. The differential is then no stronger than a unit test and must not be
    read as hardware evidence.

Either way this is still ONE physical card. No PCIe hop, no second driver context, no
independent VRAM ceiling. It reproduces the SELECTION defect, not a second GPU.
"""

from __future__ import annotations

TITLE = "GPU selection with an uncovered second device (PR 8791)"
MODE = "differential"

NEEDS = ["rocm", "gpu", "integrated_gpu", "discrete_gpu",
         "multi_gpu", "multi_gpu_amd", "nvidia", "windows", "xpu", "mlx"]

_CTX: dict = {"mechanism": None, "mocked": None}


def _sel(state: dict) -> dict:
    return state.get("selection") or {}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    states = {n: v for n, v in obs.items() if not n.startswith("_")}

    mechs = {n: v.get("real_multidevice_mechanism") for n, v in states.items()}
    mocked = {n: v.get("phase2_used_mocked_count") for n, v in states.items()}
    _CTX["mechanism"] = mechs.get("head")
    _CTX["mocked"] = mocked.get("head")

    # Every state must have measured the same way, or base and head are not comparable:
    # a base that got real devices and a head that got mocked ones is not a differential.
    same = len(set(str(v) for v in mechs.values())) == 1
    out.append(("every state used the same multi-device mechanism", same,
                ", ".join(f"{k}={v}" for k, v in mechs.items())))

    # Two devices must actually have been presented, however they were obtained.
    counts = {n: len((_sel(v).get("presented_archs") or [])) for n, v in states.items()}
    out.append(("two devices were presented to the selector",
                all(c >= 2 for c in counts.values()),
                ", ".join(f"{k}={v}" for k, v in counts.items())))

    # And device 1 must really have read back as the uncovered arch, or the whole
    # scenario is two covered devices and neither side should drop anything.
    reads = {}
    for n, v in states.items():
        archs = _sel(v).get("presented_archs") or []
        reads[n] = archs[1] if len(archs) > 1 else None
    want = (obs.get("head") or {}).get("uncovered_arch")
    out.append(("device 1 presented the uncovered arch",
                all(r == want for r in reads.values()),
                ", ".join(f"{k}={v}" for k, v in reads.items())))

    b, h = obs.get("base") or {}, obs.get("head") or {}
    out.append(("base predates the arch gate and head carries it",
                b.get("has_gate_source") is False and h.get("has_gate_source") is True,
                f"base={b.get('has_gate_source')} head={h.get('has_gate_source')}"))
    return out


def table(obs: dict) -> str:
    rows = ["### Which mechanisms give torch more than one device on one AMD GPU", "",
            "| mechanism | env | torch device_count |", "|---|---|---|"]
    head = obs.get("head") or {}
    for label, rec in (head.get("mechanisms") or {}).items():
        env = ", ".join(f"{k}={v}" for k, v in (rec.get("env") or {}).items()) or "-"
        cnt = rec.get("count", rec.get("error", rec.get("child_error", "-")))
        rows.append(f"| {label} | `{env}` | {cnt} |")

    rows += ["", "### Selection with device 1 presenting the uncovered arch", "",
             "| state | presented archs | gate says uncovered | selected | mode |",
             "|---|---|---|---|---|"]
    for name, v in obs.items():
        if name.startswith("_"):
            continue
        s = _sel(v)
        unc = s.get("uncovered_ids") if s.get("gate_present") else "n/a (no gate)"
        rows.append(f"| {name} | {s.get('presented_archs')} | {unc} | "
                    f"{s.get('selected')} | {s.get('selection_mode')} |")

    rows.append("")
    if _CTX.get("mocked"):
        rows += [
            "**No mechanism produced a second real torch device on this host, so the "
            "device count was mocked.** `GGML_CUDA_DEVICES` gives llama.cpp virtual "
            "devices but is a ggml-backend feature and creates none for torch. This "
            "differential is therefore no stronger than a unit test and must not be "
            "read as hardware evidence for the fix.",
        ]
    else:
        rows += [
            f"**Two real torch devices were obtained via `{_CTX.get('mechanism')}`.** "
            "The devices, their HIP contexts, the enumeration and the whole selection "
            "path are real; only the arch string on device 1 is faked. That is what "
            "makes the defect reachable here at all.",
        ]
    rows += [
        "",
        "Still one physical card either way: no PCIe hop, no second driver context, no "
        "independent VRAM ceiling. This reproduces the SELECTION defect, not a second "
        "GPU, and says nothing about whether a real gfx1036 iGPU then behaves this way.",
    ]
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    """Before the PR nothing asks what the wheel was built for, so the uncovered
    device 1 stays in the selection."""
    sel = _sel(base).get("selected")
    return isinstance(sel, list) and 1 in sel


def head_is_fixed(head: dict) -> bool:
    """After it, device 1 is dropped and device 0 is kept. Keeping device 0 matters:
    dropping both would send a working host to CPU, which the fail-open rule exists
    to prevent."""
    sel = _sel(head).get("selected")
    return isinstance(sel, list) and 1 not in sel and 0 in sel
