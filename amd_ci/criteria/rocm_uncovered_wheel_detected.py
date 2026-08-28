#!/usr/bin/env python3
"""Criteria: on a real GPU the installed wheel has no kernels for, does PR 8791 notice?

Scoped narrowly on purpose. This is NOT the two-GPU selection defect from issue 8792,
and a CONFIRMED verdict here must not be read as that. What it is:

The runner's gfx1151 is covered by its stock wheel, so the gate has nothing to find.
Installing AMD's `gfx110X-all` build -- a real, shipping wheel for RDNA3 discrete,
which does not include RDNA3.5 -- makes the real silicon genuinely uncovered with
nothing spoofed. The question then is whether this change can tell.

  base_shows_defect -- the device is really outside the wheel's arch list and the base
                       has no mechanism that notices; the host is broken and silent.
  head_is_fixed     -- the head identifies the host as uncovered.

With a single device that is itself uncovered, the gate's fail-open rule returns the
empty set deliberately, rather than dropping the only GPU and stranding a working
machine on CPU. So "noticing" is observed through the warning it logs, not through the
returned set. That rule firing correctly on real silicon is a real result and the
reason the returned set is not scored here.

WHAT THIS STILL CANNOT SAY. One card. The iGPU-beside-a-discrete-card shape, where the
gate must drop one device and keep another, needs two AMD GPUs and is not reachable on
any runner available. No ROCm mechanism duplicates a GPU: the visibility variables only
filter and reorder, and CPX compute partitioning, which does expose one package as
several HIP devices, is CDNA2/CDNA3 only. That was measured, not assumed.
"""

from __future__ import annotations

TITLE = "Detecting a real GPU the installed torch wheel has no kernels for (PR 8791)"
MODE = "differential"

NEEDS = ["rocm", "gpu", "integrated_gpu", "discrete_gpu",
         "multi_gpu", "multi_gpu_amd", "nvidia", "windows", "xpu", "mlx"]

_NOTICED = (
    "has no kernels for any GPU on this host",   # the all-uncovered warning
    "no kernels for their architecture",         # the per-device exclusion warning
)


def _noticed(g: dict) -> bool:
    """Did the gate SAY the host is uncovered, on any stream it actually writes to?

    The backend logs through structlog, which goes to the child's real stdout/stderr,
    not the stdlib handler the probe installs. Searching only the stdlib buffer reads
    a warning that was emitted as one that never was.
    """
    blob = " ".join(str(g.get(k) or "") for k in ("log", "child_stdout", "child_stderr"))
    return any(m in blob for m in _NOTICED)


def _why_empty(g: dict) -> str:
    """Which early exit fired, when the gate returned nothing."""
    if g.get("pre_torch_available") is False:
        return "torch.cuda.is_available() was False"
    if g.get("pre__rocm_device_ordinal_active") is True:
        return "GPU_DEVICE_ORDINAL renumbers ordinals"
    if g.get("pre__rocm_visibility_masks_are_stacked") is True:
        return "ROCr and HIP masks are stacked"
    if g.get("pre_numeric_ids") is None:
        return "no numeric physical ids behind the mask"
    return "reached the arch comparison"


def _u(state: dict) -> dict:
    return state.get("uncovered") or {}


def _ug(state: dict) -> dict:
    return state.get("uncovered_gate") or {}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    states = {n: v for n, v in obs.items() if not n.startswith("_")}

    # The uncovered wheel has to have installed, or every reading below is about the
    # stock wheel and the whole run means nothing.
    bad = {n: (v.get("install") or {}).get("pip_rc") for n, v in states.items()
           if (v.get("install") or {}).get("pip_rc") != 0}
    out.append(("the uncovered wheel installed in every state", not bad,
                str(bad) if bad else "pip returned 0 everywhere"))

    # The swapped wheel must actually be a ROCm build. Without this gate a venv that
    # quietly kept a CUDA torch reports arch_list ['sm_80', ...], device_is_covered
    # False, and a matmul failure -- every surface reading of "uncovered" -- while
    # measuring nothing about AMD at all. That happened once and was caught only by
    # eye, which is not a control.
    hips = {n: _u(v).get("hip") for n, v in states.items()}
    out.append(("the swapped wheel is a ROCm build",
                all(bool(v) for v in hips.values()),
                ", ".join(f"{k}=hip {v}" for k, v in hips.items())))

    seen = {n: _u(v).get("arch_list") for n, v in states.items()}
    ok = all(any(str(a).startswith("gfx") for a in (v or [])) for v in seen.values())
    out.append(("its arch list names gfx targets, not sm_", ok,
                ", ".join(f"{k}={v}" for k, v in seen.items())))

    # The control: with the stock wheel the device must be COVERED. Without this, an
    # "uncovered" reading might just be this host, not the wheel.
    stock = {n: (v.get("stock") or {}).get("device_is_covered") for n, v in states.items()}
    out.append(("the stock wheel covers this GPU (control)",
                all(v is True for v in stock.values()),
                ", ".join(f"{k}={v}" for k, v in stock.items())))

    # And with the swapped wheel it must NOT be. This is the whole premise.
    unc = {n: _u(v).get("device_is_covered") for n, v in states.items()}
    out.append(("the gfx110X-all wheel does NOT cover this GPU",
                all(v is False for v in unc.values()),
                ", ".join(f"{k}={v}" for k, v in unc.items())))

    b, h = obs.get("base") or {}, obs.get("head") or {}
    out.append(("base predates the gate and head carries it",
                b.get("has_gate_source") is False and h.get("has_gate_source") is True,
                f"base={b.get('has_gate_source')} head={h.get('has_gate_source')}"))
    return out


def table(obs: dict) -> str:
    rows = ["| state | wheel | arch list | device arch | covered | real matmul |",
            "|---|---|---|---|---|---|"]
    for name, v in obs.items():
        if name.startswith("_"):
            continue
        for label, rec in (("stock", v.get("stock") or {}), ("gfx110X-all", _u(v))):
            mm = rec.get("matmul_error") or rec.get("matmul_ok")
            rows.append(
                f"| {name} | {label} | `{rec.get('arch_list')}` | "
                f"`{rec.get('device_archs')}` | {rec.get('device_is_covered')} | "
                f"{str(mm)[:70]} |"
            )

    rows += ["", "| state | gate present | uncovered ids | selected | noticed in the log |",
             "|---|---|---|---|---|"]
    for name, v in obs.items():
        if name.startswith("_"):
            continue
        g = _ug(v)
        rows.append(f"| {name} | {g.get('gate_present')} | {g.get('uncovered_ids')} | "
                    f"{g.get('selected')} | {_noticed(g)} |")

    rows += ["", "| state | torch available | ordinal renumbered | masks stacked | "
             "numeric ids | why empty |", "|---|---|---|---|---|---|"]
    for name, v in obs.items():
        if name.startswith("_"):
            continue
        g = _ug(v)
        rows.append(
            f"| {name} | {g.get('pre_torch_available')} | "
            f"{g.get('pre__rocm_device_ordinal_active')} | "
            f"{g.get('pre__rocm_visibility_masks_are_stacked')} | "
            f"{g.get('pre_numeric_ids')} | {_why_empty(g)} |")

    rows += [
        "",
        "The `gfx110X-all` rows use a real, shipping AMD wheel built for RDNA3 discrete "
        "(gfx1100/1101/1102/1103). This runner is gfx1151, RDNA3.5, which that build "
        "does not cover. Nothing is spoofed: the device, the arch list and the "
        "mismatch are all real.",
        "",
        "**This is the detection half only.** One uncovered device out of one means the "
        "gate's fail-open rule returns the empty set on purpose, rather than dropping "
        "the only GPU. The two-GPU selection defect from issue 8792 needs a second AMD "
        "card and is not reachable here: ROCm's visibility variables only filter and "
        "reorder, and CPX partitioning is CDNA-only. Both were measured.",
    ]
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    """A genuinely uncovered device, and nothing in the tree that knows."""
    if _u(base).get("device_is_covered") is not False:
        return False
    g = _ug(base)
    if g.get("gate_present"):
        return False
    return not _noticed(g)


def head_is_fixed(head: dict) -> bool:
    """The head has the gate and says so about this host."""
    if _u(head).get("device_is_covered") is not False:
        return False
    g = _ug(head)
    if not g.get("gate_present"):
        return False
    return _noticed(g)
