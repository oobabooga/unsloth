#!/usr/bin/env python3
"""Criteria: does PR 7682 turn bf16 off on gfx10 without turning it off anywhere else?

The defect (issue 7922) is that on RDNA1/2 `torch.cuda.is_bf16_supported()` answers
True, Unsloth loads bf16, and Triton then asks for `llvm.amdgcn.fdot2.bf16.bf16`,
which LLVM cannot lower for that target. The process dies before Python sees it.

Both halves are judged here, because a fix that disables bf16 everywhere would also
"fix" the crash:

  base_shows_defect -- under the gfx1032 spoof the base still reports bf16 supported.
  head_is_fixed     -- under the same spoof the head reports it unsupported, AND the
                       unspoofed reading on this real gfx1151 is still bf16 supported.

That second clause is the whole point. This runner is RDNA3.5, which genuinely does
have bf16, so it is the control: if the head switched bf16 off here too, the gate
over-matches and the verdict must not be CONFIRMED.

WHAT THIS CANNOT SAY. The spoof changes a string, not silicon. It proves the decision
this code makes given an arch, not that a physical RX 6600 XT then trains to
completion. No gfx10 card exists on any runner reachable here; that row stays open and
is stated in the report rather than implied away.
"""

from __future__ import annotations

TITLE = "bf16 gating on gfx10 (PR 7682), against a real ROCm host"
MODE = "differential"

# Declares what the CHANGE touches, not what the host has: the patch is process-wide
# and reaches every backend, so each of these has to appear as a stated gap.
NEEDS = [
    "rocm", "gpu", "discrete_gpu", "integrated_gpu",
    "multi_gpu", "multi_gpu_amd", "nvidia", "xpu", "mlx", "windows",
]


def _real(state: dict) -> dict:
    return state.get("real") or {}


def _spoofed(state: dict) -> dict:
    return state.get("spoofed") or {}


def _bf16(reading: dict):
    """The value Unsloth actually loads models with: models/_utils.SUPPORTS_BFLOAT16.

    Deliberately not _gpu_init's copy. Every dtype decision in llama.py, vision.py,
    loader.py and attention_dispatch reads the _utils one, so that is the number whose
    change is user-visible; the _gpu_init value only matters because it feeds it.
    """
    return reading.get("utils_bf16")


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    states = {n: v for n, v in obs.items() if not n.startswith("_")}

    # Every reading has to exist, or a missing subprocess reads as "no defect" and
    # the run would quietly land on VOID for the wrong reason.
    missing = []
    for name, st in states.items():
        for leg in ("real", "spoofed"):
            r = st.get(leg) or {}
            if _bf16(r) is None:
                err = (r.get("unsloth_error") or r.get("child_error")
                       or r.get("torch_error") or "no reading")
                missing.append(f"{name}/{leg}: {err}")
    out.append(("every state read SUPPORTS_BFLOAT16 both ways",
                not missing, "; ".join(missing) or "all four readings present"))

    # If this is not a ROCm host the whole comparison is meaningless: the HIP branch
    # never runs and both states answer from the CUDA path.
    hips = {n: (_real(v).get("torch_hip")) for n, v in states.items()}
    on_hip = all(bool(v) for v in hips.values())
    out.append(("torch is a ROCm build", on_hip,
                ", ".join(f"{k}=hip {v}" for k, v in hips.items())))

    # The spoof must actually have taken, or the "spoofed" leg is a second real leg
    # and base_shows_defect would be answering about gfx1151.
    reads = {n: _spoofed(v).get("spoof_reads_back") for n, v in states.items()}
    took = all((v or "").startswith("gfx1032") for v in reads.values())
    out.append(("the gfx1032 spoof was visible to the reader", took,
                ", ".join(f"{k}={v}" for k, v in reads.items())))

    # The base must predate arch_lacks_bf16 and the head must carry it, otherwise the
    # states are not the two sides of this PR.
    b, h = obs.get("base") or {}, obs.get("head") or {}
    shape = (b.get("has_arch_lacks_bf16") is False
             and h.get("has_arch_lacks_bf16") is True)
    out.append(("base predates arch_lacks_bf16 and head carries it", shape,
                f"base={b.get('has_arch_lacks_bf16')} head={h.get('has_arch_lacks_bf16')}"))
    return out


def table(obs: dict) -> str:
    rows = ["| state | real arch bf16 | gfx1032 spoof bf16 | torch's own answer | "
            "bf16 matmul | kwarg call |",
            "|---|---|---|---|---|---|"]
    for name, v in obs.items():
        if name.startswith("_"):
            continue
        r, s = _real(v), _spoofed(v)
        kwarg = r.get("bf16_kwarg_error") or r.get("bf16_including_emulation_false")
        rows.append(
            f"| {name} | {_bf16(r)} | {_bf16(s)} | {r.get('torch_says_bf16')} | "
            f"{r.get('bf16_matmul_ok', r.get('bf16_matmul_error', '-'))} | {kwarg} |"
        )
    rows += [
        "",
        "`real` is this runner's own gfx1151, which has bf16 and must keep it. "
        "`gfx1032 spoof` presents the arch from issue 7922 to the same ROCm runtime.",
        "",
        "The spoof exercises the decision, not the silicon. A physical gfx10 card was "
        "not available to any runner reachable from here, so 'a real RX 6600 XT now "
        "trains' is NOT established by this run.",
    ]
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    # Before the PR, models/_utils.py hardcodes SUPPORTS_BFLOAT16 = True on HIP, so
    # the spoofed gfx1032 still comes back bf16-capable. That is the defect.
    return _bf16(_spoofed(base)) is True


def head_is_fixed(head: dict) -> bool:
    # Fixed means BOTH: gfx1032 loses bf16, and this real gfx1151 keeps it.
    return _bf16(_spoofed(head)) is False and _bf16(_real(head)) is True
