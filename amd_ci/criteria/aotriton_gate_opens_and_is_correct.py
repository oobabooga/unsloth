"""Judge PR 8821's AOTriton gate on real AMD hardware.

The defect PR 8821 claims: on ROCm, `import unsloth` leaves
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL unset, so torch refuses the flash and
mem-efficient SDPA backends on an architecture AOTriton flags experimental and
attention falls back to the quadratic MATH path.

So the base must show BOTH halves: the gate unset after `import unsloth`, AND
torch declining at least one of the two backends because of it. A host whose
AOTriton does not flag this architecture experimental admits the backends with
the gate shut, which makes the defect unreproducible here -- VOID, not a pass.
That is the honest answer for gfx1151 if that is what the hardware says.

The head must open the gate, admit the backends, and -- the part a placement
test cannot reach -- produce output that still agrees with the math reference.
An experimental kernel that runs and is wrong is worse than the fallback.
"""

MODE = "differential"
TITLE = "PR 8821: does opening the AOTriton gate change SDPA, and is it still correct?"

# Declared for what the CHANGE touches, not for what this host has. The gate is a
# no-op on a CUDA build (the read sits inside `#if USE_ROCM`), but the claim is
# about ROCm architectures this box is not, and about Windows and multi-GPU
# inheritance that this job does not reach either.
NEEDS = ["rocm", "gpu", "nvidia", "windows", "multi_gpu", "xpu", "mlx"]

# fp16 attention against a math reference in fp16: a different but valid kernel
# ordering moves the last bits, a broken one does not stay anywhere near this.
_TOLERANCE = 2e-2


def _num(state, key):
    value = state.get(key)
    return value if isinstance(value, (int, float)) else None


def gates(obs):
    out = []
    for name in ("base", "head"):
        state = obs.get(name) or {}
        out.append(
            (
                f"{name}: the probe ran",
                state.get("rc") == 0 and not state.get("timeout"),
                f"rc={state.get('rc')} timeout={bool(state.get('timeout'))} "
                f"{state.get('stderr_tail', '')[-200:]}",
            )
        )
        out.append(
            (
                f"{name}: unsloth imported",
                bool(state.get("unsloth_imported")),
                str(state.get("unsloth_error", "ok")),
            )
        )
        out.append(
            (
                f"{name}: a ROCm GPU answered",
                bool(state.get("hip")) and bool(state.get("cuda_available")),
                f"hip={state.get('hip')} arch={state.get('arch')} "
                f"available={state.get('cuda_available')}",
            )
        )
    base, head = obs.get("base") or {}, obs.get("head") or {}
    out.append(
        (
            "the two states are the same GPU",
            base.get("arch") == head.get("arch") and base.get("arch") is not None,
            f"base={base.get('arch')} head={head.get('arch')}",
        )
    )
    out.append(
        (
            "only the head carries the gate",
            base.get("init_sets_gate") is False and head.get("init_sets_gate") is True,
            f"base={base.get('init_sets_gate')} head={head.get('init_sets_gate')}",
        )
    )
    return out


def table(obs):
    keys = (
        "arch", "torch", "hip", "init_sets_gate", "gate_after_import",
        "can_use_flash", "can_use_efficient",
        "flash_ran", "flash_max_abs_diff_vs_math", "flash_peak_gib",
        "efficient_ran", "efficient_max_abs_diff_vs_math", "efficient_peak_gib",
        "math_peak_gib", "default_peak_gib", "default_matches_math_exactly",
    )
    names = [n for n in obs if not n.startswith("_")]
    rows = ["| observation | " + " | ".join(names) + " |",
            "|---|" + "---|" * len(names)]
    for key in keys:
        rows.append(
            f"| `{key}` | "
            + " | ".join(str((obs[n] or {}).get(key, "-")) for n in names)
            + " |"
        )
    return "\n".join(rows)


def base_shows_defect(base):
    if base.get("gate_after_import") is not None:
        return False, (
            "the base already had the gate set after importing unsloth, so nothing "
            "about this reading is attributable to the change"
        )
    shut_out = [
        name
        for name, key in (("flash", "can_use_flash"), ("mem-efficient", "can_use_efficient"))
        if base.get(key) is False
    ]
    if not shut_out:
        return False, (
            f"with the gate SHUT, torch still admits both SDPA backends on "
            f"{base.get('arch')}. AOTriton does not flag this architecture "
            f"experimental, so the variable the PR sets is never read here and the "
            f"defect cannot be reproduced on this host"
        )
    return True, (
        f"with the gate shut, torch refuses {' and '.join(shut_out)} attention on "
        f"{base.get('arch')}"
    )


def head_is_fixed(state):
    if state.get("gate_after_import") != "1":
        return False, (
            f"importing unsloth left the gate at {state.get('gate_after_import')!r}, "
            "so it never reached torch"
        )
    if not (state.get("can_use_flash") or state.get("can_use_efficient")):
        return False, (
            "the gate is open but torch still admits neither backend, so opening it "
            "bought nothing on this architecture"
        )
    # Correctness, not just eligibility.
    for name in ("flash", "efficient"):
        if not state.get(f"{name}_ran"):
            continue
        if state.get(f"{name}_finite") is False:
            return False, f"the {name} kernel produced non-finite output"
        diff = _num(state, f"{name}_max_abs_diff_vs_math")
        if diff is None:
            return False, f"the {name} kernel ran but was never compared to math"
        if diff > _TOLERANCE:
            return False, (
                f"the {name} kernel disagrees with the math reference by {diff:.4g} "
                f"(tolerance {_TOLERANCE}), so the experimental path this PR opens is "
                f"not numerically usable on {state.get('arch')}"
            )
    return True, (
        "the gate is open, torch admits the backend, and its output still agrees "
        "with the math reference"
    )
