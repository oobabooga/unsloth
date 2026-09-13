#!/usr/bin/env python3
"""Criteria: can PR #10618's ``--load-mode dio`` collide with the planner's ``none``?

The worry is concrete. Two producers can put a ``--load-mode`` on one argv, they
disagree about the loader, and llama.cpp resolves a repeated ``--load-mode`` by
LAST ARG WINS. If the planner's ``none`` landed after the managed ``dio`` the
no-reserve setting would silently stop streaming; if it landed before, the
planner's measured 2-4.6x host-read win would be silently discarded.

The answer this judges is that they cannot collide, because the launch seam is
``_spill_plan_flags_for`` and that helper never emits a ``--load-mode`` at all --
``plan_to_args``, which does, has no production caller. That is a claim about
which helper the running code calls, so it is judged on tokens the probe read
out of a live interpreter, not on a source search.

Regression, not differential: there is no defect to demonstrate at the base. The
invariant has to hold on BOTH revisions, and the thing that would make head
worse is the PR introducing a second ``--load-mode`` producer that meets the
first.
"""

from __future__ import annotations

TITLE = "Load-mode producers on one argv, base versus head"
MODE = "regression"
# Windows because the dio policy is win32-gated and this pool's Windows half is
# the only place the decision runs natively; gpu because the APU classification
# the other producer depends on is a property of the real device.
NEEDS = ["gpu", "windows"]


def _modes(o: dict, key: str) -> list[str]:
    return list(o.get(key) or [])


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    for name in ("base", "head"):
        o = obs.get(name) or {}

        # The probe has to have reached the code at all. An import error would
        # otherwise read as "no load-mode tokens found", which is the answer the
        # criteria is looking for, arrived at by not looking.
        reached = bool(o.get("spill_seam_present")) and "spill_seam_tokens" in o
        detail = "read the seam's tokens"
        for k in ("planner_import_error", "backend_import_error", "spill_seam_error", "error"):
            if o.get(k):
                detail = f"{k}: {str(o[k])[:200]}"
        out.append((f"{name}: production spill seam was exercised", reached, detail))

        # And the plan fed to it must genuinely have wanted `none`. A plan that
        # did not ask for it proves nothing about a seam that did not emit it.
        wanted = bool(o.get("plan_load_mode_none")) and bool(o.get("plan_spills_anything"))
        out.append((
            f"{name}: the plan actually asked for load-mode none",
            wanted,
            f"load_mode_none={o.get('plan_load_mode_none')}, "
            f"spills_anything={o.get('plan_spills_anything')}",
        ))

        # The contrast has to be live too: if the dead helper did NOT emit
        # `none` for this plan, then the seam not emitting it says nothing.
        dead = _modes(o, "plan_to_args_load_modes")
        out.append((
            f"{name}: the unused helper does emit none for that plan",
            dead == ["none"],
            f"plan_to_args -> {dead or 'nothing'}",
        ))

    h = obs.get("head") or {}
    # Windows, or the dio half of the question was never really asked.
    out.append((
        "head: the dio decision ran on a native Windows interpreter",
        bool(h.get("decision_platform_is_native")),
        f"platform={h.get('platform')}, native={h.get('decision_platform_is_native')}",
    ))
    return out


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    seam = _modes(head, "spill_seam_load_modes")
    combined = _modes(head, "combined_load_modes")

    if seam:
        return True, (
            f"the launch seam emitted `--load-mode {' '.join(seam)}` for a spill plan. "
            "It never did before, so the planner is now a second load-mode producer "
            "and can meet the managed dio on one argv."
        )
    if len(combined) > 1:
        return True, (
            f"a single argv carried {len(combined)} load-mode values {combined}; "
            "llama.cpp takes the last, so one of the two producers is silently lost."
        )

    base_seam = _modes(base, "spill_seam_load_modes")
    if base_seam and not seam:
        return False, (
            "the base seam emitted a load-mode and head does not; that is a narrowing, "
            "not a regression."
        )
    return False, (
        f"the spill seam emits no --load-mode on either revision "
        f"(head tokens: {head.get('spill_seam_tokens')}), while the unused "
        f"`plan_to_args` would have emitted {_modes(head, 'plan_to_args_load_modes')}. "
        f"The two producers cannot meet: combined argv carries {combined or 'no'} "
        "load-mode value(s)."
    )


def table(obs: dict) -> str:
    rows = [
        "| state | smart offload | seam tokens | seam load-modes | unused helper | dio emitted | combined |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if not o:
            continue
        rows.append(
            f"| {name} "
            f"| {o.get('smart_offload_enabled')} "
            f"| `{' '.join(o.get('spill_seam_tokens') or []) or 'none'}` "
            f"| {', '.join(_modes(o, 'spill_seam_load_modes')) or 'NONE'} "
            f"| {', '.join(_modes(o, 'plan_to_args_load_modes')) or 'none'} "
            f"| {o.get('policy_emitted_dio')} "
            f"| {', '.join(_modes(o, 'combined_load_modes')) or 'NONE'} |"
        )
    return "\n".join(rows)
