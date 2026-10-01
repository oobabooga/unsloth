#!/usr/bin/env python3
"""Criteria: does the MiniMax-H3 resident check over-estimate free VRAM on this host?

The risk: `_h3_card_free_bytes` reads free memory out of process (amd-smi here). On a
unified-memory APU that read can report the whole pool as free, or ignore what another process
holds; a render would then drop `--offload-to-cpu` and OOM. Judged against the driver
(`torch.cuda.mem_get_info()` in the bystander probe):

  1. the head read is not None on a GPU host;
  2. it does not exceed driver free by more than max(1 GiB, 5% of total);
  3. with a holder resident, it drops by roughly what the holder took (vs the no-holder run of
     the same state): within max(1 GiB, 25% of the holder) of the driver's own drop, which a
     gate requires to be ~ the holder.

The merge base has no such function, so this is a regression-mode, head-correctness check:
the base is recorded N/A and nothing is claimed about a fix. Pairs with probes/h3_free_probe.py
and (holder state) probes/vram_holder_fixture.py.
"""

from __future__ import annotations

GIB = 1024 ** 3
TITLE = "H3 resident check: live free VRAM read vs driver free"
MODE = "regression"
NEEDS = ["gpu", "rocm", "integrated_gpu", "discrete_gpu", "nvidia", "multi_gpu",
         "windows_rocm_wddm", "xpu", "mlx"]

_CTX: dict = {"held_gib": 0.0, "holder": False}


def _gib(b):
    return None if b is None else b / GIB


def _states(obs):
    return {n: v for n, v in obs.items() if not n.startswith("_")}


def _head_read(state):
    return (state.get("h3_free_post_ctx") or {}).get("none")


def _driver_free(state):
    return state.get("driver_free_bytes_after", state.get("driver_free_bytes"))


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    fx = obs.get("_fixture") or {}
    held = float(fx.get("allocated_gib") or 0.0)
    _CTX["holder"] = bool(fx)
    _CTX["held_gib"] = held
    out = []
    for name, v in _states(obs).items():
        total = v.get("driver_total_bytes") or 0
        out.append((f"{name}: torch sees a GPU", total > 0,
                    v.get("torch_error") or f"total {total / GIB:.2f} GiB, {v.get('arch')}"))
        out.append((f"{name}: observer stayed a bystander",
                    (v.get("observer_allocated_bytes") or 0) < GIB // 2,
                    f"{(v.get('observer_allocated_bytes') or 0) / GIB:.3f} GiB"))
    head = obs.get("head") or {}
    out.append(("head checkout has _h3_card_free_bytes",
                bool(head.get("h3_free_fn_available")),
                head.get("video_import_error") or "imported"))
    if fx:
        out.append(("holder really held >= 2 GiB", held >= 2.0, f"{held:.2f} GiB"))
        b = head.get("baseline") or {}
        ok = _driver_free(b) is not None and _driver_free(head) is not None
        drop = (_driver_free(b) - _driver_free(head)) / GIB if ok else None
        tol = max(1.0, 0.25 * held)
        out.append(("driver free itself dropped by ~holder (holder visible to the reference)",
                    ok and abs(drop - held) <= tol,
                    f"driver drop {drop:.2f} GiB vs held {held:.2f}" if ok
                    else "no no-holder baseline for head"))
    return out


def table(obs: dict) -> str:
    rows = ["| state | driver free GiB | head read GiB (pre ctx / post ctx, ord None / 0) | "
            "total GiB | read - driver | resident flags (head read) | resident (driver free) |",
            "|---|---|---|---|---|---|---|"]
    for name, v in _states(obs).items():
        df, tot = _gib(_driver_free(v)), _gib(v.get("driver_total_bytes"))
        if not v.get("h3_free_fn_available"):
            rows.append(f"| {name} | {df:.2f} | N/A (function absent at this checkout) | "
                        f"{tot:.2f} | N/A | N/A | N/A |" if df is not None and tot is not None
                        else f"| {name} | {v.get('torch_error')} | N/A | | | | |")
            continue
        pre, post = v.get("h3_free_pre_ctx") or {}, v.get("h3_free_post_ctx") or {}

        def f(x):
            return "None" if x is None else f"{x / GIB:.2f}"
        rd = _head_read(v)
        delta = "" if rd is None or df is None else f"{rd / GIB - df:+.2f}"
        rows.append(
            f"| {name} | {df:.2f} | {f(pre.get('none'))} / {f(post.get('none'))}, "
            f"{f(post.get('none'))} / {f(post.get('zero'))} | {tot:.2f} | {delta} | "
            f"{v.get('resident')} {v.get('flags')} | {v.get('driver_resident')} |")
    rows.append("")
    need = next((v.get("need_bytes") for v in _states(obs).values() if v.get("need_bytes")), None)
    if need:
        rows.append(f"need_bytes = h3_native_resident_bytes(24 GiB, 960, 544, 124) = {need / GIB:.2f} GiB")
    if _CTX["holder"]:
        rows.append(f"Holder resident: {_CTX['held_gib']:.2f} GiB.")
    else:
        rows.append("No holder.")
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    problems, notes = [], ["base: function absent, recorded N/A (no-regression / head "
                           "correctness check, not fix confirmation)"]
    total = head.get("driver_total_bytes") or 0
    df = _driver_free(head)
    for key in ("none", "zero"):
        rd = (head.get("h3_free_post_ctx") or {}).get(key)
        if rd is None:
            problems.append(f"head read (ordinal {key}) is None on a GPU host")
            continue
        tol = max(GIB, 0.05 * total)
        if df is not None and rd - df > tol:
            problems.append(f"head read (ordinal {key}) {rd / GIB:.2f} GiB exceeds driver free "
                            f"{df / GIB:.2f} GiB by more than {tol / GIB:.2f} GiB")
        else:
            notes.append(f"ordinal {key}: read {rd / GIB:.2f} vs driver {df / GIB:.2f} GiB")
    if _CTX["holder"]:
        b = head.get("baseline") or {}
        r0, r1 = _head_read(b), _head_read(head)
        d0, d1 = _driver_free(b), _driver_free(head)
        held = _CTX["held_gib"]
        tol = max(1.0, 0.25 * held)
        if None in (r0, r1, d0, d1):
            problems.append("cannot compute head read drop (missing no-holder or holder read)")
        else:
            drop, ddrop = (r0 - r1) / GIB, (d0 - d1) / GIB
            # Judged against the driver's own drop over the same interval, so anything else
            # moving the card between the two runs moves both figures alike; the gate already
            # requires the driver drop itself to be ~ the holder.
            if abs(drop - ddrop) > tol:
                side = "UNDER-counts the holder (over-estimates free)" if drop < ddrop \
                    else "over-counts the holder (conservative)"
                problems.append(f"head read dropped {drop:.2f} GiB while driver free dropped "
                                f"{ddrop:.2f} GiB ({held:.2f} GiB held): {side}")
            else:
                notes.append(f"head read dropped {drop:.2f} GiB, driver {ddrop:.2f} GiB, "
                             f"holder {held:.2f} GiB")
    if problems:
        return True, "; ".join(problems)
    return False, "; ".join(notes)
