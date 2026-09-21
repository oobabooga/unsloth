#!/usr/bin/env python3
"""Criteria: on a REAL AMD card, does unsloth#11068 leave a working ROCm host alone?

The PR adds a ROCm -> Vulkan rung for AMD cards whose generic ROCm sd.cpp prebuilt
cannot run (#9278 gfx1201, #8814 gfx1100). Neither card is in this pool. gfx1151 is,
so the question this host CAN answer is the mirror image of the reporters':

  if the generic ROCm build runs here, the fallback must not fire, and the head must
  select `rocm` exactly as the base did.

That is a regression question, not a differential one, and it is stated as such
deliberately. Running it in differential mode would demand the base exhibit the
defect, and a gfx1151 whose ROCm build works cannot: the honest verdict there is
VOID, which says nothing at all. In regression mode the same reading is a real
result -- the strongest non-regression evidence this change can get, because the
host is a genuine AMD card that the fallback is entitled to divert and does not.

If the ROCm build does NOT run here, the run instead yields the real error text,
and the table shows which of the PR's two marker tiers it lands in. The tiers were
written from issue reports rather than from a machine, so that reading is the
point whichever way it goes: a real failure that matches no marker is reported,
not judged away.

Pairs with probes/sd_cpp_accelerator_reality_probe.py.
"""

from __future__ import annotations

TITLE = "sd.cpp ROCm vs Vulkan on a real gfx1151, base versus head"
MODE = "regression"

# Authored, not computed: every capability this CHANGE touches. The two reporter
# cards are in here precisely because this host is neither of them.
NEEDS = ["gpu", "rocm", "vulkan", "amd_smi", "discrete_gpu",
         "gfx1201_rdna4", "gfx1100_rdna3",
         "windows", "windows_rocm_wddm", "amdvlk",
         "multi_gpu", "nvidia", "xpu", "mlx"]


# ------------------------------------------------------------------ small readers

def _states(obs: dict) -> dict:
    return {n: v for n, v in obs.items() if not n.startswith("_")}


def _shared(obs: dict) -> dict:
    """The host-level half. Identical for every state by construction; the first
    non-empty one is taken and the sameness is checked in a gate."""
    for v in _states(obs).values():
        s = v.get("shared") or {}
        if s:
            return s
    return {}


def _val(entry) -> object:
    if isinstance(entry, dict):
        return entry.get("value")
    return None


def _present(entry) -> bool:
    return bool(isinstance(entry, dict) and entry.get("present"))


def _gens(obs: dict, accel: str) -> dict:
    return {k: g for k, g in (_shared(obs).get("generations") or {}).items()
            if k.startswith(accel + ":")}


def _rendered(g: dict) -> bool:
    img = g.get("image") or {}
    return g.get("rc") == 0 and bool(img.get("png_magic")) and int(img.get("bytes") or 0) > 2000


def _any_rendered(obs: dict, accel: str) -> bool:
    return any(_rendered(g) for g in _gens(obs, accel).values())


def _selected(state: dict) -> str:
    """What this checkout selects for an AMD host with a clean record. The base has
    no `preferred_accelerator`, and that IS its answer: it always asks for rocm."""
    entry = (state.get("preferred_clean") or {}).get("rocm")
    if not _present(entry):
        return "rocm (no preference layer in this checkout)"
    return str(_val(entry))


def _lists(state: dict, accel: str) -> object:
    return _val(((state.get("reads") or {}).get(accel) or {}).get("lists_accelerator"))


# ------------------------------------------------------------------------- gates

def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    states = _states(obs)
    shared = _shared(obs)

    for name, v in states.items():
        mod = v.get("module") or {}
        out.append((f"{name}: sd_cpp_backend imported from its own checkout",
                    bool(mod.get("imported")), str(mod.get("error") or mod.get("file"))[:300]))

    # The states must have read the SAME machine facts, or the comparison is
    # between two different hosts wearing one report.
    own = {n: v.get("installer_sha256") for n, v in states.items()}
    same_installer = len(set(own.values())) == 1 and None not in set(own.values())
    out.append(("every state ships the same install_sd_cpp_prebuilt.py, so the bundles are "
                "comparable", same_installer, f"sha256 per state: {own}"))
    ids = {n: (v.get("shared") or {}).get("run_id") for n, v in states.items()}
    out.append(("every state read ONE shared set of host facts, not one each",
                len(set(ids.values())) == 1 and None not in set(ids.values()), f"{ids}"))
    smoke = any((v.get("shared") or {}).get("smoke") for v in states.values())
    out.append(("not a harness smoke run (AMD_CI_SD_CPP_SMOKE unset), so the renders below "
                "were really attempted", not smoke, f"smoke={smoke}"))

    rocm = (shared.get("bundles") or {}).get("rocm") or {}
    out.append(("the generic ROCm prebuilt really installed", bool(rocm.get("cli")),
                f"asset={rocm.get('asset_resolved')} rc={rocm.get('install_rc')} "
                f"cli={rocm.get('cli')} bytes={rocm.get('cli_bytes')}"))

    ld = (shared.get("list_devices") or {}).get("rocm") or {}
    ran = ld.get("rc") is not None or bool(_gens(obs, "rocm"))
    out.append(("the ROCm binary was actually executed on this card", ran,
                f"--list-devices rc={ld.get('rc')}; generations attempted="
                f"{sorted(_gens(obs, 'rocm'))}"))

    # THE non-vacuity gate. If no accelerator build renders at all, a ROCm failure
    # here is evidence about this harness's command line, not about the card.
    rendered = _any_rendered(obs, "rocm") or _any_rendered(obs, "vulkan")
    detail = "; ".join(f"{k} rc={g.get('rc')} bytes={(g.get('image') or {}).get('bytes')}"
                       for k, g in (shared.get("generations") or {}).items()) or "none attempted"
    out.append(("at least one accelerator build rendered a real image on this card, so the "
                "model, the arguments and the CLI harness are known to work", rendered, detail))
    return out


# ------------------------------------------------------------------------- table

def table(obs: dict) -> str:
    shared = _shared(obs)
    states = _states(obs)
    rows: list[str] = []

    rows.append("**What the real card did** "
                f"(node `{(shared.get('platform') or {}).get('node')}`, "
                f"{(shared.get('platform') or {}).get('system')} "
                f"{(shared.get('platform') or {}).get('machine')})")
    rows.append("")
    rows.append("| build | asset | sd-cli | `--list-devices` rc | devices reported |")
    rows.append("|---|---|---|---|---|")
    for accel in ("rocm", "vulkan"):
        b = (shared.get("bundles") or {}).get(accel) or {}
        ld = (shared.get("list_devices") or {}).get(accel) or {}
        devices = " / ".join(
            ln.strip() for ln in (ld.get("output") or "").splitlines() if ln.strip())[:200]
        rows.append(f"| {accel} | `{b.get('asset_resolved')}` | "
                    f"`{(b.get('cli') or 'MISSING').split('/')[-1]}` | {ld.get('rc')} | "
                    f"{devices or '-'} |")
    rows.append("")

    for accel in ("rocm", "vulkan"):
        wl = ((shared.get("bundles") or {}).get(accel) or {}).get("windows_loader")
        if not wl:
            continue
        failed = {k: v for k, v in (wl.get("bundled") or {}).items() if v != "loaded"}
        rows.append(f"Windows loader, `{accel}` bundle: {wl.get('bundle_dll_count')} DLLs shipped; "
                    f"{len(failed)} of them will not load: {sorted(failed)[:8]}. Resolved by name "
                    f"from PATH: "
                    f"{ {k: ('yes' if 'loaded' in str(v) else 'NO') for k, v in (wl.get('by_name') or {}).items()} }.")
        for k, v in sorted(failed.items())[:4]:
            rows.append(f"    {k}: {v}")
        rows.append("")

    rows.append("| generation | rc | seconds | image bytes | PNG | device lines |")
    rows.append("|---|---|---|---|---|---|")
    for key, g in sorted((shared.get("generations") or {}).items()):
        img = g.get("image") or {}
        dev = "; ".join(g.get("device_lines") or [])[:160].replace("|", "/")
        rows.append(f"| {key} | {g.get('rc')} | {g.get('seconds')} | {img.get('bytes')} | "
                    f"{img.get('png_magic')} | {dev or '-'} |")
    rows.append("")

    rows.append("| state | selects for an AMD host | lists_accelerator(rocm) | "
                "lists_accelerator(vulkan) | fallback_for(rocm) | already diverted? |")
    rows.append("|---|---|---|---|---|---|")
    for name, v in states.items():
        rows.append(f"| {name} | `{_selected(v)}` | {_lists(v, 'rocm')} | {_lists(v, 'vulkan')} | "
                    f"{_val((v.get('fallback_for') or {}).get('rocm'))} | "
                    f"{_val(v.get('failed_clean'))} |")
    rows.append("")

    # The marker tiers against the REAL error text, whichever way it went.
    for name, v in states.items():
        for key, r in sorted((v.get("error_reading") or {}).items()):
            m = r.get("markers") or {}
            dec = (m.get("_ACCELERATOR_DECISIVE_FAILURE_MARKERS") or {}) or {}
            amb = (m.get("_ACCELERATOR_AMBIGUOUS_FAILURE_MARKERS") or {}) or {}
            cap = (m.get("_ACCELERATOR_CAPACITY_FAILURE_MARKERS") or {}) or {}
            if dec.get("markers") is None and amb.get("markers") is None:
                continue
            rows.append(f"Marker read, `{name}` on `{key}` (rc={r.get('rc')}): decisive "
                        f"{dec.get('matched')}, ambiguous {amb.get('matched')}, capacity "
                        f"{cap.get('matched')}, "
                        f"`output_shows_accelerator_failure` = "
                        f"{_val(r.get('output_shows_accelerator_failure'))}.")
            rows.append(f"    As the load path sees it (`{(r.get('engine_message') or '')[:60]}"
                        f"...`, {r.get('raw_output_chars')} chars of raw output): "
                        f"accelerator failure "
                        f"{_val(r.get('engine_message_shows_accelerator_failure'))}, decisive "
                        f"{_val(r.get('engine_message_shows_decisive_failure'))}, image-load "
                        f"{_val(r.get('engine_message_shows_image_load_failure'))}.")
    empty = {k: g.get("rc") for k, g in (shared.get("generations") or {}).items()
             if k.startswith("rocm:") and not (g.get("output") or "").strip()}
    if empty:
        rows.append(f"The ROCm build printed NOTHING on {sorted(empty)}: exit status "
                    f"{sorted(set(empty.values()))}, zero bytes of output. Every marker tier "
                    f"reads text, so none of them can fire on this failure however it is "
                    f"classified.")
    rows.append("")

    for name, v in states.items():
        fp = v.get("fingerprint") or {}
        if not fp.get("present"):
            continue
        nb = fp.get("inventory_nonblocking") or {}
        bl = fp.get("inventory_blocking") or {}
        rows.append(f"Fingerprint on `{name}`: cold `{fp.get('fingerprint_cold')}`, warm "
                    f"`{fp.get('fingerprint_warm')}`, torch `{fp.get('torch_version')}`, "
                    f"torch.version.hip `{fp.get('torch_version_hip')}`, bundle tag "
                    f"`{((fp.get('fingerprint_cold') or {}) or {}).get('bundle')}`.")
        rows.append(f"`get_physical_gpu_inventory(block=False)` on `{name}`: unknown="
                    f"{nb.get('unknown')}, cards={nb.get('names')}, vendors={nb.get('vendors')}, "
                    f"sources={nb.get('sources')} (blocking: unknown={bl.get('unknown')}, "
                    f"cards={bl.get('names')}).")
        rows.append(f"Record written here: persisted="
                    f"{bool(_val(fp.get('stored_after_note')))}, "
                    f"diverting={_val(fp.get('failed_after_note'))}; after "
                    f"installing bundle `{(fp.get('retirement_install') or {}).get('tag_requested')}` "
                    f"over the managed tree: {_val(fp.get('failed_after_bundle_change'))}; after "
                    f"the reset route's clear: {_val(fp.get('failed_after_clear'))}.")
    return "\n".join(rows)


# ----------------------------------------------------------------------- verdict

def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    obs = {"base": base, "head": head}
    rocm_ok = _any_rendered(obs, "rocm")
    vulkan_ok = _any_rendered(obs, "vulkan")
    sel_base, sel_head = _selected(base), _selected(head)

    if (base.get("module") or {}).get("imported") and not (head.get("module") or {}).get("imported"):
        return True, ("the head's sd_cpp_backend does not import on this host while the base's "
                      f"does: {(head.get('module') or {}).get('error')}")

    if _val(head.get("failed_clean")) is True:
        return True, ("the head reports this card as already diverted with no record ever "
                      "written, so a clean AMD host would never try its own ROCm build")

    if rocm_ok and not str(sel_head).startswith("rocm"):
        return True, (f"the ROCm build renders on this card, and the head still selects "
                      f"`{sel_head}` where the base selects `{sel_base}`: a working ROCm host "
                      f"is being diverted")

    for accel in ("rocm", "vulkan"):
        b, h = _lists(base, accel), _lists(head, accel)
        if b is True and h is False:
            return True, (f"the base reads the real {accel} build as offering an accelerator "
                          f"and the head does not, so the head loses a GPU this host has")

    if rocm_ok:
        return False, (
            f"the generic ROCm sd.cpp prebuilt RUNS on this gfx1151: it rendered a real image "
            f"{'and Vulkan did too' if vulkan_ok else 'while Vulkan did not'}. The head still "
            f"selects `{sel_head}`, the base selects `{sel_base}`, and the head's record is "
            f"empty, so the new rung did not fire on a host that does not need it. That is the "
            f"non-regression this card can prove; it is not evidence about gfx1201 or gfx1100")

    if vulkan_ok:
        return False, (
            f"the generic ROCm prebuilt does NOT render on this gfx1151 while the Vulkan one "
            f"does, which is the shape #9278 and #8814 report on other cards. The head selects "
            f"`{sel_head}` on a clean record and the base selects `{sel_base}`; see the marker "
            f"table for whether the real error text lands in the PR's tiers. Not worse than base")

    return False, ("neither build rendered, so there is nothing for the head to be worse at; "
                   "the gate above should have caught this")
