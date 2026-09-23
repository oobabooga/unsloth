#!/usr/bin/env python3
"""Criteria: what the two upstream sd.cpp prebuilts really do on a gfx1151, and
what that bounds about unsloth#11068.

This is stated as an OBSERVATION, not as "does the fix fix the defect", and the
distinction is the whole design. PR 11068 adds a ROCm -> Vulkan rung for AMD cards
whose generic ROCm sd.cpp prebuilt cannot run (#9278 gfx1201, #8814 gfx1100).
Neither card is in this pool. gfx1151 IS, and gfx1151 is in the ROCm build's
target list, so on a runner that has a ROCm runtime the defect cannot reproduce.
Run as a differential, this would therefore land on VOID -- a correct verdict and
a wasted run.

So the question is turned around into the three this host can actually answer:

  1. do the ROCm sonames the archive NEEDS but does not bundle resolve here, and
     which ROCm does this machine have;
  2. does the generic ROCm sd-cli start and enumerate this card;
  3. does the Vulkan sd-cli start and enumerate it, which is the load-bearing and
     (as far as the PR argues it) unverified assumption of the whole change;
  4. does each of them actually RENDER, so "enumerates a device" is not mistaken
     for "works".

Those four are facts about the HOST and the ARCHIVES. They are expected to be
IDENTICAL at the base and the head, because `studio/install_sd_cpp_prebuilt.py` is
byte-identical across the PR and the prebuilts do not know which checkout
downloaded them. That identity is not a weakness of the run and is not dressed up
as a defect differential: it is the finding. What the states are compared on is
the only thing that does differ -- what each checkout's own backend SELECTS and
RECORDS when handed those same real binaries and that same real error text.

MODE is "regression" for that reason. The vocabulary (CONFIRMED / VOID /
FIX_INCOMPLETE / NO_REGRESSION / REGRESSION / INCONCLUSIVE) has no word for "an
observation that bounds a design decision", and NO_REGRESSION is the least
misleading of the six: it is true, it is what the state comparison measures, and
it does not claim the defect was reproduced or repaired. A reader who wants the
observation reads the tables, which is why they are printed before the verdict.

Pairs with probes/sd_cpp_prebuilt_runtime_probe.py.
"""

from __future__ import annotations

TITLE = "The upstream sd.cpp ROCm and Vulkan prebuilts on a real gfx1151 (observation)"
MODE = "regression"

# Authored, not computed: every capability this CHANGE touches. The two reporter
# cards are in here precisely because this host is neither of them, and
# `no_rocm_runtime_host` is in here because a pool of ROCm development boxes
# cannot observe what the ROCm archive does on a machine with no ROCm at all --
# which is the population the new rung exists to serve.
NEEDS = ["gpu", "rocm", "vulkan", "amd_smi", "discrete_gpu",
         "rocm_runtime_libs", "no_rocm_runtime_host",
         "gfx1201_rdna4", "gfx1100_rdna3",
         "windows", "windows_rocm_wddm", "windows_docker", "amdvlk",
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

    # The dependency half must have been READ, or questions 1 and 2 are answered by
    # an empty dict and the report would quietly say nothing while looking complete.
    dep = (shared.get("dependencies") or {}).get("rocm") or {}
    read = bool(dep.get("needed_by")) or bool(
        [f for f in (dep.get("files") or {}).values() if f.get("ldd")])
    out.append(("the ROCm archive's own dependency list was read from the archive, not "
                "assumed", read,
                f"readelf={dep.get('have_readelf')} ldd={dep.get('have_ldd')} objects="
                f"{dep.get('object_count')} sonames needed by="
                f"{ {k: len(v) for k, v in (dep.get('rocm_sonames_needed') or {}).items()} }"))

    host = shared.get("host_rocm") or {}
    out.append(("the loader's view of the ROCm sonames was really queried on this host",
                host.get("ldconfig") is not None,
                f"ldconfig={host.get('ldconfig')}; resolved="
                f"{ {k: bool(v) for k, v in (host.get('soname_resolution') or {}).items()} }"))

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

    rows.append("**These prebuilt observations are HOST facts, identical at every state by "
                "construction**: the installer is byte-identical across this PR and an "
                "archive does not know which checkout fetched it. The identity IS the "
                "finding; only the per-state selection table further down differs.")
    rows.append("")

    # ---- question 1: the host runtime the ROCm archive depends on.
    host = shared.get("host_rocm") or {}
    rows.append("**Q1. Does the ROCm runtime the archive needs exist on this host?**")
    rows.append("")
    rows.append(f"Host ROCm version file `/opt/rocm/.info/version`: "
                f"`{host.get('/opt/rocm/.info/version')}`. "
                f"`hipconfig --version`: `{str(host.get('hipconfig --version') or '')[:80]}`. "
                f"KFD `gfx_target_version`: `{host.get('kfd_gfx_target_version')}`.")
    rows.append("")
    rows.append("| soname the archive NEEDS | resolved by the loader | `ldconfig -p` entry |")
    rows.append("|---|---|---|")
    for soname, hit in (host.get("soname_resolution") or {}).items():
        rows.append(f"| `{soname}` | {'yes' if hit else 'NO'} | `{(hit or '-')[:110]}` |")
    rows.append("")

    rows.append(f"`LD_LIBRARY_PATH` in the probe's environment: "
                f"`{host.get('LD_LIBRARY_PATH') or '(unset)'}`. ROCm lines in the ldconfig "
                f"cache: {len(host.get('ldconfig_rocm_lines') or [])}. "
                f"`/etc/ld.so.conf.d` entries mentioning ROCm: "
                f"{sorted(k for k, v in (host.get('ld_so_conf_d') or {}).items() if 'rocm' in (k + v).lower()) or 'none'}.")
    rows.append("")

    for accel in ("rocm", "vulkan"):
        dep = (shared.get("dependencies") or {}).get(accel) or {}
        if not dep or dep.get("error"):
            continue
        needed = {k: v for k, v in (dep.get("rocm_sonames_needed") or {}).items() if v}
        unresolved = {k: v for k, v in (dep.get("rocm_sonames_unresolved") or {}).items() if v}
        rows.append(f"`{accel}` archive: {dep.get('object_count')} ELF objects. ROCm sonames "
                    f"NEEDED by something in it: "
                    f"{ {k: v[0] for k, v in needed.items()} or 'none'}. Unresolved on this "
                    f"host: {sorted(unresolved) or 'none'}. Bundled ROCm runtime libraries: "
                    f"{dep.get('bundles_any_rocm_runtime') or 'none'}.")
        for name, entry in sorted((dep.get("files") or {}).items()):
            gfx = entry.get("gfx") or {}
            if not gfx.get("targets"):
                continue
            here = [t for t in gfx["targets"] if t in ("gfx1151", "gfx1100", "gfx1201")]
            rows.append(f"    `{name}`: RUNPATH {entry.get('runpath')}, "
                        f"{len(gfx['targets'])} gfx targets carrying {gfx.get('entries')} "
                        f"offload-bundle entries; of the cards at issue it carries {here}. "
                        f"Targets: {', '.join(gfx['targets'])}.")
            rows.append(f"    `{name}` NEEDED: {entry.get('needed')}. The loader resolves "
                        f"them to: "
                        f"{[ln.strip() for ln in (entry.get('ldd') or '').splitlines() if any(s in ln for s in ('hipblas', 'rocblas', 'amdhip64'))]}.")
            if "ldd_clean_env_missing" in entry:
                clean = entry["ldd_clean_env_missing"] or [
                    "nothing, so the resolution is a property of the MACHINE and not of "
                    "this job's prepared environment"]
                rows.append("    With `LD_LIBRARY_PATH` removed from the environment, "
                            "unresolved becomes: " + "; ".join(str(x) for x in clean) + ".")
        rows.append("")

    rows.append("**What the real card did** "
                f"(node `{(shared.get('platform') or {}).get('node')}`, "
                f"{(shared.get('platform') or {}).get('system')} "
                f"{(shared.get('platform') or {}).get('machine')})")
    rows.append("")
    rows.append("")
    rows.append("**Q2 (rocm row) and Q3 (vulkan row). Does each sd-cli start and enumerate "
                "this GPU?**")
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

    rows.append("**Q4. Does each build actually RENDER, rather than merely enumerate?** "
                "4 steps, 256x256, SD1.5 fp16 and the Q4_0 GGUF.")
    rows.append("")
    rows.append("| generation | rc | seconds | image bytes | PNG | device lines |")
    rows.append("|---|---|---|---|---|---|")
    for key, g in sorted((shared.get("generations") or {}).items()):
        img = g.get("image") or {}
        dev = "; ".join(g.get("device_lines") or [])[:160].replace("|", "/")
        rows.append(f"| {key} | {g.get('rc')} | {g.get('seconds')} | {img.get('bytes')} | "
                    f"{img.get('png_magic')} | {dev or '-'} |")
    rows.append("")

    rows.append("**The only part that CAN differ between the states**: what each checkout's "
                "own backend selects and records, handed the identical binaries above.")
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

    host = _shared(obs).get("host_rocm") or {}
    resolved = {k: bool(v) for k, v in (host.get("soname_resolution") or {}).items()}
    # Two readings of the same question, and they can legitimately disagree: the
    # ldconfig CACHE is a host-wide index, while `ldd` is what the process actually
    # gets. The loader's answer is the operative one; the cache line is reported
    # beside it rather than being silently preferred either way.
    dep = (_shared(obs).get("dependencies") or {}).get("rocm") or {}
    loader_missing = sorted(k for k, v in (dep.get("rocm_sonames_unresolved") or {}).items() if v)
    cache_missing = sorted(k for k, v in resolved.items() if not v)
    all_resolved = not loader_missing
    if all_resolved and not cache_missing:
        runtime = "every ROCm soname the archive needs resolves here, cache and loader agreeing"
    elif all_resolved:
        runtime = (f"the loader resolves every ROCm soname the archive needs, though the "
                   f"ldconfig cache does not itself list {cache_missing}")
    else:
        runtime = f"the loader cannot resolve {loader_missing} for the archive"
    dependency_reading = (
        "so on this host neither half of the dependency is missing and the defect the PR "
        "targets cannot reproduce" if all_resolved else
        "so the host runtime, not the kernel coverage, is what is missing here")

    if rocm_ok:
        return False, (
            f"OBSERVATION, reported under the least misleading verdict this vocabulary has. "
            f"The generic ROCm sd.cpp prebuilt RUNS on this gfx1151: it enumerated the card and "
            f"rendered a real image, {'and Vulkan did too' if vulkan_ok else 'while Vulkan did not'}. "
            f"{runtime}, and the archive's own HIP fatbinary carries gfx1151 -- {dependency_reading}. "
            f"The state comparison, which is the only thing that differs here, shows "
            f"the head selecting `{sel_head}` where the base selects `{sel_base}` with an empty "
            f"record, so the new rung does not divert a host that does not need it. This bounds "
            f"the change; it is not evidence about gfx1201, about gfx1100, or about an AMD "
            f"machine with no ROCm runtime at all")

    if vulkan_ok:
        return False, (
            f"OBSERVATION. The generic ROCm prebuilt does NOT render on this gfx1151 while the "
            f"Vulkan one does, which is the shape #9278 and #8814 report on other cards. On this "
            f"host {runtime}. The head selects `{sel_head}` on a clean record and the base "
            f"selects `{sel_base}`; see the marker table for whether the real error text lands "
            f"in the PR's tiers. Not worse than base")

    return False, ("neither build rendered, so there is nothing for the head to be worse at; "
                   "the gate above should have caught this")
