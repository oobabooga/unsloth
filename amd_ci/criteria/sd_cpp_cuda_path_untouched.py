#!/usr/bin/env python3
"""Criteria: does unsloth#11068 leave a real NVIDIA host's diffusion path alone?

The other half of the non-regression claim, and the half no spoofed matrix can
make. The PR adds a rung below ROCm; the eleven corners of its own table that do
not move are asserted from stubs. On a machine with real CUDA cards the same
question is measurable: the sd.cpp CUDA prebuilt is installed and really renders,
and then each checkout is asked what it selects, what it reads from that binary,
and -- the part a table cannot show -- what reaching that answer COSTS.

"Untouched" is taken literally here:

  * the head must select `cuda`, exactly as the base does;
  * it must read the real CUDA binary the same way the base does;
  * and it must get there without consulting the fingerprint or the settings
    store even once, because those are a database read on every diffusion load
    for a host the feature is not about.

Pairs with probes/sd_cpp_accelerator_reality_probe.py, run with
`--accelerators cuda,vulkan --host-accelerator cuda`.
"""

from __future__ import annotations

TITLE = "The CUDA diffusion path on real NVIDIA cards, base versus head"
MODE = "regression"

NEEDS = ["gpu", "nvidia", "multi_gpu", "discrete_gpu",
         "rocm", "gfx1201_rdna4", "gfx1100_rdna3", "windows", "mlx", "xpu", "amdvlk"]


def _states(obs: dict) -> dict:
    return {n: v for n, v in obs.items() if not n.startswith("_")}


def _shared(obs: dict) -> dict:
    for v in _states(obs).values():
        if v.get("shared"):
            return v["shared"]
    return {}


def _val(entry):
    return entry.get("value") if isinstance(entry, dict) else None


def _rendered(g: dict) -> bool:
    img = g.get("image") or {}
    return g.get("rc") == 0 and bool(img.get("png_magic")) and int(img.get("bytes") or 0) > 2000


def _reads(state: dict) -> dict:
    out = {}
    for accel, r in (state.get("reads") or {}).items():
        out[accel] = (_val(r.get("verdict")), _val(r.get("lists_accelerator")))
    return out


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    states = _states(obs)
    shared = _shared(obs)
    for name, v in states.items():
        mod = v.get("module") or {}
        out.append((f"{name}: sd_cpp_backend imported from its own checkout",
                    bool(mod.get("imported")), str(mod.get("error") or mod.get("file"))[:200]))

    host = shared.get("host_accelerator")
    out.append(("the probe was run for the CUDA host path", host == "cuda", f"host={host}"))

    b = (shared.get("bundles") or {}).get("cuda") or {}
    out.append(("the CUDA sd.cpp prebuilt really installed", bool(b.get("cli")),
                f"asset={b.get('asset_resolved')} rc={b.get('install_rc')} cli={b.get('cli')}"))

    gens = {k: g for k, g in (shared.get("generations") or {}).items() if k.startswith("cuda:")}
    rendered = any(_rendered(g) for g in gens.values())
    out.append(("the CUDA build rendered a real image on a card of this box, so the reads "
                "below are about a binary that works here", rendered,
                "; ".join(f"{k} rc={g.get('rc')} bytes={(g.get('image') or {}).get('bytes')} "
                          f"{g.get('seconds')}s" for k, g in sorted(gens.items())) or "none"))

    inv = {}
    for v in states.values():
        inv = ((v.get("fingerprint") or {}).get("inventory_nonblocking") or {})
        if inv:
            break
    real = bool(inv.get("names")) and "nvidia" in [str(x) for x in (inv.get("vendors") or [])]
    out.append(("the host enumerates REAL NVIDIA cards, so the fingerprint and the routes are "
                "reading hardware and not a stub", real,
                f"vendors={inv.get('vendors')} n={len(inv.get('names') or [])} "
                f"unknown={inv.get('unknown')}"))

    smoke = any((v.get("shared") or {}).get("smoke") for v in states.values())
    out.append(("not a harness smoke run", not smoke, f"smoke={smoke}"))
    return out


def table(obs: dict) -> str:
    shared = _shared(obs)
    rows = ["| generation | rc | seconds | bytes | PNG | device |", "|---|---|---|---|---|---|"]
    for key, g in sorted((shared.get("generations") or {}).items()):
        img = g.get("image") or {}
        dev = "; ".join(g.get("device_lines") or [])[:120].replace("|", "/")
        rows.append(f"| {key} | {g.get('rc')} | {g.get('seconds')} | {img.get('bytes')} | "
                    f"{img.get('png_magic')} | {dev or '-'} |")
    rows += ["", "| state | selects | reads (verdict, lists) | fingerprint calls | "
                 "settings reads | record written for its own accelerator |",
             "|---|---|---|---|---|---|"]
    for name, v in _states(obs).items():
        iso = v.get("isolation") or {}
        calls = iso.get("calls_for_own_accelerator") or {}
        stored = _val(iso.get("stored_after_noting_own"))
        rows.append(f"| {name} | `{_val(iso.get('selected')) or 'n/a (no preference layer)'}` | "
                    f"{_reads(v)} | {calls.get('fingerprint', 'n/a')} | "
                    f"{calls.get('stored', 'n/a')} | {stored if stored is not None else 'n/a'} |")
    rows.append("")
    for name, v in _states(obs).items():
        iso = v.get("isolation") or {}
        if not iso.get("present"):
            continue
        rows.append(f"On `{name}`, after a ROCm record is written on this NVIDIA box, "
                    f"`preferred_accelerator('cuda')` is "
                    f"`{_val(iso.get('selected_own_after_rocm_note'))}`; the reported state is "
                    f"{_val(iso.get('state'))}.")
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    hi = head.get("isolation") or {}
    sel = _val(hi.get("selected"))

    if (base.get("module") or {}).get("imported") and not (head.get("module") or {}).get("imported"):
        return True, f"the head does not import here: {(head.get('module') or {}).get('error')}"

    if hi.get("present") and sel != "cuda":
        return True, f"the head selects `{sel}` for a CUDA host where the base selects `cuda`"

    if _reads(base) != _reads(head):
        return True, (f"the two checkouts read the same real binaries differently: base "
                      f"{_reads(base)} vs head {_reads(head)}")

    calls = hi.get("calls_for_own_accelerator") or {}
    if calls.get("fingerprint") or calls.get("stored"):
        return True, (f"selecting `cuda` on the head consults the fingerprint "
                      f"{calls.get('fingerprint')} time(s) and the settings store "
                      f"{calls.get('stored')} time(s); on the base it consults neither, so "
                      f"every diffusion load on an NVIDIA host would pay a new read")

    stored = _val(hi.get("stored_after_noting_own"))
    if stored:
        return True, (f"the head persisted a fallback record for a CUDA host: {stored}. There is "
                      f"no rung below cuda, so nothing would ever read it")

    if _val(hi.get("selected_own_after_rocm_note")) != "cuda":
        return True, ("a ROCm record written on this box moved the CUDA selection to "
                      f"`{_val(hi.get('selected_own_after_rocm_note'))}`")

    return False, (
        f"on real NVIDIA hardware the head selects `cuda` exactly as the base does, reads the "
        f"real CUDA sd.cpp binary identically ({_reads(head)}), reaches that answer with "
        f"{calls.get('fingerprint')} fingerprint reads and {calls.get('stored')} settings reads, "
        f"writes no record for an accelerator with no rung below it, and is unmoved by a ROCm "
        f"record written on the same machine. The CUDA path is untouched here; that is a "
        f"statement about this box's cards, not about any AMD card")
