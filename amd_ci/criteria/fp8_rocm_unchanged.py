#!/usr/bin/env python3
"""Criteria: PR 10790 must change nothing on ROCm.

This is deliberately a REGRESSION question, not a differential one. The defect the PR
fixes is a triton compile failure for fp8e4nv below sm89, which is an NVIDIA capability
gate; AMD's triton backend lists fp8e4nv unconditionally, and gfx1151 reports a
capability well above the threshold anyway. So the base does not exhibit the defect here
and never could, which under the one rule would make a differential run VOID rather than
a pass. The answerable question on this host is whether the change perturbs ROCm at all.

It also pins the clause that IS ROCm-specific: the guard reads torch.version.hip, so the
probe records that it is genuinely non-None on this host and that routing is therefore
identical at every capability including spoofed pre-sm89 ones. Without that, the hip
clause would be asserted only from source.

Pairs with probes/fp8_rocm_routing_probe.py.
"""

from __future__ import annotations

TITLE = "fp8 block dequant routing on ROCm, base versus head"
MODE = "regression"
NEEDS = ["rocm", "gpu"]


def _caps(o: dict) -> list[str]:
    return sorted(set(o.get("routes", {})) | set(o.get("checksums", {})) | set(o.get("errors", {})))


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        err = o.get("error")
        out.append((f"{name} probe imported unsloth", not err, err or f"fp8 from {o.get('fp8_file')}"))

        hip = o.get("hip")
        out.append((f"{name} really ran on a ROCm build", bool(hip),
                    f"torch.version.hip = {hip!r}; a None here would mean this ran on CUDA "
                    f"and the ROCm claim is unbacked"))

        # Compare against what the probe asked for, not a hardcoded count: the real card's
        # capability is one of the entries, so the requested set dedupes on some hosts.
        routes = o.get("routes") or {}
        want = set(o.get("caps_requested") or [])
        missing = sorted(want - set(routes))
        out.append((f"{name} observed routing at every capability",
                    bool(want) and not missing,
                    f"{len(routes)}/{len(want)} observed: {sorted(routes)}"
                    + (f"; MISSING {missing}" if missing else "")))

        errs = o.get("errors") or {}
        out.append((f"{name} evaluated without kernel errors", not errs,
                    "; ".join(f"{k}: {v}" for k, v in errs.items()) or "none"))
    return out


def table(obs: dict) -> str:
    caps = _caps(obs.get("head") or {})
    rows = ["| capability | base route | head route | base checksum | head checksum |",
            "|---|---|---|---|---|"]
    for c in caps:
        b, h = obs.get("base") or {}, obs.get("head") or {}
        rows.append(f"| {c} | {b.get('routes', {}).get(c, '-')} | {h.get('routes', {}).get(c, '-')} "
                    f"| {b.get('checksums', {}).get(c, '-')} | {h.get('checksums', {}).get(c, '-')} |")
    h = obs.get("head") or {}
    extra = [
        "",
        f"Card: `{h.get('gpu')}`, real capability `{h.get('real_capability')}`, "
        f"`torch.version.hip = {h.get('hip')!r}`, torch `{h.get('torch')}`.",
        "",
        "What each predicate would divert here (the head guard is the second row; both are "
        "all-False on ROCm because the `torch.version.hip is None` clause short-circuits):",
        "",
        f"- `capability[0] < 9`: `{h.get('major_only_would_divert')}`",
        f"- `capability < (8, 9)`: `{h.get('full_tuple_would_divert')}`",
    ]
    return "\n".join(rows + extra)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    diffs = []
    for cap in sorted(set(base.get("routes", {})) | set(head.get("routes", {}))):
        b, h = base.get("routes", {}).get(cap), head.get("routes", {}).get(cap)
        if b != h:
            diffs.append(f"{cap}: base routed {b}, head routed {h}")
    for cap in sorted(set(base.get("checksums", {})) | set(head.get("checksums", {}))):
        b, h = base.get("checksums", {}).get(cap), head.get("checksums", {}).get(cap)
        if b != h:
            diffs.append(f"{cap}: base checksum {b}, head checksum {h}")
    new_errs = sorted(set(head.get("errors", {})) - set(base.get("errors", {})))
    if new_errs:
        diffs.append("new errors at the head: " + ", ".join(new_errs))
    if diffs:
        return True, "; ".join(diffs)
    n = len(head.get("routes", {}))
    return False, (f"identical routing and identical values at all {n} capabilities, including "
                   f"spoofed pre-sm89 ones: on ROCm this change is a no-op, as the "
                   f"torch.version.hip clause intends")
