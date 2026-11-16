#!/usr/bin/env python3
"""Criteria: diffusion LoRA training end to end on this GPU (judges probes/diffusion_lora_train_probe.py).

Differential. The defect is "Studio cannot train a LoRA for the target family on this host":

base_shows_defect: the base refuses the target family for training (``normalized()`` raises), which is what a
commit before the family's training PR does.

head_is_fixed: the head accepts the family and, for EVERY precision ``train_precision_modes()`` offers here among the
ones the probe was asked to train (bf16 / nf4), the run
  - completed every step (a ``complete`` event, not stopped, steps_run == configured steps, no error event),
  - logged only finite losses,
  - saved an adapter whose lora_B tensors are finite and not all zero (it learned something),
  - published it to the Images LoRA catalog,
and the deploy pipeline, loaded fresh, rendered with that LoRA by catalog id differently from the no-LoRA render by
more than the no-LoRA A/A pair (LPIPS > max(0.02, 3 x A/A)), and the export route's copy was byte-identical,
rediscovered by id with the family tags intact, and rendered the same image as the catalog copy
(LPIPS <= A/A + 0.01).

"Loss falls" is tabulated (first-quarter mean -> last-quarter mean), not gated: per-step flow-matching loss is
dominated by the sampled timestep, so a short run's trend is noise-limited, and gating on it would make the verdict
a coin flip. The control family (already trainable) is tabulated, not judged, so a failure can be attributed.
Speed and peak memory are reported, never judged (the job may run with --shared-gpu).
"""

from __future__ import annotations

import math

TITLE = "gfx1151: diffusion LoRA training, reload, deploy and export through Studio's own code"
MODE = "differential"
NEEDS = ["gpu", "rocm", "linux", "exclusive_gpu", "nvidia", "discrete_gpu", "windows", "windows_rocm_wddm"]
TRAINED = ("bf16", "nf4")


def _fam(state: dict, role: str = "target") -> dict:
    return ((state or {}).get("families") or {}).get(role) or {}


def _q(xs: list, first: bool) -> float | None:
    xs = [x for x in xs if isinstance(x, (int, float))]
    if len(xs) < 4:
        return None
    k = max(1, len(xs) // 4)
    part = xs[:k] if first else xs[-k:]
    return sum(part) / len(part)


def _cell_ok(cell: dict, steps: int) -> tuple[bool, str]:
    if not cell:
        return False, "not run"
    if cell.get("stage_error") or cell.get("raised"):
        return False, str(cell.get("stage_error") or cell.get("raised"))[:300]
    if cell.get("errors"):
        return False, "error event: " + str(cell["errors"][0])[:300]
    c = cell.get("complete") or {}
    if not c:
        return False, f"no complete event (rc={cell.get('_rc')})"
    if c.get("stopped") or c.get("steps_run") != steps:
        return False, f"stopped={c.get('stopped')} steps_run={c.get('steps_run')} of {steps}"
    losses = cell.get("losses") or []
    if len(losses) != steps or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in losses):
        return False, f"{len(losses)} losses logged, finite={all(isinstance(x, (int, float)) and math.isfinite(x) for x in losses)}"
    n = cell.get("lora_norms") or {}
    if not n.get("finite") or not n.get("lora_B_nonzero"):
        return False, f"adapter tensors: {n}"
    if not cell.get("catalog_id"):
        return False, "not published to the LoRA catalog"
    return True, "trained"


def _render_ok(render: dict, lid: str) -> tuple[bool, str]:
    if not render or render.get("stage_error") or render.get("skipped"):
        return False, str((render or {}).get("stage_error") or (render or {}).get("skipped") or "no render stage")[:300]
    rs = render.get("renders") or {}
    for tag in ("none", "none_repeat", f"lora_{lid}", f"export_{lid}"):
        if not (rs.get(tag) or {}).get("ok"):
            return False, f"render {tag}: {(rs.get(tag) or {}).get('error', 'missing')}"[:300]
    cmp = render.get("compare") or {}
    # LPIPS when the scorer installed; else mean abs pixel difference (0-255) with its own floors.
    metric = "lpips" if (cmp.get("aa_none_vs_repeat") or {}).get("lpips") is not None else "mad"
    floor, slack = (0.02, 0.01) if metric == "lpips" else (2.0, 0.5)
    aa = (cmp.get("aa_none_vs_repeat") or {}).get(metric)
    lv = (cmp.get(f"lora_vs_none::{lid}") or {}).get(metric)
    ex = (cmp.get(f"export_vs_lora::{lid}") or {}).get(metric)
    if aa is None or lv is None or ex is None:
        return False, f"{metric} missing (aa={aa}, lora={lv}, export={ex}; {render.get('lpips_error', '')})"
    if not lv > max(floor, 3 * aa):
        return False, f"LoRA does not change the render beyond noise: {metric} {lv} vs A/A {aa}"
    info = (render.get("loras") or {}).get(lid) or {}
    if not info.get("export_bytes_identical") or not info.get("rediscovered_id"):
        return False, f"export round trip: {info.get('export_error') or info}"[:300]
    if set(info.get("rediscovered_families") or []) != set((info.get("catalog_entry") or {}).get("families") or []):
        return False, "export lost the family tags"
    if not ex <= aa + slack:
        return False, f"exported copy renders differently: {metric} {ex} vs A/A {aa}"
    return True, "ok"


def _cells_to_judge(state: dict) -> list[str]:
    offered = ((state or {}).get("host") or {}).get("train_precision_modes") or []
    return [p for p in TRAINED if p in offered]


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        host = o.get("host") or {}
        ran = o.get("_probe_rc") == 0 and not o.get("_missing_output") and not o.get("_parse_error")
        out.append((f"{name}: probe ran", ran, f"rc={o.get('_probe_rc')} {o.get('_parse_error', '')}"))
        hok = bool(host) and not host.get("stage_error") and host.get("cuda_available")
        out.append((f"{name}: Studio imported on a visible GPU", hok,
                    f"{host.get('device_name')} {host.get('arch')} torch {host.get('torch')} hip {host.get('hip')} "
                    f"{str(host.get('stage_error', ''))[:200]}"))
        acc = _fam(o).get("accepted")
        out.append((f"{name}: target family decision observed", acc in (True, False),
                    f"accepted={acc} {str(_fam(o).get('refusal') or '')[:200]}"))
    head = obs.get("head") or {}
    hh = head.get("host") or {}
    out.append(("head: gfx1151 ROCm build", bool(hh.get("hip")) and str(hh.get("arch", "")).startswith("gfx1151"),
                f"arch={hh.get('arch')} hip={hh.get('hip')}"))
    ds = (head.get("dataset") or {}).get("images") or 0
    out.append(("head: training images present", ds >= 4, f"{ds} images ({head.get('dataset_error', '')})"))
    judged = _cells_to_judge(head)
    out.append(("head: at least one precision offered and attempted", bool(judged)
                and all(p in (_fam(head).get("cells") or {}) for p in judged),
                f"offered here {hh.get('train_precision_modes')}; attempted {sorted((_fam(head).get('cells') or {}))};"
                f" skipped {_fam(head).get('skipped')}"))
    return out


def base_shows_defect(base: dict):
    f = _fam(base)
    return (f.get("accepted") is False, f"base refusal: {f.get('refusal')}")


def head_is_fixed(head: dict):
    f = _fam(head)
    if not f.get("accepted"):
        return (False, f"head refuses: {f.get('refusal')}")
    steps = head.get("steps")
    for p in _cells_to_judge(head):
        cell = (f.get("cells") or {}).get(p) or {}
        ok, why = _cell_ok(cell, steps)
        if not ok:
            return (False, f"{p}: {why}")
        ok, why = _render_ok(f.get("render") or {}, cell.get("catalog_id"))
        if not ok:
            return (False, f"{p}: {why}")
    return (True, "every offered precision trained, reloaded, changed the render and round-tripped through export")


def _fmt(x, nd = 3):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def table(obs: dict) -> str:
    parts = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        h = o.get("host") or {}
        parts.append(f"**{name}**: torch {h.get('torch')} hip {h.get('hip')} {h.get('device_name')} {h.get('arch')}; "
                     f"mem_get_info {h.get('mem_get_info_gb')} GB, trusted {h.get('trusted_mem_get_info_gb')}, "
                     f"host {h.get('meminfo')} GiB, disk free {h.get('disk_free_gb')} GB; triton "
                     f"{h.get('triton_version')}, bitsandbytes {h.get('bitsandbytes_version')}, torchao "
                     f"{h.get('torchao_version')} (functional {h.get('has_functional_torchao')}), diffusers "
                     f"{h.get('diffusers_version')}; train_precision_modes {h.get('train_precision_modes')} "
                     f"(recommended {h.get('train_precision_recommended')}); kernel check {h.get('kernel_check')}")
        parts.append("")
    rows = ["| state | role | family | precision | offered | preflight | trained | loss q1 -> q4 | fell | samples/s "
            "| trainer peak GB | proc max reserved GB | host min avail GiB | compiled | auto pick | LPIPS lora vs none "
            "| A/A | export ok (LPIPS vs catalog) | wall s | note |",
            "|" + "---|" * 20]
    for name in ("base", "head"):
        o = obs.get(name) or {}
        offered = ((o.get("host") or {}).get("train_precision_modes")) or []
        for role in ("target", "control"):
            f = _fam(o, role)
            if not f:
                continue
            fam = f.get("resolved_family") or "-"
            if not f.get("accepted"):
                rows.append(f"| {name} | {role} | {fam} | - | - | - | refused | - | - | - | - | - | - | - | - | - | - "
                            f"| - | - | {str(f.get('refusal') or '')[:160]} |")
                continue
            pre = f.get("preflight") or {}
            render = f.get("render") or {}
            cmp = render.get("compare") or {}
            aa = (cmp.get("aa_none_vs_repeat") or {}).get("lpips")
            if aa is None:
                aa = (cmp.get("aa_none_vs_repeat") or {}).get("mad")
            for p in ("bf16", "bf16-eager", "nf4", "int8", "fp8", "mxfp8", "auto"):
                cell = (f.get("cells") or {}).get(p)
                if p == "bf16-eager" and p not in (f.get("cells") or {}):
                    continue
                pf = pre.get(p.split("-")[0])
                pf_s = "ok" if pf is None else str(pf)[:70]
                if not cell:
                    why = next((s["why"] for s in f.get("skipped") or [] if s.get("precision") == p), "")
                    rows.append(f"| {name} | {role} | {fam} | {p} | {'yes' if p.split('-')[0] in offered else 'no'} | {pf_s} "
                                f"| not run | - | - | - | - | - | - | - | - | - | - | - | - | {why} |")
                    continue
                ok, why = _cell_ok(cell, o.get("steps"))
                q1, q4 = _q(cell.get("losses") or [], True), _q(cell.get("losses") or [], False)
                fell = "-" if q1 is None or q4 is None else ("yes" if q4 < q1 else "no")
                lid = cell.get("catalog_id")
                lv = (cmp.get(f"lora_vs_none::{lid}") or {}).get("lpips")
                ex = (cmp.get(f"export_vs_lora::{lid}") or {}).get("lpips")
                if lv is None:
                    lv = (cmp.get(f"lora_vs_none::{lid}") or {}).get("mad")
                    ex = (cmp.get(f"export_vs_lora::{lid}") or {}).get("mad")
                rok, rwhy = _render_ok(render, lid) if lid else (False, "no adapter")
                info = (render.get("loras") or {}).get(lid) or {}
                exp = "-" if not info else (f"{'yes' if info.get('export_bytes_identical') and info.get('rediscovered_id') else 'NO'}"
                                            f" ({_fmt(ex, 4)})")
                note = why if not ok else ("" if rok else rwhy)
                rows.append(
                    f"| {name} | {role} | {fam} | {p} | {'yes' if p.split('-')[0] in offered else 'no'} | {pf_s} "
                    f"| {'yes' if ok else 'NO'} | {_fmt(q1)} -> {_fmt(q4)} | {fell} "
                    f"| {_fmt(cell.get('samples_per_second'))} | {_fmt(cell.get('trainer_peak_memory_gb'))} "
                    f"| {_fmt(cell.get('max_reserved_gb'))} | {_fmt(cell.get('host_min_available_gib'))} "
                    f"| {cell.get('compiled')} | {cell.get('auto_pick')} | {_fmt(lv, 4)} | {_fmt(aa, 4)} | {exp} "
                    f"| {cell.get('wall_s')} | {str(note)[:200]} |")
    parts.append("\n".join(rows))
    parts.append("")
    for name in ("base", "head"):
        o = obs.get(name) or {}
        for role in ("target", "control"):
            r = _fam(o, role).get("render") or {}
            if r:
                parts.append(f"- {name} {role} deploy `{r.get('deploy')}`: load {r.get('load_s')} s, "
                             f"{r.get('load_status')}, renders "
                             f"{ {k: v.get('s') if v.get('ok') else 'FAILED' for k, v in (r.get('renders') or {}).items()} }, "
                             f"max reserved {r.get('max_reserved_gb')} GB, compare {r.get('compare')}, "
                             f"{str(r.get('stage_error') or r.get('lpips_error') or '')[:300]}")
            for p, cell in (_fam(o, role).get("cells") or {}).items():
                if cell.get("warnings"):
                    parts.append(f"- {name} {role} {p} warnings: {cell['warnings'][:4]}")
    return "\n".join(parts)
