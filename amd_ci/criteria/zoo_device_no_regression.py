#!/usr/bin/env python3
"""Criteria: is an unsloth-zoo backend-dispatch change a strict no-op on THIS host?

Pairs with probes/zoo_device_probe.py. Regression mode. NO_REGRESSION requires, base
versus head on the same host in the same job:

  * identical device decisions: DEVICE_TYPE, DEVICE_TYPE_TORCH, DEVICE_COUNT,
    device_is_bf16_supported(), is_hip(), and the module-level dispatch objects the
    change edits (gradient_checkpointing.torch_gpu_stream, the custom_fwd/bwd autocast
    device, loss_utils.current_device);
  * identical Unsloth gradient-checkpointing state after
    initialize_unsloth_gradient_checkpointing() and after
    reset_unsloth_gradient_checkpointing_buffers(): buffer count, shape, dtype, device,
    stream and event types;
  * device_synchronize / device_empty_cache, vllm_utils.get_mem_info and (head)
    _device_empty_cache complete without error wherever the base's did;
  * npu_is_available() is False at the head (the new probe must not fire on HIP);
  * the LoRA SFT leg gives finite losses at both states for every step, head within
    ABS_TOL + REL_TOL * |base| of base per step (same seed, same data, same env);
  * unsloth_train completes at the head if it completed at the base, with finite
    losses within the same tolerance.

A symbol the base does not have (npu_is_available, vllm_utils._device_empty_cache,
gradient_checkpointing._amp_device_type) is compared by its EFFECTIVE value: the
base hard-coded "cuda" as the autocast device, so an absent _amp_device_type reads
as "cuda".

Gates make the comparison non-vacuous: both probes ran, both states imported the
zoo FROM THEIR OWN CHECKOUT (not the environment's installed copy), the base
really is a HIP host, the base allocated at least one GC buffer on the GPU, and
the base training leg produced the requested number of finite losses.
"""

from __future__ import annotations

import math

TITLE = "unsloth-zoo device dispatch and a short LoRA run on real gfx1151, base versus head"
MODE = "regression"
# What the change touches: the HIP path (measured here), the CUDA and XPU paths it
# edits alongside, the new Ascend NPU path itself, per-device loops (multi-GPU),
# and the discrete-GPU double-buffer branch, which is auto-off on this unified-memory
# APU, so its event_ctor dispatch never runs here.
NEEDS = ["linux", "rocm", "gpu", "nvidia", "xpu", "npu", "multi_gpu", "discrete_gpu", "windows"]

EXPECT_DEVICE_TYPE = "hip"
ABS_TOL = 0.05
REL_TOL = 0.05


def _dev(o: dict) -> dict:
    return ((o or {}).get("device")) or {}


def _train(o: dict) -> dict:
    return ((o or {}).get("train")) or {}


def _finite(xs) -> bool:
    return bool(xs) and all(isinstance(x, (int, float)) and math.isfinite(x) for x in xs)


def _gc_view(summary: dict | None) -> dict:
    s = summary or {}
    keep = ("GPU_BUFFERS", "GPU_BUFFERS_B", "CPU_BUFFERS", "BUFFER_EVENTS_A",
            "BUFFER_EVENTS_B", "EXTRA_STREAMS", "MAIN_STREAMS", "USE_DOUBLE_BUFFER",
            "MINIMUM_SIZE", "USE_UNSLOTH_GC", "NEXT_BUFFER_SLOT", "CPU_INDEX")
    return {k: s.get(k) for k in keep}


def _status(v) -> str:
    """ok / absent / <ExceptionType> for a call outcome."""
    if v == "ok" or v == "absent":
        return v
    if isinstance(v, dict) and "error" in v:
        return "error " + str(v["error"]).split(":", 1)[0]
    return str(v)


def compared(o: dict) -> dict:
    """Fields that must be IDENTICAL between states."""
    d = _dev(o)
    disp = d.get("dispatch") or {}
    amp = disp.get("gradient_checkpointing._amp_device_type")
    gc = d.get("gc") or {}
    vl = d.get("vllm_utils") or {}
    vimp = vl.get("import")
    out = {
        "DEVICE_TYPE": d.get("DEVICE_TYPE"),
        "DEVICE_TYPE_TORCH": d.get("DEVICE_TYPE_TORCH"),
        "DEVICE_COUNT": d.get("DEVICE_COUNT"),
        "device_is_bf16_supported()": d.get("device_is_bf16_supported"),
        "is_hip()": d.get("is_hip"),
        "device_synchronize()": _status(d.get("device_synchronize")),
        "device_empty_cache()": _status(d.get("device_empty_cache")),
        "gc torch_gpu_stream": disp.get("gradient_checkpointing.torch_gpu_stream"),
        "gc custom_fwd/bwd device": "cuda" if amp == "absent" else amp,
        "loss_utils.current_device": disp.get("loss_utils.current_device"),
        "gc init error": gc.get("init_error"),
        "gc reset error": gc.get("reset_error"),
        "vllm_utils import": "ok" if vimp == "ok" else ("error " + str(vimp).split(":", 1)[0]),
        "vllm get_mem_info ok": (vl.get("get_mem_info") or {}).get("ok"),
        "vllm get_mem_info total": (vl.get("get_mem_info") or {}).get("total"),
    }
    for phase in ("init", "after_reset"):
        for k, v in _gc_view(gc.get(phase)).items():
            out[f"gc {phase} {k}"] = v
    return out


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        d = _dev(o)
        ok = (o.get("_probe_rc") == 0 and not o.get("_missing_output") and not o.get("_parse_error")
              and d.get("rc") == 0 and not d.get("error") and not d.get("missing_output"))
        out.append((f"{name} device leg wrote observations", ok,
                    f"probe rc={o.get('_probe_rc')}; leg rc={d.get('rc')}"
                    + (f"; {d.get('error')}" if d.get("error") else "")
                    + ("; missing output" if d.get("missing_output") else "")))
        z = d.get("zoo") or {}
        out.append((f"{name} imported unsloth_zoo from its own checkout", bool(z.get("from_checkout")),
                    f"`{z.get('file')}`"))
    b = obs.get("base") or {}
    h = obs.get("head") or {}
    out.append(("base and head are different commits",
                bool(b.get("commit")) and b.get("commit") != h.get("commit"),
                f"{str(b.get('commit'))[:9]} vs {str(h.get('commit'))[:9]}"))
    bd = _dev(b)
    out.append((f"base is a {EXPECT_DEVICE_TYPE} host", bd.get("DEVICE_TYPE") == EXPECT_DEVICE_TYPE,
                f"DEVICE_TYPE={bd.get('DEVICE_TYPE')}; torch hip={(bd.get('torch') or {}).get('hip')}; "
                f"arch={(bd.get('torch') or {}).get('gcn_arch')}"))
    bufs = ((bd.get("gc") or {}).get("init") or {}).get("GPU_BUFFERS") or []
    out.append(("base allocated Unsloth GC buffers on the GPU",
                bool(bufs) and all("error" not in (x or {}) for x in bufs),
                f"{len(bufs)} buffer(s); {(bd.get('gc') or {}).get('init_error') or 'ok'}"))
    bt = _train(b)
    steps = bt.get("steps")
    out.append(("base training leg produced finite losses for every step",
                not bt.get("skipped") and bt.get("rc") == 0 and not bt.get("error")
                and _finite(bt.get("sft_losses")) and len(bt.get("sft_losses") or []) == steps,
                "skipped (no --train)" if bt.get("skipped") else
                f"rc={bt.get('rc')}; losses={bt.get('sft_losses')}"
                + (f"; {bt.get('error')}" if bt.get("error") else "")))
    for name in ("base", "head"):
        t = _train(obs.get(name) or {})
        if t and not t.get("skipped"):
            z = t.get("zoo") or {}
            out.append((f"{name} training leg imported unsloth_zoo from its own checkout",
                        bool(z.get("from_checkout")), f"`{z.get('file')}`"))
    return out


def _close(b: float, h: float) -> bool:
    return abs(h - b) <= ABS_TOL + REL_TOL * abs(b)


def _loss_rows(b: dict, h: dict) -> list[tuple[str, object, object, bool]]:
    rows = []
    bt, ht = _train(b), _train(h)
    rows.append(("sft losses", bt.get("sft_losses"), ht.get("sft_losses"), None))
    bu, hu = bt.get("unsloth_train") or {}, ht.get("unsloth_train") or {}
    rows.append(("unsloth_train ok", bu.get("ok"), hu.get("ok"), None))
    rows.append(("unsloth_train losses", bu.get("losses"), hu.get("losses"), None))
    rows.append(("unsloth_train error", bu.get("error"), hu.get("error"), None))
    rows.append(("train DEVICE_TYPE", bt.get("DEVICE_TYPE"), ht.get("DEVICE_TYPE"), None))
    rows.append(("unsloth commit", b.get("unsloth_commit"), h.get("unsloth_commit"), None))
    return rows


def table(obs: dict) -> str:
    b, h = obs.get("base") or {}, obs.get("head") or {}
    cb, ch = compared(b), compared(h)
    short = lambda v: str(v).replace("|", "\\|")[:180]  # noqa: E731
    rows = ["| field | base | head | same |", "|---|---|---|---|"]
    for k in cb:
        rows.append(f"| {k} | `{short(cb[k])}` | `{short(ch.get(k))}` | "
                    f"{'yes' if cb[k] == ch.get(k) else 'NO'} |")
    hd = _dev(h)
    rows.append(f"| head npu_is_available() | `{short(_dev(b).get('npu_is_available'))}` | "
                f"`{short(hd.get('npu_is_available'))}` | expect head False |")
    vb = (_dev(b).get("vllm_utils") or {}).get("_device_empty_cache")
    vh = (hd.get("vllm_utils") or {}).get("_device_empty_cache")
    rows.append(f"| vllm_utils._device_empty_cache() | `{short(_status(vb))}` | "
                f"`{short(_status(vh))}` | expect head ok/absent |")
    for name, bv, hv, _ in _loss_rows(b, h):
        rows.append(f"| {name} | `{short(bv)}` | `{short(hv)}` | |")
    if "merge" in obs:
        m = obs["merge"] or {}
        cm = compared(m)
        moved = [k for k in cb if cb[k] != cm.get(k)]
        rows.append(f"| merge state vs base | | | {'identical' if not moved else 'moved: ' + ', '.join(moved)} "
                    f"(sft losses {_train(m).get('sft_losses')}) |")
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    findings: list[str] = []
    notes: list[str] = []
    cb, ch = compared(base), compared(head)
    moved = [k for k in cb if cb[k] != ch.get(k)]
    if moved:
        findings.append("moved at the head: " + ", ".join(f"`{k}`" for k in moved))
    hd = _dev(head)
    if hd.get("npu_is_available") not in (False, "absent"):
        findings.append(f"head npu_is_available() = `{hd.get('npu_is_available')}` on a non-NPU host")
    vh = (hd.get("vllm_utils") or {}).get("_device_empty_cache")
    if (hd.get("vllm_utils") or {}).get("import") == "ok" and _status(vh) not in ("ok", "absent"):
        findings.append(f"head vllm_utils._device_empty_cache() failed: {vh}")

    bt, ht = _train(base), _train(head)
    steps = bt.get("steps")
    hl, bl = ht.get("sft_losses") or [], bt.get("sft_losses") or []
    if ht.get("rc") != 0 or ht.get("error") or not _finite(hl) or len(hl) != steps:
        findings.append(f"head SFT leg did not give {steps} finite losses "
                        f"(rc={ht.get('rc')}, losses={hl}, error={ht.get('error')})")
    elif not all(_close(b, h) for b, h in zip(bl, hl)):
        findings.append(f"head SFT losses {hl} outside {ABS_TOL} + {REL_TOL}*|base| of base {bl}")
    bu, hu = bt.get("unsloth_train") or {}, ht.get("unsloth_train") or {}
    if bu.get("ok"):
        if not hu.get("ok"):
            findings.append(f"unsloth_train completed at the base but failed at the head: {hu.get('error')}")
        else:
            bl2, hl2 = bu.get("losses") or [], hu.get("losses") or []
            if not _finite(hl2) or len(hl2) != len(bl2):
                findings.append(f"head unsloth_train losses {hl2} not finite or not matching base {bl2}")
            elif not all(_close(b, h) for b, h in zip(bl2, hl2)):
                findings.append(f"head unsloth_train losses {hl2} outside tolerance of base {bl2}")
    else:
        notes.append(f"unsloth_train did not complete at the BASE ({bu.get('error')}), so that "
                     f"sub-leg compares nothing; head: ok={hu.get('ok')} {hu.get('error') or ''}")

    if findings:
        return True, "; ".join(findings + notes)
    return False, (f"all {len(cb)} device / GC fields identical (DEVICE_TYPE `{cb['DEVICE_TYPE']}`, "
                   f"count {cb['DEVICE_COUNT']}, bf16 `{cb['device_is_bf16_supported()']}`); "
                   f"head npu_is_available() False; SFT losses base {bl} head {hl} within tolerance"
                   + (f"; unsloth_train base {bu.get('losses')} head {hu.get('losses')}" if bu.get("ok") else "")
                   + ("; " + "; ".join(notes) if notes else ""))
