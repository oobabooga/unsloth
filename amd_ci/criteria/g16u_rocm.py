#!/usr/bin/env python3
"""Criteria (G16, unsloth #18812 Inkling 2): non-Inkling LoRA SFT + save on gfx1151 must not move. REGRESSION mode.

Everything train5_rocm judges (losses, finite / decreasing, memory, GC calls, test ids), plus for the g16 cells:
  * g16_flm_save / g16_fm_save: the LoRA-wrapped module set equals base; the merged_16bit save succeeded at base
    and not at head = worse; adapter and merged tensors (dtype, shape, sha256) equal base whenever the base A/A
    repeat reproduced them byte for byte, else the merged reload loss differs from base beyond 3 x A/A + 1e-2;
    saved file sets equal base.
  * g16_targets: every target_modules x family-flag combo wraps the same module set at head as at base.
"""
from __future__ import annotations

import importlib.util
import os

_spec = importlib.util.spec_from_file_location("g16_train5_rocm", os.path.join(os.path.dirname(os.path.abspath(__file__)), "train5_rocm.py"))
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)
T.TRAIN = tuple(T.TRAIN) + ("g16_flm_save", "g16_fm_save")

TITLE = "G16 #18812: tiny Llama / Qwen3 LoRA SFT + lora / merged_16bit save + LoRA targets on gfx1151, base vs head"
MODE = "regression"
NEEDS = list(T.NEEDS)
SAVE_CELLS = ("g16_flm_save", "g16_fm_save")


def gates(obs):
    out = T.gates(obs)
    b = T._cells(obs.get("base"))
    for c in SAVE_CELLS:
        if c in b:
            out.append((f"base {c} saved merged_16bit and reloaded it", bool((b.get(c) or {}).get("merged_ok")) and
                        isinstance((b.get(c) or {}).get("merged_reload_rel"), float),
                        str((b.get(c) or {}).get("merged_error") or (b.get(c) or {}).get("merged_reload_rel"))[:300]))
    if "g16_targets" in b:
        combos = (b.get("g16_targets") or {}).get("combos") or {}
        out.append(("base g16_targets resolved every combo", bool(combos) and all("modules" in v for v in combos.values()),
                    str({k: (v.get("error") or len(v.get("modules") or [])) for k, v in combos.items()})[:300]))
    return out


def _tensors(c, key):
    return ((c or {}).get(key) or {}).get("tensors") or {}


def head_is_worse(base, head):
    worse_t5, detail = T.head_is_worse(base, head)
    bad = []
    bc, hc = T._cells(base), T._cells(head)
    for c in SAVE_CELLS:
        b, h, aa = bc.get(c), hc.get(c), bc.get(c + "@aa")
        if not b or not h:
            continue
        if b.get("lora_modules") != h.get("lora_modules"):
            bad.append(f"{c}: LoRA module set differs ({b.get('lora_modules_n')} base vs {h.get('lora_modules_n')} head)")
        if b.get("merged_ok") and not h.get("merged_ok"):
            bad.append(f"{c}: merged_16bit save failed at head: {h.get('merged_error')}"[:400])
            continue
        for key in ("lora_save", "merged_save"):
            fb, fh = set(((b.get(key) or {}).get("files") or {})), set(((h.get(key) or {}).get("files") or {}))
            if fb != fh:
                bad.append(f"{c} {key}: file set differs base {sorted(fb)} head {sorted(fh)}"[:400])
            tb, th = _tensors(b, key), _tensors(h, key)
            deterministic = aa is not None and _tensors(aa, key) == tb and bool(tb)
            h[f"_{key}_deterministic_at_base"] = deterministic
            if set(tb) != set(th):
                bad.append(f"{c} {key}: tensor names differ ({len(tb)} base vs {len(th)} head)")
            elif deterministic and tb != th:
                diff = [k for k in tb if tb[k] != th.get(k)]
                bad.append(f"{c} {key}: {len(diff)} tensors differ from base (base A/A identical): {diff[:4]}")
            elif any(tb[k][:2] != th[k][:2] for k in tb):
                bad.append(f"{c} {key}: dtype / shape differs from base")
        rb, rh = b.get("merged_reload_rel"), h.get("merged_reload_rel")
        lb, lh = b.get("merged_reload_loss"), h.get("merged_reload_loss")
        if isinstance(lb, float) and isinstance(lh, float):
            spread = abs(lb - (aa or {}).get("merged_reload_loss", lb)) / max(abs(lb), 1e-6)
            if abs(lb - lh) / max(abs(lb), 1e-6) > 3 * spread + 1e-2:
                bad.append(f"{c}: merged reload loss {lh} at head vs {lb} at base")
        if isinstance(rh, float) and isinstance(rb, float) and rh > max(5e-2, 3 * rb):
            bad.append(f"{c}: merged reload vs in-memory loss rel {rh} at head (base {rb})")
    bt, ht = (bc.get("g16_targets") or {}).get("combos") or {}, (hc.get("g16_targets") or {}).get("combos") or {}
    for k, v in bt.items():
        w = ht.get(k) or {}
        if v.get("modules") != w.get("modules"):
            bad.append(f"g16_targets {k}: base {len(v.get('modules') or [])} modules ({v.get('error')}) vs head "
                       f"{len(w.get('modules') or [])} ({w.get('error')})"[:300])
    if bad:
        return True, "; ".join(bad) + (". " + detail if worse_t5 else "")
    return worse_t5, detail + ("" if worse_t5 else "; g16 LoRA module sets, adapter / merged tensors and target "
                                                   "combos equal base")


def table(obs):
    rows = [T.table(obs), "", "| g16 cell | state | LoRA modules | lora tensors | merged tensors | merged ok | reload loss | "
            "A/A identical (lora / merged) |", "|---|---|---|---|---|---|---|---|"]
    for c in SAVE_CELLS:
        for st in ("base", "head"):
            x = T._cells(obs.get(st)).get(c) or {}
            rows.append(f"| {c} | {st} | {x.get('lora_modules_n')} | {len(_tensors(x, 'lora_save'))} | "
                        f"{len(_tensors(x, 'merged_save'))} | {x.get('merged_ok')} {str(x.get('merged_error') or '')[:80]} | "
                        f"{x.get('merged_reload_loss')} | {x.get('_lora_save_deterministic_at_base')} / "
                        f"{x.get('_merged_save_deterministic_at_base')} |")
    for st in ("base", "head"):
        combos = (T._cells(obs.get(st)).get("g16_targets") or {}).get("combos") or {}
        rows.append(f"\ng16_targets {st}: " + ", ".join(f"{k}={v.get('error') or len(v.get('modules') or [])}"
                                                      for k, v in combos.items()))
    return "\n".join(rows)
