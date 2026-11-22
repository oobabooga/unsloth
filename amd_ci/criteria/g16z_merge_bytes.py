#!/usr/bin/env python3
"""Criteria (unsloth-zoo #2655): CPU merged save / GGUF staging streamed from the live model. REGRESSION mode.
head_is_worse: any saved tensor (name, dtype, shape, sha256) differs from base in any cell / save; a cell that saved
at base errors at head; a streamable cell (plain LoRA on Linear, full finetune; Qwen3MoE only on transformers 4.x)
did not take the streamed path at head; a cell that must fall back (DoRA, LoRA on embeddings, LoRA on a tied
weight, safe_serialization=False, a 4.x VLM with a checkpoint conversion mapping) did stream; the head save changed
the live model. File-byte differences (safetensors layout, shard grouping, index formatting) are reported in the
table, not judged worse: loaders read the same tensors.
Gates: each state imported unsloth_zoo from its own checkout (state import and the cell's own check), base produced
outputs for every cell, the head exposes the streamed save and base does not.
"""
from __future__ import annotations

TITLE = "unsloth-zoo #2655: CPU merged save bytes, base (deepcopy + save_pretrained) vs head (streamed)"
MODE = "regression"
NEEDS = ["windows"]
SAVES = ("merged_16bit", "gguf_stage")


def _norm(p):
    return str(p or "").replace("\\", "/").lower()


def _own(o):
    f, ck = _norm(((o or {}).get("zoo_import") or {}).get("file")), _norm((o or {}).get("checkout"))
    cell = (o or {}).get("cell") or {}
    return bool(f) and bool(ck) and f.startswith(ck + "/") and cell.get("own_tree") is True


def _cells(o):
    return ((o or {}).get("cell") or {}).get("cells") or {}


def _major(o):
    try:
        return int(str(((o or {}).get("cell") or {}).get("transformers") or "0").split(".")[0])
    except ValueError:
        return 0


def _expected(expect, major):
    if expect == "tf4_only":
        return major < 5
    if expect == "tf5_only":
        return None  # depends on the model's conversion mapping: recorded, not judged
    return expect


def gates(obs):
    b, h = obs.get("base") or {}, obs.get("head") or {}
    out = []
    for name, o in (("base", b), ("head", h)):
        out.append((f"{name}: unsloth_zoo imported from its own checkout", _own(o),
                    f"import={((o.get('zoo_import') or {}).get('file'))} cell_own={(o.get('cell') or {}).get('own_tree')} "
                    f"checkout={o.get('checkout')} err={str((o.get('zoo_import') or {}).get('error') or '')[:200]}"))
    bc = _cells(b)
    saved = [c for c, r in bc.items() if isinstance(r.get("merged_16bit"), dict) and r["merged_16bit"].get("tensors")]
    out.append(("base produced a merged checkpoint for every cell", bool(bc) and len(saved) == len(bc),
                f"{len(saved)}/{len(bc)} cells; cell rc={(b.get('cell') or {}).get('rc')} "
                f"tail={str((b.get('cell') or {}).get('log_tail') or '')[-300:]}".replace("\n", " ").replace("|", "/")))
    out.append(("head exposes the streamed save, base does not",
                (h.get("cell") or {}).get("arm") == "head" and (b.get("cell") or {}).get("arm") == "base",
                f"base arm={(b.get('cell') or {}).get('arm')} head arm={(h.get('cell') or {}).get('arm')}"))
    out.append(("same transformers / torch in both states",
                bool((b.get("cell") or {}).get("transformers"))
                and (b.get("cell") or {}).get("transformers") == (h.get("cell") or {}).get("transformers")
                and (b.get("cell") or {}).get("torch") == (h.get("cell") or {}).get("torch"),
                f"base {(b.get('cell') or {}).get('transformers')} / {(b.get('cell') or {}).get('torch')}, head "
                f"{(h.get('cell') or {}).get('transformers')} / {(h.get('cell') or {}).get('torch')}"))
    return out


def _compare(rb, rh):
    """(tensor_diffs, file_diffs) for one save."""
    if "error" in rh and "error" not in rb:
        return [f"head error: {rh.get('error')}"], []
    if "error" in rb or "error" in rh:
        return [], []
    tb, th = rb.get("tensors") or {}, rh.get("tensors") or {}
    td = []
    if set(tb) != set(th):
        td.append(f"tensor names base-only {sorted(set(tb) - set(th))[:4]} head-only {sorted(set(th) - set(tb))[:4]}")
    for k in sorted(set(tb) & set(th)):
        for f in ("dtype", "shape", "sha"):
            if tb[k].get(f) != th[k].get(f):
                td.append(f"{k} {f}")
                break
    fb, fh = rb.get("files") or {}, rh.get("files") or {}
    fd = sorted(n for n in set(fb) | set(fh) if fb.get(n) != fh.get(n))
    return td, fd


def head_is_worse(base, head):
    bad, files, engaged = [], 0, 0
    bc, hc = _cells(base), _cells(head)
    major = _major(head)
    for cell, rb in bc.items():
        rh = hc.get(cell)
        if rh is None:
            bad.append(f"{cell}: missing at head")
            continue
        if "build_error" in rh and "build_error" not in rb:
            bad.append(f"{cell}: head build error {rh['build_error']}")
            continue
        exp = _expected(rb.get("expect_engaged"), major)
        for save in SAVES:
            if save not in rb:
                continue
            if save not in rh:
                bad.append(f"{cell}/{save}: not run at head")
                continue
            td, fd = _compare(rb[save], rh[save])
            if td:
                bad.append(f"{cell}/{save}: tensors differ: {td[:3]}")
            files += bool(fd)
            got = rh[save].get("engaged")
            if exp is True and got is not True:
                bad.append(f"{cell}/{save}: streamable but engaged={got}")
            if exp is False and got is True:
                bad.append(f"{cell}/{save}: must fall back but streamed")
            engaged += got is True
            if rh[save].get("live_unchanged") is False:
                bad.append(f"{cell}/{save}: head changed the live model")
    return (bool(bad), "; ".join(bad[:12]) or
            f"every tensor of every cell identical to base; streamed path engaged in {engaged} saves; "
            f"{files} saves differ only in file bytes (layout / index), see table")


def table(obs):
    b, h = obs.get("base") or {}, obs.get("head") or {}
    bc, hc = _cells(b), _cells(h)
    major = _major(h)
    L = ["| cell | save | tensors identical | file bytes identical | engaged (head) | expected | differing files |",
         "|---|---|---|---|---|---|---|"]
    for cell, rb in bc.items():
        rh = hc.get(cell) or {}
        for save in SAVES:
            if save not in rb or save not in rh:
                continue
            td, fd = _compare(rb[save], rh[save])
            err = rb[save].get("error") or rh[save].get("error")
            L.append(f"| {cell} | {save} | {'ERR ' + str(err)[:60] if err else ('Y' if not td else 'N')} | "
                     f"{'Y' if not fd else 'N'} | {rh[save].get('engaged')} | {_expected(rb.get('expect_engaged'), major)} | "
                     f"{', '.join(fd)[:120]} |")
    for name, o in (("base", b), ("head", h)):
        c = o.get("cell") or {}
        if c.get("rc") not in (0, None) or not c.get("cells"):
            L.append(f"\n{name} cell rc={c.get('rc')} log={str(c.get('log_tail'))[-800:]}")
    L.append(f"\nversions {h.get('versions')}; zoo base {(b.get('zoo_import') or {}).get('file')} head "
             f"{(h.get('zoo_import') or {}).get('file')}; commits base {b.get('commit')} head {h.get('commit')}; "
             f"peak RSS MB base {(b.get('cell') or {}).get('peak_rss_mb')} head {(h.get('cell') or {}).get('peak_rss_mb')}")
    return "\n".join(L)
