#!/usr/bin/env python3
"""Criteria: a Studio-style Qwen3.5 fine-tune GGUF fails Q4NX conversion at base and converts at head.

Pairs with probes/q4nx_convert_probe.py. Head counts as fixed only when the converter's own
HF-vs-q4nx checker prints ALL PASS, so a q4nx with layers silently dropped is not a fix.
"""

from __future__ import annotations

TITLE = "Q4NX conversion of a Qwen3.5 fine-tune GGUF, base versus head"
MODE = "differential"
NEEDS: list[str] = []


def _gguf_ok(o: dict) -> tuple[bool, str]:
    g = o.get("gguf") or {}
    blk0 = g.get("blk0") or []
    ok = (g.get("architecture") == "qwen35" and g.get("name") == "Model" and g.get("basename") is None
          and "blk.0.attn_qkv.weight" in blk0)
    return ok, f"name={g.get('name')!r} basename={g.get('basename')!r} arch={g.get('architecture')} " \
               f"blocks={g.get('block_count')} nextn={g.get('nextn')} {o.get('error') or ''}"


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        ok, detail = _gguf_ok(o)
        out.append((f"{name}: Studio-style unmerged qwen35 GGUF was built", ok, detail))
        conv = o.get("convert") or {}
        out.append((f"{name}: conversion ran", conv.get("result") not in (None, "NO_RESULT"),
                    str(conv)[:300]))
    return out


def table(obs: dict) -> str:
    rows = ["| state | result | seconds | checker | q4nx sha256 | NPU seen |", "|---|---|---|---|---|---|"]
    for name in ("base", "head"):
        o = obs.get(name) or {}
        c = o.get("convert") or {}
        ch = o.get("checker") or {}
        rows.append(f"| {name} | {str(c.get('result'))[:120]} | {c.get('seconds')} | "
                    f"{(ch.get('tail') or ['-'])[-1]} ({ch.get('n_pass', '-')} PASS) | "
                    f"{(o.get('q4nx_sha256') or '-')[:16]} | {o.get('npu')} |")
    return "\n".join(rows)


def base_shows_defect(base: dict) -> tuple[bool, str]:
    r = str((base.get("convert") or {}).get("result"))
    return r != "OK", r[:200]


def head_is_fixed(head: dict) -> tuple[bool, str]:
    r = str((head.get("convert") or {}).get("result"))
    ch = head.get("checker") or {}
    ok = r == "OK" and ch.get("rc") == 0 and (ch.get("tail") or [""])[-1].strip() == "ALL PASS"
    return ok, f"{r[:120]}; checker {ch.get('tail')}; fails {ch.get('fails')}"
