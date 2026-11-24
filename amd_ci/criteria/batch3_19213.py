#!/usr/bin/env python3
"""PR 19213: the fixes key on torch.cuda.is_current_stream_capturing(); it must read True inside a real HIP graph
capture and False outside, and the head's H3 _capturing() must follow it. Regression mode (torch behaviour is the
same at both trees; the head's helpers are gated)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import base_gates, states  # noqa: E402

TITLE = "PR 19213: is_current_stream_capturing inside a real HIP graph capture (gfx1151)"
MODE = "regression"
NEEDS = ["rocm", "gpu"]


def gates(obs):
    out = base_gates(obs)
    for name, o in states(obs).items():
        c = o.get("capture") or {}
        out.append((f"{name}: capture ran", "error" not in c, str(c.get("error", ""))[-400:]))
        out.append((f"{name}: False outside / True inside / False after",
                    c.get("outside") is False and c.get("inside") is True and c.get("inside_device_ctx") is True
                    and c.get("after") is False, str(c)))
        out.append((f"{name}: replay computed the recorded op", c.get("replay_value_ok") is True, str(c.get("replay_value_ok"))))
        if name != "base":
            out.append((f"{name}: H3 _capturing True inside, False outside",
                        c.get("h3_capturing") is True and c.get("h3_capturing_outside") is False, str(c)))
            out.append((f"{name}: AdaLN uninstall has _drop_graphs_then_free", bool(o.get("adaln_has_drop_graphs")), ""))
    return out


def table(obs):
    rows = ["| state | outside | inside | inside device ctx | H3 _capturing in/out | recording_here | after |",
            "|---|---|---|---|---|---|---|"]
    for name, o in states(obs).items():
        c = o.get("capture") or {}
        rows.append(f"| {name} | {c.get('outside')} | {c.get('inside')} | {c.get('inside_device_ctx')} "
                    f"| {c.get('h3_capturing', '-')}/{c.get('h3_capturing_outside', '-')} | {c.get('recording_here')} | {c.get('after')} |")
    return "\n".join(rows)


def head_is_worse(base, head):
    b, h = base.get("capture") or {}, head.get("capture") or {}
    if h.get("inside") != b.get("inside"):
        return True, f"capture flag differs: base {b.get('inside')} head {h.get('inside')}"
    return False, f"capture flag True inside a real capture at both trees; head H3 _capturing {h.get('h3_capturing')}"
