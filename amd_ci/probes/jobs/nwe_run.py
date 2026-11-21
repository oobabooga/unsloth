"""unsloth #18625 nwe: run jobs/sft.py (next to this file) unchanged, with UNSLOTH_TP_VOCAB_PARALLEL unset, and record
engagement at exit into $NWE_EXTRA (JSON): tp_native.VOCAB_PARALLEL_CALLS (head only; "absent" at base), calls of
llama.unsloth_fused_ce_loss (the default loss path) and llama._vocab_parallel_active (head only). Observes only."""
import atexit
import json
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXTRA = os.environ["NWE_EXTRA"]
os.environ.pop("UNSLOTH_TP_VOCAB_PARALLEL", None)
cwd = os.getcwd()
for var, sub in (("TORCHINDUCTOR_CACHE_DIR", "inductor"), ("TRITON_CACHE_DIR", "triton")):
    os.environ[var] = os.path.join(cwd, sub)  # fresh per run: no compile reuse across arms / runs
extra = {"env_UNSLOTH_TP_VOCAB_PARALLEL": os.environ.get("UNSLOTH_TP_VOCAB_PARALLEL"), "fused_ce_calls": 0,
         "vocab_parallel_active_calls": None}


def _dump():
    try:
        import unsloth.models.tp_native as t
        extra["tp_native_file"] = getattr(t, "__file__", None)
        extra["vocab_parallel_calls"] = dict(t.VOCAB_PARALLEL_CALLS) if hasattr(t, "VOCAB_PARALLEL_CALLS") else "absent"
    except Exception as e:  # noqa: BLE001
        extra["vocab_parallel_calls"] = f"ERR {e!r}"[:300]
    try:
        import unsloth.models.llama as L
        extra["llama_file"] = L.__file__
        extra["has_vocab_parallel_active"] = hasattr(L, "_vocab_parallel_active")
    except Exception as e:  # noqa: BLE001
        extra["llama_file"] = f"ERR {e!r}"[:300]
    with open(EXTRA, "w", encoding = "utf-8") as fh:
        json.dump(extra, fh)


atexit.register(_dump)

import unsloth  # noqa: E402,F401
import unsloth.models.llama as L  # noqa: E402

_orig_ce = L.unsloth_fused_ce_loss


def _ce(*a, **k):
    extra["fused_ce_calls"] += 1
    return _orig_ce(*a, **k)


L.unsloth_fused_ce_loss = _ce
if hasattr(L, "_vocab_parallel_active"):
    extra["vocab_parallel_active_calls"] = 0
    _orig_vp = L._vocab_parallel_active

    def _vp(*a, **k):
        extra["vocab_parallel_active_calls"] += 1
        return _orig_vp(*a, **k)

    L._vocab_parallel_active = _vp

sft = os.path.join(HERE, "sft.py")
sys.argv = [sft] + sys.argv[1:]
runpy.run_path(sft, run_name = "__main__")
