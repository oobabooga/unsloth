"""unsloth #18745 ppd: run jobs/sft.py (next to this file) unchanged except that FastLanguageModel.get_peft_model
gets layers_to_transform=[top layer] (tiny Qwen3 has 2 layers -> LoRA on layer 1 only), so under gradient
checkpointing 'unsloth' the frozen-prefix skip has a cut (k = 1). Records at exit into $PPD_EXTRA (JSON):
_frozen_prefix.FROZEN_PREFIX_STATS, _plan calls / returns, the arm's file paths, and whether pp_native's
_PPPassThrough carries the new _unsloth_pipeline_stand_in marker (head True, base False). Observes only."""
import atexit
import json
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXTRA = os.environ["PPD_EXTRA"]
os.environ.pop("UNSLOTH_FROZEN_PREFIX_SKIP", None)
cwd = os.getcwd()
for var, sub in (("TORCHINDUCTOR_CACHE_DIR", "inductor"), ("TRITON_CACHE_DIR", "triton")):
    os.environ[var] = os.path.join(cwd, sub)  # fresh per run: no compile reuse across arms / runs
extra = {"env_UNSLOTH_FROZEN_PREFIX_SKIP": os.environ.get("UNSLOTH_FROZEN_PREFIX_SKIP"), "plan_calls": 0,
         "plan_cuts": [], "peft_layers_to_transform": None, "lora_layers": None}


def _dump():
    try:
        import unsloth.models._frozen_prefix as fp
        extra["frozen_prefix_file"] = fp.__file__
        extra["frozen_prefix_stats"] = dict(fp.FROZEN_PREFIX_STATS)
    except Exception as e:  # noqa: BLE001
        extra["frozen_prefix_stats"] = f"ERR {e!r}"[:300]
    try:
        import unsloth.models.pp_native as pp
        extra["pp_native_file"] = pp.__file__
        extra["pp_stand_in_marker"] = bool(getattr(getattr(pp, "_PPPassThrough", None), "_unsloth_pipeline_stand_in", False))
    except Exception as e:  # noqa: BLE001
        extra["pp_stand_in_marker"] = f"ERR {e!r}"[:300]
    extra["plan_cuts"] = sorted(set(extra["plan_cuts"]), key = str)
    with open(EXTRA, "w", encoding = "utf-8") as fh:
        json.dump(extra, fh)


atexit.register(_dump)

import unsloth  # noqa: E402,F401
import unsloth.models._frozen_prefix as fp  # noqa: E402
from unsloth import FastLanguageModel  # noqa: E402

_orig_plan = fp._plan


def _plan(*a, **k):
    r = _orig_plan(*a, **k)
    extra["plan_calls"] += 1
    extra["plan_cuts"].append(None if r is None else r[1])
    return r


fp._plan = _plan  # install_frozen_prefix_skip's closure looks _plan up in module globals at call time

_orig_peft = FastLanguageModel.get_peft_model


def _peft(model, *a, **k):
    n = model.config.num_hidden_layers
    k["layers_to_transform"] = [n - 1]
    extra["peft_layers_to_transform"] = [n - 1]
    out = _orig_peft(model, *a, **k)
    extra["lora_layers"] = sorted({int(x) for name, p in out.named_parameters() if p.requires_grad
                                   for x in name.split(".") if x.isdigit()})
    extra["trainable_tensors"] = sum(1 for _, p in out.named_parameters() if p.requires_grad)
    return out


FastLanguageModel.get_peft_model = staticmethod(_peft)

sft = os.path.join(HERE, "sft.py")
sys.argv = [sft] + sys.argv[1:]
runpy.run_path(sft, run_name = "__main__")
