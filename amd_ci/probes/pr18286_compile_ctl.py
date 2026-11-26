#!/usr/bin/env python3
"""Control for tests/test_qat_weight_memo.py::test_compile_trace_unchanged, runnable at the BASE too (no memo module
needed). Observes only; JSON to argv[1].

Same block, LoRA, int8-int4 TorchAOConfig and fullgraph CompileCounter compile as the test, with the memo set by
UNSLOTH_QAT_WEIGHT_MEMO from the caller (the base has no memo at all). Records whether dynamo traced it
(frame_count) or raised, and whether the compiled output equals eager. Answers: does the fullgraph trace of a QAT
int8-int4 LoRA block fail on this torch / torchao with no memo involved?
"""
import json
import os
import sys
import traceback

out = sys.argv[1]
res = {"memo_env": os.environ.get("UNSLOTH_QAT_WEIGHT_MEMO")}
try:
    import torch
    import peft
    import torchao
    import unsloth  # noqa: F401
    from unsloth.models._utils import TorchAOConfig, _prepare_model_for_qat
    from torchao.quantization import Int8DynamicActivationIntxWeightConfig
    from torchao.quantization.granularity import PerGroup

    res["versions"] = {"torch": torch.__version__, "torchao": torchao.__version__, "peft": peft.__version__}
    res["unsloth_file"] = unsloth.__file__
    try:
        import unsloth.models._qat_weight_memo  # noqa: F401
        res["memo_module"] = True
    except ImportError:
        res["memo_module"] = False
    DIM = 128

    class _Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.up_proj = torch.nn.Linear(DIM, 2 * DIM, bias = False)
            self.down_proj = torch.nn.Linear(2 * DIM, DIM, bias = False)

        def forward(self, x):
            return self.down_proj(torch.relu(self.up_proj(x)))

    torch.manual_seed(0)
    model = peft.get_peft_model(_Block(), peft.LoraConfig(r = 16, lora_alpha = 16, target_modules = ["up_proj", "down_proj"],
                                                          lora_dropout = 0.0))
    for name, p in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(p, std = 0.02)
    config = Int8DynamicActivationIntxWeightConfig(weight_dtype = torch.int4, weight_granularity = PerGroup(32))
    only_base = lambda m, fqn: isinstance(m, torch.nn.Linear) and fqn.endswith("base_layer")  # noqa: E731
    model = _prepare_model_for_qat(model, TorchAOConfig(qat_scheme = None, base_config_and_filter_fns = [(config, only_base)]))
    p = torch.nn.Parameter(torch.zeros(1))
    p.grad = torch.zeros(1)
    torch.optim.SGD([p], lr = 0.0).step()
    x = torch.randn(4, DIM)
    for _ in range(2):
        model.zero_grad(set_to_none = True)
        model(x).float().square().mean().backward()
    from torch._dynamo.testing import CompileCounter

    torch._dynamo.reset()
    counter = CompileCounter()
    compiled = torch.compile(model, backend = counter, fullgraph = True)
    try:
        with torch.no_grad():
            for _ in range(2):
                o = compiled(x)
            want = model(x)
        res["traced"] = True
        res["frame_count"] = counter.frame_count
        res["equal"] = bool(torch.equal(o, want))
    except Exception as e:  # noqa: BLE001
        res["traced"] = False
        res["trace_error"] = f"{type(e).__name__}: {str(e)[:600]}"
except BaseException as e:  # noqa: BLE001
    res["error"] = f"{type(e).__name__}: {str(e)[:800]}"
    res["traceback"] = traceback.format_exc()[-2500:]
with open(out, "w", encoding = "utf-8") as f:
    json.dump(res, f, indent = 1, default = str)
