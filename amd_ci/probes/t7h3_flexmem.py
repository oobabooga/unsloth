"""T7h.3 follow-up (observes only): GPT-OSS flex prefill peak memory across VARIED prompt lengths in one fresh process,
flex route only (no eager kill-switch arm), with Dynamo recompile messages captured. argv: out.json mode
mode = "default" | "static" (torch._dynamo.config.automatic_dynamic_shapes = False) | "alternate" (eager / flex rounds).
"""
import json
import logging
import os
import sys
import traceback

OUT, MODE = sys.argv[1], sys.argv[2]
rec = {"mode": MODE, "steps": [], "log": []}


def save():
    with open(OUT, "w", encoding = "utf-8") as fh:
        json.dump(rec, fh, indent = 1, default = str)


os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
os.environ.setdefault("UNSLOTH_MODEL_NAME", "unsloth/gpt-oss-test")
try:
    import unsloth  # noqa: F401
except Exception as e:  # noqa: BLE001
    rec["unsloth_import_error"] = f"{type(e).__name__}: {e}"[:500]
import torch  # noqa: E402
import torch._dynamo  # noqa: E402


class _H(logging.Handler):
    def emit(self, r):
        m = r.getMessage()
        if any(k in m for k in ("ecompil", "triggered by", "fallback", "Fallback", "flex", "graph break", "Graph break")):
            rec["log"].append(m[:600])


for name in ("torch._dynamo", "torch._inductor", "torch._dynamo.guards"):
    lg = logging.getLogger(name)
    lg.addHandler(_H())
try:
    torch._logging.set_logs(recompiles = True, graph_breaks = True)
except Exception as e:  # noqa: BLE001
    rec["set_logs_error"] = str(e)
if MODE == "static":
    torch._dynamo.config.automatic_dynamic_shapes = False
rec["torch"] = {"version": torch.__version__, "hip": torch.version.hip}
try:
    import triton

    rec["triton"] = triton.__version__
except Exception:  # noqa: BLE001
    rec["triton"] = None
try:
    from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
    import transformers.models.gpt_oss.modeling_gpt_oss as M
    from unsloth_zoo.temporary_patches import gpt_oss as G
    import unsloth_zoo

    rec["unsloth_zoo_file"] = unsloth_zoo.__file__
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation = False) else torch.float16
    torch.manual_seed(0)
    L = 4
    c = GptOssConfig(hidden_size = 512, intermediate_size = 512, num_hidden_layers = L, num_attention_heads = 32,
                     num_key_value_heads = 8, head_dim = 64, num_local_experts = 4, num_experts_per_tok = 2,
                     vocab_size = 1024, max_position_embeddings = 8192, sliding_window = 128,
                     layer_types = ["sliding_attention", "full_attention"] * (L // 2))
    c._attn_implementation = "eager"
    model = M.GptOssForCausalLM(c).to("cuda", dtype).eval()
    G.patch_GptOssAttention(); G.patch_GptOssModel()
    ids = torch.randint(1, 1024, (1, 4096), device = "cuda")
    plan = [("flex", T) for T in (2048, 4096, 1024, 3072, 2048, 4096, 1536, 4096)]
    if MODE == "alternate":
        plan = [("eager", 2048), ("eager", 4096), ("flex", 2048), ("flex", 4096), ("eager", 2048), ("eager", 4096),
                ("flex", 2048), ("flex", 4096), ("flex", 4096)]
    import torch._dynamo.utils as DU

    for arm, T in plan:
        os.environ["UNSLOTH_GPT_OSS_EAGER_INFERENCE"] = "1" if arm == "eager" else "0"
        n_log = len(rec["log"])
        step = {"arm": arm, "T": T}
        try:
            with torch.no_grad():
                torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
                model(input_ids = ids[:, :T], use_cache = False, logits_to_keep = 1)
                torch.cuda.synchronize()
                step["peak_mib"] = round((torch.cuda.max_memory_allocated() - base) / 2**20, 1)
        except Exception as e:  # noqa: BLE001
            step["error"] = f"{type(e).__name__}: {e}"[:600]
        step["unique_graphs"] = DU.counters["stats"].get("unique_graphs")
        step["new_log"] = rec["log"][n_log:][:6]
        rec["steps"].append(step)
        save()
except Exception as e:  # noqa: BLE001
    rec["error"] = f"{type(e).__name__}: {e}"[:800] + " | " + traceback.format_exc()[-1500:]
rec["done"] = True
save()
