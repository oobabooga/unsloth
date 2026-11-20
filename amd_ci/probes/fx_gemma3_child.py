"""unsloth w20c-fp32-flex AMD child: FastModel load of tiny Gemma3ForConditionalGeneration (text head_dim 256) in
W20C_DTYPE, two no-grad forwards, one train forward + backward. Writes JSON to argv[1] after every stage."""
import json
import os
import sys
import time

t0 = time.time()
out = sys.argv[1]
res = {"stage": "import"}


def dump(stage):
    res["stage"], res["t"] = stage, round(time.time() - t0, 1)
    with open(out, "w", encoding = "utf-8") as fh:
        json.dump(res, fh, default = str)


dump("import")
import unsloth  # noqa: E402
from unsloth import FastModel  # noqa: E402
import torch  # noqa: E402

res["unsloth_file"] = unsloth.__file__
MODEL = "trl-internal-testing/tiny-Gemma3ForConditionalGeneration"
DT = getattr(torch, os.environ.get("W20C_DTYPE", "float32"))
torch.manual_seed(0)
dump("load")
model, tok = FastModel.from_pretrained(MODEL, load_in_4bit = False, dtype = DT, max_seq_length = 256)
text_cfg = model.config.get_text_config()
res["attn"] = text_cfg._attn_implementation
res["top_attn"] = getattr(model.config, "_attn_implementation", None)
res["param_dtype"] = str(next(model.parameters()).dtype)
res["load_s"] = round(time.time() - t0, 1)
dump("forward")
g = torch.Generator().manual_seed(1)
ids = torch.randint(10, 5000, (1, 57), generator = g).cuda()
model.eval()
for _ in range(2):
    with torch.no_grad():
        logits = model(input_ids = ids, use_cache = False).logits
torch.cuda.synchronize()
res["logits_sum"] = float(logits.float().sum())
res["logits_absmax"] = float(logits.float().abs().max())
res["forward_s"] = round(time.time() - t0, 1)
dump("train")
model.train()
o = model(input_ids = ids, labels = ids, use_cache = False)
o.loss.backward()
torch.cuda.synchronize()
res["loss"] = float(o.loss.item())
res["train_s"] = round(time.time() - t0, 1)
dump("done")
