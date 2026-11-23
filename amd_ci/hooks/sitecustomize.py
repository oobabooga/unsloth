"""PR 18825 recheck instrumentation (active only with ACC_HASH_DIR): per optimizer step, a bit checksum of every
parameter's gradient (int32 view sum) and its float sum, plus the gc budget's plan log line. JSON lines only."""
import os

_DIR = os.environ.get("ACC_HASH_DIR")
if _DIR:
    try:
        import json
        import logging
        import sys
        import time

        os.makedirs(_DIR, exist_ok = True)
        _path = os.path.join(_DIR, f"hash_{os.getpid()}_{int(time.time())}.jsonl")
        _fh = open(_path, "a", encoding = "utf-8")
        _fh.write(json.dumps({"argv": sys.argv}) + "\n")
        _fh.flush()

        class _H(logging.Handler):
            def emit(self, rec):
                try:
                    _fh.write(json.dumps({"log": rec.getMessage()}) + "\n")
                    _fh.flush()
                except Exception:  # noqa: BLE001
                    pass

        for name in ("core.training.diffusion_gc_budget", "core.training.diffusion_lora_fast"):
            lg = logging.getLogger(name)
            lg.setLevel(logging.INFO)
            lg.addHandler(_H())

        import torch
        from torch.optim.optimizer import register_optimizer_step_pre_hook

        _n = [0]

        def _hook(opt, args, kwargs):
            _n[0] += 1
            bits, sums = [], []
            for g in opt.param_groups:
                for p in g["params"]:
                    if p.grad is None:
                        continue
                    gr = p.grad.detach()
                    if gr.dtype == torch.float32:
                        bits.append(gr.contiguous().view(torch.int32).sum(dtype = torch.int64))
                    else:
                        bits.append(gr.float().contiguous().view(torch.int32).sum(dtype = torch.int64))
                    sums.append(gr.float().sum())
            b = torch.stack(bits).tolist() if bits else []
            s = torch.stack(sums).tolist() if sums else []
            _fh.write(json.dumps({"step": _n[0], "bits": b, "sums": s}) + "\n")
            _fh.flush()

        register_optimizer_step_pre_hook(_hook)
    except Exception as exc:  # noqa: BLE001
        try:
            open(os.path.join(_DIR, f"hook_error_{os.getpid()}.txt"), "w").write(repr(exc))
        except Exception:  # noqa: BLE001
            pass
