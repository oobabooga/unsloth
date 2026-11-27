"""One process: direct fp8 _scaled_mm calls on the device, then unsloth's fp8_lora.rowwise_supported(). JSON to --out."""
import argparse, json, sys, traceback

ap = argparse.ArgumentParser(); ap.add_argument("--out", required = True); a = ap.parse_args()
R = {}
try:
    import torch
    R.update(torch = torch.__version__, hip = torch.version.hip, cuda_available = torch.cuda.is_available())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        R["device"] = torch.cuda.get_device_name(0); R["arch"] = getattr(p, "gcnArchName", None)
        dev = torch.device("cuda", 0)
        direct = {}
        for name in ("float8_e4m3fn", "float8_e4m3fnuz"):
            dt = getattr(torch, name, None)
            if dt is None:
                direct[name] = {"missing_dtype": True}; continue
            d = {}
            try:
                x = torch.ones((64, 128), device = dev).to(dt)
            except Exception as e:
                direct[name] = {"cast_error": f"{type(e).__name__}: {e}"[:300]}; continue
            one = torch.ones((), device = dev)
            rows = torch.ones((64, 1), device = dev)
            for kind, sa, sb in (("tensorwise", one, one), ("rowwise", rows, rows.t())):
                try:
                    out = torch._scaled_mm(x, x.t(), scale_a = sa, scale_b = sb, out_dtype = torch.bfloat16)
                    out = out[0] if isinstance(out, tuple) else out
                    d[kind] = {"ok": True, "correct": bool((out == 128).all())}
                except Exception as e:
                    d[kind] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            direct[name] = d
        R["direct"] = direct
        import unsloth  # noqa: F401  (the state's checkout is first on PYTHONPATH)
        from unsloth.kernels import fp8_lora as FL
        R["unsloth_file"] = unsloth.__file__
        calls = []
        real = torch._scaled_mm
        def counting(*args, **kw):
            calls.append(1); return real(*args, **kw)
        torch._scaled_mm = counting
        R["rowwise_supported"] = bool(FL.rowwise_supported(dev))
        torch._scaled_mm = real
        R["probe_scaled_mm_calls"] = len(calls)
        R["probe_cached"] = {str(k): v for k, v in getattr(FL, "_ROWWISE_PROBED", {}).items()}
        try:
            R["context_ok_after"] = float(torch.ones(1, device = dev).add_(1).item()) == 2.0
        except Exception as e:
            R["context_ok_after"] = f"{type(e).__name__}: {e}"[:200]
except Exception as e:
    R["error"] = f"{type(e).__name__}: {e}"[:500]; R["trace"] = traceback.format_exc()[-1500:]
open(a.out, "w", encoding = "utf-8").write(json.dumps(R, indent = 1))
