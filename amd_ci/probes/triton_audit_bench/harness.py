"""Common micro-bench harness for the Triton kernel audit.

Every bench module b_<group>.py:
  * imports `from harness import *`
  * defines CASES: list of dict(name=..., ...) and `def run(case, dtype, H)` that calls
    H.time_arm / H.parity / H.compile_check, and returns nothing (rows are emitted as JSON lines).
  * ends with `if __name__ == "__main__": main(CASES, run)`.

CLI (same for all modules):
  python b_x.py --dtype bf16|fp16 [--quick] [--only substr] [--rep K]
Each invocation is one fresh process = one repeat. The driver (run_all.py) interleaves
modules and repeats.

Output: one JSON per line prefixed with "@@ROW " on stdout.
"""
import argparse, json, os, sys, time, types, traceback, statistics, importlib

UNSLOTH_SRC = os.environ.get("AUDIT_UNSLOTH_SRC", "")
ZOO_SRC = os.environ.get("AUDIT_ZOO_SRC", "")


def _stub_pkg(name, path):
    """Register a package without running its __init__ (avoids unsloth / zoo global patching)."""
    if name in sys.modules:
        return sys.modules[name]
    m = types.ModuleType(name)
    m.__path__ = [path]
    m.__file__ = os.path.join(path, "__init__.py")
    import importlib.machinery
    spec = importlib.machinery.ModuleSpec(name, None, is_package=True)
    spec.submodule_search_locations = [path]
    spec.origin = m.__file__
    m.__spec__ = spec
    sys.modules[name] = m
    parent, _, child = name.rpartition(".")
    if parent:
        setattr(sys.modules[parent], child, m)
    return m


def setup_paths():
    os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
    os.environ.setdefault("UNSLOTH_ZOO_IS_PRESENT", "1")
    os.environ.setdefault("UNSLOTH_DISABLE_AUTO_UPDATES", "1")
    if UNSLOTH_SRC and UNSLOTH_SRC not in sys.path:
        sys.path.insert(0, UNSLOTH_SRC)
    if ZOO_SRC and ZOO_SRC not in sys.path:
        sys.path.insert(0, ZOO_SRC)


def import_unsloth(modname, stub=("unsloth", "unsloth.kernels")):
    """import unsloth.kernels.<x> without unsloth/__init__ and kernels/__init__ side effects."""
    setup_paths()
    import importlib.util
    base = os.path.dirname(importlib.util.find_spec("unsloth").origin) if "unsloth" not in sys.modules else sys.modules["unsloth"].__path__[0]
    for s in stub:
        rel = s.split(".")[1:]
        _stub_pkg(s, os.path.join(base, *rel))
    return importlib.import_module(modname)


def import_zoo(modname, stub=("unsloth_zoo",)):
    setup_paths()
    import importlib.util
    base = os.path.dirname(importlib.util.find_spec("unsloth_zoo").origin) if "unsloth_zoo" not in sys.modules else sys.modules["unsloth_zoo"].__path__[0]
    for s in stub:
        rel = s.split(".")[1:]
        _stub_pkg(s, os.path.join(base, *rel))
    return importlib.import_module(modname)


class H:
    def __init__(self, bench, dtype_name, quick=False, rep=0):
        import torch
        self.torch = torch
        self.bench = bench
        self.dtype_name = dtype_name
        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_name]
        self.quick = quick
        self.rep = rep
        self.case = None
        self.meta = dict(
            gpu=torch.cuda.get_device_name(0),
            cap="%d.%d" % torch.cuda.get_device_capability(0),
            torch=torch.__version__,
            triton=__import__("triton").__version__,
        )

    def emit(self, **kw):
        row = dict(bench=self.bench, case=self.case, dtype=self.dtype_name, rep=self.rep, **self.meta)
        row.update(kw)
        print("@@ROW " + json.dumps(row, default=str), flush=True)

    # ---------------------------------------------------------------- timing
    def _l2_flush_buf(self):
        if not hasattr(self, "_l2"):
            self._l2 = self.torch.empty(64 * 1024 * 1024, dtype=self.torch.int8, device="cuda")
        return self._l2

    def measure(self, fn, iters=None, warmup=None, flush_l2=False, budget_ms=None):
        """Median kernel time (ms) by CUDA events around each call; returns (median, p10, p90, n)."""
        torch = self.torch
        warmup = warmup if warmup is not None else (3 if self.quick else 10)
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        # estimate
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        est = max(s.elapsed_time(e), 1e-3)
        budget = budget_ms if budget_ms is not None else (100 if self.quick else 400)
        n = iters or int(min(max(budget / est, 10), 2000))
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        buf = self._l2_flush_buf() if flush_l2 else None
        for i in range(n):
            if buf is not None:
                buf.zero_()
            starts[i].record(); fn(); ends[i].record()
        torch.cuda.synchronize()
        t = sorted(a.elapsed_time(b) for a, b in zip(starts, ends))
        return t[len(t) // 2], t[int(len(t) * 0.1)], t[min(int(len(t) * 0.9), len(t) - 1)], n

    def host_us(self, fn, n=200):
        """Host (CPU) microseconds per call, launches back to back, sync only at end. At tiny
        shapes this is launch / Python overhead."""
        torch = self.torch
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        return (t1 - t0) / n * 1e6, (t2 - t0) / n * 1e6

    def time_arm(self, arm, phase, fn, shape=None, host=False, **kw):
        """arm: 'ours', 'ours_aa' (second measurement for A/A), 'eager', 'compiled', 'cublas', 'fla_latest', ...
        phase: 'fwd', 'bwd', 'fwdbwd'."""
        try:
            med, p10, p90, n = self.measure(fn, **kw)
            row = dict(arm=arm, phase=phase, shape=shape, ms=med, p10=p10, p90=p90, n=n)
            if host:
                hu, wu = self.host_us(fn)
                row.update(host_us=hu, wall_us=wu)
            self.emit(kind="time", **row)
            return med
        except Exception as ex:
            self.emit(kind="error", arm=arm, phase=phase, shape=shape, err=repr(ex)[:500], tb=traceback.format_exc()[-1500:])
            return None

    def interleave(self, arms, phase, shape=None, rounds=2, host=False, **kw):
        """arms: dict name->fn. ABBA-ish interleave: each arm measured `rounds` times alternating order.
        The 'ours' arm is also emitted as 'ours_aa' for the second round to give an A/A band."""
        names = list(arms)
        for r in range(rounds):
            order = names if r % 2 == 0 else names[::-1]
            for nme in order:
                tag = nme if r == 0 else nme + "_r%d" % r
                self.time_arm(tag, phase, arms[nme], shape=shape, host=host and r == 0, **kw)

    # ---------------------------------------------------------------- parity
    def parity(self, name, ours, ref32, ref_lowp=None, shape=None):
        """ours: tensor (or tuple) from our kernel; ref32: fp32/fp64 reference; ref_lowp: same-dtype
        torch reference (its error vs ref32 is the dtype-rounding baseline)."""
        torch = self.torch

        def err(a, b):
            a = a.float(); b = b.float()
            d = (a - b).abs()
            fin = torch.isfinite(a).all().item()
            denom = b.abs().max().clamp_min(1e-12)
            return dict(max_abs=d.max().item(), mean_abs=d.mean().item(), max_rel=(d.max() / denom).item(), finite=fin)

        if not isinstance(ours, (tuple, list)):
            ours, ref32 = (ours,), (ref32,)
            ref_lowp = (ref_lowp,) if ref_lowp is not None else None
        for i, (o, r) in enumerate(zip(ours, ref32)):
            if o is None or r is None:
                continue
            row = dict(kind="parity", what="%s[%d]" % (name, i), shape=shape, ours=err(o, r))
            if ref_lowp is not None and ref_lowp[i] is not None:
                row["ref_lowp"] = err(ref_lowp[i], r)
                b = row["ref_lowp"]["mean_abs"]
                row["ratio_mean_err"] = row["ours"]["mean_abs"] / b if b > 0 else None
            self.emit(**row)

    # ---------------------------------------------------------------- compile
    def compile_check(self, name, fn, inputs_list, dynamic=None, fullgraph=False):
        """torch.compile(fn) over several inputs (different shapes) -> graph breaks + frame/recompile counts."""
        torch = self.torch
        try:
            import torch._dynamo
            from torch._dynamo.utils import counters
            torch._dynamo.reset()
            counters.clear()
            cf = torch.compile(fn, dynamic=dynamic, fullgraph=fullgraph)
            t0 = time.time()
            for args in inputs_list:
                out = cf(*args)
                if isinstance(out, torch.Tensor) and out.requires_grad:
                    out.float().sum().backward()
            torch.cuda.synchronize()
            gb = {k: v for k, v in counters["graph_break"].items()}
            self.emit(kind="compile", what=name, graph_breaks=sum(gb.values()), break_reasons=list(gb)[:5],
                      frames_ok=counters["frames"].get("ok", 0), frames_total=counters["frames"].get("total", 0),
                      unique_graphs=counters["stats"].get("unique_graphs", 0),
                      recompiles=counters.get("recompiles", {}) and sum(counters["recompiles"].values()),
                      compile_s=time.time() - t0, n_inputs=len(inputs_list))
        except Exception as ex:
            self.emit(kind="compile", what=name, error=repr(ex)[:500], tb=traceback.format_exc()[-1500:])

    # ---------------------------------------------------------------- autotune
    def autotune_report(self, name, kernel):
        """kernel: an @triton.autotune Autotuner. Emits picked config and measured spread."""
        try:
            timings = getattr(kernel, "configs_timings", None) or {}
            best = getattr(kernel, "best_config", None)
            vals = []
            for c, t in timings.items():
                v = t[0] if isinstance(t, (list, tuple)) else t
                vals.append((float(v), str(c)))
            vals.sort()
            cache = {str(k): str(v) for k, v in list(getattr(kernel, "cache", {}).items())[:6]}
            self.emit(kind="autotune", what=name, n_configs=len(getattr(kernel, "configs", [])),
                      best=str(best), timings=vals[:8], worst=vals[-1] if vals else None, cache=cache)
        except Exception as ex:
            self.emit(kind="autotune", what=name, error=repr(ex)[:300])


def main(CASES, run, bench=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    bench = bench or os.path.basename(sys.argv[0]).replace(".py", "")
    if a.list:
        for c in CASES:
            print(c["name"])
        return
    import torch
    torch.manual_seed(3407)
    h = H(bench, a.dtype, a.quick, a.rep)
    for c in CASES:
        if a.only and a.only not in c["name"]:
            continue
        if a.dtype not in c.get("dtypes", ("bf16", "fp16")):
            continue
        need = c.get("min_cap")
        if need and torch.cuda.get_device_capability(0) < tuple(need):
            h.case = c["name"]; h.emit(kind="skip", why="cap<%s" % (need,))
            continue
        h.case = c["name"]
        t0 = time.time()
        try:
            run(c, a.dtype, h)
        except Exception as ex:
            h.emit(kind="error", arm="case", err=repr(ex)[:500], tb=traceback.format_exc()[-2000:])
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        h.emit(kind="case_done", secs=time.time() - t0)


__all__ = ["H", "main", "import_unsloth", "import_zoo", "setup_paths"]
