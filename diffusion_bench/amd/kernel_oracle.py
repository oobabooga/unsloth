#!/usr/bin/env python3
"""Hand-written kernel vs Inductor vs eager, each against an fp64 oracle, from one spec file.

    python kernel_oracle.py kernel_oracle_specs/example_groupnorm.py --out outputs/kernel_oracle/gn
    python kernel_oracle.py SPEC.py --spec-arg backend=wt_r12078/studio/backend --only gn_ --perf-gate

A spec module defines ``cases(args: dict) -> iterable of dict``; ``args`` holds the ``--spec-arg K=V`` pairs.
Each case dict:

    name     str, unique
    ours     () -> Tensor   the kernel under test (a fused Triton path, a custom op, ...)
    stock    () -> Tensor   the code it replaces, written with stock torch / library modules; compiled by
                            Inductor as the second arm, so Inductor's kernels come from the SAME math
    ref      Tensor | () -> Tensor   fp64 evaluation of the stock formula
    expect   "fused" (default) | "stock" | "golden"   "stock": ours must equal the eager stock output bit for
                            bit (kernels that deliberately keep stock rounding); "golden": ours must equal golden
                            (e.g. a flush-to-zero library kernel, where the fp64 error comparison is meaningless)
    note     str, optional
    kernels  list, optional: Triton JITFunctions (or source strings) of ours, digested beside Inductor's
    golden   () -> Tensor, optional: an existing implementation ours must equal bit for bit (e.g. the library
                            kernel being replaced, which Inductor cannot compile); any differing element FAILs
    inputs   list of leaf tensors with requires_grad, optional: adds one backward row per input,
                            "<name>.grad[i]", with torch.autograd.grad through ours, stock and Inductor's stock
    grad_out Tensor, optional: the incoming gradient (default: seeded randn); pass an expanded (stride-0) one
                            to cover losses such as out.sum()
    ref_grad () -> tuple of Tensors: fp64 gradients of inputs (required with inputs)
    watch    list of tensors, optional: watched for mutation like inputs
    traceable bool, optional: torch.compile(ours) must run with 0 graph breaks, never mutate inputs, watch or
                            grad_out, and match eager ours bit for bit, forward and backward; a user Triton
                            kernel re-emitted by Inductor runs with fp fusion ON, so mul + add can become an FMA
                            that eager (fusion off) never shows
    pure     bool, optional: eager ours must not mutate inputs, watch or grad_out either (off by default: an
                            eager path may keep in-place buffer reuse on purpose)
    inference bool, optional: ours must also run under torch.inference_mode(), eager and compiled, and match
                            eager ours bit for bit (inference callers: Studio renders and generates under it, and a
                            torchao weight subclass cannot be detached / viewed there, unsloth#12083)

Per case: max / mean |err| vs fp64 for ours, Inductor and eager; exact count of elements that differ from
Inductor and from eager; CUDA-event (or wall-clock on CPU) medians; Inductor kernel count; an op digest of the generated
code. Inductor's source is written to OUT/inductor/<case>.py.

Compiled-ours arm (every case unless --no-compile-ours): torch.compile(ours) forward (and backward with
inputs); flags BREAKS n, COMPILED_DRIFT n, COMPILE_ERROR. Mutation flags: MUTATES_INPUT, MUTATES_GRAD (the
incoming gradient was overwritten), *_COMPILED for the compiled arm. They FAIL a case only when it declares
traceable (compiled arm) / pure (eager); GOLDEN_DIFF always fails. Inference arm (every case): ours under
torch.inference_mode(), eager and compiled; flags INFERENCE_ERROR, INFERENCE_DRIFT n, fail only with inference=True.
Peak memory (CUDA): extra MiB allocated by one call of ours and of Inductor's stock; MORE_MEMORY flag like SLOWER. special_values(dtype) gives specs subnormals, signed zeros and range edges (FTZ kernels, e.g.
libdevice under Triton or fast-math builds, flush subnormal products that a plain multiply keeps).

Verdicts: IDENTICAL (== Inductor everywhere), MORE_ACCURATE (max and mean error <= Inductor's), WITHIN_TOL
(both <= --tol x the larger of Inductor / eager; 2 ULP of the output dtype when both are exact), STOCK_EXACT (expect=stock, == eager), GOLDEN_EXACT (expect=golden, == golden), FAIL, VOID (a
callable raised, or Inductor could not compile the stock arm). SLOWER is a flag (ours slower than Inductor
by more than --perf-tol); --perf-gate makes it fail.

Exit: 0 every case passed, 1 a FAIL (or SLOWER under --perf-gate), 2 VOID / nothing ran.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Optional

PASS_VERDICTS = ("IDENTICAL", "MORE_ACCURATE", "WITHIN_TOL", "STOCK_EXACT", "GOLDEN_EXACT")

# Ops worth reading side by side: accumulation strategy, normalisation constant, activation, precision casts.
DIGEST_PATTERNS = {
    "triton_kernels": r"@triton\.jit",
    "reductions": r"def triton_(?:red|per)_",
    "pointwise": r"def triton_poi_",
    "cpp_kernels": r"cpp_fused_",
    "extern_calls": r"extern_kernels\.\w+",
    "welford": r"welford",
    "tl_sum": r"tl\.sum\(",
    "tl_max": r"tl\.max\(|triton_helpers\.max2",
    "rsqrt": r"rsqrt",
    "sqrt": r"(?<!r)sqrt\(",
    "exp": r"\bexp\(|tl\.exp\(|libdevice\.exp\(",
    "sigmoid": r"sigmoid",
    "maximum": r"maximum\(",
    "where": r"tl\.where\(",
    "to_fp32": r"\.to\(tl\.float32\)",
    "to_half": r"\.to\(tl\.(?:bfloat16|float16)\)",
    "int64_index": r"\.to\(tl\.int64\)|int64",
}


def digest(source: str) -> dict:
    return {k: len(re.findall(p, source or "")) for k, p in DIGEST_PATTERNS.items()}


def _src_of(k: Any) -> str:
    if isinstance(k, str):
        return k
    for attr in ("src", "fn"):
        v = getattr(k, attr, None)
        if isinstance(v, str):
            return v
        if callable(v):
            try:
                import inspect
                return inspect.getsource(v)
            except (OSError, TypeError):
                pass
    return ""


def differ(n: int, numel: int) -> str:
    """Exact count: a rounded percentage once hid 36 differing elements behind "100.0%"."""
    return "identical" if n == 0 else f"{n} of {numel} differ ({n / max(numel, 1) * 100:.3g}%)"


def verdict(row: dict, tol: float) -> str:
    """Accuracy verdict from one measured row (keys as written by run_case)."""
    if row.get("error"):
        return "VOID"
    if not row.get("finite", True) or not row.get("shape_ok", True):
        return "FAIL"
    if row.get("expect") == "stock":
        return "STOCK_EXACT" if row.get("differ_from_eager") == 0 else "FAIL"
    if row.get("expect") == "golden":
        return "GOLDEN_EXACT" if row.get("differ_from_golden") == 0 else "FAIL"
    if row.get("differ_from_inductor") == 0:
        return "IDENTICAL"
    o_max, o_mean = row["ours_max"], row["ours_mean"]
    i_max, i_mean = row["inductor_max"], row["inductor_mean"]
    if o_max <= i_max and o_mean <= i_mean:
        return "MORE_ACCURATE"
    worst_max = tol * max(i_max, row["eager_max"])
    worst_mean = tol * max(i_mean, row["eager_mean"])
    if i_max == 0.0 and row["eager_max"] == 0.0:
        # Both arms ARE the reference (fp64 inputs): allow 2 ULP of the output dtype, only here.
        worst_max, worst_mean = 2 * row.get("ulp_max", 0.0), 2 * row.get("ulp_mean", 0.0)
    else:
        # One rounding step at max magnitude is luck, not a failure; the mean bound catches systematic error.
        worst_max = max(worst_max, row.get("ulp_max", 0.0))
    if o_max <= worst_max and o_mean <= worst_mean:
        return "WITHIN_TOL"
    return "FAIL"


def load_spec(path: str) -> Any:
    spec = importlib.util.spec_from_file_location(f"kernel_oracle_spec_{Path(path).stem}", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load spec {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not callable(getattr(mod, "cases", None)):
        raise SystemExit(f"{path}: spec must define cases(args) -> iterable of case dicts")
    return mod


def _sync(torch: Any) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time(torch: Any, fn: Any, repeats: int, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    _sync(torch)
    ts = []
    cuda = torch.cuda.is_available()
    for _ in range(repeats):
        if cuda:
            s, e = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        else:
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def _peak_mib(torch: Any, fn: Any) -> Optional[float]:
    """Extra MiB one call allocates at its peak (CUDA only): a fusion that drops an HBM intermediate shows here."""
    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**20


def apply_inductor_config(torch: Any, cfg: dict) -> None:
    """--inductor-config K=V: the settings the product compiles with (Studio's regional compile sets
    emulate_precision_casts=True, which moves Inductor from ~5% of int8 codes off eager to ~0.002%)."""
    import torch._inductor.config as ic

    for k, v in cfg.items():
        if not hasattr(ic, k):
            raise SystemExit(f"--inductor-config: torch._inductor.config has no {k!r} on torch {torch.__version__}")
        setattr(ic, k, v)


def parse_config_value(v: str) -> Any:
    low = v.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def _err(torch: Any, out: Any, ref: Any) -> tuple:
    """Max / mean |out - ref| over the elements where ref is finite (NaN / inf positions are checked by
    _nonfinite_match): one NaN in ref (a fully masked attention row) would otherwise make every
    comparison False and FAIL a kernel that matches stock exactly."""
    r = ref.detach().double()
    m = torch.isfinite(r)
    if not bool(m.any().item()):
        return 0.0, 0.0
    d = (out.detach().double() - r).abs()[m]
    return float(d.max().item()), float(d.mean().item())


def _nonfinite_match(torch: Any, out: Any, ref: Any) -> bool:
    """out is finite where ref is, and has the same NaN / +-inf where ref does not."""
    o, r = out.detach().double(), ref.detach().double()
    fin = torch.isfinite(r)
    if not bool(torch.isfinite(o)[fin].all().item()):
        return False
    nf = ~fin
    return bool(((o[nf] == r[nf]) | (o[nf].isnan() & r[nf].isnan())).all().item())


def _n_differ(a: Any, b: Any) -> int:
    import torch

    a, b = a.detach().contiguous(), b.detach().contiguous()
    # Bit-level: -0.0 and 0.0 differ (FTZ and fused paths disagree on signed zeros); NaNs match any NaN.
    eq = ((a == b) & (torch.signbit(a) == torch.signbit(b))) | (a.isnan() & b.isnan())
    return int((~eq).sum().item())


def special_values(dtype: Any, device: Any = "cpu") -> Any:
    """Edge inputs for a spec: signed zeros, the smallest and largest subnormal, the smallest normal, 1 +- 1 ULP,
    the largest finite value and half of it, each with both signs. Products of subnormals (or of a subnormal and a
    small scale) separate flush-to-zero kernels from IEEE ones."""
    import torch

    fi = torch.finfo(dtype)
    tiny = fi.tiny
    sub_min = float(torch.tensor(tiny, dtype = dtype) * torch.tensor(fi.eps, dtype = dtype)) if dtype != torch.float64 \
        else tiny * fi.eps  # fmt: skip
    vals = [0.0, sub_min, tiny - sub_min, tiny, 1.0 - fi.eps / 2, 1.0, 1.0 + fi.eps, fi.max / 2, fi.max]
    t = torch.tensor(vals, dtype = torch.float64)
    return torch.cat([t, -t]).to(dtype).to(device)


def _snapshot(ts: Any) -> list:
    return [t.detach().clone() for t in ts]


def _detached(t: Any) -> Any:
    return None if t is None else t.detach().clone()


def _mutated(before: list, ts: Any) -> bool:
    """True if any tensor changed since the snapshot; changed tensors are restored so later arms see the inputs."""
    import torch

    changed = False
    for b, t in zip(before, ts):
        if _n_differ(b, t) != 0:
            changed = True
            with torch.no_grad():
                t.copy_(b)
    return changed


def apply_gates(row: dict, base_verdict: str) -> str:
    """Flags from the compiled-ours arm, the mutation checks and the golden arm, and the verdict they force."""
    flags = row.setdefault("flags", [])
    if row.get("compiled_error"):
        flags.append("COMPILE_ERROR")
    if row.get("compiled_breaks"):
        flags.append(f"BREAKS {row['compiled_breaks']}")
    if row.get("differ_compiled"):
        flags.append(f"COMPILED_DRIFT {row['differ_compiled']}")
    if row.get("mutates_input"):
        flags.append("MUTATES_INPUT")
    if row.get("mutates_grad"):
        flags.append("MUTATES_GRAD")
    if row.get("traceable") and row.get("compiled_skipped"):
        flags.append("TRACEABLE_UNCHECKED")
    if row.get("mutates_input_compiled"):
        flags.append("MUTATES_INPUT_COMPILED")
    if row.get("mutates_grad_compiled"):
        flags.append("MUTATES_GRAD_COMPILED")
    if row.get("differ_from_golden"):
        flags.append(f"GOLDEN_DIFF {row['differ_from_golden']}")
    if row.get("inference_error"):
        flags.append("INFERENCE_ERROR")
    if row.get("differ_inference"):
        flags.append(f"INFERENCE_DRIFT {row['differ_inference']}")
    if base_verdict == "VOID":
        return base_verdict
    fail = bool(row.get("differ_from_golden"))
    if row.get("traceable") and (row.get("compiled_error") or row.get("compiled_breaks") or row.get("differ_compiled")
                                 or row.get("mutates_input_compiled") or row.get("mutates_grad_compiled")):
        fail = True
    if row.get("pure") and (row.get("mutates_input") or row.get("mutates_grad")):
        fail = True
    if row.get("inference") and (row.get("inference_error") or row.get("differ_inference")):
        fail = True
    if fail:
        return "FAIL"
    if row.get("traceable") and row.get("compiled_skipped") and base_verdict in PASS_VERDICTS:
        return "VOID"  # a traceability claim nobody checked is not a pass
    return base_verdict


def _fill_accuracy(torch: Any, row: dict, o: Any, s: Any, i_out: Any, ref: Any) -> bool:
    """Shape, non-finite, error and differ-count fields of one compared output; False on a shape mismatch."""
    row["shape_ok"] = tuple(o.shape) == tuple(s.shape) == tuple(i_out.shape) == tuple(ref.shape)
    if not row["shape_ok"]:
        row["shapes"] = {"ours": list(o.shape), "stock": list(s.shape), "inductor": list(i_out.shape),
                         "ref": list(ref.shape)}  # fmt: skip
        return False
    row["finite"] = _nonfinite_match(torch, o, ref)
    eps = torch.finfo(o.dtype).eps if o.is_floating_point() else 0.0
    ref_abs = ref.detach().double().abs()
    ref_abs = ref_abs[torch.isfinite(ref_abs)]
    row["ulp_max"] = float(eps * ref_abs.max().item()) if ref_abs.numel() else 0.0
    row["ulp_mean"] = float(eps * ref_abs.mean().item()) if ref_abs.numel() else 0.0
    row["ours_max"], row["ours_mean"] = _err(torch, o, ref)
    row["inductor_max"], row["inductor_mean"] = _err(torch, i_out, ref)
    row["eager_max"], row["eager_mean"] = _err(torch, s, ref)
    row["numel"] = int(o.numel())
    row["differ_from_inductor"] = _n_differ(o, i_out)
    row["differ_from_eager"] = _n_differ(o, s)
    return True


def _graph_breaks(torch: Any) -> int:
    import torch._dynamo.utils as du

    return int(sum(du.counters["graph_break"].values()))


def _disable_compile_caches(torch: Any) -> None:
    """Every compile in this process builds fresh code. Re-applied per case: a spec may import a library that turns
    the caches back on (Unsloth sets TORCHINDUCTOR_FX_GRAPH_CACHE=1). Seen on torch 2.13: the FX graph cache key
    holds a triton_op call but not the Triton source behind it, so after editing the kernel (or checking out another
    tree) a warm cache returns the previous kernel's code and a compiled-arm verdict describes the wrong kernel."""
    import importlib

    for mod, key in (("torch._inductor.config", "fx_graph_cache"), ("torch._inductor.config", "fx_graph_remote_cache"),
                     ("torch._functorch.config", "enable_autograd_cache"),
                     ("torch._functorch.config", "enable_remote_autograd_cache")):  # fmt: skip
        try:
            setattr(importlib.import_module(mod), key, False)
        except (ImportError, AttributeError):  # an older torch without that cache
            pass


def run_case(torch: Any, case: dict, *, out_dir: Path, repeats: int, inductor: bool, dynamic: bool,
             compile_ours: bool = True, compile_backend: str = "inductor") -> list:
    """The forward row, then one "<name>.grad[i]" row per entry of case["inputs"]."""
    name = case["name"]
    traceable = bool(case.get("traceable", False))
    common = {"expect": case.get("expect", "fused"), "note": case.get("note", ""), "traceable": traceable,
              "pure": bool(case.get("pure", False)), "inference": bool(case.get("inference", False))}  # fmt: skip
    row = {"case": name, **common}
    inputs = list(case.get("inputs") or [])
    watched = inputs + list(case.get("watch") or [])
    try:
        with torch.no_grad():
            ref = case["ref"]() if callable(case["ref"]) else case["ref"]
            before = _snapshot(watched)
            o = _detached(case["ours"]())  # a copy: ours may return (a view of) an input it rotated in place
            row["mutates_input"] = _mutated(before, watched)
            s = case["stock"]()
            codes: list = []
            if inductor:
                from torch._inductor.utils import run_and_get_code

                torch._dynamo.reset()
                compiled = torch.compile(case["stock"], fullgraph = True, dynamic = dynamic)
                i_out, codes = run_and_get_code(compiled)
            else:
                compiled, i_out = case["stock"], s
            g = case["golden"]() if case.get("golden") else None
    except Exception as exc:  # noqa: BLE001 - reported as VOID, never a pass
        row["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:300] if str(exc) else ''}"
        row["verdict"] = "VOID"
        return [row]
    source = "\n\n# ==== next graph ====\n\n".join(codes)
    if source:
        (out_dir / "inductor").mkdir(parents = True, exist_ok = True)
        (out_dir / "inductor" / f"{name}.py").write_text(source)
    ours_src = "\n\n".join(_src_of(k) for k in case.get("kernels", []) or [])
    if not _fill_accuracy(torch, row, o, s, i_out, ref):
        row["verdict"] = "FAIL"
        return [row]
    if g is not None:
        row["differ_from_golden"] = _n_differ(o, g) if tuple(g.shape) == tuple(o.shape) else int(o.numel())

    # Grad mode and no_grad lower differently (a triton_op re-emitted in one, opaque in the other): run both.
    co = None
    if compile_ours:
        import torch._dynamo.utils as du

        row["differ_compiled"], row["compiled_breaks"] = 0, 0
        for mode in ("grad", "no_grad"):
            torch._dynamo.reset()
            du.counters.clear()
            before = _snapshot(watched)
            try:
                co = torch.compile(case["ours"], dynamic = dynamic, backend = compile_backend)
                with torch.no_grad() if mode == "no_grad" else torch.enable_grad():
                    o_c = _detached(co())
                n = _n_differ(o_c, o) if tuple(o_c.shape) == tuple(o.shape) else int(o.numel())
                row[f"differ_compiled_{mode}"] = n
                row["differ_compiled"] = max(row["differ_compiled"], n)
            except Exception as exc:  # noqa: BLE001
                row["compiled_error"] = f"{mode}: {type(exc).__name__}: {str(exc).splitlines()[0][:300] if str(exc) else ''}"
                co = None
            finally:
                row["mutates_input_compiled"] = _mutated(before, watched) or row.get("mutates_input_compiled", False)
            row[f"compiled_breaks_{mode}"] = _graph_breaks(torch)
            row["compiled_breaks"] += row[f"compiled_breaks_{mode}"]
            if co is None:
                break
    else:
        row["compiled_skipped"] = common["compiled_skipped"] = True
    # Weight-subclass detach / version-counter writes can raise only under inference_mode, never under no_grad.
    for label, fn in (("eager", case["ours"]), ("compiled", "compile")):
        if label == "compiled" and not compile_ours:
            continue
        before = _snapshot(watched)
        try:
            torch._dynamo.reset()
            call = torch.compile(case["ours"], dynamic = dynamic, backend = compile_backend) if fn == "compile" else fn
            with torch.inference_mode():
                o_inf = call()
            o_inf = _detached(o_inf)
            n = _n_differ(o_inf, o) if tuple(o_inf.shape) == tuple(o.shape) else int(o.numel())
            row["differ_inference"] = max(row.get("differ_inference", 0), n)
            row[f"differ_inference_{label}"] = n
        except Exception as exc:  # noqa: BLE001
            msg = f"{label}: {type(exc).__name__}: {str(exc).splitlines()[0][:300] if str(exc) else ''}"
            row["inference_error"] = row.get("inference_error") or msg
        finally:
            _mutated(before, watched)
    before = _snapshot(watched)  # timing repeats ours; an in-place kernel must not drift the inputs of later arms
    with torch.no_grad():
        row["ours_peak_mib"] = _peak_mib(torch, case["ours"])
        _mutated(before, watched)
        row["inductor_peak_mib"] = _peak_mib(torch, compiled)
        row["ours_ms"] = _time(torch, case["ours"], repeats)
        _mutated(before, watched)
        row["inductor_ms"] = _time(torch, compiled, repeats)
        row["eager_ms"] = _time(torch, case["stock"], repeats)
    row["inductor_digest"] = digest(source)
    if ours_src:
        row["ours_digest"] = digest(ours_src)
    rows = [row]
    if inputs:
        rows += _grad_rows(torch, case, common, o, co, inputs, watched, inductor, dynamic, row)
    return rows


def _grad_rows(torch: Any, case: dict, common: dict, o: Any, co: Any, inputs: list, watched: list, inductor: bool,
               dynamic: bool, fwd_row: dict) -> list:
    name = case["name"]
    if case.get("grad_out") is not None:
        grad_out = case["grad_out"]
    else:
        gen = torch.Generator(device = "cpu").manual_seed(1234)
        grad_out = torch.randn(o.shape, generator = gen, dtype = torch.float32).to(o.device, o.dtype)

    def grads(fn: Any) -> tuple:
        with torch.enable_grad():
            out = fn()
            # Copies: a backward may hand back the incoming gradient's buffer as its result.
            return tuple(_detached(g) for g in torch.autograd.grad(out, inputs, grad_out, allow_unused = True))

    if common.get("expect") == "golden":  # golden is a forward-only comparison
        common = {**common, "expect": "fused"}
    try:
        ref = tuple(case["ref_grad"]())
        if len(ref) != len(inputs):
            raise ValueError(f"ref_grad returned {len(ref)} tensors for {len(inputs)} inputs")
        gb, before = _snapshot([grad_out]), _snapshot(watched)
        g_ours = grads(case["ours"])
        mutates_grad = _mutated(gb, [grad_out])
        fwd_row["mutates_input"] = _mutated(before, watched) or fwd_row.get("mutates_input", False)
        g_stock = grads(case["stock"])
        g_ind = grads(torch.compile(case["stock"], dynamic = dynamic)) if inductor else g_stock
        g_comp, comp_err, mutates_grad_compiled = None, None, False
        if co is not None:
            before = _snapshot(watched)
            gb = _snapshot([grad_out])
            n0 = _graph_breaks(torch)
            try:
                g_comp = grads(co)
            except Exception as exc:  # noqa: BLE001
                comp_err = f"{type(exc).__name__}: {str(exc).splitlines()[0][:300] if str(exc) else ''}"
            # grads(co) recompiles the grad-mode forward, whose breaks the forward row already counted.
            extra = _graph_breaks(torch) - n0 - fwd_row.get("compiled_breaks_grad", 0)
            fwd_row["compiled_breaks"] = fwd_row.get("compiled_breaks", 0) + max(0, extra)
            mutates_grad_compiled = _mutated(gb, [grad_out])
            fwd_row["mutates_input_compiled"] = fwd_row.get("mutates_input_compiled") or _mutated(before, watched)
    except Exception as exc:  # noqa: BLE001
        return [{"case": f"{name}.grad", **common, "verdict": "VOID",
                 "error": f"{type(exc).__name__}: {str(exc).splitlines()[0][:300] if str(exc) else ''}"}]  # fmt: skip
    rows = []
    for i, r in enumerate(ref):
        row = {"case": f"{name}.grad[{i}]", **common, "mutates_grad": mutates_grad,
               "mutates_grad_compiled": mutates_grad_compiled}
        if g_ours[i] is None or g_stock[i] is None or g_ind[i] is None:
            row.update(verdict = "VOID", error = "no gradient reached this input")
            rows.append(row)
            continue
        if not _fill_accuracy(torch, row, g_ours[i], g_stock[i], g_ind[i], r):
            row["verdict"] = "FAIL"
            rows.append(row)
            continue
        if comp_err:
            row["compiled_error"] = comp_err
        elif g_comp is not None:
            row["differ_compiled"] = _n_differ(g_comp[i], g_ours[i]) if g_comp[i] is not None else int(r.numel())
        rows.append(row)
    return rows


def to_markdown(rows: list, meta: dict) -> str:
    lines = [f"# kernel_oracle: {meta.get('spec')}", "",
             f"torch {meta.get('torch')}, triton {meta.get('triton')}, device {meta.get('device')}", "",
             "| case | verdict | ours max / mean err | Inductor max / mean | eager max / mean | vs Inductor (elements) | "
             "ms ours / Inductor / eager | peak MiB ours / Inductor | flags |",
             "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in rows:
        if r.get("verdict") == "VOID" and "ours_max" not in r:
            lines.append(f"| {r['case']} | VOID | {r.get('error', '')} | | | | | | {' '.join(r.get('flags', []))} |")
            continue
        if "ours_max" not in r:
            lines.append(f"| {r['case']} | {r['verdict']} | shape mismatch {r.get('shapes', '')} | | | | | | |")
            continue
        lines.append(
            f"| {r['case']} | {r['verdict']} | {r['ours_max']:.2e} / {r['ours_mean']:.2e} | "
            f"{r['inductor_max']:.2e} / {r['inductor_mean']:.2e} | {r['eager_max']:.2e} / {r['eager_mean']:.2e} | "
            f"{differ(r['differ_from_inductor'], r['numel'])} | "
            + (f"{r['ours_ms']:.3f} / {r['inductor_ms']:.3f} / {r['eager_ms']:.3f}" if "ours_ms" in r else "")
            + " | "
            + (f"{r['ours_peak_mib']:.1f} / {r['inductor_peak_mib']:.1f}" if r.get("ours_peak_mib") is not None else "")
            + f" | {' '.join(r.get('flags', []))} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description = __doc__.split("\n\n")[0])
    ap.add_argument("spec", help = "spec .py defining cases(args)")
    ap.add_argument("--out", default = None, help = "output dir (default outputs/kernel_oracle/<spec stem>)")
    ap.add_argument("--spec-arg", action = "append", default = [], metavar = "K=V", help = "passed to cases()")
    ap.add_argument("--only", default = "", help = "comma-separated substrings of case names to run")
    ap.add_argument("--repeats", type = int, default = 25)
    ap.add_argument("--tol", type = float, default = 1.5,
                    help = "WITHIN_TOL bound: ours error <= tol x max(Inductor, eager) error (default 1.5)")
    ap.add_argument("--perf-tol", type = float, default = 0.10,
                    help = "flag SLOWER when ours_ms > (1 + perf_tol) x inductor_ms (default 0.10)")
    ap.add_argument("--perf-gate", action = "store_true", help = "SLOWER fails the case")
    ap.add_argument("--no-inductor", action = "store_true",
                    help = "skip torch.compile (the Inductor arm becomes eager); for hosts without a compiler")
    ap.add_argument("--dynamic", action = "store_true", help = "torch.compile(dynamic=True) for the Inductor arm")
    ap.add_argument("--no-compile-ours", action = "store_true",
                    help = "skip the torch.compile(ours) arm (graph breaks, compiled drift)")
    ap.add_argument("--compile-backend", default = "inductor",
                    help = "backend for the compiled-ours arm (aot_eager: Dynamo + AOT autograd without a compiler)")
    ap.add_argument("--inductor-cache", action = "store_true",
                    help = "keep Inductor's FX graph / AOT autograd caches (off by default: a torch.library.triton_op "
                           "call is cached without its kernel source, so an edited kernel can be served stale)")
    ap.add_argument("--nondeterministic", action = "store_true",
                    help = "leave cuDNN algorithm choice free; by default deterministic=True, benchmark=False so a "
                           "conv in both arms cannot pick different algorithms (and roundings) run to run")
    ap.add_argument("--inductor-config", action = "append", default = [], metavar = "K=V",
                    help = "torch._inductor.config setting for every compile, as the product sets it (repeatable), "
                           "e.g. emulate_precision_casts=True for Studio's regional compile")
    a = ap.parse_args(argv)

    import torch

    spec_args = {}
    for kv in a.spec_arg:
        if "=" not in kv:
            ap.error(f"--spec-arg needs K=V, got {kv!r}")
        k, v = kv.split("=", 1)
        spec_args[k] = v
    inductor_cfg = {}
    for kv in a.inductor_config:
        if "=" not in kv:
            ap.error(f"--inductor-config needs K=V, got {kv!r}")
        k, v = kv.split("=", 1)
        inductor_cfg[k.strip()] = parse_config_value(v)
    if not a.nondeterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    mod = load_spec(a.spec)
    out_dir = Path(a.out or Path("outputs") / "kernel_oracle" / Path(a.spec).stem)
    out_dir.mkdir(parents = True, exist_ok = True)
    only = [s for s in a.only.split(",") if s]
    rows = []
    seen = set()
    for case in mod.cases(spec_args):
        name = case["name"]
        if name in seen:
            raise SystemExit(f"duplicate case name {name!r}")
        seen.add(name)
        if only and not any(s in name for s in only):
            continue
        if not a.inductor_cache:
            _disable_compile_caches(torch)
        if inductor_cfg:
            apply_inductor_config(torch, inductor_cfg)
        for row in run_case(torch, case, out_dir = out_dir, repeats = a.repeats, inductor = not a.no_inductor,
                            dynamic = a.dynamic, compile_ours = not a.no_compile_ours,
                            compile_backend = a.compile_backend):  # fmt: skip
            base = row["verdict"] if "verdict" in row else verdict(row, a.tol)
            flags = []
            if row.get("ours_ms") and row.get("inductor_ms") and row["ours_ms"] > (1 + a.perf_tol) * row["inductor_ms"]:
                flags.append("SLOWER")
                if a.perf_gate and base in PASS_VERDICTS:
                    base = "FAIL"
            op, ip = row.get("ours_peak_mib"), row.get("inductor_peak_mib")
            if op is not None and ip is not None and op > (1 + a.perf_tol) * ip + 1.0:
                flags.append("MORE_MEMORY")
                if a.perf_gate and base in PASS_VERDICTS:
                    base = "FAIL"
            row["flags"] = flags
            row["verdict"] = apply_gates(row, base)
            rows.append(row)
            extra = f" {' '.join(row['flags'])}" if row["flags"] else ""
            if "ours_max" in row:
                print(f"{row['verdict']:13s} {row['case']:36s} err ours {row['ours_max']:.2e} ind {row['inductor_max']:.2e} "
                      f"eager {row['eager_max']:.2e} | vs Inductor {differ(row['differ_from_inductor'], row['numel'])}"
                      + (f" | ms ours {row['ours_ms']:.3f} ind {row['inductor_ms']:.3f}" if "ours_ms" in row else "")
                      + extra, flush = True)  # fmt: skip
            else:
                print(f"{row['verdict']:13s} {row['case']:36s} {row.get('error', row.get('shapes', ''))}{extra}", flush = True)
    try:
        import triton
        triton_version = triton.__version__
    except Exception:  # noqa: BLE001
        triton_version = None
    meta = {"spec": str(a.spec), "spec_args": spec_args, "torch": torch.__version__, "triton": triton_version,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "tol": a.tol,
            "perf_tol": a.perf_tol, "perf_gate": a.perf_gate, "inductor": not a.no_inductor,
            "cudnn_deterministic": not a.nondeterministic, "compile_ours": not a.no_compile_ours, "inductor_cache": a.inductor_cache,
            "compile_backend": a.compile_backend, "inductor_config": inductor_cfg}  # fmt: skip
    (out_dir / "results.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent = 1, default = str))
    (out_dir / "results.md").write_text(to_markdown(rows, meta))
    counts: dict = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    print(f"KERNEL_ORACLE {json.dumps(counts, sort_keys = True)} -> {out_dir}/results.md")
    if not rows:
        return 2
    if any(r["verdict"] == "FAIL" for r in rows):
        return 1
    if any(r["verdict"] == "VOID" for r in rows):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
