"""Load the zoo-vendored fla (0.5.1) and upstream fla latest side by side in ONE process.

Both trees use absolute `import fla...`, so each is imported with the top-level `fla` package
stubbed (skips fla/__init__, which on latest pulls fla.models -> transformers 5) and its
sys.modules entries are snapshotted. `use(name)` swaps the snapshot back in, so lazy
function-level `from fla...` imports resolve to the tree being timed.
"""
import os, sys, contextlib, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import _stub_pkg, setup_paths

_SNAP = {}
_NS = {}


def _clear():
    for k in [k for k in sys.modules if k == "fla" or k.startswith("fla.")]:
        del sys.modules[k]


def _load(name, root, version):
    _clear()
    t0 = time.time()
    _stub_pkg("fla", os.path.join(root, "fla")).__version__ = version
    import importlib
    ns = {}
    gdr = importlib.import_module("fla.ops.gated_delta_rule")
    ns["chunk_gated_delta_rule"] = gdr.chunk_gated_delta_rule
    ns["fused_recurrent_gated_delta_rule"] = gdr.fused_recurrent_gated_delta_rule
    try:
        mods = importlib.import_module("fla.modules")
        ns["FusedRMSNormGated"] = getattr(mods, "FusedRMSNormGated", None)
    except Exception as ex:  # noqa
        ns["FusedRMSNormGated"] = None
        ns["modules_err"] = repr(ex)
    for cand in ("fla.modules.l2norm", "fla.modules.norm.l2norm"):
        try:
            m = importlib.import_module(cand)
            ns["l2norm"] = getattr(m, "l2norm", None) or getattr(m, "l2_norm", None)
            break
        except Exception:
            pass
    for cand in ("fla.modules.causal_conv1d", "fla.modules.conv"):
        try:
            m = importlib.import_module(cand)
            ns["causal_conv1d"] = getattr(m, "causal_conv1d", None)
            ns["causal_conv1d_update"] = getattr(m, "causal_conv1d_update", None)
            break
        except Exception:
            pass
    ns["import_s"] = time.time() - t0
    ns["root"] = root
    _SNAP[name] = {k: v for k, v in sys.modules.items() if k == "fla" or k.startswith("fla.")}
    _NS[name] = ns
    return ns


def load_both():
    """returns dict impl -> namespace; impl in ('vendored', 'latest' [if AUDIT_FLA_LATEST])."""
    setup_paths()
    if _NS:
        return _NS
    import importlib.util
    zoo = os.path.dirname(importlib.util.find_spec("unsloth_zoo").origin)
    os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "0")
    _load("vendored", os.path.join(zoo, "_vendored"), "0.5.1")
    lat = os.environ.get("AUDIT_FLA_LATEST")
    if lat and os.path.isdir(os.path.join(lat, "fla")):
        try:
            _load("latest", lat, "0.6.0")
        except Exception as ex:
            _NS["latest_err"] = repr(ex)[:400]
    _NS["autotune_cap"] = cap_autotune()
    use("vendored")
    return _NS


def small_gpu():
    import torch
    return os.environ.get("AUDIT_FORCE_SMALL") == "1" or torch.cuda.get_device_properties(0).total_memory < 20 * 2**30


def cap_autotune():
    """Bound fla's first-call autotune cost on small / slow tiers (Colab T4: 2 vCPUs, every config is a JIT
    compile). AUDIT_FLA_AT = full | fast | trim:N (default trim:4 on < 20 GB cards, else full).
    fast: do_bench warmup 5 ms / rep 20 ms instead of 25 / 100. trim:N: also keep N configs, evenly strided
    over the original list (identical trimming for vendored and latest). Applied to every Autotuner of both trees."""
    import math, triton
    mode = os.environ.get("AUDIT_FLA_AT") or ("trim:4" if small_gpu() else "full")
    if mode == "full":
        return mode
    n_cap = int(mode.split(":")[1]) if mode.startswith("trim:") else None
    seen = set()
    for snap in _SNAP.values():
        for m in snap.values():
            if m is None:
                continue
            for obj in list(vars(m).values()):
                o = obj
                for _ in range(4):
                    if isinstance(o, triton.runtime.autotuner.Autotuner):
                        break
                    o = getattr(o, "fn", None)
                    if o is None:
                        break
                if not isinstance(o, triton.runtime.autotuner.Autotuner) or id(o) in seen:
                    continue
                seen.add(id(o))
                o._do_bench = (lambda kc, quantiles: triton.testing.do_bench(kc, warmup=5, rep=20, quantiles=quantiles))
                if n_cap and len(o.configs) > n_cap:
                    step = math.ceil(len(o.configs) / n_cap)
                    o.configs = o.configs[::step][:n_cap]
    return mode


def use(name):
    _clear()
    sys.modules.update(_SNAP[name])


def wrap(name, fn):
    """fn bound to its tree: swap modules once on first call of a timing block is the caller's job;
    this wrapper swaps only if the active tree differs (cheap identity check)."""
    snap = _SNAP[name]
    flamod = snap.get("fla")

    def f(*a, **k):
        if sys.modules.get("fla") is not flamod:
            use(name)
        return fn(*a, **k)
    return f


def autotuners(name):
    """All triton Autotuner objects in a tree's loaded modules."""
    import triton
    out = {}
    for mn, m in _SNAP.get(name, {}).items():
        for an, obj in list(vars(m).items()) if m is not None else []:
            if isinstance(obj, triton.runtime.autotuner.Autotuner):
                out["%s.%s" % (mn.replace("fla.", ""), an)] = obj
    return out
