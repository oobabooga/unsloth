"""Shared helpers for the dense-kernel bench modules (fwd + bwd timing of autograd entry points,
config sweeps). Kept out of harness.py on purpose."""
import torch


def as_tuple(o):
    return tuple(o) if isinstance(o, (tuple, list)) else (o,)


def fwd_bwd(H, arms, make_inputs, shape, host=False, do_bwd=True, grad_scale=1.0):
    """arms: name -> fn(*inputs). make_inputs(): fresh tuple of inputs; tensors that need grad
    carry requires_grad (callers turn in-place-mutated inputs into non-leaves themselves).
    Times 'fwd' (grad mode on, as in training) and 'bwd' (autograd.grad, retain_graph) per arm,
    interleaved. Upstream grads are identical across arms (seeded)."""
    states = {}
    for name, fn in arms.items():
        ins = make_inputs()
        states[name] = [fn, ins, None, None, None]
    fwd = {n: (lambda s=s: s[0](*s[1])) for n, s in states.items()}
    H.interleave(fwd, "fwd", shape=shape, host=host)
    if not do_bwd:
        return
    for name, s in states.items():
        fn, ins = s[0], s[1]
        outs = [o for o in as_tuple(fn(*ins)) if torch.is_tensor(o) and o.requires_grad]
        g = torch.Generator(device="cuda").manual_seed(1234)
        grads = [torch.randn(o.shape, device=o.device, dtype=o.dtype, generator=g) * grad_scale
                 if o.dim() > 0 else torch.ones_like(o) for o in outs]
        leaves = [i for i in s[1] if torch.is_tensor(i) and i.requires_grad]
        s[2], s[3], s[4] = outs, grads, leaves
    bwd = {n: (lambda s=s: torch.autograd.grad(s[2], s[4], s[3], retain_graph=True)) for n, s in states.items()}
    H.interleave(bwd, "bwd", shape=shape, host=host)
    for s in states.values():
        s[2] = s[3] = None


def grads_of(fn, ins, seed=1234):
    outs = [o for o in as_tuple(fn(*ins)) if torch.is_tensor(o) and o.requires_grad]
    g = torch.Generator(device="cuda").manual_seed(seed)
    grads = [torch.randn(o.shape, device=o.device, dtype=o.dtype, generator=g) if o.dim() > 0
             else torch.ones_like(o) for o in outs]
    leaves = [i for i in ins if torch.is_tensor(i) and i.requires_grad]
    return [o.detach() for o in outs], list(torch.autograd.grad(outs, leaves, grads, allow_unused=True))


def config_sweep(H, what, configs, shape, budget_ms=150):
    """configs: list of (label, fn, is_default). Emits kind=config_sweep rows."""
    res = []
    for label, fn, default in configs:
        try:
            ms = H.measure(fn, budget_ms=budget_ms)[0]
        except Exception as ex:  # a config may not compile (shared memory, registers)
            H.emit(kind="config_sweep", what=what, config=label, default=default, shape=shape, err=repr(ex)[:300])
            continue
        res.append((ms, label, default))
        H.emit(kind="config_sweep", what=what, config=label, default=default, shape=shape, ms=ms)
    if res:
        best = min(res)
        dflt = [r for r in res if r[2]]
        if dflt:
            H.emit(kind="config_sweep_summary", what=what, shape=shape, best=best[1], best_ms=best[0],
                   default_ms=dflt[0][0], default_over_best=dflt[0][0] / best[0])


_ONCE = set()


def once(key):
    """True the first time per process: compile checks run on one case per module (each torch.compile
    costs tens of seconds on a 2-vCPU Colab host; graph breaks do not depend on the shape)."""
    if key in _ONCE:
        return False
    _ONCE.add(key)
    return True


def vram_budget_bytes():
    """Per-case peak budget: 12 GB on cards under 20 GB (T4 14.5 GB), else 85% of the card."""
    total = torch.cuda.get_device_properties(0).total_memory
    return 12e9 if total < 20e9 else 0.85 * total


def small_vram():
    return torch.cuda.get_device_properties(0).total_memory < 20e9
