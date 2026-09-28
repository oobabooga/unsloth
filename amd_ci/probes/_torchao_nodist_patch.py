"""Prototype: let real torchao import on a torch without torch.distributed (AMD Windows ROCm).

torchao imports DTensor / DeviceMesh / functional collectives at module scope and keys dispatch
tables on torch.ops c10d ops, none of which exist on such a build (pytorch/ao#3452, #4761). Both
only feed isinstance() checks and collective handlers that can never run there, so for the
duration of `import torchao` only: serve inert stand-ins for those modules and inert keys for the
missing ops, then remove the stand-ins so nothing else sees a fake torch.distributed.
"""

import importlib
import importlib.util
import sys
import types

_DIST_MODULES = (
    "torch.distributed._functional_collectives",
    "torch.distributed._tensor",
    "torch.distributed.tensor",
    "torch.distributed.device_mesh",
)
_OP_NAMESPACES = frozenset({"c10d_functional", "_c10d_functional", "c10d", "_dtensor"})
_MISSING = object()


class _NeverMeta(type):
    def __instancecheck__(cls, obj):
        return False

    def __subclasscheck__(cls, sub):
        return False

    def __call__(cls, *a, **k):
        raise RuntimeError(f"{cls.__name__} needs torch.distributed, which this torch build lacks.")


def _stub_module(name):
    mod = types.ModuleType(name)
    cache = {}

    def __getattr__(attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        if attr not in cache:
            if attr[:1].isupper():
                cache[attr] = _NeverMeta(attr, (), {"__module__": name})
            else:
                def _unavailable(*a, _attr = attr, **k):
                    raise RuntimeError(f"{name}.{_attr} needs torch.distributed, which this torch build lacks.")
                cache[attr] = _unavailable
        return cache[attr]

    mod.__getattr__ = __getattr__
    mod.__unsloth_nodist_stub__ = True
    return mod


class _InertOverload:
    __slots__ = ("qualname",)

    def __init__(self, qualname):
        self.qualname = qualname

    def __repr__(self):
        return f"<absent op {self.qualname}>"


class _InertPacket:
    def __init__(self, qualname):
        self._qualname = qualname
        self._overloads = {}

    def __getattr__(self, overload):
        if overload.startswith("__"):
            raise AttributeError(overload)
        return self._overloads.setdefault(overload, _InertOverload(f"{self._qualname}.{overload}"))

    def __call__(self, *a, **k):
        raise RuntimeError(f"{self._qualname} needs torch.distributed, which this torch build lacks.")


def fix_torchao_without_torch_distributed():
    """Import real torchao on a torch whose torch.distributed is unavailable. Returns True if
    torchao was imported through the shim; no-op (False) anywhere torchao imports on its own."""
    if "torchao" in sys.modules or importlib.util.find_spec("torchao") is None:
        return False
    import torch

    try:
        if torch.distributed.is_available():
            return False
    except Exception:
        return False

    op_ns_cls = torch._ops._OpNamespace
    original_getattr = op_ns_cls.__getattr__
    packets = {}

    def __getattr__(self, op_name):
        try:
            return original_getattr(self, op_name)
        except AttributeError:
            if self.name not in _OP_NAMESPACES or op_name.startswith("__"):
                raise
            key = f"{self.name}.{op_name}"
            return packets.setdefault(key, _InertPacket(key))

    saved = {name: sys.modules.get(name, _MISSING) for name in _DIST_MODULES}
    parent = sys.modules.get("torch.distributed")
    saved_attrs = {n.rsplit(".", 1)[1]: getattr(parent, n.rsplit(".", 1)[1], _MISSING) for n in _DIST_MODULES}
    before = set(sys.modules)
    ok = False
    try:
        for name in _DIST_MODULES:
            sys.modules[name] = _stub_module(name)
        op_ns_cls.__getattr__ = __getattr__
        importlib.import_module("torchao")
        ok = True
    except Exception:
        for name in set(sys.modules) - before:
            if name == "torchao" or name.startswith("torchao."):
                sys.modules.pop(name, None)
    finally:
        op_ns_cls.__getattr__ = original_getattr
        for name, mod in saved.items():
            if mod is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod
        for attr, value in saved_attrs.items():
            if value is _MISSING:
                if getattr(getattr(parent, attr, None), "__unsloth_nodist_stub__", False):
                    delattr(parent, attr)
            else:
                setattr(parent, attr, value)
        # _OpNamespace caches resolved ops as attributes; drop any inert packet cached there.
        for ns in _OP_NAMESPACES:
            ns_obj = torch.ops.__dict__.get(ns)
            if ns_obj is not None:
                for attr, value in list(vars(ns_obj).items()):
                    if isinstance(value, _InertPacket):
                        delattr(ns_obj, attr)
    return ok
