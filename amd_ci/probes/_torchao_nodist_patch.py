"""Prototype: let real torchao import on a torch without torch.distributed (AMD Windows ROCm).

torchao imports DTensor / DeviceMesh / functional collectives at module scope and keys dispatch
tables on torch.ops c10d ops, none of which exist on such a build (pytorch/ao#3452, #4761). Both
only feed isinstance() checks and collective handlers that can never run there. So every torchao
module (now or imported lazily later, e.g. transformers' quantizer_torchao pulling
torchao.prototype.mx_formats) executes its top level inside a window that serves inert stand-ins
for those modules and inert keys for the missing ops; outside torchao nothing changes.
"""

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import threading
import types

_OP_NAMESPACES = frozenset({"c10d_functional", "_c10d_functional", "c10d", "_dtensor"})
_MISSING = object()


class _NeverMeta(type):
    def __instancecheck__(cls, obj):
        return False

    def __subclasscheck__(cls, sub):
        return False

    def __call__(cls, *a, **k):
        raise RuntimeError(f"{cls.__name__} needs torch.distributed, which this torch build lacks.")


_IMPORTING = [0]  # depth of torchao module bodies executing right now


def _passthrough(fn):
    return fn


def _unavailable(qualname):
    def fn(*a, **k):
        # torchao applies distributed decorators (register_sharding, ...) while its modules load:
        # there they wrap nothing, since no sharded tensor can exist. Any later call is real
        # distributed work and must fail loudly.
        if _IMPORTING[0]:
            # @deco(fn) gets fn back; a factory call (register_sharding(op)) gets a no-op decorator.
            return a[0] if len(a) == 1 and not k and isinstance(a[0], types.FunctionType) else _passthrough
        raise RuntimeError(f"{qualname} needs torch.distributed, which this torch build lacks.")
    return fn


def _make_stub(module):
    """Turn a (possibly half-executed) module into an inert stand-in, in place."""
    name = module.__name__
    cache = {}

    def __getattr__(attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        if attr not in cache:
            cache[attr] = (_NeverMeta(attr, (), {"__module__": name}) if attr[:1].isupper()
                           else _unavailable(f"{name}.{attr}"))
        return cache[attr]

    module.__getattr__ = __getattr__
    if not hasattr(module, "__path__"):
        module.__path__ = []
    module.__unsloth_nodist_stub__ = True
    return module


class _StubLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return types.ModuleType(spec.name)

    def exec_module(self, module):
        _make_stub(module)


STUB_IMPORTERS = []  # (stubbed module, first non-importlib importer module) for diagnostics


def _importer():
    f = sys._getframe(2)
    while f is not None:
        name = f.f_globals.get("__name__", "")
        if not name.startswith(("importlib", "_frozen_importlib")) and name != __name__:
            return name
        f = f.f_back
    return None


class _DistStubFinder(importlib.abc.MetaPathFinder):
    """Inside the window only: any torch.distributed.* not already loaded is an inert stand-in."""

    def find_spec(self, fullname, path = None, target = None):
        if fullname.startswith("torch.distributed.") and fullname not in sys.modules:
            STUB_IMPORTERS.append((fullname, _importer()))
            return importlib.machinery.ModuleSpec(fullname, _StubLoader(), is_package = True)
        return None


class _InertPacket:
    """Stands in for a missing c10d OpOverloadPacket: every overload is a unique hashable key
    that no real op can equal, so dispatch-table entries keyed on it are unreachable."""

    def __init__(self, qualname):
        self._qualname = qualname
        self._overloads = {}

    def __getattr__(self, overload):
        if overload.startswith("__"):
            raise AttributeError(overload)
        return self._overloads.setdefault(overload, _unavailable(f"{self._qualname}.{overload}"))

    def __call__(self, *a, **k):
        raise RuntimeError(f"{self._qualname} needs torch.distributed, which this torch build lacks.")


class _Window:
    """Reentrant: stand-ins exist only while at least one torchao module body is executing."""

    def __init__(self, torch):
        self.torch = torch
        self.lock = threading.RLock()
        self.depth = 0
        self.packets = {}
        self.dist_finder = _DistStubFinder()

    def __enter__(self):
        self.lock.acquire()
        self.depth += 1
        _IMPORTING[0] = self.depth
        if self.depth == 1:
            torch = self.torch
            self.before = set(sys.modules)
            sys.meta_path.insert(0, self.dist_finder)
            ns_cls = torch._ops._OpNamespace
            self.original_getattr = original = ns_cls.__getattr__
            packets = self.packets

            def __getattr__(ns, op_name):
                try:
                    return original(ns, op_name)
                except AttributeError:
                    if ns.name not in _OP_NAMESPACES or op_name.startswith("__"):
                        raise
                    key = f"{ns.name}.{op_name}"
                    return packets.setdefault(key, _InertPacket(key))

            ns_cls.__getattr__ = __getattr__
        return self

    def __exit__(self, *exc):
        try:
            self.depth -= 1
            _IMPORTING[0] = self.depth
            if self.depth == 0:
                torch = self.torch
                torch._ops._OpNamespace.__getattr__ = self.original_getattr
                sys.meta_path.remove(self.dist_finder)
                for n in set(sys.modules) - self.before:
                    mod = sys.modules.get(n)
                    if getattr(mod, "__unsloth_nodist_stub__", False):
                        del sys.modules[n]
                        parent_name, _, attr = n.rpartition(".")
                        parent = sys.modules.get(parent_name)
                        if parent is not None and getattr(parent, attr, None) is mod:
                            delattr(parent, attr)
        finally:
            self.lock.release()
        return False


class _WindowedLoader(importlib.abc.Loader):
    def __init__(self, loader, window):
        self.loader = loader
        self.window = window

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        with self.window:
            self.loader.exec_module(module)

    def __getattr__(self, name):
        return getattr(self.loader, name)


class _TorchaoWindowFinder(importlib.abc.MetaPathFinder):
    def __init__(self, window):
        self.window = window

    def find_spec(self, fullname, path = None, target = None):
        if fullname != "torchao" and not fullname.startswith("torchao."):
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                break
        else:
            return None
        if spec.loader is not None and not isinstance(spec.loader, _WindowedLoader):
            spec.loader = _WindowedLoader(spec.loader, self.window)
        return spec


def fix_torchao_without_torch_distributed():
    """Make real torchao importable on a torch whose torch.distributed is unavailable. Returns
    True when the hook is installed and torchao imported through it; False (and nothing left
    behind) anywhere torchao imports on its own or still cannot import."""
    if any(isinstance(f, _TorchaoWindowFinder) for f in sys.meta_path):
        return "torchao" in sys.modules
    if "torchao" in sys.modules or importlib.util.find_spec("torchao") is None:
        return False
    import torch

    try:
        if torch.distributed.is_available():
            return False
    except Exception:
        return False

    finder = _TorchaoWindowFinder(_Window(torch))
    sys.meta_path.insert(0, finder)
    before = set(sys.modules)
    try:
        importlib.import_module("torchao")
        return True
    except Exception:
        sys.meta_path.remove(finder)
        for name in set(sys.modules) - before:
            if name == "torchao" or name.startswith("torchao."):
                sys.modules.pop(name, None)
        return False
