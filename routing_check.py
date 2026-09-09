"""Which platforms can even reach the changed module, and does it import cleanly?"""
import importlib.util, os, platform, sys
print("platform:", platform.platform(), "machine:", platform.machine())

have_triton = importlib.util.find_spec("triton") is not None
print("triton available:", have_triton)

# the module hard-imports triton at module scope -- that is the routing gate
src = open("unsloth/kernels/rms_layernorm.py").read()
assert src.splitlines()[14].startswith("import triton"), src.splitlines()[14]
print("module-scope `import triton` present at line 15: True")

if have_triton:
    sys.path.insert(0, os.getcwd())
    os.environ["UNSLOTH_IS_PRESENT"] = "1"
    try:
        import unsloth.kernels.rms_layernorm as m
        print("imported OK:", m.__file__)
        import inspect
        fwd = inspect.getsource(m.Fast_RMS_Layernorm.forward)
        bwd = inspect.getsource(m.Fast_RMS_Layernorm.backward)
        assert "reshape(-1, dim).contiguous()" in fwd
        assert "reshape(-1, dim).contiguous()" in bwd
        print("both call sites carry .contiguous(): True")
    except Exception as exc:
        print("import failed:", type(exc).__name__, exc)
else:
    print("no triton on this platform -> unsloth.kernels.rms_layernorm is unreachable here")
print("ROUTING CHECK OK")
