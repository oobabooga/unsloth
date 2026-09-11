# Throwaway check for the unslothai/unsloth#10741 mirror: loads raw_text.py the two ways unsloth does.
import importlib.util, os, pickle, platform, sys, types, unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH = os.path.join(ROOT, "unsloth", "dataprep", "raw_text.py")
stub = types.ModuleType("datasets"); stub.Dataset = object; sys.modules.setdefault("datasets", stub)

def load(name, register):
    spec = importlib.util.spec_from_file_location(name, PATH)
    mod = importlib.util.module_from_spec(spec)
    if register:
        sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

mlx = load("unsloth._mlx_raw_text", register=False)   # unsloth/__init__.py MLX branch
cuda = load("unsloth_dataprep_raw_text", register=True)  # importable module, like unsloth.dataprep.raw_text
cases = {
    "Le café était très bon.": "Le café était très bon.",
    "机器学习很有趣。": "机器学习很有趣。",
    "नमस्ते दुनिया।": "नमस्ते दुनिया।",
    "I ❤️ you © 1️⃣": "I you 1",
    "a\x00b\r\n\r\n\r\nc d﻿": "ab\n\nc d",
}
for mod in (mlx, cuda):
    p = mod.TextPreprocessor()
    for raw, want in cases.items():
        got = p.clean_text(raw)
        assert got == want, (mod.__name__, raw, got, want)
        assert p.clean_text(got) == got
table = cuda.TextPreprocessor._TEXT_CHARS
assert len(table) > 0 and len(pickle.loads(pickle.dumps(table))) == 0
q = pickle.loads(pickle.dumps(cuda.TextPreprocessor()))
assert q.clean_text("café ©") == "café"
print(f"OK {platform.system()} {platform.machine()} py{platform.python_version()} ucd={unicodedata.unidata_version}")
