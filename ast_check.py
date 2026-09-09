"""Structural invariants of the PR, checked without needing full git history.

1. The changed module carries exactly two `.contiguous()` calls, both appended to a
   `reshape(-1, dim)` inside Fast_RMS_Layernorm.forward / .backward.
2. Its import list is exactly the pre-PR one.
3. The PR (per the GitHub API) touches no studio/, packaging, workflow or installer file.
"""
import ast, json, sys, urllib.request

SRC = "unsloth/kernels/rms_layernorm.py"
EXPECTED_IMPORTS = sorted([
    "LlamaRMSNorm", "LlamaRMSNorm", "MllamaTextRMSNorm", "calculate_settings",
    "torch", "torch_gpu_device",
    "transformers.models.llama.modeling_llama", "transformers.models.llama.modeling_llama",
    "transformers.models.mllama.modeling_mllama", "transformers.models.mllama.modeling_mllama",
    "triton", "triton.language",
])

tree = ast.parse(open(SRC, encoding = "utf-8").read())

imports = []
for n in ast.walk(tree):
    if isinstance(n, ast.Import):
        imports += [a.name for a in n.names]
    elif isinstance(n, ast.ImportFrom):
        imports += [a.name for a in n.names]
print("module imports:", sorted(imports))
assert sorted(imports) == sorted(EXPECTED_IMPORTS), sorted(imports)

cls = next(n for n in tree.body
           if isinstance(n, ast.ClassDef) and n.name == "Fast_RMS_Layernorm")
found = {}
for fn in cls.body:
    if not isinstance(fn, ast.FunctionDef):
        continue
    for stmt in fn.body:
        txt = ast.unparse(stmt)
        if "reshape(-1, dim)" in txt:
            found[fn.name] = txt
print("reshape statements:", found)
assert found.get("forward") == "X = X.reshape(-1, dim).contiguous()", found
assert found.get("backward") == "dY = dY.reshape(-1, dim).contiguous()", found

total = sum(1 for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "contiguous")
print("total .contiguous() call sites in module:", total)
assert total == 2, total

import os
url = "https://api.github.com/repos/unslothai/unsloth/pulls/10617/files?per_page=100"
headers = {"Accept": "application/vnd.github+json", "User-Agent": "pr10617-probe"}
tok = os.environ.get("GITHUB_TOKEN")
if tok:
    headers["Authorization"] = f"Bearer {tok}"
req = urllib.request.Request(url, headers = headers)
files = [f["filename"] for f in json.load(urllib.request.urlopen(req, timeout = 60))]
print("PR files:", files)
FORBIDDEN = ("studio/", ".github/", "pyproject.toml", "setup.py", "setup.cfg",
             "MANIFEST.in", "install.sh", "install.ps1", "requirements")
bad = [f for f in files if f.startswith(FORBIDDEN) or f in FORBIDDEN]
assert not bad, f"PR touches Studio/packaging/installer files: {bad}"
assert set(files) == {"tests/test_rmsnorm_gradient_layout.py",
                      "unsloth/kernels/rms_layernorm.py"}, files
print("AST CHECK OK")
