# SPDX-License-Identifier: AGPL-3.0-only
"""Verify the approved sandbox change does not alter unrelated Studio code."""
import ast
import subprocess

BASE = "73eb6fb29d3a3a013d4c730ea092f384ab2eaa74"
HEAD = "842c826a95e99a5fd31a4e546604e3d42265fd32"
changed = subprocess.check_output(["git", "diff", "--name-only", BASE, HEAD], text=True).splitlines()
allowed = {".github/workflows/studio-backend-ci.yml", "studio/backend/core/inference/tools.py"}
allowed.update("studio/backend/core/inference/" + name + ".py" for name in
               ("os_sandbox", "sandbox_linux", "sandbox_macos", "sandbox_probe", "sandbox_seccomp", "sandbox_landlock"))
assert all(path in allowed or path.startswith("studio/backend/tests/") for path in changed), changed

def tree(ref):
    return ast.parse(subprocess.check_output(["git", "show", ref + ":studio/backend/core/inference/tools.py"], text=True))
old, new = tree(BASE), tree(HEAD)
def definitions(node):
    return {n.name: ast.dump(n, include_attributes=False) for n in node.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
before, after = definitions(old), definitions(new)
modified = {name for name in before if before[name] != after.get(name)}
assert modified == {"execute_tool", "_python_exec", "_bash_exec"}, modified
def bindings(node):
    result = {}
    for item in node.body:
        if isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name):
                    result[target.id] = ast.dump(item.value, include_attributes=False)
    return result
old_bindings, new_bindings = bindings(old), bindings(new)
assert all(new_bindings.get(name) == value for name, value in old_bindings.items())
print("PASS: only execute_tool, _python_exec and _bash_exec change existing executable definitions")
print("PASS: existing module bindings, tool schemas and prompt constants are unchanged")
print("PASS: installer, updater, Desktop, shortcuts, auth, persistence, frontend and hardware-routing files are byte-identical to base")
