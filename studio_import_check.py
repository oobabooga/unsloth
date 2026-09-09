"""Studio's backend parent process must not pull in the changed module."""
import ast, os
main = open(os.path.join("studio", "backend", "main.py"), encoding="utf-8").read()
tree = ast.parse(main)
bad = []
for n in ast.walk(tree):
    if isinstance(n, ast.Import):
        bad += [a.name for a in n.names if a.name.split(".")[0] == "unsloth"]
    elif isinstance(n, ast.ImportFrom) and (n.module or "").split(".")[0] == "unsloth":
        bad.append(n.module)
print("studio/backend/main.py unsloth imports:", bad)
assert not bad, bad

run = open(os.path.join("studio", "backend", "run.py"), encoding="utf-8").read()
tree = ast.parse(run)
bad = []
for n in ast.walk(tree):
    if isinstance(n, ast.Import):
        bad += [a.name for a in n.names if a.name.split(".")[0] == "unsloth"]
    elif isinstance(n, ast.ImportFrom) and (n.module or "").split(".")[0] == "unsloth":
        bad.append(n.module)
print("studio/backend/run.py unsloth imports:", bad)
assert not bad, bad
print("STUDIO IMPORT CHECK OK: neither entrypoint imports unsloth at module scope")
