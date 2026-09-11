"""AST proof: main vs PR raw_text.py differ only in two If conditions inside smart_chunk_text."""
import ast, sys

m = ast.parse(open("ci_pr10734/main_raw_text.py", encoding = "utf-8").read())
h = ast.parse(open("unsloth/dataprep/raw_text.py", encoding = "utf-8").read())
assert len(m.body) == len(h.body)
changed = []
for a, b in zip(m.body, h.body):
    if ast.dump(a) == ast.dump(b):
        continue
    assert isinstance(a, ast.ClassDef) and a.name == "RawTextDataLoader", ast.unparse(a)[:200]
    fa = {f.name: f for f in a.body if isinstance(f, ast.FunctionDef)}
    fb = {f.name: f for f in b.body if isinstance(f, ast.FunctionDef)}
    assert fa.keys() == fb.keys()
    for name in fa:
        if ast.dump(fa[name]) == ast.dump(fb[name]):
            continue
        assert name == "smart_chunk_text", name
        assert ast.dump(fa[name].args) == ast.dump(fb[name].args)
        ia = [n for n in ast.walk(fa[name]) if isinstance(n, ast.If)]
        ib = [n for n in ast.walk(fb[name]) if isinstance(n, ast.If)]
        assert len(ia) == len(ib)
        for x, y in zip(ia, ib):
            if ast.dump(x.test) != ast.dump(y.test):
                changed.append((ast.unparse(x.test), ast.unparse(y.test)))
                x.test = y.test
        assert ast.dump(fa[name]) == ast.dump(fb[name]), "non-condition change in smart_chunk_text"
print("changed conditions:", changed)
assert changed == [
    ("end_idx == len(tokens) or len(chunk_tokens_list) == chunk_size", "end_idx == len(tokens)"),
    ("end_idx == len(tokens) or len(chunk_tokens) == chunk_size", "end_idx == len(tokens)"),
], changed
print("AST CHECK PASS")
