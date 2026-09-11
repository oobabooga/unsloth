"""Real gpt2 tokenizer, main vs PR, both output modes, through the CLI's loader helper."""
import importlib.util, sys


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
cli = load("unsloth_cli_script", "unsloth-cli.py")
text = " ".join(f"The quick brown fox number {i} jumps over the lazy dog." for i in range(40))
full = tok(text, add_special_tokens = False)["input_ids"]
CS, ST = 32, 8
res = {}
for v, path in [("main", "ci_pr10734/main_raw_text.py"), ("head", "unsloth/dataprep/raw_text.py")]:
    rt = load(f"rt_{v}", path)
    loader = cli._raw_text_loader_for_backend(rt.RawTextDataLoader, tok, False, CS, ST)
    ch = loader.chunk_text(text, return_tokenized = True)
    ids = [c["input_ids"] for c in ch]
    eos = [int(x[-1] == tok.eos_token_id) for x in ids]
    body = [x[:-1] if x[-1] == tok.eos_token_id else x for x in ids]
    rec = body[0] + [t for b in body[1:] for t in b[ST:]]
    tc = loader.chunk_text(text, return_tokenized = False)
    res[v] = dict(n = len(ids), lens = [len(x) for x in ids], eos = eos,
                  text_eos = [int(c.endswith(tok.eos_token)) for c in tc], rec = rec == full)
    print(v, res[v])
assert res["main"]["rec"] and res["head"]["rec"]
assert sum(res["main"]["eos"]) == res["main"]["n"] and sum(res["main"]["text_eos"]) == res["main"]["n"]
assert res["head"]["eos"] == [0] * (res["head"]["n"] - 1) + [1]
assert res["head"]["text_eos"] == [0] * (res["head"]["n"] - 1) + [1]
assert all(l == CS for l in res["head"]["lens"][:-1])
print("COMPARE PASS")
