"""Download the smallest GGUF from a small repo into a scan folder."""
import os
import sys

from huggingface_hub import HfApi, hf_hub_download

target = sys.argv[1]
repo = sys.argv[2] if len(sys.argv) > 2 else "unsloth/Qwen3.5-0.8B-GGUF"
os.makedirs(target, exist_ok=True)

api = HfApi()
files = [f for f in api.list_repo_files(repo) if f.lower().endswith(".gguf")]
if not files:
    print("no gguf in", repo)
    raise SystemExit(1)

info = api.model_info(repo, files_metadata=True)
sizes = {s.rfilename: (s.size or 0) for s in info.siblings}
files.sort(key=lambda f: sizes.get(f, 0) or 1 << 60)
pick = files[0]
print("picking", pick, "size", sizes.get(pick))

local_dir = os.path.join(target, repo.replace("/", os.sep))
path = hf_hub_download(repo_id=repo, filename=pick, local_dir=local_dir)
print("downloaded to", path)
print("dir listing:", os.listdir(os.path.dirname(path)))
