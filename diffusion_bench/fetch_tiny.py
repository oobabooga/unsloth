#!/usr/bin/env python3
"""Snapshot the hf-internal-testing tiny random pipelines the tiny tier reads into $WORKSPACE/hf_tiny/<name>.

The tiny tier (edge/run_edge.py --tier tiny, edge/edge_comfy_sdcpp.py --tier tiny, specs/tiny_plumbing.json,
specs/studio_tiny.json) expects one directory per repo name under $WORKSPACE/hf_tiny. This fills it: public repos
only, no token (the ambient one is dropped), local_dir so nothing lands in ~/.cache/huggingface, and a repo that is
already complete is skipped. All 14 together are about 340 MB. DERIVED variants (below) are built locally from a
fetched pipe, never downloaded.

  python diffusion_bench/fetch_tiny.py                    # every pipe any tiny spec or check names
  python diffusion_bench/fetch_tiny.py --set edge         # only what run_edge.py --tier tiny needs
  python diffusion_bench/fetch_tiny.py --only tiny-wan-pipe --root /elsewhere --force
  python diffusion_bench/fetch_tiny.py --list

Exit 0 when every requested pipe is present, 1 when any download failed.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402

ORG = "hf-internal-testing"
# Edge suite (run_edge.py --tier tiny): the img_a / img_b / vid roles plus tiny.family_matrix and
# tiny.rope_overflow_isolated. Missing matrix rows are reported as "missing", not failed, so a partial set still runs.
EDGE_CORE = ["tiny-stable-diffusion-xl-pipe", "tiny-flux-pipe", "tiny-wan-pipe"]
EDGE_MATRIX = ["tiny-zimage-pipe", "tiny-qwenimage-pipe", "tiny-qwenimage21-pipe", "tiny-lumina2-pipe",
               "tiny-flux-kontext-pipe", "tiny-qwenimage-edit-pipe", "tiny-krea2-turbo-modular-pipe",
               "tiny-sd3-pipe", "tiny-sana-pipe", "tiny-random-hunyuanvideo", "tiny-cogvideox-pipe"]
SETS = {
    "core": EDGE_CORE,
    "edge": EDGE_CORE + EDGE_MATRIX,
    # specs/tiny_plumbing.json, specs/studio_tiny.json, specs/amd_strix_small.json tier 1
    "matrix": ["tiny-stable-diffusion-xl-pipe", "tiny-flux-pipe", "tiny-wan-pipe", "tiny-zimage-pipe",
               "tiny-qwenimage-pipe", "tiny-qwenimage21-pipe", "tiny-lumina2-pipe", "tiny-sd3-pipe"],
}
SETS["all"] = list(dict.fromkeys(SETS["edge"] + SETS["matrix"]))
DONE = ".fetch_complete"

# Local variants derived from a fetched pipe (never downloaded). tiny-wan-pipe ships rope_max_seq_len 32, so its
# rope table covers at most a 32 x 32 patch grid; Studio's HTTP /video route only accepts the family presets
# (1280x704 / 704x1280 for Wan2.2-TI2V-5B), a 44 x 80 grid, and diffusers raises "shape '[1, 44, 1, -1]' is invalid".
# The rope buffers are non-persistent (rebuilt from the config at load), so raising the limit changes no weight and
# leaves every position under 32 bit-identical: the same random pipe, now able to render a preset over HTTP.
DERIVED = {
    "tiny-wan-pipe-rope128": ("tiny-wan-pipe", {"transformer/config.json": {"rope_max_seq_len": 128}}),
}
for _set in ("core", "edge", "all"):
    SETS[_set] = SETS[_set] + [n for n, (src, _) in DERIVED.items() if src in SETS[_set]]


def fetch(name: str, root: Path, force: bool = False, retries: int = 4) -> dict:
    from huggingface_hub import snapshot_download

    target = root / name
    marker = target / DONE
    if marker.exists() and not force:
        return {"name": name, "status": "present", "path": str(target)}
    t0 = time.time()
    last = None
    for attempt in range(retries):
        try:
            rev = snapshot_download(repo_id = f"{ORG}/{name}", local_dir = str(target), token = False)
            last = None
            break
        except Exception as exc:  # noqa: BLE001 - hub 5xx happen; retry, then report
            last = f"{type(exc).__name__}: {str(exc)[:300]}"
            C.log(f"{name}: attempt {attempt + 1} failed: {last}")
            time.sleep(5 * (attempt + 1))
    if last:
        return {"name": name, "status": "failed", "error": last}
    marker.write_text(f"{ORG}/{name} {Path(rev).name if rev else ''} {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
    size = sum(p.stat().st_size for p in target.rglob("*") if p.is_file() and ".cache" not in p.parts)
    return {"name": name, "status": "downloaded", "path": str(target), "mb": round(size / 1e6, 1),
            "s": round(time.time() - t0, 1)}


def derive(name: str, root: Path, force: bool = False) -> dict:
    """Build a DERIVED variant from its fetched source: a copy with a few config keys overridden."""
    import json
    import shutil

    src_name, patches = DERIVED[name]
    src, target = root / src_name, root / name
    if not (src / DONE).exists():
        return {"name": name, "status": "failed", "error": f"{src_name} is not fetched"}
    if (target / DONE).exists() and not force:
        return {"name": name, "status": "present", "path": str(target)}
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(src, target, ignore = shutil.ignore_patterns(".cache", DONE))
    for rel, keys in patches.items():
        cfg = json.loads((target / rel).read_text())
        cfg.update(keys)
        (target / rel).write_text(json.dumps(cfg, indent = 2) + "\n")
    (target / DONE).write_text(f"derived from {src_name}: {json.dumps(patches)}\n")
    return {"name": name, "status": "derived", "path": str(target), "from": src_name}


def main() -> int:
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default = str(C.WS / "hf_tiny"), help = "target root (default $WORKSPACE/hf_tiny)")
    ap.add_argument("--set", choices = sorted(SETS), default = "all")
    ap.add_argument("--only", action = "append", default = [], help = "repo name(s) under hf-internal-testing")
    ap.add_argument("--force", action = "store_true", help = "re-download even when complete")
    ap.add_argument("--list", action = "store_true")
    args = ap.parse_args()

    names = args.only or SETS[args.set]
    root = Path(args.root).resolve()
    if args.list:
        for n in names:
            src = f"derived from {DERIVED[n][0]}" if n in DERIVED else f"{ORG}/{n}"
            print(f"{n:<34} {'present' if (root / n / DONE).exists() else 'missing':<8} {src}")
        return 0
    C.scrub_tokens()
    os.environ.pop("HUGGINGFACE_HUB_TOKEN", None)
    os.environ.pop("HF_HUB_TOKEN", None)
    # Keep the hub's own lock / metadata cache inside the workspace too, never ~/.cache.
    os.environ.setdefault("HF_HOME", str(C.WS / "temp" / "hf_home"))
    root.mkdir(parents = True, exist_ok = True)
    bad = 0
    for n in names:
        r = derive(n, root, force = args.force) if n in DERIVED else fetch(n, root, force = args.force)
        bad += r["status"] == "failed"
        C.log(" ".join(f"{k}={v}" for k, v in r.items()))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
