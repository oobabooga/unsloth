"""#19621 a real hosted download sample: start a video load into a scratch Hugging Face cache (set by the driver,
deleted afterwards) and record what the load progress counter says against the bytes really on disk, then stop.
usage: video_dl_probe.py --out FILE [--seconds 90] [--repo R] [--tq fp8] [--te fp8]"""

from __future__ import annotations

import os
import sys
import threading
import time

import _dc


def disk_bytes(root: str) -> int:
    total = 0
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            p = os.path.join(dirpath, f)
            try:
                if not os.path.islink(p):
                    total += os.path.getsize(p)
            except OSError:
                pass
    return total


def main() -> None:
    out = _dc.out_path()
    args = sys.argv[3:]

    def opt(name, default):
        return args[args.index(name) + 1] if name in args else default

    seconds = float(opt("--seconds", "90"))
    repo = opt("--repo", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    rec = {"tree": _dc.enter_tree(), **_dc.cuda_info(), "repo": repo, "samples": []}
    cache = os.environ.get("HF_HUB_CACHE") or ""
    rec["scratch_cache"] = cache
    try:
        from core.inference.video import VideoBackend

        eng = VideoBackend()
        kw = {"hf_token": os.environ.get("HF_TOKEN") or None,
              "transformer_quant": opt("--tq", "fp8"), "text_encoder_quant": opt("--te", "fp8")}
        err_box = {}

        def start():
            try:
                eng.begin_load(repo, **kw)
            except Exception as exc:  # noqa: BLE001
                err_box["error"] = _dc.err(exc, 1500)

        threading.Thread(target = start, daemon = True).start()
        t0 = time.time()
        while time.time() - t0 < seconds:
            time.sleep(3)
            try:
                p = eng.load_progress() or {}
            except Exception as exc:  # noqa: BLE001
                p = {"progress_error": _dc.err(exc, 300)}
            rec["samples"].append({"t": round(time.time() - t0, 1), "phase": p.get("phase"),
                                   "downloaded": p.get("downloaded_bytes"), "expected": p.get("expected_bytes"),
                                   "on_disk": disk_bytes(cache) if cache else None, "error": p.get("error")})
            if p.get("phase") in ("ready", "error"):
                break
        if err_box:
            rec["begin_load_error"] = err_box["error"]
    except Exception as exc:  # noqa: BLE001
        rec["fatal"] = _dc.err(exc)
        rec["tb"] = _dc.tb()
    _dc.write(out, rec)
    os._exit(0)


if __name__ == "__main__":
    main()
