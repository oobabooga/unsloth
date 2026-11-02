# SPDX-License-Identifier: AGPL-3.0-only
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import urllib.request
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from capability import detect, untested_section

URL = "https://github.com/unslothai/llama.cpp/releases/download/b11985-mix-c116c32/app-b11985-mix-c116c32-windows-x64-cpu.zip"


def string(value):
    value = value.encode("utf-8")
    return struct.pack("<Q", len(value)) + value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    output = root / "out"
    output.mkdir(exist_ok=True)
    archive = root / "cpu.zip"
    urllib.request.urlretrieve(URL, archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(root / "cpu")
    binary = next((root / "cpu").rglob("llama-server.exe"))
    body = struct.pack("<IIQQ", 0x46554747, 3, 1, 1)
    body += string("general.architecture") + struct.pack("<I", 8) + string("llama")
    body += string("token_embd.weight") + struct.pack("<IQQIQ", 2, 1, 1, 0, 0)
    body += b"\0" * (-len(body) % 32) + b"\0" * 4
    name = "DeepSeek-V4.1-Flash-1017-Uncensored-3.58bpw-Q2_K-00001-of-00008.gguf"
    short = root / "short.gguf"
    prefix = root / "cache"
    target = max(269, len(str(prefix / name)) + 20)
    long = prefix / ("a" * (target - len(str(prefix / name)) - 1)) / name
    long.parent.mkdir(parents=True, exist_ok=True)
    short.write_bytes(body)
    long.write_bytes(body)
    extended = chr(92) * 2 + "?" + chr(92) + str(long)
    observations = {}
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="")
    for label, path in (("short", str(short)), ("long", str(long)), ("extended", extended)):
        command = [str(binary), "-m", path, "-ngl", "0", "--no-warmup", "--port", "0"]
        try:
            proc = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60, env=environment)
            text = proc.stdout + proc.stderr
            record = {"rc": proc.returncode, "path": path, "path_length": len(path), "file_exists_python": Path(path).is_file(), "open_failed": "failed to open GGUF file" in text, "metadata_seen": "general.architecture" in text, "command": command}
        except subprocess.TimeoutExpired as exc:
            text = str(exc)
            record = {"timeout": True, "path_length": len(path)}
        (output / f"{label}.log").write_text(text, encoding="utf-8")
        observations[label] = record
        print(label, json.dumps(record), flush=True)
    result = {"url": URL, "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "fixture_bytes": len(body), "observations": observations}
    (output / "path_observations.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    gaps = untested_section(detect(require_torch=False), ["windows", "nvidia", "multi_gpu"])
    (output / "capability_gaps.md").write_text(gaps, encoding="utf-8")
    print(gaps, flush=True)
    return int(any(r.get("timeout") for r in observations.values()) or not observations["short"].get("metadata_seen"))


if __name__ == "__main__":
    raise SystemExit(main())
