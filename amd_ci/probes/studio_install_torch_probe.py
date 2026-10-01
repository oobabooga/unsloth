#!/usr/bin/env python3
"""Probe: what torch does this checkout's install.sh put into a fresh Studio home, and does a second
run over that home keep it?

Observes only. Runs `install.sh --local` twice into a home under $RUNNER_TEMP: once fresh, once over
the result (the update path). A PyPI JSON fixture that admits torch 2.13 is passed through
UNSLOTH_PYPI_JSON_URL, so the cu130 new-install route is open wherever it applies; on ROCm it never
applies, and that is what the criteria checks. Then one short LoRA SFT step runs on the GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

FIXTURE = '{"info": {"version": "2026.10.1", "requires_dist": ["numpy", "torch<2.15.0,>=2.4.0", "torchvision"]}}'

SFT = r"""
import json, torch
from unsloth import FastLanguageModel
model, tok = FastLanguageModel.from_pretrained("trl-internal-testing/tiny-Qwen3ForCausalLM", max_seq_length=128, load_in_4bit=False)
model = FastLanguageModel.get_peft_model(model, r=8, target_modules=["q_proj", "v_proj"])
batch = tok(["hello world, this is a short training row"] * 2, return_tensors="pt").to(model.device)
out = model(**batch, labels=batch["input_ids"])
out.loss.backward()
print("SFT_RESULT " + json.dumps({"loss": float(out.loss), "finite": bool(torch.isfinite(out.loss)), "device": str(model.device)}))
"""


def torch_info(python: Path) -> dict:
    code = ("import json, torch; print('TORCH ' + json.dumps({'version': torch.__version__, "
            "'hip': getattr(torch.version, 'hip', None), 'cuda': torch.version.cuda, "
            "'available': torch.cuda.is_available()}))")
    try:
        out = subprocess.run([str(python), "-c", code], capture_output = True, text = True, timeout = 600)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
    for line in out.stdout.splitlines():
        if line.startswith("TORCH "):
            return json.loads(line[6:])
    return {"error": (out.stderr or out.stdout)[-400:]}


def install(checkout: Path, base: Path, log: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "VIRTUAL_ENV")}
    env.update({
        "HOME": str(base / "home"),
        "UNSLOTH_STUDIO_HOME": str(base / "studio"),
        "XDG_CACHE_HOME": str(base / "home" / ".cache"),
        "UNSLOTH_SKIP_AUTOSTART": "1",
        "UNSLOTH_PYPI_JSON_URL": (base / "pypi.json").as_uri(),
    })
    start = time.time()
    with log.open("a", encoding = "utf-8") as fh:
        rc = subprocess.run(["bash", "install.sh", "--local"], cwd = checkout, env = env,
                            stdout = fh, stderr = subprocess.STDOUT, timeout = 3 * 3600).returncode
    text = log.read_text(encoding = "utf-8", errors = "replace")
    return {"rc": rc, "seconds": round(time.time() - start),
            "kept_line": any("keeping it" in line for line in text.splitlines()[-4000:])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()

    base = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / f"studio_install_{args.state}"
    (base / "home").mkdir(parents = True, exist_ok = True)
    (base / "pypi.json").write_text(FIXTURE, encoding = "utf-8")
    python = base / "studio" / "unsloth_studio" / "bin" / "python"
    obs: dict = {"state": args.state}
    try:
        obs["fresh"] = install(args.checkout, base, base / "fresh.log")
        obs["fresh"]["torch"] = torch_info(python)
        obs["rerun"] = install(args.checkout, base, base / "rerun.log")
        obs["rerun"]["torch"] = torch_info(python)
        sft = subprocess.run([str(python), "-c", SFT], capture_output = True, text = True, timeout = 1800,
                             env = {**os.environ, "HOME": str(base / "home")})
        line = next((l for l in sft.stdout.splitlines() if l.startswith("SFT_RESULT ")), None)
        obs["sft"] = json.loads(line[11:]) if line else {"error": (sft.stderr or sft.stdout)[-600:]}
    except Exception as exc:  # noqa: BLE001
        obs["error"] = f"{type(exc).__name__}: {exc}"
    for name in ("fresh.log", "rerun.log"):
        path = base / name
        if path.exists():
            obs[name.replace(".log", "_tail")] = path.read_text(encoding = "utf-8", errors = "replace")[-1500:]
    args.out.write_text(json.dumps(obs, indent = 1), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
