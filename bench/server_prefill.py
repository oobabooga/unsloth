"""Multi-turn llama-server prefill probe (issue unslothai/unsloth#10656).

Launches one llama-server, sends a Home-Assistant-like chat (large tool/system
prompt, then follow-up turns that extend it), records llama-server's own timings
per turn and the device placement lines from its log. Writes JSON to --out.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def build_system_prompt(n_entities: int) -> str:
    rooms = ["kitchen", "living room", "bedroom", "office", "garage", "hallway", "bathroom", "garden"]
    kinds = ["light", "switch", "sensor", "climate", "cover", "media_player", "fan", "lock"]
    lines = [
        "You are a voice assistant controlling a smart home. Use the tools to act.",
        "Available entities (entity_id, name, state, attributes):",
    ]
    for i in range(n_entities):
        room = rooms[i % len(rooms)]
        kind = kinds[(i // len(rooms)) % len(kinds)]
        eid = f"{kind}.{room.replace(' ', '_')}_{i}"
        lines.append(
            f"- {eid}: '{room.title()} {kind.replace('_', ' ')} {i}', state="
            f"{'on' if i % 3 else 'off'}, brightness={(i * 37) % 255}, "
            f"temperature={18 + (i % 9)}.{i % 10}, last_changed=2026-09-{1 + i % 28:02d}T{i % 24:02d}:{i % 60:02d}:00"
        )
    return "\n".join(lines)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "HassTurnOn",
            "description": "Turns on/opens a device or entity",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "area": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "HassTurnOff",
            "description": "Turns off/closes a device or entity",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "area": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "GetLiveContext",
            "description": "Provides real-time information about the CURRENT state of devices",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

USER_TURNS = [
    "Turn on the kitchen light please.",
    "What is the temperature in the living room right now?",
    "Close the garage cover and lock the hallway lock.",
    "Is anything still on in the office?",
]


def post(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url, data = json.dumps(payload).encode(), headers = {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout = timeout) as r:
        return json.loads(r.read().decode())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required = True)
    ap.add_argument("--model", required = True)
    ap.add_argument("--port", type = int, default = 18091)
    ap.add_argument("--entities", type = int, default = 160)
    ap.add_argument("--log", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--label", required = True)
    ap.add_argument("extra", nargs = argparse.REMAINDER)
    a = ap.parse_args()
    extra = [x for x in a.extra if x != "--"]

    argv = [a.server, "-m", a.model, "--port", str(a.port), "--host", "127.0.0.1", *extra]
    result = {"label": a.label, "argv": argv, "turns": [], "error": None}
    logf = open(a.log, "w", encoding = "utf-8", errors = "replace")
    t0 = time.time()
    p = subprocess.Popen(argv, stdout = logf, stderr = subprocess.STDOUT)
    base = f"http://127.0.0.1:{a.port}"
    try:
        ready = False
        while time.time() - t0 < 900:
            if p.poll() is not None:
                result["error"] = f"server exited {p.returncode} before ready"
                break
            try:
                with urllib.request.urlopen(base + "/health", timeout = 5) as r:
                    if r.status == 200:
                        ready = True
                        break
            except Exception:
                pass
            time.sleep(2)
        result["load_s"] = round(time.time() - t0, 1)
        if ready:
            messages = [{"role": "system", "content": build_system_prompt(a.entities)}]
            for turn, text in enumerate(USER_TURNS):
                messages.append({"role": "user", "content": text})
                t1 = time.time()
                resp = post(
                    base + "/v1/chat/completions",
                    {
                        "messages": messages,
                        "tools": TOOLS,
                        "max_tokens": 48,
                        "temperature": 0,
                        "cache_prompt": True,
                    },
                    timeout = 1800,
                )
                wall = time.time() - t1
                tm = resp.get("timings", {})
                msg = resp["choices"][0]["message"]
                result["turns"].append(
                    {
                        "turn": turn,
                        "wall_s": round(wall, 2),
                        "prompt_n": tm.get("prompt_n"),
                        "cache_n": tm.get("cache_n"),
                        "prompt_ms": tm.get("prompt_ms"),
                        "prompt_per_second": tm.get("prompt_per_second"),
                        "predicted_n": tm.get("predicted_n"),
                        "predicted_per_second": tm.get("predicted_per_second"),
                    }
                )
                print(a.label, json.dumps(result["turns"][-1]), flush = True)
                messages.append(
                    {"role": "assistant", "content": msg.get("content") or "Done."}
                )
    except Exception as e:  # recorded, the caller judges
        result["error"] = f"{type(e).__name__}: {e}"
    finally:
        p.terminate()
        try:
            p.wait(timeout = 60)
        except Exception:
            p.kill()
        logf.close()
    keep = []
    for line in Path(a.log).read_text(encoding = "utf-8", errors = "replace").splitlines():
        low = line.lower()
        if any(
            k in low
            for k in (
                "model buffer size",
                "kv buffer size",
                "compute buffer size",
                "ggml_vulkan:",
                "matrix cores",
                "offloaded",
                "found ",
                "device ",
                "n_ctx",
                "flash_attn",
                "n_ubatch",
                "swa",
                "checkpoint",
                "error",
                "warn",
            )
        ):
            keep.append(line[:400])
    result["log_lines"] = keep[:300]
    Path(a.out).write_text(json.dumps(result, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
