"""Backend A/B of interrupted NPU downloads, main vs branch, on real NPU hardware.

Drives Studio's LemonadeNpuBackend from the given checkout in child processes, so a child can be
killed mid-download the way closing Studio kills it.

  parent:  python probe.py            env SRC_MAIN, SRC_FIX, WORK, OUT
  child:   python probe.py child <mode> [model] [stop_percent]   env SRC, NPU_ROOT

Scenarios per variant:
  kill:  download KILL_MODEL, kill the whole process tree at 40% of its weights, download again,
         then load it and ask for one word.
  drops: download DROP_MODEL through a proxy that closes every Hugging Face connection after
         CUT_MB, then load it and ask for one word.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

KILL_MODEL = os.environ.get("KILL_MODEL", "gemma3-4b-FLM")
DROP_MODEL = os.environ.get("DROP_MODEL", "qwen3-1.7b-FLM")
CUT_MB = int(os.environ.get("CUT_MB", "300"))
BIG_MODEL = os.environ.get("BIG_MODEL", "qwen3.6-moe-35b-a3b-FLM")
BIG_CUT_MB = int(os.environ.get("BIG_CUT_MB", "3000"))
WIN = sys.platform == "win32"
T0 = time.monotonic()


def log(*parts):
    print(f"[{time.monotonic() - T0:8.1f}s]", *parts, flush = True)


# ---------------------------------------------------------------- child


def child(mode, model = None):
    sys.path.insert(0, os.path.join(os.environ["SRC"], "studio", "backend"))
    from core.inference.npu_backend import LemonadeNpuBackend

    npu = LemonadeNpuBackend(root = Path(os.environ["NPU_ROOT"]))
    out = {"mode": mode, "model": model}
    try:
        status = npu.enable()
        out["enabled"] = status["ready"]
        if mode == "download":
            started = time.monotonic()
            first = last = None
            events = 0
            last_print = 0.0
            try:
                for event in npu.download(model):
                    events += 1
                    pct = event.get("percent")
                    if first is None and isinstance(pct, (int, float)):
                        first = pct
                    last = event
                    now = time.monotonic()
                    if now - last_print > 2 or event.get("event") != "progress":
                        last_print = now
                        print("EVENT " + json.dumps(event), flush = True)
                out.update(result = "complete")
            except Exception as exc:  # noqa: BLE001
                out.update(result = "error", error = str(exc)[:600])
            out.update(
                seconds = round(time.monotonic() - started, 1),
                first_percent = first,
                events = events,
                last = last,
            )
            if npu._server is not None:
                out["lemond_tail"] = npu._server.log_tail(25)
        elif mode == "chat":
            try:
                npu.load(model)
                response = npu._server.request(
                    "POST",
                    "/v1/chat/completions",
                    json_body = {
                        "model": model,
                        "messages": [{"role": "user", "content": "Reply with the single word: ready"}],
                        "max_tokens": 24,
                    },
                    timeout = 600.0,
                )
                body = response.json()
                out["reply"] = body["choices"][0]["message"]["content"] if response.status_code == 200 else body
            except Exception as exc:  # noqa: BLE001
                out["error"] = str(exc)[:600]
                if npu._server is not None:
                    out["lemond_tail"] = npu._server.log_tail(25)
        out["catalog"] = {
            m.id: {"downloaded": m.downloaded, "resume_percent": getattr(m, "resume_percent", None)}
            for m in npu.catalog()
            if model is None or m.id == model
        }
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"[:600]
    finally:
        try:
            npu.shutdown()
        except Exception:  # noqa: BLE001
            pass
    print("RESULT " + json.dumps(out), flush = True)


# ---------------------------------------------------------------- parent


def kill_tree(proc, root):
    if WIN:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output = True)
        for image in ("flm.exe", "lemond.exe"):
            subprocess.run(["taskkill", "/F", "/IM", image], capture_output = True)
    else:
        import signal

        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        subprocess.run(["pkill", "-9", "-f", str(root)], check = False)
    proc.wait(60)


def run_child(src, root, mode, model = None, env_extra = None, kill_when = None):
    env = {**os.environ, "SRC": str(src), "NPU_ROOT": str(root), "PYTHONIOENCODING": "utf-8"}
    env.update(env_extra or {})
    kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if WIN else {"start_new_session": True}
    proc = subprocess.Popen(
        [sys.executable, "-u", __file__, "child", mode] + ([model] if model else []),
        env = env, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, text = True,
        encoding = "utf-8", errors = "replace", **kwargs,
    )
    result = {}
    killed_at = None
    for line in proc.stdout:
        line = line.rstrip()
        if line.startswith("RESULT "):
            result = json.loads(line[7:])
            continue
        if line.startswith("EVENT "):
            event = json.loads(line[6:])
            log(mode, model, json.dumps(event)[:300])
            if kill_when and killed_at is None and kill_when(event):
                killed_at = event
                log("killing the download's process tree at", json.dumps(event)[:300])
                kill_tree(proc, root)
                break
        elif line.strip():
            print("   |", line[:400], flush = True)
    if killed_at is None:
        proc.wait()
    else:
        result = {"result": "killed", "killed_at": killed_at}
    log(mode, model, "->", json.dumps({k: v for k, v in result.items() if k != "lemond_tail"})[:1500])
    return result


def files_under(root):
    base = root / "flm" / "models"
    return {
        str(p.relative_to(base)): p.stat().st_size
        for p in sorted(base.rglob("*"))
        if p.is_file()
    } if base.exists() else {}


def big_file(event):
    return event.get("file") == "model.q4nx"


def kill_at_40(event):
    pct = event.get("percent")
    if not isinstance(pct, (int, float)) or not big_file(event):
        return False
    # main relays lemond's percent for the one file; the branch reports the whole model's.
    return pct >= 40


def variant(name, src, work):
    root = work / f"npu-{name}"
    res = {}
    log(f"===== {name}")
    res["enable"] = run_child(src, root, "enable").get("enabled")

    scenarios = os.environ.get("SCENARIOS", "kill,drops").split(",")
    if "kill" in scenarios:
        log("--- kill scenario")
        res["kill"] = {"first": run_child(src, root, "download", KILL_MODEL, kill_when = kill_at_40)}
        time.sleep(3)
        res["kill"]["files_after_kill"] = files_under(root)
        after = run_child(src, root, "download", KILL_MODEL)
        res["kill"]["second"] = {k: after.get(k) for k in ("result", "error", "seconds", "first_percent", "events", "catalog")}
        res["kill"]["files_after_second"] = files_under(root)
        chat = run_child(src, root, "chat", KILL_MODEL)
        res["kill"]["chat"] = {k: chat.get(k) for k in ("reply", "error", "lemond_tail")}
    if "drops" in scenarios:
        res["drops"] = drops(src, root, DROP_MODEL, CUT_MB)
    if "big" in scenarios:
        res["big"] = drops(src, root, BIG_MODEL, BIG_CUT_MB)
    return res


def drops(src, root, model, cut_mb):
    log(f"--- drops scenario: {model}, cut every {cut_mb} MB")
    sys.path.insert(0, str(Path(__file__).parent))
    from cut_proxy import CutProxy

    proxy = CutProxy(cut_mb << 20)
    env = {
        "HTTPS_PROXY": proxy.url, "https_proxy": proxy.url,
        "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
    }
    drop = run_child(src, root, "download", model, env_extra = env)
    res = {
        "model": model,
        "cut_mb": cut_mb,
        **{k: drop.get(k) for k in ("result", "error", "seconds", "events", "catalog")},
        "proxy": proxy.summary(),
        "lemond_tail": drop.get("lemond_tail"),
        "files": files_under(root),
    }
    if drop.get("result") == "complete":
        chat = run_child(src, root, "chat", model)
        res["chat"] = {k: chat.get(k) for k in ("reply", "error", "lemond_tail")}
    return res


def main():
    work = Path(os.environ["WORK"])
    out = Path(os.environ["OUT"])
    out.mkdir(parents = True, exist_ok = True)
    summary = {}
    for name in os.environ.get("VARIANTS", "main,fix").split(","):
        try:
            summary[name] = variant(name, Path(os.environ[f"SRC_{name.upper()}"]), work)
        except Exception as exc:  # noqa: BLE001
            log(name, "FAILED", repr(exc))
            summary[name] = {"error": repr(exc)}
        (out / f"probe-{sys.platform}.json").write_text(json.dumps(summary, indent = 1))
    print(json.dumps(summary, indent = 1))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        child(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    else:
        main()
