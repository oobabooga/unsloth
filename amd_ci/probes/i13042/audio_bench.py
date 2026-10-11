# Start audiocpp_server on one backend, run one ASR or TTS request, report wall time as JSON.
# usage: audio_bench.py --server BIN --backend cpu|vulkan --family F --model PATH --task asr|tts
#        [--audio WAV] [--text T] [--out WAV] [--port N] [--json OUT]
import argparse, json, os, subprocess, sys, tempfile, time, urllib.request, uuid
from pathlib import Path


def post_multipart(url, fields, file_field, file_path, timeout):
    boundary = uuid.uuid4().hex
    body = b""
    for k, v in fields.items():
        body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
    data = Path(file_path).read_bytes()
    body += (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"{file_field}\"; filename=\"a.wav\"\r\n"
        "Content-Type: audio/wav\r\n\r\n"
    ).encode() + data + b"\r\n" + f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(url, data = body, headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout = timeout) as r:
        return r.status, r.read()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server", required = True)
    p.add_argument("--backend", required = True)
    p.add_argument("--family", required = True)
    p.add_argument("--model", required = True)
    p.add_argument("--task", required = True, choices = ["asr", "tts"])
    p.add_argument("--audio")
    p.add_argument("--text", default = "Hello there, this is a test of the audio runtime.")
    p.add_argument("--voice", default = "af_heart")
    p.add_argument("--out")
    p.add_argument("--port", type = int, default = 18790)
    p.add_argument("--timeout", type = float, default = 900)
    p.add_argument("--json")
    a = p.parse_args()
    work = Path(tempfile.mkdtemp(prefix = "abench-"))
    cfg = {
        "host": "127.0.0.1", "port": a.port, "backend": a.backend, "lazy_load": False,
        "max_loaded_models": 1, "busy_timeout_ms": 0,
        "models": [{"id": "m", "family": a.family, "path": str(Path(a.model).resolve()), "task": a.task, "mode": "offline"}],
    }
    (work / "server.json").write_text(json.dumps(cfg), encoding = "utf-8")
    log = open(work / "server.log", "wb")
    t0 = time.monotonic()
    proc = subprocess.Popen([a.server, "--config", str(work / "server.json"), "--no-ui"], stdout = log, stderr = subprocess.STDOUT, cwd = str(Path(a.server).parent))
    result = {"backend": a.backend, "family": a.family, "model": Path(a.model).name, "ok": False}
    try:
        health = None
        while time.monotonic() - t0 < 600:
            if proc.poll() is not None:
                break
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout = 2) as r:
                    health = json.loads(r.read())
                    break
            except Exception:
                time.sleep(0.5)
        result["load_s"] = round(time.monotonic() - t0, 2)
        result["health"] = health
        if health is None:
            raise RuntimeError("server never became healthy")
        t1 = time.monotonic()
        if a.task == "asr":
            status, body = post_multipart(f"http://127.0.0.1:{a.port}/v1/audio/transcriptions", {"model": "m"}, "file", a.audio, a.timeout)
            result["text"] = body.decode("utf-8", "replace")[:400]
        else:
            req = urllib.request.Request(
                f"http://127.0.0.1:{a.port}/v1/audio/speech",
                data = json.dumps({"model": "m", "input": a.text, "voice": a.voice}).encode(),
                headers = {"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout = a.timeout) as r:
                status, body = r.status, r.read()
            if a.out:
                Path(a.out).write_bytes(body)
            result["bytes"] = len(body)
        result["request_s"] = round(time.monotonic() - t1, 2)
        result["http"] = status
        result["ok"] = status == 200
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["fail_after_s"] = round(time.monotonic() - t0, 2)
        if hasattr(exc, "read"):
            try:
                result["error_body"] = exc.read().decode("utf-8", "replace")[:2000]
            except Exception:
                pass
    finally:
        proc.terminate()
        try:
            proc.wait(20)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
        if a.json:
            try:
                Path(a.json).with_suffix(".server.log").write_bytes((work / "server.log").read_bytes())
            except OSError:
                pass
        tail = (work / "server.log").read_bytes()[-1500:].decode("utf-8", "replace")
        result["log_tail"] = "\n".join(l for l in tail.splitlines() if "ggml_vulkan" in l or "fail" in l.lower() or "error" in l.lower())[-800:]
    print(json.dumps(result, indent = 1))
    if a.json:
        Path(a.json).write_text(json.dumps(result), encoding = "utf-8")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
