"""The terminal path: what a user tries from the CLI to run an NPU model, against the normal GGUF path."""
import json, os, re, shutil, subprocess, sys, threading, time

sys.path.insert(0, os.path.dirname(__file__))
import uxlib as U
from uxlib import step, note, run, poll

EXE = shutil.which("unsloth")
PASSWORD = (U.ROOT / "password.txt").read_text().strip() if (U.ROOT / "password.txt").exists() else None


def kill_tree(p):
    if U.WIN:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True)
    else:
        try:
            os.killpg(p.pid, 15)
        except Exception:
            p.terminate()
        try:
            p.wait(30)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, 9) if not U.WIN else None


def spawn_until(args, name, *, ready=None, timeout=900, then=None):
    """Start a long-running CLI command, collect output until it exits, a ready pattern appears
    (then run `then()` while it serves), or the timeout. Always stops it."""
    log_path = U.OUT / "cmd" / f"{name}.log"
    log_path.parent.mkdir(exist_ok=True)
    lines = []
    kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if U.WIN else {"start_new_session": True}
    env = dict(os.environ, PYTHONUNBUFFERED="1", COLUMNS="160", NO_COLOR="1")
    t0 = time.monotonic()
    p = subprocess.Popen([EXE, *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         text=True, encoding="utf-8", errors="replace", env=env, **kw)
    th = threading.Thread(target=lambda: [lines.append(l.rstrip("\n")) for l in p.stdout], daemon=True)
    th.start()
    hit = None
    while time.monotonic() - t0 < timeout:
        if p.poll() is not None:
            break
        text = "\n".join(lines)
        if ready and (m := re.search(ready, text, re.I | re.M)):
            hit = m.group(0)
            break
        time.sleep(1)
    seconds = round(time.monotonic() - t0, 1)
    extra = None
    if hit and then:
        try:
            extra = then("\n".join(lines))
        except Exception as exc:  # noqa: BLE001
            extra = {"error": str(exc)}
    rc = p.poll()
    if rc is None:
        kill_tree(p)
    th.join(5)
    text = "\n".join(lines)
    log_path.write_text(f"$ unsloth {' '.join(args)}\nrc={rc} seconds={seconds} ready={hit!r}\n\n{text}", encoding="utf-8")
    return {"args": args, "rc": rc, "seconds": seconds, "ready": hit, "then": extra, "tail": text[-2500:]}


def chat_via(text):
    """When `unsloth run` serves, find its base URL + key in its output and chat once."""
    url = re.search(r"Unsloth Studio running at (https?://\S+)", text)
    key = re.search(r"API Key:\s+(\S+)", text)
    if not url:
        return {"error": "no URL printed"}
    base = url.group(1).rstrip("/").replace("0.0.0.0", "127.0.0.1")
    h = {"Authorization": "Bearer " + key.group(1)} if key else {}
    models = U.requests.get(base + "/v1/models", headers=h, timeout=30)
    mid = None
    try:
        data = models.json().get("data") or []
        mid = data[0]["id"] if data else None
    except ValueError:
        pass
    r = U.requests.post(base + "/v1/chat/completions", headers=h, timeout=300,
                        json={"model": mid or "default", "messages": [{"role": "user", "content": "What is 17 + 25? Reply with just the number."}], "max_tokens": 200})
    loaded = re.search(r"Model loaded:\s*(.+)", text)
    return {"base": base, "key_found": bool(key), "model_loaded_line": loaded.group(1) if loaded else None,
            "models_status": models.status_code, "model_ids": mid,
            "chat_status": r.status_code, "chat": r.text[:1500]}


def main():
    step("which_unsloth", lambda: EXE, shoot=False)
    if not EXE:
        return
    for args, name in ((["--help"], "help"), (["run", "--help"], "run-help"), (["studio", "--help"], "studio-help"),
                       (["start", "--help"], "start-help"), (["chat", "--help"], "chat-help")):
        def h(args=args, name=name):
            rc, text = run([EXE, *args], name=name, timeout=180)
            return {"rc": rc, "mentions_npu": bool(re.search(r"\bNPU\b|FastFlow|Lemonade|Ryzen", text, re.I)), "head": text[:1800]}
        step("cli_" + name, h, shoot=False)

    cat = json.loads((U.ROOT / "catalog.json").read_text()) if (U.ROOT / "catalog.json").exists() else []
    small = next((m for m in cat if m["checkpoint"] == "qwen3:0.6b"), None)
    npu_id = small["id"] if small else "qwen3-0.6b-FLM"
    ready = r"OpenAI / Anthropic SDK base URL:\s*\n\s*\S+"

    # What an NPU user would try first: the id Studio shows, its model_path, and FastFlowLM's own name.
    for label, model in (("model_path", "lemonade:" + npu_id), ("studio_id", npu_id), ("flm_name", "qwen3:0.6b")):
        step(f"unsloth_run_npu_{label}", lambda model=model, label=label:
             spawn_until(["run", "--model", model, "-p", "8899"], f"run-npu-{label}", ready=ready, timeout=900, then=chat_via), shoot=False)

    # The GGUF baseline (`unsloth run --model unsloth/Qwen3-0.6B-GGUF:Q4_K_M`) served and chatted in 21-25 s on
    # both OSes in runs 3-8; not repeated.

    # `unsloth chat` is the terminal chat: does it offer the NPU model at all?
    def chat_cli():
        # An invalid pick then EOF makes it print the list and exit.
        rc, text = run([EXE, "chat"], name="chat-list", timeout=600, input="x\n")
        if npu_id not in text:
            note("`unsloth chat` (pick from your local models) does not list the downloaded NPU model.")
        return {"rc": rc, "lists_npu_model": npu_id in text, "tail": text[-2500:]}
    step("unsloth_chat_model_list", chat_cli, shoot=False)
    def chat_npu():
        rc, text = run([EXE, "chat", "lemonade:" + npu_id], name="chat-npu", timeout=900, input="What is 17 + 25?\n")
        return {"rc": rc, "tail": text[-2500:]}
    step("unsloth_chat_npu_model", chat_npu, shoot=False)
    print("NOTES:", json.dumps(U.NOTES, indent=1))


if __name__ == "__main__":
    main()
