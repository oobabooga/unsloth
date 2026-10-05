"""Run vLLM with each Studio precision variant on the AMD runner and report pass/fail."""
import json, os, subprocess, sys, time, urllib.request

PY = sys.argv[1]
W = sys.argv[2]
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
torchao = lambda t, v: ["--quantization", "torchao", "--hf-overrides", json.dumps({"quantization_config_dict_json": json.dumps({"_type": t, "_version": v, "_data": {"set_inductor_config": True}})})]
CASES = {
    "fp16": (MODEL, ["--dtype", "float16"]),
    "bnb_int4": (MODEL, ["--quantization", "bitsandbytes", "--load-format", "bitsandbytes"]),
    "bnb_prequant": ("unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit", ["--load-format", "bitsandbytes"]),
    "torchao_int8": (MODEL, torchao("Int8WeightOnlyConfig", 1)),
    "torchao_fp8": (MODEL, torchao("Float8WeightOnlyConfig", 2)),
    "native_fp8": (MODEL, ["--quantization", "fp8"]),
    "awq": ("Qwen/Qwen2.5-0.5B-Instruct-AWQ", []),
    "gptq": ("Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4", []),
}
only = sys.argv[3].split(",") if len(sys.argv) > 3 else list(CASES)
results = {}
for name in only:
    model, extra = CASES[name]
    log = f"{W}/case-{name}.log"
    cmd = [PY, "-m", "vllm.entrypoints.openai.api_server", "--model", model, "--max-model-len", "4096",
           "--gpu-memory-utilization", "0.3", "--host", "127.0.0.1", "--port", "8011", "--generation-config", "vllm", *extra]
    t0 = time.time()
    with open(log, "w") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
    status, answer = "timeout", None
    while time.time() - t0 < 900:
        if proc.poll() is not None:
            status = f"exited rc={proc.returncode}"
            break
        try:
            urllib.request.urlopen("http://127.0.0.1:8011/health", timeout=2)
            body = json.dumps({"model": model, "messages": [{"role": "user", "content": "What is the capital of France? One word."}], "max_tokens": 12, "temperature": 0}).encode()
            req = urllib.request.Request("http://127.0.0.1:8011/v1/chat/completions", body, {"Content-Type": "application/json"})
            answer = json.load(urllib.request.urlopen(req, timeout=120))["choices"][0]["message"]["content"]
            status = "ok"
            break
        except Exception:
            time.sleep(3)
    proc.terminate()
    try:
        proc.wait(30)
    except subprocess.TimeoutExpired:
        proc.kill(); proc.wait()
    time.sleep(3)
    results[name] = {"status": status, "answer": answer, "seconds": round(time.time() - t0)}
    print(name, results[name], flush=True)
    if status != "ok":
        tail = open(log, errors="replace").read().splitlines()
        errs = [l for l in tail if "Error" in l or "error" in l or "Traceback" in l][-15:]
        print("\n".join(errs + tail[-25:]), flush=True)
json.dump(results, open(f"{W}/results.json", "w"), indent=1)
print(json.dumps(results, indent=1))
