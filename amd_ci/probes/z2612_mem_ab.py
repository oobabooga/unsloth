"""zoo PR 2612 A/B: conv wrapper output dtype under autocast. Parent installs once, then runs
alternating base/head worker subprocesses (B,H,B,H,B,H) swapping only unsloth_zoo/compiler.py.

usage: python mem_ab.py --model whisper|whisper4|qwen3asr|wav2vec2|qwenvl|lfm2 [--dtype bf16|fp16]
       [--reps 3] --base compiler_base.txt --head compiler_head.txt
"""
import argparse, importlib.util, io, json, os, shutil, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))


def parent(a):
    spec = importlib.util.find_spec("unsloth_zoo")
    zoo_dir = os.path.dirname(spec.origin)
    target = os.path.join(zoo_dir, "compiler.py")
    print("zoo at", zoo_dir, flush=True)
    results = []
    arms = []
    for _ in range(a.reps):
        arms += ["base", "head"]
    for i, arm in enumerate(arms):
        src = a.base if arm == "base" else a.head
        if not os.path.isabs(src):
            src = os.path.join(HERE, os.path.basename(src)) if not os.path.exists(src) else src
        shutil.copyfile(src, target)
        for pyc in os.listdir(os.path.join(zoo_dir, "__pycache__")) if os.path.isdir(os.path.join(zoo_dir, "__pycache__")) else []:
            if pyc.startswith("compiler."):
                os.remove(os.path.join(zoo_dir, "__pycache__", pyc))
        env = dict(os.environ)
        env["UNSLOTH_COMPILE_LOCATION"] = tempfile.mkdtemp(prefix=f"ucc_{arm}_{i}_")
        env["Z2612_ARM"] = arm
        t0 = time.time()
        p = subprocess.run([sys.executable, "-u", os.path.abspath(__file__), "--worker", "--model", a.model,
                            "--dtype", a.dtype], env=env, capture_output=True, text=True)
        res = None
        for line in p.stdout.splitlines():
            if line.startswith("RESULT "):
                res = json.loads(line[7:])
        if res is None:
            res = {"arm": arm, "error": f"exit {p.returncode}", "tail": (p.stdout + p.stderr)[-3000:]}
        res["run"] = i
        res["wall_s"] = round(time.time() - t0, 1)
        print("ARM", json.dumps(res), flush=True)
        if "error" in res:
            print((p.stdout + p.stderr)[-6000:], flush=True)
        results.append(res)
    os.makedirs("out", exist_ok=True)
    with open(f"out/z2612_{a.model}_{a.dtype}.json", "w") as f:
        json.dump(results, f, indent=1)
    print("DONE", flush=True)


def load_audio(n=4):
    import pandas as pd, soundfile as sf
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("hf-internal-testing/librispeech_asr_dummy", "clean/validation-00000-of-00001.parquet",
                           repo_type="dataset")
    df = pd.read_parquet(path)
    audios, texts = [], []
    for _, row in df.head(n).iterrows():
        wav, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        assert sr == 16000
        audios.append(wav)
        texts.append(row["text"].lower())
    return audios, texts


def wer(ref, hyp):
    r, h = ref.lower().split(), "".join(c for c in hyp.lower() if c.isalnum() or c == " " or c == "'").split()
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(1, len(r))


def worker(a):
    import torch
    from unsloth import FastModel, FastLanguageModel  # noqa: F401
    import unsloth_zoo.compiler as zc
    dt = torch.bfloat16 if a.dtype == "bf16" else torch.float16
    out = {"arm": os.environ.get("Z2612_ARM"), "model": a.model, "dtype": a.dtype,
           "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "pr_code_present": hasattr(zc, "_NO_AUTOCAST_DEVICE_TYPES")}
    import transformers
    out["transformers"] = transformers.__version__
    torch.manual_seed(0)
    gen_texts = None
    m = a.model
    if m.startswith("whisper"):
        from transformers import WhisperForConditionalGeneration
        model, proc = FastModel.from_pretrained("unsloth/whisper-large-v3-turbo", dtype=dt, load_in_4bit=(m == "whisper4"),
                                                auto_model=WhisperForConditionalGeneration,
                                                whisper_language="English", whisper_task="transcribe")
        model = FastModel.get_peft_model(model, r=16, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                                         use_gradient_checkpointing="unsloth", random_state=0, task_type=None)
        audios, texts = load_audio()
        feats = proc.feature_extractor(audios, sampling_rate=16000, return_tensors="pt").input_features.cuda()
        lab = proc.tokenizer(texts, padding=True, return_tensors="pt")
        labels = lab.input_ids.masked_fill(lab.attention_mask == 0, -100).cuda()
        batch = {"input_features": feats, "labels": labels}
        FastModel.for_inference(model)
        with torch.no_grad():
            ids = model.generate(input_features=feats, max_new_tokens=96)
        gen_texts = proc.tokenizer.batch_decode(ids, skip_special_tokens=True)
        FastModel.for_training(model)
    elif m == "wav2vec2":
        from transformers import Wav2Vec2ForCTC
        model, proc = FastModel.from_pretrained("facebook/wav2vec2-base-960h", dtype=dt, load_in_4bit=False,
                                                auto_model=Wav2Vec2ForCTC)
        model = FastModel.get_peft_model(model, r=16, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                                         use_gradient_checkpointing="unsloth", random_state=0, task_type=None)
        audios, texts = load_audio()
        iv = proc(audios, sampling_rate=16000, return_tensors="pt", padding=True).input_values.cuda()
        lab = proc.tokenizer([t.upper() for t in texts], padding=True, return_tensors="pt")
        labels = lab.input_ids.masked_fill(lab.attention_mask == 0, -100).cuda()
        batch = {"input_values": iv, "labels": labels}
        FastModel.for_inference(model)
        with torch.no_grad(), torch.autocast("cuda", dtype=dt):
            logits = model(input_values=iv).logits
        gen_texts = proc.batch_decode(logits.argmax(-1))
        FastModel.for_training(model)
    elif m == "qwen3asr":
        model, proc = FastModel.from_pretrained("Qwen/Qwen3-ASR-0.6B", dtype=dt, load_in_4bit=False)
        model = FastModel.get_peft_model(model, r=16, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                                         use_gradient_checkpointing="unsloth", random_state=0)
        audios, texts = load_audio()
        convs = [[{"role": "user", "content": [{"type": "audio", "audio": x}]},
                  {"role": "assistant", "content": [{"type": "text", "text": f"language English<asr_text>{t}"}]}]
                 for x, t in zip(audios, texts)]
        prompts = proc.apply_chat_template(convs, tokenize=False)
        enc = proc(text=prompts, audio=audios, output_labels=True, padding=True, return_tensors="pt")
        T = enc["input_features"].shape[-1]
        padT = (-T) % 100
        if padT:
            enc["input_features"] = torch.nn.functional.pad(enc["input_features"], (0, padT))
            enc["input_features_mask"] = torch.nn.functional.pad(enc["input_features_mask"], (0, padT))
        batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in enc.items()}
        out["input_features_dtype"] = str(batch["input_features"].dtype)
        FastModel.for_inference(model)
        try:
            req = proc.apply_transcription_request(audio=audios, language="English", return_tensors="pt", padding=True)
            req = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in req.items()}
            with torch.no_grad():
                ids = model.generate(**req, max_new_tokens=96)
            gen_texts = proc.decode(ids[:, req["input_ids"].shape[1]:], return_format="transcription_only")
        except Exception as e:  # noqa: BLE001
            out["gen_error"] = f"{type(e).__name__}: {str(e)[:200]}"
        FastModel.for_training(model)
    elif m == "qwenvl":
        import numpy as np
        from PIL import Image
        model, proc = FastModel.from_pretrained("unsloth/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit", load_in_4bit=True, dtype=dt)
        model = FastModel.get_peft_model(model, r=16, lora_alpha=16, finetune_vision_layers=True,
                                         use_gradient_checkpointing="unsloth", random_state=0)
        rng = np.random.default_rng(0)
        imgs = [Image.fromarray(rng.integers(0, 255, (336, 336, 3), dtype=np.uint8)) for _ in range(2)]
        msgs = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image."}]},
                 {"role": "assistant", "content": [{"type": "text", "text": "A noisy picture of random colourful pixels."}]}]
                for _ in imgs]
        prompts = [proc.apply_chat_template(x, tokenize=False) for x in msgs]
        enc = proc(text=prompts, images=imgs, return_tensors="pt", padding=True)
        enc["labels"] = enc["input_ids"].masked_fill(enc["attention_mask"] == 0, -100)
        batch = {k: v.cuda() for k, v in enc.items()}
    elif m == "gemma3n":
        # fp16 GPU: Unsloth forces float32 weights + fp16 autocast for gemma3n (FORCE_FLOAT32)
        from PIL import Image
        from huggingface_hub import hf_hub_download
        model, proc = FastModel.from_pretrained("unsloth/gemma-3n-E2B-it", load_in_4bit=True, dtype=dt, max_seq_length=1024)
        out["forced_float32"] = os.environ.get("UNSLOTH_FORCE_FLOAT32")
        out["model_dtype"] = str(model.dtype)
        model = FastModel.get_peft_model(model, r=8, lora_alpha=8, finetune_vision_layers=False,
                                         finetune_audio_layers=False, use_gradient_checkpointing="unsloth", random_state=0)
        img = Image.open(hf_hub_download("huggingface/documentation-images", "bee.jpg", repo_type="dataset")).convert("RGB")
        audios, texts = load_audio(1)
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "audio", "audio": audios[0]},
                                             {"type": "text", "text": "Describe the image and transcribe the audio."}]},
                {"role": "assistant", "content": [{"type": "text", "text": "A bee on a flower. " + texts[0]}]}]
        enc = proc.apply_chat_template(msgs, tokenize=True, return_dict=True, return_tensors="pt")
        enc["labels"] = enc["input_ids"].clone()
        batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in enc.items()}
        out["batch_dtypes"] = {k: str(v.dtype) for k, v in batch.items() if torch.is_tensor(v)}
        # tower outputs: autocast vs no autocast reference (the no-autocast path is untouched by the PR)
        towers = {}
        for name in ("vision_tower", "audio_tower"):
            mod = None
            for n_, mm in model.named_modules():
                if n_.endswith(name):
                    mod = mm
                    break
            if mod is not None:
                towers[name] = mod
        caught = {}

        def first_tensor(o):
            if torch.is_tensor(o):
                return o
            if hasattr(o, "last_hidden_state"):
                return o.last_hidden_state
            if isinstance(o, (tuple, list)):
                for x in o:
                    if torch.is_tensor(x):
                        return x
            return None
        hs = [mm.register_forward_hook(lambda mod_, i_, o_, n_=n: caught.__setitem__(n_, first_tensor(o_).detach().float().clone()))
              for n, mm in towers.items()]
        model.eval()
        res = {}
        for ac in (True, False):
            caught.clear()
            with torch.no_grad(), torch.autocast("cuda", dtype=dt, enabled=ac):
                lo = model(**batch).loss
            res[ac] = (float(lo), dict(caught))
        for h in hs:
            h.remove()
        out["tower_cmp"] = {}
        for n in towers:
            a_, r_ = res[True][1].get(n), res[False][1].get(n)
            if a_ is None or r_ is None:
                continue
            out["tower_cmp"][n] = {"finite_autocast": bool(torch.isfinite(a_).all()),
                                   "maxabs_autocast": float(a_.abs().max()), "maxabs_ref": float(r_.abs().max()),
                                   "rel_err_vs_fp32_ref": float((a_ - r_).norm() / r_.norm())}
        out["loss_autocast_eval"], out["loss_fp32_eval"] = res[True][0], res[False][0]
        model.train()
    elif m == "lfm2":
        model, tok = FastLanguageModel.from_pretrained("unsloth/LFM2-350M", dtype=dt, load_in_4bit=False, max_seq_length=1024)
        model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16, use_gradient_checkpointing="unsloth", random_state=0,
                                                 target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"])
        g = torch.Generator().manual_seed(0)
        ids = torch.randint(100, 30000, (4, 512), generator=g).cuda()
        batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}
    else:
        raise SystemExit(f"unknown model {m}")

    out["gen_texts"] = gen_texts
    if gen_texts is not None:
        out["wer"] = [round(wer(r, h), 4) for r, h in zip(texts, gen_texts)]

    conv_dtypes = {}

    def hook(mod, inp, o):
        if torch.is_tensor(o):
            conv_dtypes[f"{type(mod).__name__}:{inp[0].dtype}->{o.dtype}"] = conv_dtypes.get(
                f"{type(mod).__name__}:{inp[0].dtype}->{o.dtype}", 0) + 1
    handles = [mm.register_forward_hook(hook) for mm in model.modules()
               if isinstance(mm, torch.nn.modules.conv._ConvNd)]
    out["n_conv"] = len(handles)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4)
    model.train()
    losses, times = [], []
    torch.cuda.synchronize()
    weights_alloc = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    for step in range(10):
        if step == 2:
            for h in handles:
                h.remove()
            torch.cuda.synchronize()
            warm_alloc = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        if step == 0:
            pass
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=dt):
            loss = model(**batch).loss
        loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
        losses.append(float(loss))
    out["conv_io_dtypes_first2steps"] = conv_dtypes
    out["losses"] = [round(x, 6) for x in losses]
    out["peak_warm_mib"] = round((torch.cuda.max_memory_allocated() - warm_alloc) / 2**20, 1)
    out["peak_abs_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    out["weights_mib"] = round(weights_alloc / 2**20, 1)
    st = sorted(times[2:])
    out["step_ms_median"] = round(st[len(st) // 2] * 1000, 1)
    # eval without autocast must be unchanged by the PR
    model.eval()
    try:
        with torch.no_grad():
            ev = {k: v for k, v in batch.items()}
            if m.startswith("whisper") or m == "wav2vec2" or m == "qwen3asr":
                pass
            out["eval_loss_autocast"] = float(_eval(model, ev, dt, True))
            out["eval_loss_no_autocast"] = float(_eval(model, ev, dt, False))
    except Exception as e:  # noqa: BLE001
        out["eval_error"] = f"{type(e).__name__}: {str(e)[:300]}"
    print("RESULT " + json.dumps(out), flush=True)


def _eval(model, batch, dt, ac):
    import torch
    with torch.autocast("cuda", dtype=dt, enabled=ac):
        return model(**batch).loss


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--base", default="compiler_base.txt")
    ap.add_argument("--head", default="compiler_head.txt")
    ap.add_argument("--worker", action="store_true")
    a = ap.parse_args()
    if a.worker:
        try:
            worker(a)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print("RESULT " + json.dumps({"arm": os.environ.get("Z2612_ARM"), "error": f"{type(e).__name__}: {str(e)[:500]}"}), flush=True)
    else:
        parent(a)
