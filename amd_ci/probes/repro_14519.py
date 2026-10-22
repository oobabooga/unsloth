"""Repro for unslothai/unsloth#14519 against one studio/backend tree.

Real CPU SentenceTransformer (bge-small, the default archive embedder) behind the real
conversation_archive.archive_turns, on a synthetic long tool-heavy thread.

  python repro_14519.py --backend-dir <tree>/studio/backend --label base --out base.json
"""

import argparse
import inspect
import json
import os
import random
import sys
import tempfile
import threading
import time

ap = argparse.ArgumentParser()
ap.add_argument("--backend-dir", required = True)
ap.add_argument("--label", required = True)
ap.add_argument("--out", required = True)
ap.add_argument("--rounds", type = int, default = 120)
ap.add_argument("--tool-words", type = int, default = 2400)
ap.add_argument("--stop-after", type = float, default = 8.0)
ap.add_argument("--model", default = "unsloth/bge-small-en-v1.5")
args = ap.parse_args()

sys.path.insert(0, os.path.abspath(args.backend_dir))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

ST = SentenceTransformer(args.model, device = "cpu")
TOK = ST.tokenizer

ENCODED = {"texts": 0, "calls": 0, "after_stop": 0}
STOP = threading.Event()
LOCK = threading.Lock()


def real_encode(texts, *, model_name = None, normalize = True):
    # Same call as embeddings._st_encode: one SentenceTransformer.encode per request.
    with LOCK:
        ENCODED["texts"] += len(texts)
        ENCODED["calls"] += 1
        if STOP.is_set():
            ENCODED["after_stop"] += len(texts)
    with torch.inference_mode():
        out = ST.encode(
            list(texts),
            normalize_embeddings = normalize,
            convert_to_numpy = True,
            show_progress_bar = False,
        )
    return out.astype("float32", copy = False)


WORDS = (
    "def class return import self value config tensor path index buffer offset token "
    "request response handler worker thread lock queue cache model layer weight grad "
    "batch shard device stream kernel event state error retry timeout socket parse"
).split()


def tool_output(i, rng):
    lines = []
    n = 0
    while n < args.tool_words:
        k = rng.randint(6, 14)
        line = " ".join(rng.choice(WORDS) for _ in range(k)) + f" {i}_{n} {rng.randint(0, 99999)}"
        lines.append(line)
        n += k + 2
    return "\n".join(lines)


def build_thread(rng):
    msgs = []
    for i in range(args.rounds):
        msgs.append({"role": "user", "content": f"Step {i}: read src/module_{i}.py and summarise it."})
        body = tool_output(i, rng)
        if i == 3:
            body += "\nNOTE: the deploy password hint for service-7 is MAGENTA-PELICAN-42."
        msgs.append(
            {
                "role": "assistant",
                "content": f"Tool output for src/module_{i}.py:\n{body}\n\nSummary: module {i} wires the worker queue.",
            }
        )
    msgs.append({"role": "user", "content": "What was the deploy password hint for service-7?"})
    return msgs


def fresh_home():
    home = tempfile.mkdtemp(prefix = f"repro14519_{args.label}_")
    os.environ["UNSLOTH_STUDIO_HOME"] = home
    from storage import rag_db

    rag_db._schema_ready = set()
    return home


def persist(thread_id, msgs):
    from storage import studio_db

    studio_db.upsert_chat_thread(
        {"id": thread_id, "title": "t", "modelType": "base", "modelId": "local-model", "createdAt": 1}
    )
    rows = [
        {
            "id": f"{thread_id}-{i}",
            "threadId": thread_id,
            "role": m["role"],
            "content": [{"type": "text", "text": m["content"]}],
            "createdAt": i + 2,
        }
        for i, m in enumerate(msgs)
    ]
    studio_db.sync_chat_messages(thread_id, rows, prune_missing = True)


def setup():
    from core.rag import config, embeddings

    config.EMBED_BACKEND = "sentence-transformers"
    embeddings.encode = real_encode
    embeddings.dim = lambda model_name = None: ST.get_sentence_embedding_dimension()
    embeddings.max_tokens = lambda model_name = None: ST.max_seq_length
    embeddings.token_counter = lambda model_name = None: (
        lambda t: len(TOK(t, add_special_tokens = False)["input_ids"])
    )
    embeddings.warm = lambda model_name = None: None
    config.effective_embedding_model = lambda fresh = False: args.model


def request(thread_id, msgs, live_n, cancel_check = None):
    from core.rag import conversation_archive as ca

    live = msgs[-live_n:]
    evicted = msgs[:-live_n]
    kwargs = {"live": live, "branch": msgs}
    if cancel_check is not None and "cancel_check" in inspect.signature(ca.archive_turns).parameters:
        kwargs["cancel_check"] = cancel_check
    before = ENCODED["texts"]
    t0 = time.monotonic()
    written = ca.archive_turns(thread_id, evicted, **kwargs)
    return {
        "written_groups": written,
        "chunks_embedded": ENCODED["texts"] - before,
        "seconds": round(time.monotonic() - t0, 2),
    }


def main():
    setup()
    rng = random.Random(3407)
    msgs = build_thread(rng)
    live_n = 5
    out = {
        "label": args.label,
        "torch_threads": torch.get_num_threads(),
        "cpu_count": os.cpu_count(),
        "rounds": args.rounds,
        "has_cancel_check": "cancel_check"
        in inspect.signature(
            __import__("core.rag.conversation_archive", fromlist = ["x"]).archive_turns
        ).parameters,
    }
    # Warm the model so the first timed request is not paying the load.
    real_encode(["warmup"] * 4)
    ENCODED.update(texts = 0, calls = 0, after_stop = 0)

    # S1: cold first request, then the follow-ups a user would send, until nothing is left.
    fresh_home()
    persist("t1", msgs)
    reqs = []
    for _ in range(40):
        r = request("t1", msgs, live_n)
        reqs.append(r)
        if r["chunks_embedded"] == 0:
            break
    out["s1_requests"] = reqs
    out["s1_first_request_s"] = reqs[0]["seconds"]
    out["s1_total_chunks"] = sum(r["chunks_embedded"] for r in reqs)
    out["s1_total_groups"] = sum(r["written_groups"] for r in reqs)
    out["s1_requests_to_converge"] = len(reqs) - 1

    from core.rag import conversation_archive as ca

    hit = ca.recall("t1", "deploy password hint for service-7", branch_messages = msgs)
    text = hit[0] if hit else ""
    out["s1_recall_finds_fact"] = "MAGENTA-PELICAN-42" in (text or "")

    # S2: Chat Stop during the cold pass.
    fresh_home()
    persist("t2", msgs)
    STOP.clear()
    ENCODED.update(after_stop = 0)
    res = {}

    def run():
        res.update(request("t2", msgs, live_n, cancel_check = STOP.is_set))
        res["returned_at"] = time.monotonic()

    t0 = time.monotonic()
    th = threading.Thread(target = run)
    th.start()
    th.join(args.stop_after)
    stopped_at = None
    if th.is_alive():
        STOP.set()
        stopped_at = time.monotonic()
        th.join()
    out["s2"] = {
        **res,
        "stop_pressed_after_s": round(stopped_at - t0, 2) if stopped_at else None,
        "busy_after_stop_s": round(res["returned_at"] - stopped_at, 2) if stopped_at else 0.0,
        "chunks_embedded_after_stop": ENCODED["after_stop"],
    }
    out["s2"].pop("returned_at", None)
    STOP.clear()
    print("RESULT", json.dumps(out), flush = True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent = 1)


main()
