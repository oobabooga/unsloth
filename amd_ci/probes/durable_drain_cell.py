#!/usr/bin/env python3
"""One durable-run drain cell: the real ChatGenerationSupervisor._produce over a mocked upstream stream.

Run with cwd = a checkout's studio/backend and UNSLOTH_STUDIO_HOME = a fresh directory. Times every REAL
append_events call (the SQLite commit the host actually does), optionally adds a fixed delay before it, and
writes one JSON object to --out. Observes only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--latency-ms", type = float, default = 0.0)
    ap.add_argument("--chunks", type = int, default = 2000)
    ap.add_argument("--mode", choices = ["burst", "paced"], default = "paced")
    ap.add_argument("--rate", type = float, default = 125.0)
    ap.add_argument("--prefill-events", type = int, default = 0)
    args = ap.parse_args()

    sys.path.insert(0, os.getcwd())
    os.environ.setdefault("UNSLOTH_DISABLE_SANDBOX_WARMUP", "1")

    from core.inference.chat_generation_runs import ChatGenerationSupervisor
    from routes import inference
    from state import active_generations
    from storage import chat_generation_runs_db as runs_db
    from storage import studio_db

    studio_db.open_wal_keeper()
    for tid, uid, rid in (("thread-old", "user-old", "run-old"), ("thread-1", "user-1", "run-1")):
        studio_db.upsert_chat_thread(
            {"id": tid, "title": "Chat", "modelType": "base", "modelId": "local.gguf", "createdAt": 1}
        )
        studio_db.upsert_chat_message(
            {"id": uid, "threadId": tid, "role": "user", "content": [{"type": "text", "text": "Hi"}], "createdAt": 2}
        )
        runs_db.create_run(
            run_id = rid,
            owner_subject = "alice",
            thread_id = tid,
            user_message_id = uid,
            assistant_message_id = "assistant-" + rid,
            request_payload = {
                "model": "local.gguf",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
                "cancel_id": rid,
                "thread_id": tid,
                "generation_run_id": rid,
            },
        )
        if rid == "run-old" and args.prefill_events:
            # History from earlier chats in the same studio.db, as a long-lived install has.
            conn = studio_db.get_connection()
            payload = json.dumps({"choices": [{"delta": {"content": "old token "}}]})
            start = conn.execute("SELECT last_event_seq FROM chat_generation_runs WHERE id=?", (rid,)).fetchone()[0]
            conn.executemany(
                "INSERT INTO chat_generation_events (run_id, seq, event_type, payload_json, created_at)"
                " VALUES (?, ?, 'chunk', ?, 1)",
                ((rid, start + i + 1, payload) for i in range(args.prefill_events)),
            )
            conn.execute(
                "UPDATE chat_generation_runs SET last_event_seq=? WHERE id=?", (start + args.prefill_events, rid)
            )
            conn.commit()
            conn.close()
    active_generations.reset_for_tests()

    delay_s = args.latency_ms / 1000.0
    real_append = runs_db.append_events
    calls: list[int] = []
    commit_ms: list[float] = []

    def timed_append(run_id, worker_token, events):
        batch = list(events)
        if delay_s > 0:
            time.sleep(delay_s)
        t = time.perf_counter()
        try:
            return real_append(run_id, worker_token, batch)
        finally:
            commit_ms.append((time.perf_counter() - t) * 1000.0)
            calls.append(len(batch))

    runs_db.append_events = timed_append

    chunks = [{"choices": [{"delta": {"content": f"t{i} "}, "finish_reason": None}]} for i in range(args.chunks)]
    final = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
    marks: dict[str, float] = {}

    async def body():
        interval = 1.0 / args.rate
        start = time.perf_counter()
        for i, chunk in enumerate(chunks):
            if args.mode == "paced":
                delay = start + i * interval - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
            yield f"data: {json.dumps(chunk)}\n\n"
        yield f"data: {json.dumps(final)}\n\n"
        marks["drained"] = time.perf_counter()
        yield "data: [DONE]\n\n"

    async def fake(*_args, **_kwargs):
        return SimpleNamespace(status_code = 200, body_iterator = body())

    inference.produce_openai_chat_completions = fake

    async def run():
        supervisor = ChatGenerationSupervisor(SimpleNamespace(state = SimpleNamespace()))
        t0 = time.perf_counter()
        await supervisor._produce("run-1")
        return t0, time.perf_counter()

    t0, t1 = asyncio.run(run())
    run_row = runs_db.get_run("run-1", "alice")
    stored = [e["payload"] for e in runs_db.list_events("run-1", limit = 10**9) if e["type"] == "chunk"]
    obs = {
        "mode": args.mode,
        "latency_ms": args.latency_ms,
        "prefill_events": args.prefill_events,
        "chunks": args.chunks,
        "status": run_row["status"],
        "persisted_in_order": stored == chunks + [final],
        "upstream_drained_s": round(marks["drained"] - t0, 3),
        "run_completed_s": round(t1 - t0, 3),
        "ideal_upstream_s": (args.chunks / args.rate) if args.mode == "paced" else 0.0,
        "append_calls": len(calls),
        "events_per_call_median": statistics.median(calls) if calls else 0,
        "commit_ms_p50": _pct(commit_ms, 0.50),
        "commit_ms_p95": _pct(commit_ms, 0.95),
        "commit_ms_p99": _pct(commit_ms, 0.99),
        "commit_ms_max": max(commit_ms) if commit_ms else None,
        "commits_over_one_frame": sum(1 for m in commit_ms if m >= 1000.0 / 60.0),
    }
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
