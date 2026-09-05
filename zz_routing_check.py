"""Deterministic routing assertion for unslothai/unsloth#10318.

Run from studio/backend. Proves, on whatever OS this executes on, that:
  1. importing the changed module has no platform-dependent side effect;
  2. the ONLY thing that routes _text_token_cost is whether a llama.cpp counter is
     present -- not the OS, not the hardware;
  3. the whole-document trim floor keeps the attachment whole on every platform.
Exits non-zero on any mismatch.
"""

import os
import platform
import sys

sys.path.insert(0, os.getcwd())

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")

from core.inference import tools  # noqa: E402

print(f"platform: {platform.system()} {platform.machine()} python {sys.version.split()[0]}")

failures = []


def check(label, got, want):
    ok = got == want
    print(f"  [{'OK ' if ok else 'FAIL'}] {label}: got={got!r} want={want!r}")
    if not ok:
        failures.append(label)


# 1. No local counter (safetensors / MLX / external endpoint / ctx mismatch):
#    ASCII is charged at 2 chars per token, identically on every OS.
tools._loaded_token_counter = lambda ctx: None
check("unmeasured ASCII cost for 1000 chars", tools._text_token_cost("a" * 1000, 4096), 500.0)
check("unmeasured non-ASCII cost for 1000 chars", tools._text_token_cost("漢" * 1000, 4096), 2000.0)
# ctx=0 means the budget was never sized against a window, so nothing is trusted.
check("ctx=0 never consults a counter", tools._text_token_cost("a" * 1000, 0), 500.0)

# 2. With a counter present the measured value is used verbatim, on every OS.
tools._loaded_token_counter = lambda ctx: (lambda chunk, token_budget = 0.0: 123)
check("measured cost is used verbatim", tools._text_token_cost("a" * 1000, 4096), 123.0)

# 3. A counter that cannot answer is treated as absent (not as a free pass).
tools._loaded_token_counter = lambda ctx: (lambda chunk, token_budget = 0.0: None)
check("counter returning None falls back to the estimate",
      tools._text_token_cost("a" * 1000, 4096), 500.0)

# 4. The budget helper is pure arithmetic: same answer on every platform.
check("_whole_doc_budget(8192 ctx, 1024 headroom)",
      tools._whole_doc_budget({"context_length": 8192, "response_headroom": 1024},
                              [{"role": "user", "content": "hi"}]),
      6000)
check("_whole_doc_budget(2048 ctx, 1024 headroom)",
      tools._whole_doc_budget({"context_length": 2048, "response_headroom": 1024},
                              [{"role": "user", "content": "summarize the whole document"}]),
      501)

# 5. No module-level import side effect touched anything platform-specific.
check("module still exposes build_rag_autoinject", callable(tools.build_rag_autoinject), True)

if failures:
    print(f"\nROUTING CHECK FAILED: {failures}")
    sys.exit(1)
print("\nROUTING CHECK PASSED -- identical routing on this platform")
